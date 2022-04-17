# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
from collections import OrderedDict

import numpy as np

import torch
import torch.multiprocessing as mp

import src.deit as deit
from src.utils import (
    AllReduceSum,
    trunc_normal_,
    gpu_timer,
    init_distributed,
    WarmupCosineSchedule,
    CosineWDSchedule,
    CSVLogger,
    grad_logger,
    AverageMeter
)
from src.losses import init_msn_loss
from src.data_manager import (
    init_data,
    make_transforms
)

from torch.nn.parallel import DistributedDataParallel

# --
log_timings = True
log_freq = 10
checkpoint_freq = 25
checkpoint_freq_itr = 2500
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    model_name = args['meta']['model_name']
    two_layer = False if 'two_layer' not in args['meta'] else args['meta']['two_layer']
    bottleneck = 1 if 'bottleneck' not in args['meta'] else args['meta']['bottleneck']
    output_dim = args['meta']['output_dim']
    hidden_dim = args['meta']['hidden_dim']
    load_model = args['meta']['load_checkpoint']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    use_pred_head = args['meta']['use_pred_head']
    use_bn = args['meta']['use_bn']
    drop_path_rate = args['meta']['drop_path_rate']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- CRITERTION
    memax_weight = 1 if 'memax_weight' not in args['criterion'] else args['criterion']['memax_weight']
    ent_weight = 1 if 'ent_weight' not in args['criterion'] else args['criterion']['ent_weight']
    freeze_proto = False if 'freeze_proto' not in args['criterion'] else args['criterion']['freeze_proto']
    use_ent = False if 'use_ent' not in args['criterion'] else args['criterion']['use_ent']
    reg = args['criterion']['me_max']
    use_sinkhorn = args['criterion']['use_sinkhorn']
    num_proto = args['criterion']['num_proto']
    # --
    batch_size = args['criterion']['batch_size']
    temperature = args['criterion']['temperature']
    _start_T = args['criterion']['start_sharpen']
    _final_T = args['criterion']['final_sharpen']

    # -- DATA
    label_smoothing = args['data']['label_smoothing']
    pin_mem = False if 'pin_mem' not in args['data'] else args['data']['pin_mem']
    num_workers = 1 if 'num_workers' not in args['data'] else args['data']['num_workers']
    color_jitter = args['data']['color_jitter_strength']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    patch_drop = args['data']['patch_drop']
    rand_size = args['data']['rand_size']
    rand_views = args['data']['rand_views']
    focal_views = args['data']['focal_views']
    focal_size = args['data']['focal_size']
    # --

    # -- OPTIMIZATION
    clip_grad = args['optimization']['clip_grad']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    # if rank > 0:
    #     logger.setLevel(logging.ERROR)

    # -- proto details
    assert num_proto > 0, 'unsupervised pre-training requires specifying prototypes'

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'msn'),
                           ('%.5f', 'me_max'),
                           ('%.5f', 'ent'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder = init_model(
        device=device,
        model_name=model_name,
        two_layer=two_layer,
        use_pred=use_pred_head,
        use_bn=use_bn,
        bottleneck=bottleneck,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        drop_path_rate=drop_path_rate)
    target_encoder = copy.deepcopy(encoder)
    if (world_size > 1):
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        target_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(target_encoder)

    # -- init losses
    msn = init_msn_loss(
        num_views=focal_views+rand_views,
        tau=temperature,
        me_max=reg,
        return_preds=True)

    def one_hot(targets, num_classes, smoothing=label_smoothing):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        targets = targets.long().view(-1, 1).to(device)
        return torch.full((len(targets), num_classes), off_value, device=device).scatter_(1, targets, on_value)

    # -- make data transforms
    transform = make_transforms(
        rand_size=rand_size,
        focal_size=focal_size,
        rand_views=rand_views+1,
        focal_views=focal_views,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         transform=transform,
         batch_size=batch_size,
         pin_mem=pin_mem,
         num_workers=num_workers,
         world_size=world_size,
         rank=rank,
         root_path=root_path,
         image_folder=image_folder,
         training=True,
         copy_data=copy_data)
    ipe = len(unsupervised_loader)
    logger.info(f'iterations per epoch: {ipe}')

    # -- make prototypes
    prototypes, proto_labels = None, None
    if num_proto > 0:
        with torch.no_grad():
            prototypes = torch.empty(num_proto, output_dim)
            _sqrt_k = (1./output_dim)**0.5
            torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
            prototypes = torch.nn.parameter.Parameter(prototypes).to(device)

            # -- init prototype labels
            proto_labels = one_hot(torch.tensor([i for i in range(num_proto)]), num_proto)

        if not freeze_proto:
            prototypes.requires_grad = True
        logger.info(f'Created prototypes: {prototypes.shape}')
        logger.info(f'Requires grad: {prototypes.requires_grad}')

    # -- init optimizer and scheduler
    encoder, optimizer, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        prototypes=prototypes,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs)
    if world_size > 1:
        encoder = DistributedDataParallel(encoder)
        target_encoder = DistributedDataParallel(target_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

    # -- momentum schedule
    _start_m, _final_m = 0.996, 1.0
    _increment = (_final_m - _start_m) / (ipe * num_epochs * 1.25)
    momentum_scheduler = (_start_m + (_increment*i) for i in range(int(ipe*num_epochs*1.25)+1))

    # -- sharpening schedule
    _increment_T = (_final_T - _start_T) / (ipe * num_epochs * 1.25)
    sharpen_scheduler = (_start_T + (_increment_T*i) for i in range(int(ipe*num_epochs*1.25)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, target_encoder, prototypes, optimizer, start_epoch = load_checkpoint(
            device=device,
            prototypes=prototypes,
            r_path=load_path,
            encoder=encoder,
            target_encoder=target_encoder,
            opt=optimizer)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            next(sharpen_scheduler)

    def save_checkpoint(epoch):

        if target_encoder is not None:
            target_encoder_state_dict = target_encoder.state_dict()
        else:
            target_encoder_state_dict = None

        save_dict = {
            'encoder': encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'prototypes': prototypes.data,
            'target_encoder': target_encoder_state_dict,
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'temperature': temperature
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0 \
                    or (epoch + 1) % 10 == 0 and epoch < checkpoint_freq:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        ploss_meter = AverageMeter()
        rloss_meter = AverageMeter()
        eloss_meter = AverageMeter()
        np_meter = AverageMeter()
        maxp_meter = AverageMeter()
        time_meter = AverageMeter()
        data_meter = AverageMeter()

        for itr, (udata, _) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = [u.to(device, non_blocking=True) for u in udata]
                return imgs
            imgs, dtime = gpu_timer(load_imgs)
            data_meter.update(dtime)

            def train_step():
                optimizer.zero_grad()

                # --
                # h: representations of 'imgs' before head
                # z: representations of 'imgs' after head
                # -- If use_pred_head=False, then encoder.pred (prediction
                #    head) is None, and _forward_head just returns the
                #    identity, z=h
                h, z = encoder(imgs[1:], return_before_head=True, patch_drop=patch_drop)
                with torch.no_grad():
                    h, _ = target_encoder(imgs[0], return_before_head=True)

                # Step 1. convert representations to fp32
                h, z = h.float(), z.float()

                # Step 2. determine anchor views/supports and their
                #         corresponding target views/supports
                # --
                anchor_views, target_views = z, h.detach()
                T = next(sharpen_scheduler)

                # Step 3. compute msn loss with me-max regularization
                (ploss, me_max, ent, logs, _) = msn(
                    T=T,
                    use_sinkhorn=use_sinkhorn,
                    use_entropy=use_ent,
                    anchor_views=anchor_views,
                    target_views=target_views,
                    proto_labels=proto_labels,
                    prototypes=prototypes)
                loss = ploss + memax_weight*me_max + ent_weight*ent

                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                # Step 4. Optimization step
                loss.backward()
                with torch.no_grad():
                    prototypes.grad.data = AllReduceSum.apply(prototypes.grad.data)
                grad_stats = grad_logger(encoder.named_parameters())
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                optimizer.step()

                # Step 5. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), float(ploss), float(me_max), float(ent),
                        logs, _new_lr, _new_wd, grad_stats)
            (loss, ploss, rloss, eloss,
             _logs, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            ploss_meter.update(ploss)
            rloss_meter.update(rloss)
            eloss_meter.update(eloss)
            maxp_meter.update(_logs['max_t'])
            np_meter.update(_logs['np'])

            time_meter.update(etime)

            # -- Save Checkpoint
            if itr % checkpoint_freq_itr == 0:
                save_checkpoint(epoch)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, ploss, rloss, eloss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f (%.3f %.3f %.3f) '
                                '(np: %.1f, max-t: %.3f) '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%d ms; %d ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   ploss_meter.avg,
                                   rloss_meter.avg,
                                   eloss_meter.avg,
                                   np_meter.avg,
                                   maxp_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg,
                                   data_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


def load_checkpoint(
    device,
    r_path,
    prototypes,
    encoder,
    target_encoder,
    opt
):
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    if ('scaling_module.bias' not in pretrained_dict) and ('scaling_bias' in pretrained_dict):
        pretrained_dict['scaling_module.bias'] = pretrained_dict['scaling_bias']
        del pretrained_dict['scaling_bias']
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint['target_encoder']
        if ('scaling_module.bias' not in pretrained_dict) and ('scaling_bias' in pretrained_dict):
            pretrained_dict['scaling_module.bias'] = pretrained_dict['scaling_bias']
            del pretrained_dict['scaling_bias']
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading prototypes
    if (prototypes is not None) and ('prototypes' in checkpoint):
        with torch.no_grad():
            prototypes.data = checkpoint['prototypes'].to(device)
        logger.info(f'loaded prototypes from epoch {epoch}')

    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, target_encoder, prototypes, opt, epoch


def init_model(
    device,
    model_name='resnet50',
    use_pred=False,
    use_bn=False,
    two_layer=False,
    bottleneck=1,
    hidden_dim=2048,
    output_dim=128,
    drop_path_rate=0.1,
):
    encoder = deit.__dict__[model_name](drop_path_rate=drop_path_rate)
    emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280

    # -- projection head
    encoder.fc = None
    fc = OrderedDict([])
    fc['fc1'] = torch.nn.Linear(emb_dim, hidden_dim)
    if use_bn:
        fc['bn1'] = torch.nn.BatchNorm1d(hidden_dim)
    fc['gelu1'] = torch.nn.GELU()
    fc['fc2'] = torch.nn.Linear(hidden_dim, hidden_dim)
    if use_bn:
        fc['bn2'] = torch.nn.BatchNorm1d(hidden_dim)
    fc['gelu2'] = torch.nn.GELU()
    fc['fc3'] = torch.nn.Linear(hidden_dim, output_dim)
    encoder.fc = torch.nn.Sequential(fc)

    for m in encoder.modules():
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    encoder.to(device)
    logger.info(encoder)
    return encoder


def init_opt(
    encoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    prototypes=None,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0
):
    param_groups = [
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
         'WD_exclude': True,
         'weight_decay': 0}
    ]
    if prototypes is not None:
        param_groups.append({
            'params': [prototypes],
            'lr': ref_lr,
            'LARS_exclude': True,
            'WD_exclude': True,
            'weight_decay': 0
        })

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(1.25*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(1.25*num_epochs*iterations_per_epoch))
    return encoder, optimizer, scheduler, wd_scheduler


if __name__ == "__main__":
    main()
