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

import logging
import sys

import numpy as np

import torch
import torchvision.transforms as transforms

import src.deit as deit
from src.utils import (
    AllReduce,
    init_distributed,
    WarmupCosineSchedule
)
from src.data_manager import init_data
from src.sgd import SGD
from torch.nn.parallel import DistributedDataParallel

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # -- META
    model_name = args['meta']['model_name']
    port = args['meta']['master_port']
    load_checkpoint = args['meta']['load_checkpoint']
    training = args['meta']['training']
    copy_data = args['meta']['copy_data']
    device = torch.device(args['meta']['device'])
    if 'cuda' in args['meta']['device']:
        torch.cuda.set_device(device)

    # -- DATA
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    num_classes = args['data']['num_classes']

    # -- OPTIMIZATION
    wd = float(args['optimization']['weight_decay'])
    ref_lr = args['optimization']['lr']
    num_epochs = args['optimization']['epochs']
    num_blocks = args['optimization']['num_blocks']
    l2_normalize = args['optimization']['normalize']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    r_file_enc = args['logging']['pretrain_path']

    # -- log/checkpointing paths
    r_enc_path = os.path.join(folder, r_file_enc)
    w_enc_path = os.path.join(folder, f'{tag}-lin-eval.pth.tar')

    # -- init distributed
    world_size, rank = init_distributed(port)
    logger.info(f'initialized rank/world-size: {rank}/{world_size}')

    # -- optimization/evaluation params
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    if training:
        batch_size = 256
    else:
        batch_size = 128
        load_checkpoint = True
        num_epochs = 1

    # -- init loss
    criterion = torch.nn.CrossEntropyLoss()

    # -- make train data transforms and data loaders/samples
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])
    data_loader, dist_sampler = init_data(
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        training=training,
        copy_data=copy_data)

    ipe = len(data_loader)
    logger.info(f'initialized data-loader (ipe {ipe})')

    # -- make val data transforms and data loaders/samples
    val_data_loader, val_dist_sampler = init_data(
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        training=False,
        copy_data=copy_data)
    logger.info(f'initialized val data-loader (ipe {len(val_data_loader)})')

    # -- init model and optimizer
    encoder, linear_classifier, optimizer, scheduler = init_model(
        device=device,
        device_str=args['meta']['device'],
        num_classes=num_classes,
        num_blocks=num_blocks,
        normalize=l2_normalize,
        training=training,
        r_enc_path=r_enc_path,
        iterations_per_epoch=ipe,
        world_size=world_size,
        ref_lr=ref_lr,
        weight_decay=wd,
        num_epochs=num_epochs,
        model_name=model_name)
    logger.info(encoder)

    best_acc = None
    start_epoch = 0
    # -- load checkpoint
    if not training or load_checkpoint:
        encoder, linear_classifier, optimizer, scheduler, start_epoch, best_acc = load_from_path(
            r_path=w_enc_path,
            encoder=encoder,
            linear_classifier=linear_classifier,
            opt=optimizer,
            sched=scheduler,
            device_str=args['meta']['device'])
    if not training:
        logger.info('putting model in eval mode')
        encoder.eval()
        logger.info(sum(p.numel() for n, p in encoder.named_parameters()
                        if p.requires_grad and ('fc' not in n)))
        start_epoch = 0

    encoder.eval()

    for epoch in range(start_epoch, num_epochs):

        def train_step():
            # -- update distributed-data-loader epoch
            dist_sampler.set_epoch(epoch)
            top1_correct, top5_correct, total = 0, 0, 0
            for i, data in enumerate(data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    with torch.no_grad():
                        outputs = encoder.forward_blocks(inputs, num_blocks)
                outputs = linear_classifier(outputs)
                loss = criterion(outputs, labels)
                total += inputs.shape[0]
                top5_correct += float(outputs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum())
                top1_correct += float(outputs.max(dim=1).indices.eq(labels).sum())
                top1_acc = 100. * top1_correct / total
                top5_acc = 100. * top5_correct / total
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                if i % log_freq == 0:
                    logger.info('[%d, %5d] %.3f%% %.3f%% (loss: %.3f)'
                                % (epoch + 1, i, top1_acc, top5_acc, loss))
            return 100. * top1_correct / total

        def val_step():
            top1_correct, total = 0, 0
            for i, data in enumerate(val_data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = encoder.forward_blocks(inputs, num_blocks)
                outputs = linear_classifier(outputs)
                total += inputs.shape[0]
                top1_correct += outputs.max(dim=1).indices.eq(labels).sum()
                top1_acc = 100. * top1_correct / total

            top1_acc = AllReduce.apply(top1_acc)
            logger.info('[%d, %5d] %.3f%%' % (epoch + 1, i, top1_acc))
            return top1_acc

        train_top1 = 0.
        train_top1 = train_step()
        with torch.no_grad():
            val_top1 = val_step()

        log_str = 'train:' if training else 'test:'
        logger.info('[%d] (%s %.3f%%) (val: %.3f%%)'
                    % (epoch + 1, log_str, train_top1, val_top1))

        # -- logging/checkpointing
        if training and (rank == 0) and ((best_acc is None) or (best_acc < val_top1)):
            best_acc = val_top1
            save_dict = {
                'teacher': encoder.state_dict(),
                'classifier': linear_classifier.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch + 1,
                'world_size': world_size,
                'best_top1_acc': best_acc,
                'batch_size': batch_size,
                'lr': ref_lr,
            }
            torch.save(save_dict, w_enc_path)

    return train_top1, val_top1


class LinearClassifier(torch.nn.Module):

    def __init__(self, dim, num_labels=1000, normalize=True):
        super(LinearClassifier, self).__init__()
        self.normalize = normalize
        self.norm = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.norm(x)
        if self.normalize:
            x = torch.nn.functional.normalize(x)
        return self.linear(x)


def load_pretrained(
    r_path,
    encoder,
    linear_classifier,
    device_str
):
    checkpoint = torch.load(r_path, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                f'path: {r_path}')

    if linear_classifier is not None:
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['classifier'].items()}
        for k, v in linear_classifier.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = linear_classifier.load_state_dict(pretrained_dict, strict=False)
        logger.info(f'loaded pretrained model with msg: {msg}')
        logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                    f'path: {r_path}')

    del checkpoint
    return encoder, linear_classifier


def load_from_path(
    r_path,
    encoder,
    linear_classifier,
    opt,
    sched,
    device_str
):
    encoder, linear_classifier = load_pretrained(r_path, encoder, linear_classifier, device_str)
    checkpoint = torch.load(r_path, map_location=device_str)

    best_acc = None
    if 'best_top1_acc' in checkpoint:
        best_acc = checkpoint['best_top1_acc']

    epoch = checkpoint['epoch']
    if opt is not None:
        opt.load_state_dict(checkpoint['opt'])
        sched.load_state_dict(checkpoint['sched'])
        logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, opt, sched, epoch, best_acc


def init_model(
    device,
    device_str,
    num_classes,
    num_blocks,
    training,
    r_enc_path,
    iterations_per_epoch,
    world_size,
    ref_lr,
    num_epochs,
    normalize,
    model_name='resnet50',
    warmup_epochs=0,
    weight_decay=0
):
    # -- init model
    encoder = deit.__dict__[model_name]()
    emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280
    emb_dim *= num_blocks
    encoder.fc = None
    encoder.norm = None

    encoder.to(device)
    encoder, _ = load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        linear_classifier=None,
        device_str=device_str)

    linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)

    # -- init optimizer
    optimizer, scheduler = None, None
    param_groups = [
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
         'weight_decay': 0}
    ]
    optimizer = SGD(
        param_groups,
        nesterov=True,
        weight_decay=weight_decay,
        momentum=0.9,
        lr=ref_lr)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_epochs*iterations_per_epoch,
        start_lr=ref_lr,
        ref_lr=ref_lr,
        T_max=num_epochs*iterations_per_epoch)
    if world_size > 1:
        linear_classifier = DistributedDataParallel(linear_classifier)

    return encoder, linear_classifier, optimizer, scheduler


if __name__ == "__main__":
    main()
