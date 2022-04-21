# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import logging
import pprint

import numpy as np
import torch
import torchvision.transforms as transforms
import cyanure as cyan

import src.deit as deit
from src.data_manager import (
    init_data,
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lambd', type=float,
    default=0.00025,
    help='regularization')
parser.add_argument(
    '--penalty', type=str,
    help='regularization for logistic classifier',
    default='l2',
    choices=[
        'l2',
        'elastic-net'
    ])
parser.add_argument(
    '--mask', type=float,
    default=0.0,
    help='regularization')
parser.add_argument(
    '--preload', action='store_true',
    help='whether to preload embs if possible')
parser.add_argument(
    '--fname', type=str,
    help='model architecture')
parser.add_argument(
    '--model-name', type=str,
    help='model architecture')
parser.add_argument(
    '--pretrained', type=str,
    help='path to pretrained model',
    default='')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--normalize', type=bool,
    default=True,
    help='whether to standardize images before feeding to nework')
parser.add_argument(
    '--root-path', type=str,
    default='/datasets/',
    help='root directory to data')
parser.add_argument(
    '--image-folder', type=str,
    default='imagenet_full_size/061417/',
    help='image directory inside root_path')
parser.add_argument(
    '--subset-path', type=str,
    default=None,
    help='name of dataset to evaluate on')

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(
    blocks,
    lambd,
    mask_frac,
    preload,
    pretrained,
    fname,
    subset_path,
    root_path,
    image_folder,
    penalty='l2',
    model_name=None,
    normalize=True,
    device_str='cuda:0'
):
    device = torch.device(device_str)
    if 'cuda' in device_str:
        torch.cuda.set_device(device)

    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    subset_tag = '-'.join(subset_path.split('/')).split('.txt')[0] if subset_path is not None else 'imagenet_subses1-100percent'
    train_embs_path = os.path.join(pretrained, f'train-features-{subset_tag}-{fname}')
    test_embs_path = os.path.join(pretrained, f'val-features-{fname}')
    logger.info(train_embs_path)
    logger.info(test_embs_path)

    pretrained = os.path.join(pretrained, fname)

    # -- Function to make train/test dataloader
    def init_pipe(training):
        # -- make data transforms
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        # -- init data-loaders/samplers
        subset_file = subset_path if training else None
        data_loader, _ = init_data(
            transform=transform,
            batch_size=16,
            num_workers=0,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=False,
            drop_last=False,
            subset_file=subset_file)
        return data_loader

    # -- Initialize the model
    encoder = init_model(
        device=device,
        pretrained=pretrained,
        model_name=model_name)
    encoder.eval()

    # -- If train embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(train_embs_path):
        checkpoint = torch.load(train_embs_path, map_location='cpu')
        embs, labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded embs of shape {embs.shape}')
    else:
        data_loader = init_pipe(True)
        embs, labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=mask_frac,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': embs,
            'labs': labs
        }, train_embs_path)
        logger.info(f'saved train embs of shape {embs.shape}')
    # -- Normalize embeddings
    cyan.preprocess(embs, normalize=normalize, columns=False, centering=True)

    # -- Fit Logistic Regression Classifier
    classifier = cyan.MultiClassifier(loss='multiclass-logistic', penalty=penalty, fit_intercept=False)
    lambd /= len(embs)
    classifier.fit(
        embs.numpy(),
        labs.numpy(),
        it0=10,
        lambd=lambd,
        lambd2=lambd,
        nthreads=-1,
        tol=1e-3,
        solver='auto',
        seed=0,
        max_epochs=300)

    # -- Evaluate and log
    train_score = classifier.score(embs.numpy(), labs.numpy())
    # -- (save train score)
    logger.info(f'train score: {train_score}')

    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(test_embs_path):
        checkpoint = torch.load(test_embs_path, map_location='cpu')
        test_embs, test_labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded test embs of shape {test_embs.shape}')
    else:
        data_loader = init_pipe(False)
        test_embs, test_labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=0.0,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': test_embs,
            'labs': test_labs
        }, test_embs_path)
        logger.info(f'saved test embs of shape {test_embs.shape}')
    # -- Normalize embeddings
    cyan.preprocess(test_embs, normalize=normalize, columns=False, centering=True)

    # -- Evaluate and log
    test_score = classifier.score(test_embs.numpy(), test_labs.numpy())
    # -- (save test score)
    logger.info(f'test score: {test_score}\n\n')

    return test_score


def make_embeddings(
    blocks,
    device,
    mask_frac,
    data_loader,
    encoder,
    epochs=1
):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for _ in range(epochs):
        for itr, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            with torch.no_grad():
                z = encoder.forward_blocks(imgs, blocks, mask_frac).cpu()
            labels = labels.cpu()
            z_mem.append(z)
            l_mem.append(labels)
            if itr % 50 == 0:
                logger.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)
    logger.info(z_mem.shape)
    logger.info(l_mem.shape)

    return z_mem, l_mem


def load_pretrained(
    encoder,
    pretrained
):
    checkpoint = torch.load(pretrained, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    try:
        logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                    f'path: {pretrained}')
    except Exception:
        pass
    del checkpoint
    return encoder


def init_model(
    device,
    pretrained,
    model_name,
):
    encoder = deit.__dict__[model_name]()
    encoder.fc = None
    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained)

    return encoder


if __name__ == '__main__':
    """'main' for launching script using params read from command line"""
    global args
    args = parser.parse_args()
    pp.pprint(args)
    main(
        blocks=1,
        lambd=args.lambd,
        penalty=args.penalty,
        mask_frac=args.mask,
        preload=args.preload,
        pretrained=args.pretrained,
        fname=args.fname,
        subset_path=args.subset_path,
        root_path=args.root_path,
        image_folder=args.image_folder,
        model_name=args.model_name,
        normalize=args.normalize,
        device_str=args.device
    )
