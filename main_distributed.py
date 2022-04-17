# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import submitit
import argparse
import logging
import pprint
import yaml
import sys
import os

from src.msn_train import main as msn
from linear_eval import main as linear_eval

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--linear-eval', action='store_true',
    help='whether to run linear evaluation or pre-training')
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='/checkpoint/submitit/')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs to per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')


class Trainer:

    def __init__(self, fname='configs.yaml', lin_eval=False, load_model=None):
        self.lin_eval = lin_eval
        self.fname = fname
        self.load_model = load_model

    def __call__(self):
        lin_eval = self.lin_eval
        fname = self.fname
        load_model = self.load_model
        logger.info(f'called-params {fname} {lin_eval} {load_model}')

        # -- load script params
        params = None
        with open(fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            if load_model is not None:
                params['meta']['load_checkpoint'] = load_model
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

        _sel = 'lineval' if lin_eval else 'pretrain'
        dump = os.path.join(params['logging']['folder'], f'params-{_sel}.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

        if lin_eval:
            logger.info('Running linear-evaluation')
            return linear_eval(params)
        else:
            logger.info('Running pre-training')
            return msn(params)

    def checkpoint(self):
        fb_trainer = Trainer(self.fname, self.lin_eval, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch():
    executor = submitit.AutoExecutor(folder=args.folder)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu='55G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node)

    config_file = args.fname

    jobs, trainers = [], []
    with executor.batch():
        fb_trainer = Trainer(config_file, args.linear_eval)
        job = executor.submit(fb_trainer,)
        trainers.append(fb_trainer)
        jobs.append(job)

    for job in jobs:
        print(job.job_id)


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
