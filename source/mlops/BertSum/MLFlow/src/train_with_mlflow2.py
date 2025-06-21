#!/usr/bin/env python
"""
    Main training workflow with MLflow logging
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch
import mlflow
import mlflow.pytorch
from pytorch_pretrained_bert import BertConfig

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer
from models.trainer import build_trainer
from others.logging import logger, init_logger

# Flags in the saved model config
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder',
               'ff_actv', 'use_interval', 'rnn_size']


def validate(args, device_id, pt, step):
    """ Validate on the validation set and log metric to MLflow """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt:
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

    # rebuild model
    model_opt = checkpoint['opt']
    for k in model_flags:
        if k in model_opt:
            setattr(args, k, model_opt[k])
    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device_id, load_pretrained_bert=False, bert_config=config)
    model.load_cp(checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(
        args, load_dataset(args, 'valid', shuffle=False),
        args.batch_size, "cpu" if args.visible_gpus == '-1' else "cuda",
        shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)

    # Log validation loss to MLflow
    if mlflow.active_run():
        mlflow.log_metric('valid_xent', stats.xent(), step=step)
    return stats.xent()


def test(args, device_id, pt, step):
    """ Test mode, generate summaries """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt:
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

    # rebuild model
    model_opt = checkpoint['opt']
    for k in model_flags:
        if k in model_opt:
            setattr(args, k, model_opt[k])
    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device_id, load_pretrained_bert=False, bert_config=config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(
        args, load_dataset(args, 'test', shuffle=False),
        args.batch_size, device,
        shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)


def wait_and_validate(args, device_id):
    """ Perform validation over checkpoints and log to MLflow """
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    xent_lst = []
    for i, cp in enumerate(cp_files):
        step = int(cp.split('.')[-2].split('_')[-1])
        xent = validate(args, device_id, cp, step)
        xent_lst.append((xent, cp))
        # stop when validation loss stops improving
        best = min(xent_lst, key=lambda x: x[0])
        if i - xent_lst.index(best) > 10:
            break
    # log best checkpoint
    best_steps = sorted(xent_lst, key=lambda x: x[0])[:3]
    logger.info('Best PPL %s' % str(best_steps))
    return


def train(args, device_id):
    """ Main training loop with MLflow logging """
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    # Log parameters to MLflow (only on rank 0)
    if mlflow.active_run() and device_id == 0:
        mlflow.log_params({k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))})

    # Build model and optimizer
    config = BertConfig.from_json_file(args.bert_config_path)
    if args.train_from:
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        model = Summarizer(args, device_id, load_pretrained_bert=False, bert_config=config)
        model.load_cp(checkpoint)
        optim = model_builder.build_optim(args, model, checkpoint)
    else:
        model = Summarizer(args, device_id, load_pretrained_bert=True, bert_config=config)
        optim = model_builder.build_optim(args, model, None)

    trainer = build_trainer(args, device_id, model, optim)

    # Begin training
    logger.info('Starting training...')
    trainer.train(lambda: data_loader.Dataloader(
        args, load_dataset(args, 'train', shuffle=True),
        args.batch_size, device,
        shuffle=True, is_test=False),
        args.train_steps)

    # Log final model artifact
    if mlflow.active_run() and device_id == 0:
        # Use mlflow.pytorch to save the model
        model_to_log = trainer.model.module if hasattr(trainer, 'model') and hasattr(trainer.model, 'module') else trainer.model
        mlflow.pytorch.log_model(model_to_log, artifact_path="model")


def multi_main(args):
    """ Spawns one process per GPU """
    init_logger(args.log_file)
    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    procs = []
    for i in range(nb_gpu):
        device_id = i
        p = mp.Process(target=run, args=(args, device_id, error_queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ Wrapper for distributed execution """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])
    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        logger.info('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("Distributed initialization error")
        train(args, device_id)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune BERT for Extractive Summarization with MLflow')
    parser.add_argument('-bert_config_path', help="Path to BERT config json", required=True)
    parser.add_argument('-mode', choices=['train', 'validate', 'lead', 'oracle', 'test'], default='train')
    parser.add_argument('-model_path', default='models/', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-max_pos', default=512, type=int)
    parser.add_argument('-use_interval', action='store_true')
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-seed', default=666, type=int)
    parser.add_argument('-train_steps', default=1000, type=int)
    parser.add_argument('-report_every', default=1, type=int)
    parser.add_argument('-save_checkpoint_steps', default=5, type=int)
    parser.add_argument('-world_size', default=1, type=int)
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-test_from', default='', type=str)
    parser.add_argument('-test_all', action='store_true')
    parser.add_argument('-report_rouge', action='store_true')
    parser.add_argument('-block_trigram', action='store_true')
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-lr', default=1, type=float)
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.999, type=float)
    parser.add_argument('-decay_method', default='', type=str)
    parser.add_argument('-warmup_steps', default=8000, type=int)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-recall_eval', action='store_true')
    parser.add_argument('-mlflow_experiment', default='bert_summarization', type=str,
                        help='Name of the MLflow experiment')
    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]

    # Set MLflow experiment
    mlflow.set_experiment(args.mlflow_experiment)
    # Enable automatic logging for PyTorch
    mlflow.pytorch.autolog()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    # Start MLflow run
    with mlflow.start_run():
        if args.world_size > 1:
            multi_main(args)
        elif args.mode == 'train':
            train(args, device_id)
        elif args.mode == 'validate':
            wait_and_validate(args, device_id)
        elif args.mode == 'lead':
            baseline(args, cal_lead=True)
        elif args.mode == 'oracle':
            baseline(args, cal_oracle=True)
        elif args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test(args, device_id, cp, step)
