#!/usr/bin/env python
"""
    MLflow-integrated single-GPU training for Extractive Summarization with real-time metrics
"""
import argparse
import os
import random
import time
import mlflow
import mlflow.pytorch
import torch
from pytorch_pretrained_bert import BertConfig

from models import model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer
from models.trainer import build_trainer
from others.logging import logger, init_logger
from others.utils import test_rouge

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers',
               'encoder', 'ff_actv', 'use_interval', 'rnn_size']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def validate_and_log_metrics(args, model_path, step):
    logger.info("Running evaluation...")
    result = test_rouge(args, model_path, step)
    logger.info("Evaluation metrics: %s", result)
    for k, v in result.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            continue

def train_with_mlflow(args):
    init_logger()
    device = "cuda" if args.visible_gpus != '-1' and torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Using device: {device}")

    mlflow.set_experiment("bert_extractive_summarization")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.set_device(int(args.visible_gpus))
            torch.cuda.manual_seed_all(args.seed)

        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage) if args.train_from else None

        config = BertConfig.from_json_file(args.bert_config_path)
        model = Summarizer(args, device, checkpoint, bert_config=config)
        logger.info(model)

        if args.sep_optim:
            optim_bert, optim_dec = model_builder.build_optim_bert(args, model, checkpoint)
            optim = [optim_bert, optim_dec]
        else:
            optim = model_builder.build_optim(args, model, checkpoint)

        train_dataloader = load_dataset(args, 'train', shuffle=True)
        valid_dataloader = load_dataset(args, 'valid', shuffle=False)

        trainer = build_trainer(args, device_id=0, model=model, optim=optim)

        for step in range(0, args.train_steps, args.report_every):
            print("Report every:",args.report_every)
            trainer.train(train_iter_fct=train_dataloader, valid_iter_fct=valid_dataloader, train_steps=int(args.report_every))
            mlflow.log_metric("train_loss", trainer.stats.xent(), step)
            if args.log_rouge_every and (step % args.log_rouge_every == 0 or step + args.report_every >= args.train_steps):
                model_path = os.path.join(args.model_path, f"model_step_{step+args.report_every}.pt")
                torch.save(model.state_dict(), model_path)
                validate_and_log_metrics(args, model_path, step + args.report_every)

        if args.save_checkpoint:
            model_path = os.path.join(args.model_path, "model_final.pt")
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            mlflow.pytorch.log_model(model, "model")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext')
    parser.add_argument("-mode", default='train')
    parser.add_argument("-encoder", default='bert')
    parser.add_argument("-bert_data_path", default='../bert_data/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-sep_optim", type=str2bool, default=True)
    parser.add_argument("-lr_bert", type=float, default=0.002)
    parser.add_argument("-lr_dec", type=float, default=0.2)
    parser.add_argument("-use_interval", type=str2bool, default=True)
    parser.add_argument("-visible_gpus", default='0')
    parser.add_argument("-log_rouge_every", type=int, default=1000)
    parser.add_argument("-save_checkpoint", type=str2bool, default=True)
    parser.add_argument("-train_steps", type=int, default=5000)
    parser.add_argument("-batch_size", type=int, default=3000)
    parser.add_argument("-bert_config_path", default='bert_config.json')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-seed", type=int, default=666)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='', type=str)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-dataset', default='')

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)
    train_with_mlflow(args)

if __name__ == "__main__":
    main()
