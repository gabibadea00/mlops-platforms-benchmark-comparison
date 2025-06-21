from metaflow import FlowSpec, step, Parameter, card, current, pypi_base
from metaflow.cards import Markdown, ProgressBar
import os
import sys
import random
import torch
from pytorch_pretrained_bert import BertConfig
from types import SimpleNamespace

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import BertSum components
from models.data_loader import load_dataset, Dataloader
from models.model_builder import Summarizer, build_optim
from models.trainer import build_trainer
from others.logging import init_logger

class BertSumFlow(FlowSpec):
    """
    Metaflow pipeline for fine-tuning BERT extractive summarizer with live metrics.
    """
    encoder = Parameter(
        'encoder',
        default='transformer',
        help="Encoder type to use"
    )

    mode = Parameter(
        'mode',
        default='train',
        help="Mode of operation"
    )

    bert_data_path = Parameter('bert_data_path', default='../bert_data/cnndm')
    model_path = Parameter('model_path', default='../models/')
    result_path = Parameter('result_path', default='../results/cnndm')
    temp_dir = Parameter('temp_dir', default='../temp')
    bert_config_path = Parameter('bert_config_path', default='../bert_config_uncased_base.json')

    batch_size = Parameter('batch_size', default=1000, type=int)

    use_interval = True
    hidden_size = Parameter('hidden_size', default=128, type=int)
    ff_size = Parameter('ff_size', default=512, type=int)
    heads = Parameter('heads', default=4, type=int)
    inter_layers = Parameter('inter_layers', default=2, type=int)
    rnn_size = Parameter('rnn_size', default=512, type=int)

    param_init = Parameter('param_init', default=0.0, type=float)
    param_init_glorot = True
    dropout = Parameter('dropout', default=0.1, type=float)
    optim = Parameter('optim', default='adam')
    lr = Parameter('lr', default=1.0, type=float)
    beta1 = Parameter('beta1', default=0.9, type=float)
    beta2 = Parameter('beta2', default=0.999, type=float)
    decay_method = Parameter('decay_method', default='')
    warmup_steps = Parameter('warmup_steps', default=8000, type=int)
    max_grad_norm = Parameter('max_grad_norm', default=0.0, type=float)

    save_checkpoint_steps = Parameter('save_checkpoint_steps', default=50, type=int)
    accum_count = Parameter('accum_count', default=1, type=int)
    world_size = Parameter('world_size', default=1, type=int)
    report_every = Parameter('report_every', default=1, type=int)
    train_steps = Parameter('train_steps', default=1000, type=int)
    recall_eval = False

    visible_gpus = Parameter('visible_gpus', default='-1')
    gpu_ranks = Parameter('gpu_ranks', default='0')
    log_file = Parameter('log_file', default='../logs/cnndm.log')
    dataset = Parameter('dataset', default='')
    seed = Parameter('seed', default=666, type=int)

    test_all = False
    test_from = Parameter('test_from', default='')
    train_from = Parameter('train_from', default='')
    report_rouge = True
    block_trigram = True

    @step
    def start(self):
        """
        Prepare environment and directories.
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpus)
        for d in [self.model_path, self.result_path, self.temp_dir]:
            os.makedirs(d, exist_ok=True)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.next(self.train)

    @card(type='blank', refresh_interval=10)
    @step
    def train(self):
        """
        Fine-tune BERT with live-progress dynamic card.
        """
        # Initialize logger
        log_file = os.path.join(self.temp_dir, 'train.log')
        init_logger(log_file)

        # Device setup
        device = 'cpu' if self.visible_gpus == '-1' else 'cuda'
        device_id = 0 if device == 'cuda' else -1

        # Copy parameters onto args
        args = SimpleNamespace(**{
            attr: getattr(self, attr)
            for attr in ['bert_data_path','model_path','result_path','temp_dir', 'ff_size', 'heads', 'dropout', 'use_interval',
                        'bert_config_path','batch_size','lr','seed', 'encoder', 'inter_layers', 'rnn_size', 'hidden_size', 'optim',
                        'train_steps','save_checkpoint_steps', 'param_init', 'param_init_glorot', 'train_from', 'beta1', 'beta2',
                        'visible_gpus','world_size', 'report_every', 'warmup_steps', 'decay_method', 'mode', 'max_grad_norm', \
                        'accum_count','recall_eval','log_file', 'dataset', 'test_all', 'test_from', 'report_rouge', 'block_trigram']
        })
        args.gpu_ranks = [int(r) for r in self.gpu_ranks.split(',')]

        # Data loader function
        # train_iter_fct = lambda: Dataloader(
        #     args, load_dataset(args, 'train', shuffle=True),
        #     args.batch_size, device, shuffle=True, is_test=False
        # )

        def train_iter_fct():
            return Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                                 shuffle=True, is_test=False)

        # Model, optimizer, trainer
        model = Summarizer(args, device, load_pretrained_bert=True)
        optim = build_optim(args, model, None)
        trainer = build_trainer(args, device_id, model, optim)

        # Dynamic card setup
        m = Markdown("## Training Progress")
        p = ProgressBar(max=args.train_steps, label="Step")
        current.card.append(m)
        current.card.append(p)
        current.card.refresh()

        # Training loop with live metrics
        step_count = 0
        self.train_metrics = []
        while step_count < args.train_steps:
            for batch in train_iter_fct():
                step_count += 1
                stats = trainer.train(batch, args.train_steps)
                xent = stats.xent() if stats else None
                dps  = getattr(stats, 'docs_per_sec', None)
                self.train_metrics.append({
                    'step': step_count,
                    'xent': xent,
                    'docs_per_sec': dps
                })
                # … update your card …
                if step_count >= args.train_steps:
                    break
        self.step_count = step_count
        self.next(self.validate)

    @step
    def validate(self):
        """
        Run validation on the latest checkpoint.
        """
        # similar integration or fallback to subprocess
        self.next(self.test)

    @step
    def test(self):
        """
        Run test on the latest checkpoint.
        """
        self.next(self.end)

    @step
    def end(self):
        """
        Final reporting.
        """
        print(f"Training completed {self.step_count} steps.")
        print("Metrics captured in `train_metrics` attribute.")

if __name__ == '__main__':
    BertSumFlow()