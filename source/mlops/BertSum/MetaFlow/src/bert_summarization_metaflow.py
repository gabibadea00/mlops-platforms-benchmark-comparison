from metaflow import FlowSpec, step, Parameter, BoolParameter
from metaflow.card import Markdown, Card
import os
import sys
import subprocess
from train import train

# Ensure we can run train.py from the repo root
dir_path = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = dir_path
PYTHON_EXEC = sys.executable

class BertSumMetaflow(FlowSpec):
    """
    Metaflow pipeline to fine-tune and evaluate BERT summarization using BertSum's train.py.
    """
    encoder = Parameter(
        'encoder',
        default='classifier',
        choices=['classifier', 'transformer', 'rnn', 'baseline'],
        help="Encoder type to use"
    )

    mode = Parameter(
        'mode',
        default='train',
        choices=['train', 'validate', 'test'],
        help="Mode of operation"
    )

    bert_data_path = Parameter('bert_data_path', default='../bert_data/cnndm')
    model_path = Parameter('model_path', default='../models/')
    result_path = Parameter('result_path', default='../results/cnndm')
    temp_dir = Parameter('temp_dir', default='../temp')
    bert_config_path = Parameter('bert_config_path', default='../bert_config_uncased_base.json')

    batch_size = Parameter('batch_size', default=1000, type=int)

    use_interval = BoolParameter('use_interval', default=True)
    hidden_size = Parameter('hidden_size', default=128, type=int)
    ff_size = Parameter('ff_size', default=512, type=int)
    heads = Parameter('heads', default=4, type=int)
    inter_layers = Parameter('inter_layers', default=2, type=int)
    rnn_size = Parameter('rnn_size', default=512, type=int)

    param_init = Parameter('param_init', default=0.0, type=float)
    param_init_glorot = BoolParameter('param_init_glorot', default=True)
    dropout = Parameter('dropout', default=0.1, type=float)
    optim = Parameter('optim', default='adam')
    lr = Parameter('lr', default=1.0, type=float)
    beta1 = Parameter('beta1', default=0.9, type=float)
    beta2 = Parameter('beta2', default=0.999, type=float)
    decay_method = Parameter('decay_method', default='')
    warmup_steps = Parameter('warmup_steps', default=8000, type=int)
    max_grad_norm = Parameter('max_grad_norm', default=0.0, type=float)

    save_checkpoint_steps = Parameter('save_checkpoint_steps', default=5, type=int)
    accum_count = Parameter('accum_count', default=1, type=int)
    world_size = Parameter('world_size', default=1, type=int)
    report_every = Parameter('report_every', default=1, type=int)
    train_steps = Parameter('train_steps', default=1000, type=int)
    recall_eval = BoolParameter('recall_eval', default=False)

    visible_gpus = Parameter('visible_gpus', default='-1')
    gpu_ranks = Parameter('gpu_ranks', default='0')
    log_file = Parameter('log_file', default='../logs/cnndm.log')
    dataset = Parameter('dataset', default='')
    seed = Parameter('seed', default=666, type=int)

    test_all = BoolParameter('test_all', default=False)
    test_from = Parameter('test_from', default='')
    train_from = Parameter('train_from', default='')
    report_rouge = BoolParameter('report_rouge', default=True)
    block_trigram = BoolParameter('block_trigram', default=True)

    @step
    def start(self):
        """
        Prepare environment, create output directories.
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpus)
        # make sure directories exist
        for d in [self.model_path, self.result_path, self.temp_dir]:
            os.makedirs(d, exist_ok=True)
        # Log params
        self.params = dict(
            encoder=self.encoder,
            node=self.mode,
            bert_data_path=self.bert_data_path,
            model_path=self.model_path,
            result_path=self.result_path,
            temp_dir=self.temp_dir,
            bert_config_path=self.bert_config_path,
            batch_size=self.batch_size,
            use_interval=self.use_interval,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            heads=self.heads,
            inter_layers=self.inter_layers,
            rnn_size=self.rnn_size,
            param_init=self.param_init,
            param_init_glorot=self.param_init_glorot,
            dropout=self.dropout,
            optim=self.optim,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            decay_method=self.decay_method,
            warmup_steps=self.warmup_steps,
            max_grad_norm=self.max_grad_norm,
            save_checkpoint_steps=self.save_checkpoint_steps,
            accum_count=self.accum_count,
            world_size=self.world_size,
            report_every=self.report_every,
            train_steps=self.train_steps,
            recall_eval=self.recall_eval,
            visible_gpus=self.visible_gpus,
            gpu_ranks=self.gpu_ranks,
            log_file=self.log_file,
            dataset=self.dataset,
            seed=self.seed,
            test_all=self.test_all,
            test_from=self.test_from,
            train_from=self.train_from,
            report_rouge=self.report_rouge,
            block_trigram=self.block_trigram
        )
        print("[start] Params:", self.params)
        self.next(self.train)

    @step
    def train(self):
        """
        Run train.py for fine-tuning.
        """
        train(self.params, 0)
        self.next(self.train_end)

    @step
    def validate(self):
        """
        Validate on the latest checkpoint.
        """
        cmd = [PYTHON_EXEC, os.path.join(REPO_ROOT, 'train.py'),
               '-mode', 'validate',
               '-model_path', self.model_path,
               '-batch_size', str(self.batch_size),
               '-visible_gpus', str(self.visible_gpus),
               '-gpu_ranks', str(self.gpu_ranks),
               '-world_size', str(self.world_size)
        ]
        print("[validate] Running:", ' '.join(cmd))
        subprocess.run(cmd, check=True)
        self.next(self.test)

    @step
    def test(self):
        """
        Test on the latest checkpoint.
        """
        cmd = [PYTHON_EXEC, os.path.join(REPO_ROOT, 'train.py'),
               '-mode', 'test',
               '-test_from', '',
               '-model_path', self.model_path,
               '-batch_size', str(self.batch_size),
               '-visible_gpus', str(self.visible_gpus),
               '-gpu_ranks', str(self.gpu_ranks),
               '-world_size', str(self.world_size)
        ]
        print("[test] Running:", ' '.join(cmd))
        subprocess.run(cmd, check=True)
        self.next(self.end)

    @step
    def train_end(self):
        from sklearn.metrics import accuracy_score
        acc = self.model.score(self.cleaned_df.drop('label', axis=1), self.cleaned_df['label'])

        # Save metric
        self.accuracy = acc

        # Add card
        card = Markdown(f"# Training Metrics\n\n**Accuracy:** {acc:.2f}")
        self.card = card
        self.next(self.end)

    @step
    def end(self):
        """
        Finish up.
        """
        print("Metaflow run complete. All artifacts saved under:")
        print(" Model checkpoints:", self.model_path)
        print(" Results / ROUGE outputs:", self.result_path)

if __name__ == '__main__':
    BertSumMetaflow()