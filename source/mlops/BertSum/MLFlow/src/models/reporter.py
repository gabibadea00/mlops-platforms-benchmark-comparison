''' Report manager utility with MLflow logging '''
from __future__ import print_function

import time
from datetime import datetime

import mlflow
from models.stats import Statistics
from others.logging import logger


def build_report_manager(opt):
    """
    Factory for ReportMgr.

    Args:
        opt: options object. Expected attributes:
            - report_every (int)
            - tensorboard (bool)
            - tensorboard_log_dir (str)
            - mlflow (bool)
            - mlflow_experiment_name (str, optional)
            - mlflow_run_name (str, optional)
    """
    # TensorBoard setup
    if getattr(opt, 'tensorboard', False):
        from tensorboardX import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir
        if not getattr(opt, 'train_from', None):
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S")
        tb_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    else:
        tb_writer = None

    # MLflow setup
    mlflow_run = None
    if getattr(opt, 'mlflow', False):
        if getattr(opt, 'mlflow_experiment_name', None):
            mlflow.set_experiment(opt.mlflow_experiment_name)
        run_kwargs = {}
        if getattr(opt, 'mlflow_run_name', None):
            run_kwargs['run_name'] = opt.mlflow_run_name
        mlflow_run = mlflow.start_run(**run_kwargs)

    report_mgr = ReportMgr(
        report_every=opt.report_every,
        start_time=-1,
        tensorboard_writer=tb_writer,
        mlflow_run=mlflow_run
    )
    return report_mgr


class ReportMgrBase(object):
    """
    Base class for reporting training and validation progress.
    """

    def __init__(self, report_every, start_time=-1.):
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        if self.start_time < 0:
            raise ValueError("ReportMgr needs to be started (use 'start()')")

        if step % self.report_every == 0:
            if multigpu:
                report_stats = Statistics.all_gather_stats(report_stats)
            self._report_training(step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
            return Statistics()
        return report_stats

    def _report_training(self, *args, **kwargs):
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        self._report_step(lr, step,
                          train_stats=train_stats,
                          valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1.,
                 tensorboard_writer=None, mlflow_run=None):
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer
        self.mlflow_active = mlflow_run is not None

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(prefix, self.tensorboard_writer,
                                  learning_rate, step)

    def maybe_log_mlflow(self, stats, prefix, learning_rate, step):
        if not self.mlflow_active:
            return
        # Log key metrics
        xent = stats.xent()
        elapsed = stats.elapsed_time()
        throughput = stats.n_docs / (elapsed + 1e-5)
        mlflow.log_metric(f"{prefix}/xent", xent, step=step)
        mlflow.log_metric(f"{prefix}/throughput", throughput, step=step)
        mlflow.log_metric(f"{prefix}/elapsed_time", elapsed, step=step)
        mlflow.log_metric(f"{prefix}/learning_rate", learning_rate, step=step)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        # Console output
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        # TensorBoard
        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   self.progress_step)
        # MLflow
        self.maybe_log_mlflow(report_stats,
                               "progress",
                               learning_rate,
                               self.progress_step)

        return report_stats

    def _report_step(self, lr, step,
                     train_stats=None, valid_stats=None):
        if train_stats is not None:
            self.log('Train xent: %g', train_stats.xent())
            self.maybe_log_tensorboard(train_stats, "train", lr, step)
            self.maybe_log_mlflow(train_stats, "train", lr, step)

        if valid_stats is not None:
            self.log('Validation xent: %g at step %d',
                     valid_stats.xent(), step)
            self.maybe_log_tensorboard(valid_stats, "valid", lr, step)
            self.maybe_log_mlflow(valid_stats, "valid", lr, step)
