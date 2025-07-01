from typing import Iterator, Optional, Any
from zenml import step, pipeline

# Import your existing BertSum training logic
from models.trainer import build  # adjust import path accordingly
from data_loader import get_train_iter, get_valid_iter  # adjust import path
from utils import Statistics  # adjust import path

@step
def data_loader(train_or_valid: str) -> Iterator[Any]:
    """
    ZenML step to load training or validation data.
    """
    if train_or_valid == "train":
        return get_train_iter()
    else:
        return get_valid_iter()

@step
def train_step(
    train_iter: Iterator[Any],
    valid_iter: Optional[Iterator[Any]] = None
) -> Statistics:
    """
    Wraps the BertSum `train` function inside a ZenML step.
    """
    # Hyperparameters (customize as needed)
    train_steps = 10000
    save_checkpoint_steps = 1000

    # Instantiate your trainer
    trainer = BertSumTrainer(
        grad_accum_count=4,
        save_checkpoint_steps=save_checkpoint_steps,
        # ... additional args
    )

    # Run training
    stats = trainer.train(
        train_iter_fct=lambda: train_iter,
        train_steps=train_steps,
        valid_iter_fct=(lambda: valid_iter) if valid_iter is not None else None,
        valid_steps=100
    )
    return stats

@step

def eval_step(stats: Statistics) -> None:
    """
    A placeholder evaluation step. You can log metrics or perform further evaluation here.
    """
    print(f"Training completed. Stats: {stats}")

@pipeline(
    enable_cache=True,
)
def bertsumm_pipeline():
    """
    Defines the full ZenML pipeline for BertSum training.
    """
    train_loader = data_loader(train_or_valid="train")
    valid_loader = data_loader(train_or_valid="valid")
    stats = train_step(train_iter=train_loader, valid_iter=valid_loader)
    eval_step(stats=stats)

if __name__ == "__main__":
    # Execute the pipeline
    bertsumm_pipeline()
