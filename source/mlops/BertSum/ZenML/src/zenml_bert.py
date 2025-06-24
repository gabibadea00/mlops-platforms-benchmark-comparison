from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.integrations.pytorch import PyTorchTrainerStep
from zenml.integrations.aws.s3_artifacts import S3ArtifactStore
from zenml.integrations.mlflow import MLFlowExperimentTracker

# Import your existing BertSum training logic
from train import train  # adjust import path accordingly
from data_loader import get_train_iter, get_valid_iter  # functions to produce iterators
from utils import Statistics  # any utilities you need

@step

def data_loader(train_or_valid: str) -> Output[iterator]:
    """
    ZenML step to load training or validation data.
    """
    if train_or_valid == "train":
        return get_train_iter()
    else:
        return get_valid_iter()

@step

def train_step(train_iter: iterator, valid_iter: iterator = None) -> Output[stats]:
    """
    Wraps the BertSum `train` function inside a ZenML step.  
    Expects:
      - train_iter: iterator over training batches
      - valid_iter: optional iterator for validation
    Returns:
      - total training statistics
    """
    # Define hyperparameters
    train_steps = 10000
    save_checkpoint_steps = 1000

    # Create your trainer instance (e.g., BertSumTrainer)
    trainer = YourBertSumTrainer(
        grad_accum_count=4,
        save_checkpoint_steps=save_checkpoint_steps,
        # ... other init args
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

def eval_step(stats) -> None:
    """
    A placeholder evaluation step. You can log metrics or perform further evaluation here.
    """
    print(f"Training completed. Stats: {stats}")

@pipeline(  
    enable_cache=True,
    settings={
        "experiment_tracker": MLFlowExperimentTracker(),
        "artifact_store": S3ArtifactStore(),
    }
)
def bertsumm_pipeline(
    train_loader,
    valid_loader,
    trainer_step,
    evaluator_step,
):
    """
    Defines the full ZenML pipeline for BertSum training.
    """
    stats = trainer_step(
        train_iter=train_loader,
        valid_iter=valid_loader
    )
    evaluator_step(stats)

if __name__ == "__main__":
    # Instantiate steps
    train_loader = data_loader(train_or_valid="train")
    valid_loader = data_loader(train_or_valid="valid")
    trainer = train_step()
    evaluator = eval_step()

    # Run pipeline
    bertsumm_pipeline(
        train_loader=train_loader,
        valid_loader=valid_loader,
        trainer_step=trainer,
        evaluator_step=evaluator,
    )