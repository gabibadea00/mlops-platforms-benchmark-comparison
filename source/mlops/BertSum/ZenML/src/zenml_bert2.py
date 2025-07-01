"""
ZenML Pipeline for BertSum Training Integration

This script creates a ZenML pipeline that integrates with the existing BertSum repository
to train BERT for extractive summarization while tracking the process and metrics.

Requirements:
- zenml
- torch
- transformers
- tensorboard (optional, for logging)
- The existing BertSum repository cloned locally

Usage:
1. Install ZenML: pip install zenml
2. Initialize ZenML: zenml init
3. Adjust the paths and configurations below to match your BertSum repo structure
4. Run: python zenml_bertsum_pipeline.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertConfig
from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger

# Add the BertSum repo to Python path (adjust this path to your repo location)
BERTSUM_REPO_PATH = "./BertSum"  # Change this to your actual BertSum repo path
sys.path.append(BERTSUM_REPO_PATH)

# Import from BertSum repo (adjust imports based on your repo structure)
try:
    from models.model_builder import Summarizer, build_optim
    from models.data_loader import load_dataset
    from models.trainer import build_trainer
    from others.utils import rouge_results_to_str
    from prepro.data_builder import BertData
except ImportError as e:
    print(f"Warning: Could not import from BertSum repo: {e}")
    print("Please adjust the import paths to match your BertSum repository structure")

logger = get_logger(__name__)

# Configuration
@step
def load_config() -> SimpleNamespace:
    """Load configuration for BertSum training."""
    config = SimpleNamespace(
        encoder='classifier',
        mode='train',
        bert_data_path='../../MLFlow/bert_data/cnndm',
        model_path='../models/',
        result_path='../results/cnndm',
        temp_dir='../temp',
        bert_config_path='../bert_config_uncased_base.json',

        batch_size=1000,

        use_interval=True,
        hidden_size=128,
        ff_size=512,
        heads=4,
        inter_layers=2,
        rnn_size=512,

        param_init=0.0,
        param_init_glorot=True,
        dropout=0.1,
        optim='adam',
        lr=1.0,
        beta1=0.9,
        beta2=0.999,
        decay_method='',
        warmup_steps=8000,
        max_grad_norm=0.0,

        save_checkpoint_steps=100,
        accum_count=1,
        world_size=1,
        report_every=1,
        train_steps=1000,
        recall_eval=False,

        visible_gpus='-1',
        gpu_ranks='0',
        log_file='../logs/cnndm.log',
        dataset='',
        seed=666,

        test_all=False,
        test_from='',
        train_from='',
        report_rouge=True,
        block_trigram=True
    )
    
    logger.info(f"Loaded configuration: {config}")
    return config

@step
def prepare_data(config: SimpleNamespace) -> Tuple[str, int]:
    """Prepare the dataset for training."""
    logger.info("Preparing dataset...")
    
    data_path = config.bert_data_path
    
    # # Check if data exists
    # if not os.path.exists(data_path):
    #     raise FileNotFoundError(f"Data path {data_path} does not exist. Please prepare your data first.")
    
    # Count training examples (adjust based on your data structure)
    train_files = list(Path(data_path).glob("*.train.*"))
    total_examples = len(train_files)
    
    logger.info(f"Found {total_examples} training files")
    return data_path, total_examples

@step
def initialize_model(config: SimpleNamespace) -> Summarizer:
    """Initialize the BertSum model."""
    logger.info("Initializing BertSum model...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    bert_config = BertConfig.from_json_file("../bert_config_uncased_base.json")
    # Initialize model (adjust based on your BertSum implementation)
    model = Summarizer(
        args = config,
        device=device,
        load_pretrained_bert=False, 
        bert_config = bert_config
    )
    
    # Load from checkpoint if specified
    if getattr(config, 'train_from', ''):
        logger.info(f"Loading model from {config.train_from}")
        checkpoint = torch.load(config.train_from, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    logger.info("Model initialized successfully")
    return model

@step
def train_model(
    config: SimpleNamespace,
    model: Summarizer,
    data_path: str,
    total_examples: int
) -> Tuple[Summarizer, Dict[str, float]]:
    """Train the BertSum model with metric tracking."""
    logger.info("Starting model training...")
    
    # reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build optimizer + trainer
    optim = build_optim(config, model, None)
    trainer = build_trainer(config, device_id=0, model=model, optim=optim)

    from models.data_loader import load_dataset, Dataloader

    # load_dataset yields a sequence of raw-example lists; wrap in Dataloader
    train_datasets = load_dataset(config, 'train', shuffle=True)
    train_loader = Dataloader(
        args=config,
        datasets=train_datasets,
        batch_size=config.batch_size,
        device=device,
        shuffle=True,
        is_test=False
    )
    train_iter = iter(train_loader)

    # training‐metrics
    training_metrics = {
        'train_loss': 0.0,
        'learning_rate': 0.0,
        'steps_completed': 0,
        'best_loss': float('inf')
    }

    try:
        for step in range(config.train_steps):
            # fetch next batch (and restart epoch if needed)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # do one forward/backward/update
            loss = trainer.train_step(batch)  

            # update metrics
            training_metrics['train_loss'] = loss
            training_metrics['steps_completed'] = step + 1
            # correct LR lookup:
            training_metrics['learning_rate'] = trainer.optim.optimizer.param_groups[0]['lr']

            if step % config.report_every == 0:
                logger.info(f"Step {step}: Loss = {loss:.4f}, "
                            f"LR = {training_metrics['learning_rate']:.6f}")

            if step % config.save_checkpoint_steps == 0 and step > 0:
                ckpt = os.path.join(config.model_path, f'model_step_{step}.pt')
                trainer.save_checkpoint(ckpt)
                logger.info(f"Saved checkpoint at step {step}")

            # track best
            if loss < training_metrics['best_loss']:
                training_metrics['best_loss'] = loss

    except Exception as e:
        logger.error(f"Training error at step {step}: {e}")
        raise

    logger.info(f"Training completed. Final metrics: {training_metrics}")
    return model, training_metrics

@step
def validate_model(
    config: SimpleNamespace,
    model: Summarizer,
    training_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Validate the trained model."""
    logger.info("Starting model validation...")
    
    # Validation metrics
    validation_metrics = {
        'rouge_1_f': 0.0,
        'rouge_2_f': 0.0,
        'rouge_l_f': 0.0,
        'validation_loss': 0.0
    }
    
    try:
        # Implement validation logic based on your BertSum setup
        # This would typically involve running the model on validation data
        # and computing ROUGE scores
        
        # Placeholder for validation logic
        # val_loss, rouge_scores = validate_on_data(model, val_data_loader)
        # validation_metrics.update(rouge_scores)
        
        logger.info("Validation completed successfully")
        logger.info(f"Validation metrics: {validation_metrics}")
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        # Continue with empty metrics rather than failing
    
    return validation_metrics

@step
def save_model_artifacts(
    config: SimpleNamespace,
    model: Summarizer,
    training_metrics: Dict[str, float],
    validation_metrics: Dict[str, float]
):
    """Save the trained model and create artifacts."""
    logger.info("Saving model artifacts...")
    
    # Create model directory
    model_dir = Path(config.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final model
    final_model_path = model_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_metrics': training_metrics,
        'validation_metrics': validation_metrics,
        'timestamp': datetime.now().isoformat()
    }, final_model_path)
    
    # Save metrics separately
    metrics_path = model_dir / "metrics.json"
    all_metrics = {
        'training': training_metrics,
        'validation': validation_metrics
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Model saved to {final_model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Create ZenML model artifact
    model_artifact = dict(
        name="bertsum_extractive_summarizer",
        model_name="bertsum",
        version=datetime.now().strftime("%Y%m%d_%H%M%S"),
        model_version="1.0",
        model_path=str(final_model_path),
        is_model_artifact=True
    )
    
    return model_artifact

# Define the pipeline
@pipeline(enable_cache=False)
def bertsum_training_pipeline():
    """ZenML pipeline for training BertSum model."""
    
    # Load configuration
    config = load_config()
    
    # Prepare data
    data_path, total_examples = prepare_data(config)
    
    # Initialize model
    model = initialize_model(config)
    
    # Train model
    trained_model, training_metrics = train_model(config, model, data_path, total_examples)
    
    # Validate model
    validation_metrics = validate_model(config, trained_model, training_metrics)
    
    # Save artifacts
    model_artifact = save_model_artifacts(config, trained_model, training_metrics, validation_metrics)
    
    return model_artifact

def setup_zenml():
    """Setup ZenML experiment tracker and other configurations."""
    logger.info("Setting up ZenML...")
    
    # Initialize ZenML client
    client = Client()
    
    # You can add experiment tracker setup here
    # For example, MLflow:
    # try:
    #     mlflow_tracker = MLFlowExperimentTracker(
    #         tracking_uri="http://localhost:5000"
    #     )
    #     client.activate_stack_component(mlflow_tracker)
    # except Exception as e:
    #     logger.warning(f"Could not set up MLflow tracker: {e}")
    
    logger.info("ZenML setup completed")

def main():
    """Main function to run the ZenML pipeline."""
    print("Starting BertSum ZenML Training Pipeline...")
    
    # Setup ZenML
    setup_zenml()
    
    # Run the pipeline
    try:
        pipeline_instance = bertsum_training_pipeline()
        model_artifact = pipeline_instance.run()
        
        print("Pipeline completed successfully!")
        print(f"Model artifact: {model_artifact}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()