"""
Simple script to test OlmOCR dataset loading with YAML configuration.
"""

import argparse
import logging
import os
import math
import shutil
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast
import wandb

from transformers import (
    AutoProcessor,
    get_scheduler,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from typing import Optional, Dict, Any
from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class QwenDataCollator:
    """Data collator for vision-language models that handles numpy arrays."""

    def __init__(self, max_token_len: Optional[int] = None):
        self.max_token_len = max_token_len

    def __call__(self, examples):
        # Filter out None values and extract the fields we need
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": [], "image_grid_thw": []}

        for example in examples:
            if example is not None:
                # Convert numpy arrays to tensors
                input_ids = torch.from_numpy(example["input_ids"]) if isinstance(example["input_ids"], np.ndarray) else example["input_ids"]
                attention_mask = torch.from_numpy(example["attention_mask"]) if isinstance(example["attention_mask"], np.ndarray) else example["attention_mask"]
                labels = torch.from_numpy(example["labels"]) if isinstance(example["labels"], np.ndarray) else example["labels"]
                
                # Trim to max_token_len if specified
                if self.max_token_len is not None:
                    input_ids = input_ids[:self.max_token_len]
                    attention_mask = attention_mask[:self.max_token_len]
                    labels = labels[:self.max_token_len]
                
                batch["input_ids"].append(input_ids)
                batch["attention_mask"].append(attention_mask)
                batch["labels"].append(labels)

                # Handle pixel_values which might be numpy array or already a tensor
                pixel_values = example["pixel_values"]
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = torch.from_numpy(pixel_values)
                batch["pixel_values"].append(pixel_values)

                # Handle image_grid_thw
                image_grid_thw = example["image_grid_thw"]
                if isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw)
                batch["image_grid_thw"].append(image_grid_thw)

        # Convert lists to tensors with proper padding
        # Note: For Qwen2-VL, we typically handle variable length sequences
        # The model's processor should handle the padding internally
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "pixel_values": torch.stack(batch["pixel_values"]),  # Stack into tensor
            "image_grid_thw": torch.stack(batch["image_grid_thw"]),
        }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    output_dir: str,
    save_total_limit: Optional[int] = None,
):
    """Save model, optimizer, scheduler, and training state."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(checkpoint_dir)
    
    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # Save training state
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }
    torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Enforce save_total_limit by removing oldest checkpoints
    if save_total_limit is not None and save_total_limit > 0:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        while len(checkpoints) > save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(os.path.join(output_dir, oldest))
            logger.info(f"Deleted old checkpoint: {oldest}")


def load_checkpoint(
    model_class: type,
    init_kwargs: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    checkpoint_dir: str,
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model, optimizer, scheduler, and training state from checkpoint."""
    model = model_class.from_pretrained(checkpoint_dir, **init_kwargs)
    model.to(device)
    
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=device))
    lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location=device))
    
    state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location=device)
    logger.info(f"Resumed from checkpoint: {checkpoint_dir} at epoch {state['epoch']}, step {state['global_step']}")
    return model, state


def evaluate_model(
    model: torch.nn.Module,
    eval_dataloaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on all eval datasets and return average loss per dataset."""
    model.eval()
    eval_metrics = {}
    
    for dataset_name, dataloader in eval_dataloaders.items():
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        eval_metrics[f"eval_{dataset_name}_loss"] = avg_loss
        logger.info(f"Eval {dataset_name} loss: {avg_loss:.4f}")
    
    # Compute overall eval loss as average across datasets (or customize as needed)
    if eval_metrics:
        overall_loss = sum(eval_metrics.values()) / len(eval_metrics)
        eval_metrics["eval_loss"] = overall_loss
    
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Train OlmOCR model")
    parser.add_argument("--config", type=str, default="olmocr/train/configs/example_config.yaml", help="Path to YAML configuration file")

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return

    # Set wandb project from config
    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name
        logger.info(f"Setting WANDB_PROJECT to: {config.project_name}")
    
    # Initialize wandb if reporting to it
    if "wandb" in config.training.report_to:
        wandb.init(project=config.project_name, name=config.run_name, config=config.to_dict())
    
    # Load processor for tokenization
    logger.info(f"Loading processor: {config.model.name}")
    processor = AutoProcessor.from_pretrained(
        config.model.name,
    )

    # Model init kwargs to reuse for loading checkpoints
    model_init_kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
        "device_map": config.model.device_map,
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": config.model.attn_implementation if config.model.use_flash_attention else None,
    }

    # Load model
    logger.info(f"Loading model: {config.model.name}")
    if "Qwen2.5-VL" in config.model.name:
        model_class = Qwen2_5_VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    elif "Qwen2-VL" in config.model.name:
        model_class = Qwen2VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    else:
        raise NotImplementedError()

    # Enable gradient checkpointing if configured
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs)

    # Create training datasets
    logger.info("Creating training datasets...")
    train_datasets = []
    for i, dataset_cfg in enumerate(config.dataset.train):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        logger.info(f"Creating training dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            train_datasets.append(dataset)

    # Combine all training datasets
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    logger.info(f"Total training samples: {len(train_dataset)}")

    # Create evaluation datasets
    logger.info("Creating evaluation datasets...")
    eval_datasets = {}
    for i, dataset_cfg in enumerate(config.dataset.eval):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        # Use dataset name if provided, otherwise use root_dir as name
        dataset_name = dataset_cfg.get("name", f"eval_dataset_{i+1}")

        logger.info(f"Creating evaluation dataset '{dataset_name}' from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            eval_datasets[dataset_name] = dataset

    # Log total evaluation samples across all datasets
    total_eval_samples = sum(len(dataset) for dataset in eval_datasets.values())
    logger.info(f"Total evaluation samples across {len(eval_datasets)} datasets: {total_eval_samples}")

    # Construct full output directory by appending run_name to base output_dir
    full_output_dir = os.path.join(config.training.output_dir, config.run_name)
    logger.info(f"Setting output directory to: {full_output_dir}")
    os.makedirs(full_output_dir, exist_ok=True)

    # Check for existing checkpoints if any
    found_resumable_checkpoint = None
    if os.path.exists(full_output_dir):
        # Look for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(full_output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(full_output_dir, d))]
        if checkpoint_dirs:
            # Sort by checkpoint number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(full_output_dir, checkpoint_dirs[-1])
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            found_resumable_checkpoint = latest_checkpoint
        else:
            logger.info("No existing checkpoints found in output directory")

    # Set seeds
    torch.manual_seed(config.training.seed)
    if config.training.data_seed is not None:
        torch.utils.data.dataset.random.seed(config.training.data_seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if config.training.optim == "adamw_torch":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(config.training.learning_rate),
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_epsilon,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.training.optim} not supported in custom loop")

    # Total training steps calculation
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / (config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps))
    max_train_steps = int(config.training.num_train_epochs * num_update_steps_per_epoch)

    # Set up scheduler
    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * config.training.warmup_ratio),
        num_training_steps=max_train_steps,
        scheduler_specific_kwargs=config.training.lr_scheduler_kwargs,
    )

    # Data collator
    data_collator = QwenDataCollator(max_token_len=config.training.collator_max_token_len)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config.training.dataloader_num_workers,
        drop_last=config.training.dataloader_drop_last,
    )

    eval_dataloaders = {
        name: DataLoader(
            dataset,
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=config.training.dataloader_num_workers,
            drop_last=False,
        )
        for name, dataset in eval_datasets.items()
    }

    # Resume from checkpoint if available
    start_epoch = 0
    global_step = 0
    best_metric = float("inf") if config.training.greater_is_better else -float("inf")
    if found_resumable_checkpoint:
        model, state = load_checkpoint(model_class, model_init_kwargs, optimizer, lr_scheduler, found_resumable_checkpoint, device)
        start_epoch = state["epoch"] + 1  # Start from next epoch
        global_step = state["global_step"]
        best_metric = state["best_metric"]

    # Always evaluate on start
    metrics = evaluate_model(model, eval_dataloaders, device)
    logger.info(f"Initial evaluation: {metrics}")
    if "wandb" in config.training.report_to:
        wandb.log(metrics, step=global_step)

    # Main training loop
    logger.info("Starting training...")
    model.train()
    for epoch in range(start_epoch, int(config.training.num_train_epochs)):
        epoch_start_time = time.time()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast(enabled=True, dtype=torch.bfloat16):
                outputs = model(**batch)
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()
            
            train_loss += loss.item() * config.training.gradient_accumulation_steps
            num_batches += 1
            
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
                # Step optimizer and scheduler
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if config.training.logging_steps > 0 and global_step % config.training.logging_steps == 0:
                    avg_train_loss = train_loss / num_batches
                    logs = {
                        "train_loss": avg_train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + (batch_idx / len(train_dataloader)),
                    }
                    logger.info(f"Step {global_step}: {logs}")
                    if "wandb" in config.training.report_to:
                        wandb.log(logs, step=global_step)
                    
                    train_loss = 0.0
                    num_batches = 0
            
            # Evaluation
            if config.training.eval_steps > 0 and global_step % config.training.eval_steps == 0:
                metrics = evaluate_model(model, eval_dataloaders, device)
                logger.info(f"Evaluation at step {global_step}: {metrics}")
                if "wandb" in config.training.report_to:
                    wandb.log(metrics, step=global_step)
                
                # Update best metric (for potential load_best_model_at_end, though not implemented here)
                current_metric = metrics.get(config.training.metric_for_best_model, None)
                if current_metric is not None:
                    if (config.training.greater_is_better and current_metric > best_metric) or \
                       (not config.training.greater_is_better and current_metric < best_metric):
                        best_metric = current_metric
            
            # Saving
            if config.training.save_steps > 0 and global_step % config.training.save_steps == 0:
                save_checkpoint(
                    model, optimizer, lr_scheduler, epoch, global_step, best_metric,
                    full_output_dir, config.training.save_total_limit
                )
        
        # End of epoch logging
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    # Save the final model
    logger.info("Saving final model...")
    model.save_pretrained(full_output_dir)
    
    # Final evaluation
    final_metrics = evaluate_model(model, eval_dataloaders, device)
    logger.info(f"Training completed! Final metrics: {final_metrics}")
    if "wandb" in config.training.report_to:
        wandb.log(final_metrics, step=global_step)
        wandb.finish()


if __name__ == "__main__":
    main()