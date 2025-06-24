"""
Simple script to test OlmOCR dataset loading with YAML configuration.
"""

import argparse
import logging

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
from torch.utils.data import ConcatDataset

from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_data_collator():
    """Create a data collator for vision-language models."""
    def collate_fn(examples):
        # Filter out None values and extract the fields we need
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'pixel_values': [],
            'image_grid_thw': []
        }
        
        for example in examples:
            if example is not None:
                batch['input_ids'].append(example['input_ids'])
                batch['attention_mask'].append(example['attention_mask'])
                batch['labels'].append(example['labels'])
                batch['pixel_values'].append(example['pixel_values'])
                batch['image_grid_thw'].append(example['image_grid_thw'])
        
        # Convert lists to tensors with proper padding
        # Note: For Qwen2-VL, we typically handle variable length sequences
        # The model's processor should handle the padding internally
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'labels': torch.stack(batch['labels']),
            'pixel_values': batch['pixel_values'],  # Keep as list for now
            'image_grid_thw': torch.stack(batch['image_grid_thw'])
        }
    
    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Train OlmOCR model")
    parser.add_argument(
        "--config",
        type=str,
        default="olmocr/train/configs/example_config.yaml",
        help="Path to YAML configuration file"
    )
    
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
    
    # Load processor for tokenization
    logger.info(f"Loading processor: {config.model.name}")
    processor = AutoProcessor.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.processor_trust_remote_code
    )
    
    # Load model
    logger.info(f"Loading model: {config.model.name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation=config.model.attn_implementation if config.model.use_flash_attention else None,
    )
    
    # Enable gradient checkpointing if configured
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs)
    
    # Create training datasets
    logger.info("Creating training datasets...")
    train_datasets = []
    for i, dataset_cfg in enumerate(config.dataset.train):
        root_dir = dataset_cfg['root_dir']
        pipeline_steps = config.get_pipeline_steps(dataset_cfg['pipeline'], processor)
        
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
    eval_datasets = []
    for i, dataset_cfg in enumerate(config.dataset.eval):
        root_dir = dataset_cfg['root_dir']
        pipeline_steps = config.get_pipeline_steps(dataset_cfg['pipeline'], processor)
        
        logger.info(f"Creating evaluation dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")
        
        if len(dataset) > 0:
            eval_datasets.append(dataset)
    
    # Combine all evaluation datasets
    eval_dataset = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]
    logger.info(f"Total evaluation samples: {len(eval_dataset)}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        warmup_steps=config.training.warmup_steps,
        optim=config.training.optim,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        eval_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        logging_dir=config.training.logging_dir,
        logging_strategy=config.training.logging_strategy,
        logging_steps=config.training.logging_steps,
        logging_first_step=config.training.logging_first_step,
        report_to=config.training.report_to,
        seed=config.training.seed,
        data_seed=config.training.data_seed,
        push_to_hub=config.training.push_to_hub,
        hub_model_id=config.training.hub_model_id,
        hub_strategy=config.training.hub_strategy,
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        deepspeed=config.training.deepspeed,
        dataloader_drop_last=config.training.dataloader_drop_last,
        dataloader_num_workers=config.training.dataloader_num_workers,
        remove_unused_columns=config.training.remove_unused_columns,
        run_name=config.run_name,
    )
    
    # Set up callbacks
    callbacks = []
    if config.training.use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience,
                early_stopping_threshold=config.training.early_stopping_threshold
            )
        )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=create_data_collator(),
        callbacks=callbacks,
    )
    
    # Start training
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Log metrics
    logger.info(f"Training completed! Metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()