"""
Simple script to test OlmOCR dataset loading with YAML configuration.
"""

import argparse
import logging
from pathlib import Path
from pprint import pprint

from transformers import AutoProcessor

from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def print_sample(sample, dataset_name):
    """Pretty print a dataset sample."""
    print(f"\n{'='*80}")
    print(f"Sample from: {dataset_name}")
    print(f"{'='*80}")
    
    # Print keys
    print(f"\nAvailable keys: {list(sample.keys())}")
    
    # Print path information
    if 'markdown_path' in sample:
        print(f"\nMarkdown path: {sample['markdown_path']}")
    if 'pdf_path' in sample:
        print(f"PDF path: {sample['pdf_path']}")
    
    # Print page data
    if 'page_data' in sample:
        print(f"\nPage data:")
        print(f"  Primary language: {sample['page_data'].primary_language}")
        print(f"  Is rotation valid: {sample['page_data'].is_rotation_valid}")
        print(f"  Rotation correction: {sample['page_data'].rotation_correction}")
        print(f"  Is table: {sample['page_data'].is_table}")
        print(f"  Is diagram: {sample['page_data'].is_diagram}")
        print(f"  Natural text preview: {sample['page_data'].natural_text[:200]}..." if sample['page_data'].natural_text else "  Natural text: None")
    
    # Print image info
    if 'image' in sample:
        print(f"\nImage shape: {sample['image'].size}")
    
    # Print anchor text preview
    if 'anchor_text' in sample:
        print(f"\nAnchor text preview: {sample['anchor_text'][:200]}...")
    
    # Print instruction prompt preview
    if 'instruction_prompt' in sample:
        print(f"\nInstruction prompt preview: {sample['instruction_prompt'][:200]}...")
    
    # Print response preview
    if 'response' in sample:
        print(f"\nResponse preview: {sample['response'][:200]}...")
    
    # Print tokenization info
    if 'input_ids' in sample:
        print(f"\nTokenization info:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        if 'pixel_values' in sample:
            print(f"  Pixel values shape: {sample['pixel_values'].shape}")
        if 'image_grid_thw' in sample:
            print(f"  Image grid THW: {sample['image_grid_thw']}")


def main():
    parser = argparse.ArgumentParser(description="Test OlmOCR dataset loading")
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
    
    # Process training datasets
    print(f"\n{'='*80}")
    print("TRAINING DATASETS")
    print(f"{'='*80}")
    
    for i, dataset_cfg in enumerate(config.dataset.train):
        root_dir = dataset_cfg['root_dir']
        pipeline_steps = config.get_pipeline_steps(dataset_cfg['pipeline'], processor)
        
        logger.info(f"\nCreating training dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Get first sample
            sample = dataset[0]
            print_sample(sample, f"Training Dataset {i+1}: {Path(root_dir).name}")
    
    # Process evaluation datasets
    print(f"\n\n{'='*80}")
    print("EVALUATION DATASETS")
    print(f"{'='*80}")
    
    for i, dataset_cfg in enumerate(config.dataset.eval):
        root_dir = dataset_cfg['root_dir']
        pipeline_steps = config.get_pipeline_steps(dataset_cfg['pipeline'], processor)
        
        logger.info(f"\nCreating evaluation dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Get first sample
            sample = dataset[0]
            print_sample(sample, f"Evaluation Dataset {i+1}: {Path(root_dir).name}")
    
    print(f"\n{'='*80}")
    print("Dataset loading test completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()