#!/usr/bin/env python3
"""
Compresses OlmOCR checkpoints using FP8 quantization:
1. Loads model from source (local or S3)
2. Applies FP8 dynamic quantization
3. Saves compressed model to destination (local or S3)

Usage:
    python compress_checkpoint.py <source_path> <destination_path>
    
    source_path: Path to checkpoint (local or S3)
    destination_path: Where to save compressed checkpoint (local or S3)
"""

import argparse
import json
import os
import shutil
import tempfile
from typing import Optional, Tuple, Union

import boto3
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from olmocr.s3_utils import parse_s3_path


s3_client = boto3.client("s3")


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return path.startswith("s3://")


def download_s3_to_local(bucket: str, prefix: str, local_dir: str) -> None:
    """Download all files from S3 prefix to local directory."""
    os.makedirs(local_dir, exist_ok=True)
    
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    print(f"Downloading checkpoint from s3://{bucket}/{prefix} to {local_dir}...")
    
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            
            rel_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, rel_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(bucket, key, local_path)
            print(f"  Downloaded {rel_path}")


def upload_local_to_s3(local_dir: str, bucket: str, prefix: str) -> None:
    """Upload all files from local directory to S3."""
    print(f"Uploading compressed checkpoint from {local_dir} to s3://{bucket}/{prefix}...")
    
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(prefix, rel_path)
            
            s3_client.upload_file(local_path, bucket, s3_key)
            print(f"  Uploaded {rel_path}")


def load_model_and_tokenizer(source_path: str) -> Tuple[Union[Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration], AutoTokenizer, Optional[str]]:
    """Load model and tokenizer from source path (local or S3)."""
    if is_s3_path(source_path):
        # Download from S3 to temporary directory
        temp_dir = tempfile.mkdtemp()
        bucket, prefix = parse_s3_path(source_path)
        download_s3_to_local(bucket, prefix, temp_dir)
        model_path = temp_dir
    else:
        model_path = source_path
        temp_dir = None
    
    # Read config to determine model architecture
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Get model name from config
    model_name = config.get("name_or_path", "")
    
    print(f"Loading model from {model_path}...")
    
    # Load appropriate model class based on name
    if "Qwen2.5-VL" in model_name:
        print("Detected Qwen2.5-VL model")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto"
        )
    elif "Qwen2-VL" in model_name:
        print("Detected Qwen2-VL model")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto"
        )
    else:
        # Default to checking architectures list
        architectures = config.get("architectures", [])
        if "Qwen2_5_VLForConditionalGeneration" in architectures:
            print("Detected Qwen2.5-VL model from architectures")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype="auto"
            )
        else:
            print("Detected Qwen2-VL model from architectures")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype="auto"
            )
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, temp_dir


def copy_additional_files(source_path: str, dest_path: str, temp_source_dir: Optional[str] = None) -> None:
    """Copy additional config files that are needed but not saved by save_pretrained."""
    # List of additional files to copy if they exist
    additional_files = ["preprocessor_config.json", "chat_template.json"]
    
    # Determine the actual source path (could be temp dir if downloaded from S3)
    actual_source = temp_source_dir if temp_source_dir else source_path
    
    for filename in additional_files:
        source_file = os.path.join(actual_source, filename)
        if os.path.exists(source_file):
            dest_file = os.path.join(dest_path, filename)
            print(f"Copying {filename} to destination...")
            shutil.copy2(source_file, dest_file)


def compress_checkpoint(source_path: str, dest_path: str) -> None:
    """Compress OlmOCR checkpoint using FP8 quantization."""
    # Load model and tokenizer
    model, tokenizer, temp_source_dir = load_model_and_tokenizer(source_path)
    
    try:
        # Print all model tensor names
        print("\n=== Model Tensor Names ===")
        for name, param in model.named_parameters():
            print(f"{name}: shape={list(param.shape)}, dtype={param.dtype}")
        print("=========================\n")
        
        # Configure FP8 dynamic quantization
        print("\nApplying FP8 dynamic quantization...")
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["re:.*lm_head", "re:visual.*"],
        )
        
        # Apply the quantization
        oneshot(model=model, recipe=recipe)
        print("✓ Quantization completed successfully")
        
        # Save the compressed model
        if is_s3_path(dest_path):
            # Save to temporary directory first, then upload to S3
            with tempfile.TemporaryDirectory() as temp_dest_dir:
                print(f"\nSaving compressed model to temporary directory...")
                model.save_pretrained(temp_dest_dir)
                tokenizer.save_pretrained(temp_dest_dir)
                
                # Copy additional files
                copy_additional_files(source_path, temp_dest_dir, temp_source_dir)
                
                # Upload to S3
                bucket, prefix = parse_s3_path(dest_path)
                upload_local_to_s3(temp_dest_dir, bucket, prefix)
        else:
            # Save directly to local destination
            print(f"\nSaving compressed model to {dest_path}...")
            os.makedirs(dest_path, exist_ok=True)
            model.save_pretrained(dest_path)
            tokenizer.save_pretrained(dest_path)
            
            # Copy additional files
            copy_additional_files(source_path, dest_path, temp_source_dir)
        
        print(f"\n✓ Successfully compressed checkpoint and saved to {dest_path}")
        
    finally:
        # Clean up temporary source directory if needed
        if temp_source_dir:
            print(f"Cleaning up temporary directory {temp_source_dir}...")
            shutil.rmtree(temp_source_dir)
        
        # Free up GPU memory
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Compress OlmOCR checkpoint using FP8 quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local to local
    python compress_checkpoint.py /path/to/checkpoint /path/to/compressed
    
    # S3 to S3
    python compress_checkpoint.py s3://bucket/checkpoint s3://bucket/compressed
    
    # S3 to local
    python compress_checkpoint.py s3://bucket/checkpoint /path/to/compressed
    
    # Local to S3
    python compress_checkpoint.py /path/to/checkpoint s3://bucket/compressed
        """
    )
    parser.add_argument("source", help="Source checkpoint path (local or S3)")
    parser.add_argument("destination", help="Destination path for compressed checkpoint (local or S3)")
    
    args = parser.parse_args()
    
    try:
        compress_checkpoint(args.source, args.destination)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
