# TODO Overall, this code will read in a config yaml file with omega conf
# From that config, we are going to use HuggingFace Trainer to train a model
# TODOS:
# Build a script to convert olmocr-mix to a new dataloader format
# Write a new dataloader and collator, with tests that brings in everything, only needs to support batch size 1 for this first version
# Get a basic config yaml file system working
# Get a basic hugging face trainer running, supporting Qwen2.5VL for now
# Saving and restoring training checkpoints
# Converting training checkpoints to vllm compatible checkpoinst
