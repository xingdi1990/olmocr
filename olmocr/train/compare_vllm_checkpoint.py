#!/usr/bin/env python3
"""
Batch VLM inference comparison between vLLM and HuggingFace.
Processes prompts and images from WildVision-bench until finding significant mismatch.
"""

import argparse
import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
import random
import numpy as np
from typing import List, Dict
import base64
from io import BytesIO


def image_to_base64_data_url(image):
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def load_wildvision_prompts(num_samples: int = 100, seed: int = 42, max_length: int = 2048) -> List[Dict[str, str]]:
    """Load prompts and images from WildVision-bench dataset with fixed random seed."""
    print(f"Loading WildVision-bench dataset with {num_samples} samples and seed {seed}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    dataset = load_dataset("WildVision/wildvision-bench", "vision_bench_0701", split="test", streaming=True)
    
    # Collect prompts and images
    samples = []
    examined = 0
    for example in dataset:
        examined += 1
        if len(samples) >= num_samples * 2:  # Collect extra to allow filtering
            break
        
        # Extract prompt and image from the example
        prompt = example.get('instruction', '').strip()
        image = example.get('image', None)
        
        # Filter by prompt length and ensure we have both prompt and image
        if prompt and image and 10 < len(prompt) <= max_length:
            samples.append({
                'prompt': prompt,
                'image': image  # This is already a PIL Image object
            })
        
        # Stop if we've examined too many without finding enough
        if examined > num_samples * 10:
            break
    
    # Randomly sample exactly num_samples
    if len(samples) < num_samples:
        print(f"Only found {len(samples)} valid samples out of {examined} examined")
        if len(samples) == 0:
            raise ValueError("No valid samples found in dataset")
        samples = random.choices(samples, k=num_samples)
    else:
        samples = random.sample(samples, num_samples)
    
    print(f"Selected {len(samples)} samples for comparison")
    return samples


def process_single_prompt(sample: Dict[str, any], llm, hf_model, processor, sampling_params, device, args):
    """Process a single prompt with image and return comparison results."""
    prompt = sample['prompt']
    image = sample['image']  # Already a PIL Image object
    
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt[:100]}..." if len(prompt) > 100 else f"PROMPT: {prompt}")
    print(f"IMAGE: {image.size} {image.mode}")
    
    # Generate with vLLM
    print("\n=== vLLM Generation ===")
    # Convert image to base64 data URL
    image_data_url = image_to_base64_data_url(image)
    
    # For VLMs, vLLM expects the message format with image
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": prompt}
        ]
    }]
    outputs = llm.chat(messages, sampling_params)
    output = outputs[0]
    
    # Extract prompt and generated token IDs
    prompt_token_ids = output.prompt_token_ids
    generated_token_ids = output.outputs[0].token_ids

    print(f"Prompt tokens ({len(prompt_token_ids)}): {prompt_token_ids[:10]}..." if len(prompt_token_ids) > 10 else f"Prompt tokens ({len(prompt_token_ids)}): {prompt_token_ids}")
    print(f"Generated tokens ({len(generated_token_ids)}): {generated_token_ids}")
    print(f"Generated text: {processor.decode(generated_token_ids, skip_special_tokens=True)}")
    
    # Create input tensor from concatenated token IDs
    # input_ids = torch.tensor([all_token_ids], device=device)  # Not needed for HF VLM models
    
    # HuggingFace forward pass
    print("\n=== HuggingFace Forward Pass ===")
    # Prepare inputs for HF model
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt"
    ).to(device)

    print("INPUTS", inputs)

    # Concatenate the generated tokens to the input_ids
    generated_ids_tensor = torch.tensor([generated_token_ids], device=device)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], generated_ids_tensor], dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    
    with torch.no_grad():
        outputs_hf = hf_model(**inputs)
        logits = outputs_hf.logits[0]  # [seq_len, vocab_size]
    
    # Token-by-token comparison
    print(f"\n{'Pos':>4} {'Token ID':>8} {'Token':>20} {'Type':>8} {'vLLM Prob':>12} {'HF Argmax':>10} {'HF Prob':>12} {'Match':>6} {'HF Token':>20}")
    print("-" * 125)
    
    # Get vLLM logprobs for generated tokens
    vllm_logprobs = output.outputs[0].logprobs
    
    # Track mismatch info
    first_mismatch_idx = None
    max_prob_diff = 0.0
    
    # Get all token IDs from the HF model's input
    all_token_ids = inputs["input_ids"][0].tolist()
    
    # Compare ALL tokens (prompt + generated)
    for pos, token_id in enumerate(all_token_ids):
        token_str = processor.decode([token_id], skip_special_tokens=False).replace('\n', '\\n').replace('\r', '\\r')
        
        # Determine if this is a prompt or generated token
        is_prompt = pos < len(prompt_token_ids)
        token_type = "prompt" if is_prompt else "gen"
        
        # vLLM probability (only for generated tokens)
        vllm_prob_str = "N/A"
        vllm_prob = None
        if not is_prompt:
            gen_idx = pos - len(prompt_token_ids)
            if vllm_logprobs and gen_idx < len(vllm_logprobs):
                # vLLM logprobs is a list of dicts mapping token_id to logprob
                token_logprobs = vllm_logprobs[gen_idx]
                if token_logprobs and token_id in token_logprobs:
                    # Convert logprob to probability
                    vllm_prob = torch.exp(torch.tensor(token_logprobs[token_id].logprob)).item()
                    vllm_prob_str = f"{vllm_prob:12.6f}"
        
        # HF prediction - only for generated tokens (skip prompt tokens entirely)
        if pos > 0 and not is_prompt:
            hf_logits_at_pos = logits[pos - 1]
            hf_probs = torch.softmax(hf_logits_at_pos, dim=-1)
            hf_argmax = torch.argmax(hf_logits_at_pos).item()
            hf_prob = hf_probs[token_id].item()
            
            # Check if predictions match
            match = "✓" if token_id == hf_argmax else "✗"
            
            # Track first mismatch and probability difference
            if token_id != hf_argmax:
                if first_mismatch_idx is None:
                    first_mismatch_idx = pos - len(prompt_token_ids)
                
                # Calculate probability difference
                if vllm_prob is not None:
                    prob_diff = abs(vllm_prob - hf_prob)
                    max_prob_diff = max(max_prob_diff, prob_diff)
            
            # Decode HF argmax token (only show if mismatch)
            hf_token_str = ""
            if token_id != hf_argmax:
                hf_token_str = processor.decode([hf_argmax], skip_special_tokens=False).replace('\n', '\\n').replace('\r', '\\r')
            
            print(f"{pos:>4} {token_id:>8} {token_str:>20} {token_type:>8} {vllm_prob_str:>12} {hf_argmax:>10} {hf_prob:>12.6f} {match:>6} {hf_token_str:>20}")
        else:
            # Prompt tokens or first token - no HF comparison
            print(f"{pos:>4} {token_id:>8} {token_str:>20} {token_type:>8} {vllm_prob_str:>12} {'':>10} {'':>12} {'':>6} {'':<20}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total tokens generated: {len(generated_token_ids)}")
    
    # Calculate match rate
    matches = 0
    for i, token_id in enumerate(generated_token_ids):
        pos = len(prompt_token_ids) + i
        hf_logits_at_pos = logits[pos - 1]
        hf_argmax = torch.argmax(hf_logits_at_pos).item()
        if token_id == hf_argmax:
            matches += 1
    
    match_rate = matches / len(generated_token_ids) * 100 if generated_token_ids else 0
    print(f"Token match rate: {matches}/{len(generated_token_ids)} ({match_rate:.1f}%)")
    
    # Report first mismatch index
    if first_mismatch_idx is not None:
        print(f"First mismatch at generation index: {first_mismatch_idx}")
        print(f"Max probability difference: {max_prob_diff:.6f}")
    else:
        print("No mismatches found in generated tokens")
    
    return {
        'first_mismatch_idx': first_mismatch_idx,
        'max_prob_diff': max_prob_diff,
        'match_rate': match_rate
    }


def main():
    parser = argparse.ArgumentParser(description="Batch compare VLM inference between vLLM and HuggingFace")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Model name or path")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to load from WildVision")
    parser.add_argument("--prob-threshold", type=float, default=0.20,
                        help="Probability difference threshold to stop")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt selection")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Probability threshold: {args.prob_threshold}")
    print(f"Loading {args.num_prompts} samples from WildVision-bench\n")

    # Load prompts and images
    samples = load_wildvision_prompts(num_samples=args.num_prompts, seed=args.seed)
    
    # Create vLLM engine
    print("\n=== Creating vLLM Engine ===")
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.5)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=1  # Get top-1 logprobs
    )
    
    # Get processor (VLMs use processor instead of tokenizer)
    # processor = llm.get_tokenizer()  # Not needed, we get it later
    
    # Clean up vLLM before loading HF model
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load HuggingFace model and processor
    print("\n=== Loading HuggingFace Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor_hf = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    hf_model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    hf_model.eval()
    
    # Process samples until finding significant mismatch
    print("\n=== Processing Samples ===")
    for i, sample in enumerate(samples):
        print(f"\n\n{'#'*80}")
        print(f"### Processing sample {i+1}/{len(samples)}")
        print(f"{'#'*80}")
        
        # Recreate vLLM for each prompt
        llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.5)
        
        # Process single sample
        result = process_single_prompt(sample, llm, hf_model, processor_hf, sampling_params, device, args)
        
        # Clean up vLLM after each prompt
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check if we found significant mismatch
        if result['first_mismatch_idx'] is not None and result['max_prob_diff'] > args.prob_threshold:
            print(f"\n\n{'*'*80}")
            print(f"*** FOUND SIGNIFICANT MISMATCH ***")
            print(f"*** Max probability difference: {result['max_prob_diff']:.6f} > {args.prob_threshold} ***")
            print(f"*** Stopping after sample {i+1}/{len(samples)} ***")
            print(f"{'*'*80}")
            break
    else:
        print(f"\n\n{'='*80}")
        print(f"=== Processed all {len(samples)} samples without finding significant mismatch ===")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()