#!/usr/bin/env python3
"""
Simple example: Get intermediate outputs from Qwen3-VL-8B-Instruct
"""

from vllm import LLM, SamplingParams

# Initialize the model
print("Loading Qwen3-VL-8B-Instruct...")
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    max_model_len=2048,
    tensor_parallel_size=1,  # Adjust based on your GPU
)

# Create sampling parameters with intermediate outputs enabled
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=50,
    output_hidden_states=True,  # Enable hidden states capture
    output_logits=True,          # Enable logits capture
)

# Simple text prompt
prompt = "What is machine learning? Explain in one sentence."

print(f"\nPrompt: {prompt}")
print("\nGenerating...")

# Generate
outputs = llm.generate(prompt, sampling_params)

# Get the output
output = outputs[0].outputs[0]

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

print(f"\nGenerated text:\n{output.text}")

print(f"\n{'='*80}")
print("INTERMEDIATE OUTPUTS")
print(f"{'='*80}")

# Check hidden states
if output.hidden_states is not None:
    print(f"\n✓ Hidden States captured!")
    print(f"  Shape: {output.hidden_states.shape}")
    print(f"  - Number of tokens: {output.hidden_states.shape[0]}")
    print(f"  - Hidden dimension: {output.hidden_states.shape[1]}")
    print(f"  - Dtype: {output.hidden_states.dtype}")
    print(f"  - Device: {output.hidden_states.device}")

    # Show some statistics
    print(f"\n  Statistics:")
    print(f"  - Mean: {output.hidden_states.mean().item():.4f}")
    print(f"  - Std: {output.hidden_states.std().item():.4f}")
    print(f"  - Min: {output.hidden_states.min().item():.4f}")
    print(f"  - Max: {output.hidden_states.max().item():.4f}")
else:
    print("\n✗ No hidden states (this shouldn't happen)")

# Check logits
if output.logits is not None:
    print(f"\n✓ Logits captured!")
    print(f"  Shape: {output.logits.shape}")
    print(f"  - Number of tokens: {output.logits.shape[0]}")
    print(f"  - Vocabulary size: {output.logits.shape[1]}")
    print(f"  - Dtype: {output.logits.dtype}")
    print(f"  - Device: {output.logits.device}")

    # Show top-5 predictions for the last token
    import torch
    last_logits = output.logits[-1]
    probs = torch.softmax(last_logits, dim=-1)
    top5_probs, top5_indices = probs.topk(5)

    print(f"\n  Top-5 predictions for last token:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"    {i+1}. Token ID {idx.item()}: {prob.item():.4%}")
else:
    print("\n✗ No logits (this shouldn't happen)")

print(f"\n{'='*80}")
print("Token IDs:")
print(f"{'='*80}")
print(output.token_ids)

print("\n✓ Done!")
