#!/usr/bin/env python3
"""
Example: Getting intermediate outputs from Qwen3-VL-8B-Instruct

This script demonstrates how to capture hidden states and logits from
the Qwen3-VL model during inference.
"""

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def example_text_only():
    """Example 1: Text-only input with hidden states"""
    print("=" * 80)
    print("Example 1: Text-only generation with hidden states")
    print("=" * 80)

    # Initialize the model
    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
    )

    # Create sampling params with intermediate outputs enabled
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        output_hidden_states=True,  # Get final layer hidden states
        output_logits=True,          # Get logits before sampling
    )

    # Simple text prompt
    prompt = "Describe the concept of machine learning in one sentence."

    # Generate
    outputs = llm.generate(prompt, sampling_params)

    # Access the output
    output = outputs[0].outputs[0]

    print(f"\nGenerated text: {output.text}")
    print(f"\nGenerated {len(output.token_ids)} tokens")

    # Access intermediate outputs
    if output.hidden_states is not None:
        print(f"\nHidden states shape: {output.hidden_states.shape}")
        print(f"  - Tokens: {output.hidden_states.shape[0]}")
        print(f"  - Hidden size: {output.hidden_states.shape[1]}")

        # Example: Get hidden state for the last generated token
        last_hidden = output.hidden_states[-1]
        print(f"\nLast token hidden state norm: {last_hidden.norm().item():.4f}")

    if output.logits is not None:
        print(f"\nLogits shape: {output.logits.shape}")
        print(f"  - Tokens: {output.logits.shape[0]}")
        print(f"  - Vocab size: {output.logits.shape[1]}")

        # Example: Get top-5 predictions for the last token
        import torch
        last_logits = output.logits[-1]
        top5_probs, top5_indices = torch.softmax(last_logits, dim=-1).topk(5)
        print(f"\nTop-5 predictions for last token:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            print(f"  {i+1}. Token {idx.item()}: {prob.item():.4f}")


def example_with_image():
    """Example 2: Vision-language input with intermediate outputs"""
    print("\n" + "=" * 80)
    print("Example 2: Vision-language generation with intermediate outputs")
    print("=" * 80)

    # Initialize the model
    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
    )

    # Create sampling params with intermediate outputs
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
        output_hidden_states="final",  # Same as True
        output_logits=True,
    )

    # Create a multimodal prompt with image
    # Using vLLM's placeholder format for images
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "What is shown in this image?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Load an example image
    image = ImageAsset("cherry_blossom").pil_image

    # Create multimodal data
    mm_data = {"image": image}

    # Generate
    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": mm_data},
        sampling_params
    )

    # Access the output
    output = outputs[0].outputs[0]

    print(f"\nGenerated text: {output.text}")
    print(f"\nGenerated {len(output.token_ids)} tokens")

    # Access intermediate outputs
    if output.hidden_states is not None:
        print(f"\nHidden states shape: {output.hidden_states.shape}")
        print(f"Hidden states include vision and text tokens")

    if output.logits is not None:
        print(f"\nLogits shape: {output.logits.shape}")


def example_batch_processing():
    """Example 3: Batch processing with selective intermediate outputs"""
    print("\n" + "=" * 80)
    print("Example 3: Batch processing with intermediate outputs")
    print("=" * 80)

    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_model_len=2048,
    )

    # Create different sampling params for different requests
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing briefly.",
        "Describe neural networks in simple terms.",
    ]

    # Request 1: Get hidden states only
    sampling_params_1 = SamplingParams(
        temperature=0.0,
        max_tokens=30,
        output_hidden_states=True,
        output_logits=False,
    )

    # Request 2: Get logits only
    sampling_params_2 = SamplingParams(
        temperature=0.0,
        max_tokens=30,
        output_hidden_states=False,
        output_logits=True,
    )

    # Request 3: Get both
    sampling_params_3 = SamplingParams(
        temperature=0.0,
        max_tokens=30,
        output_hidden_states=True,
        output_logits=True,
    )

    sampling_params_list = [
        sampling_params_1,
        sampling_params_2,
        sampling_params_3,
    ]

    # Generate for all prompts
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params_list,
    )

    # Process outputs
    for i, output in enumerate(outputs):
        completion = output.outputs[0]
        print(f"\n--- Request {i+1} ---")
        print(f"Prompt: {prompts[i]}")
        print(f"Response: {completion.text[:80]}...")
        print(f"Has hidden states: {completion.hidden_states is not None}")
        print(f"Has logits: {completion.logits is not None}")

        if completion.hidden_states is not None:
            print(f"Hidden states shape: {completion.hidden_states.shape}")
        if completion.logits is not None:
            print(f"Logits shape: {completion.logits.shape}")


def example_analysis():
    """Example 4: Analyzing hidden states for different tokens"""
    print("\n" + "=" * 80)
    print("Example 4: Analyzing hidden state representations")
    print("=" * 80)

    import torch

    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        output_hidden_states=True,
    )

    prompt = "The quick brown fox jumps over the lazy dog."
    outputs = llm.generate(prompt, sampling_params)

    output = outputs[0].outputs[0]
    hidden_states = output.hidden_states
    token_ids = output.token_ids

    print(f"\nGenerated text: {output.text}")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"Hidden states shape: {hidden_states.shape}")

    # Analyze hidden state statistics per token
    print("\nPer-token hidden state statistics:")
    print(f"{'Token ID':<12} {'Mean':<12} {'Std':<12} {'Norm':<12}")
    print("-" * 48)

    for i, (token_id, hidden) in enumerate(zip(token_ids[:5], hidden_states[:5])):
        mean_val = hidden.mean().item()
        std_val = hidden.std().item()
        norm_val = hidden.norm().item()
        print(f"{token_id:<12} {mean_val:<12.4f} {std_val:<12.4f} {norm_val:<12.4f}")

    if len(token_ids) > 5:
        print(f"... ({len(token_ids) - 5} more tokens)")

    # Compute similarity between first and last token representations
    if len(hidden_states) > 1:
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            first_hidden.unsqueeze(0),
            last_hidden.unsqueeze(0)
        ).item()

        print(f"\nCosine similarity (first vs last token): {similarity:.4f}")


def example_save_outputs():
    """Example 5: Saving intermediate outputs for later analysis"""
    print("\n" + "=" * 80)
    print("Example 5: Saving intermediate outputs to disk")
    print("=" * 80)

    import torch
    import json
    from pathlib import Path

    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=30,
        output_hidden_states=True,
        output_logits=True,
    )

    prompt = "Explain the importance of intermediate representations in neural networks."
    outputs = llm.generate(prompt, sampling_params)

    output = outputs[0].outputs[0]

    # Create output directory
    output_dir = Path("qwen3_vl_intermediate_outputs")
    output_dir.mkdir(exist_ok=True)

    # Save metadata
    metadata = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "prompt": prompt,
        "generated_text": output.text,
        "token_ids": output.token_ids,
        "num_tokens": len(output.token_ids),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save hidden states
    if output.hidden_states is not None:
        torch.save(
            output.hidden_states,
            output_dir / "hidden_states.pt"
        )
        print(f"\nSaved hidden states: {output.hidden_states.shape}")

    # Save logits
    if output.logits is not None:
        torch.save(
            output.logits,
            output_dir / "logits.pt"
        )
        print(f"Saved logits: {output.logits.shape}")

    print(f"\nAll outputs saved to: {output_dir.absolute()}")

    # Example: Loading back
    print("\nLoading saved outputs...")
    loaded_hidden = torch.load(output_dir / "hidden_states.pt")
    loaded_logits = torch.load(output_dir / "logits.pt")

    print(f"Loaded hidden states: {loaded_hidden.shape}")
    print(f"Loaded logits: {loaded_logits.shape}")


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Text-only", example_text_only),
        "2": ("With image", example_with_image),
        "3": ("Batch processing", example_batch_processing),
        "4": ("Analysis", example_analysis),
        "5": ("Save outputs", example_save_outputs),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            print(f"\nRunning: {name}")
            func()
        else:
            print(f"Unknown example: {choice}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        # Run all examples
        print("Running all examples...")
        print("(Pass example number as argument to run specific example)")
        print()

        for choice, (name, func) in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\nExample {choice} failed: {e}")
                import traceback
                traceback.print_exc()
