#!/usr/bin/env python3
"""
Quick performance test for intermediate outputs.
Uses a small model for fast testing.
"""

import time
import torch
from vllm import LLM, SamplingParams


def test_scenario(llm, prompt, params, name):
    """Test one scenario and return timing."""
    # Warmup
    _ = llm.generate(prompt, params)

    # Timed run
    start = time.perf_counter()
    outputs = llm.generate(prompt, params)
    elapsed = time.perf_counter() - start

    output = outputs[0].outputs[0]
    num_tokens = len(output.token_ids)

    return {
        "name": name,
        "time": elapsed,
        "tokens": num_tokens,
        "tokens_per_sec": num_tokens / elapsed,
        "has_hidden": output.hidden_states is not None,
        "has_logits": output.logits is not None,
    }


def main():
    print("Quick Performance Test - Intermediate Outputs")
    print("=" * 80)

    # Use tiny model for speed
    print("\nLoading model (facebook/opt-125m)...")
    llm = LLM("facebook/opt-125m", enforce_eager=True)

    prompt = "What is machine learning? Explain in one sentence."
    max_tokens = 30

    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"\nRunning tests...\n")

    # Test scenarios
    scenarios = [
        ("Baseline", SamplingParams(max_tokens=max_tokens)),
        ("Hidden states", SamplingParams(max_tokens=max_tokens, output_hidden_states=True)),
        ("Logits", SamplingParams(max_tokens=max_tokens, output_logits=True)),
        ("Both", SamplingParams(max_tokens=max_tokens, output_hidden_states=True, output_logits=True)),
    ]

    results = []
    for name, params in scenarios:
        result = test_scenario(llm, prompt, params, name)
        results.append(result)
        print(f"{name:20} {result['time']:.3f}s  {result['tokens_per_sec']:.1f} tok/s  "
              f"hidden={result['has_hidden']}  logits={result['has_logits']}")

    # Calculate overhead
    baseline_time = results[0]["time"]
    baseline_throughput = results[0]["tokens_per_sec"]

    print(f"\n{'='*80}")
    print("Overhead Analysis")
    print("=" * 80)

    for result in results[1:]:
        time_overhead = (result["time"] / baseline_time - 1) * 100
        throughput_loss = (1 - result["tokens_per_sec"] / baseline_throughput) * 100

        print(f"\n{result['name']}:")
        print(f"  Time overhead:     {time_overhead:+6.2f}%")
        print(f"  Throughput impact: {throughput_loss:+6.2f}%")

    print(f"\n{'='*80}")
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
