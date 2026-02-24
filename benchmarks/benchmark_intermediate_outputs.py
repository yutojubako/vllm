#!/usr/bin/env python3
"""
Performance benchmark for intermediate outputs feature.

Tests the overhead of capturing hidden states and logits during inference.
"""

import argparse
import time
from typing import Dict, List
import json
import psutil
import torch

from vllm import LLM, SamplingParams


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in GB."""
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / (1024 ** 3)

    gpu_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem[f"gpu_{i}"] = torch.cuda.memory_allocated(i) / (1024 ** 3)

    return {
        "cpu_gb": cpu_mem,
        **gpu_mem
    }


def benchmark_scenario(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    scenario_name: str,
    num_warmup: int = 2,
    num_iterations: int = 5,
) -> Dict:
    """Benchmark a specific scenario."""
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*80}")

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        _ = llm.generate(prompts, sampling_params)

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []
    mem_before = get_memory_usage()

    for i in range(num_iterations):
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")

    mem_after = get_memory_usage()

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate throughput
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_sec = total_tokens / avg_time

    # Memory delta
    mem_delta = {
        key: mem_after[key] - mem_before.get(key, 0)
        for key in mem_after.keys()
    }

    # Check intermediate outputs
    sample_output = outputs[0].outputs[0]
    has_hidden = sample_output.hidden_states is not None
    has_logits = sample_output.logits is not None

    hidden_shape = sample_output.hidden_states.shape if has_hidden else None
    logits_shape = sample_output.logits.shape if has_logits else None

    results = {
        "scenario": scenario_name,
        "num_prompts": len(prompts),
        "total_tokens": total_tokens,
        "avg_time_sec": avg_time,
        "min_time_sec": min_time,
        "max_time_sec": max_time,
        "tokens_per_sec": tokens_per_sec,
        "memory_delta_gb": mem_delta,
        "has_hidden_states": has_hidden,
        "has_logits": has_logits,
        "hidden_shape": str(hidden_shape) if hidden_shape else None,
        "logits_shape": str(logits_shape) if logits_shape else None,
    }

    print(f"\nResults:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"  Memory delta: {mem_delta}")
    print(f"  Has hidden states: {has_hidden}")
    print(f"  Has logits: {has_logits}")
    if hidden_shape:
        print(f"  Hidden shape: {hidden_shape}")
    if logits_shape:
        print(f"  Logits shape: {logits_shape}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark intermediate outputs")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to benchmark (default: facebook/opt-125m for quick testing)"
    )
    parser.add_argument(
        "--use-qwen",
        action="store_true",
        help="Use Qwen3-VL-8B-Instruct instead (requires more GPU memory)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts in batch (default: 4)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file (default: benchmark_results.json)"
    )

    args = parser.parse_args()

    # Select model
    if args.use_qwen:
        model_name = "Qwen/Qwen3-VL-8B-Instruct"
        model_kwargs = {"max_model_len": 2048}
    else:
        model_name = args.model
        model_kwargs = {}

    print(f"\n{'='*80}")
    print(f"INTERMEDIATE OUTPUTS PERFORMANCE BENCHMARK")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Warmup: {args.num_warmup}")

    # Load model
    print(f"\nLoading model...")
    start = time.time()
    llm = LLM(model=model_name, enforce_eager=True, **model_kwargs)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Create prompts
    prompts = [
        "What is machine learning? Explain briefly.",
        "Describe the concept of neural networks.",
        "What are the benefits of artificial intelligence?",
        "How does deep learning work?",
    ][:args.batch_size]

    print(f"\nPrompts ({len(prompts)}):")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p[:60]}...")

    # Define test scenarios
    scenarios = [
        {
            "name": "Baseline (no intermediate outputs)",
            "params": SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                output_hidden_states=False,
                output_logits=False,
            )
        },
        {
            "name": "Hidden states only",
            "params": SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                output_hidden_states=True,
                output_logits=False,
            )
        },
        {
            "name": "Logits only",
            "params": SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                output_hidden_states=False,
                output_logits=True,
            )
        },
        {
            "name": "Both hidden states and logits",
            "params": SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                output_hidden_states=True,
                output_logits=True,
            )
        },
    ]

    # Run benchmarks
    results = []
    for scenario in scenarios:
        result = benchmark_scenario(
            llm=llm,
            prompts=prompts,
            sampling_params=scenario["params"],
            scenario_name=scenario["name"],
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
        )
        results.append(result)

    # Calculate overhead
    baseline = results[0]
    print(f"\n{'='*80}")
    print(f"OVERHEAD ANALYSIS (compared to baseline)")
    print(f"{'='*80}")

    for result in results[1:]:
        time_overhead = (result["avg_time_sec"] / baseline["avg_time_sec"] - 1) * 100
        throughput_impact = (1 - result["tokens_per_sec"] / baseline["tokens_per_sec"]) * 100

        print(f"\n{result['scenario']}:")
        print(f"  Time overhead: {time_overhead:+.2f}%")
        print(f"  Throughput impact: {throughput_impact:+.2f}%")
        print(f"  Baseline: {baseline['tokens_per_sec']:.2f} tok/s")
        print(f"  Current:  {result['tokens_per_sec']:.2f} tok/s")

    # Save results
    output_data = {
        "model": model_name,
        "config": {
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "num_iterations": args.num_iterations,
            "num_warmup": args.num_warmup,
        },
        "results": results,
        "baseline_throughput": baseline["tokens_per_sec"],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Scenario':<40} {'Time (s)':<12} {'Tokens/s':<12} {'Overhead':<12}")
    print(f"{'-'*80}")

    for i, result in enumerate(results):
        overhead = ""
        if i > 0:
            overhead_pct = (result["avg_time_sec"] / baseline["avg_time_sec"] - 1) * 100
            overhead = f"{overhead_pct:+.1f}%"

        print(f"{result['scenario']:<40} "
              f"{result['avg_time_sec']:<12.3f} "
              f"{result['tokens_per_sec']:<12.2f} "
              f"{overhead:<12}")


if __name__ == "__main__":
    main()
