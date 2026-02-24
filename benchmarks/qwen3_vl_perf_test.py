#!/usr/bin/env python3
"""
Performance test specifically for Qwen3-VL-8B-Instruct with intermediate outputs.
"""

import argparse
import time
import torch
import psutil
from typing import Dict
from vllm import LLM, SamplingParams


def get_gpu_memory_gb() -> Dict[str, float]:
    """Get GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {}

    mem = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        mem[f"gpu_{i}_allocated"] = allocated
        mem[f"gpu_{i}_reserved"] = reserved

    return mem


def get_cpu_memory_gb() -> float:
    """Get CPU memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def run_test(llm, prompts, params, name, num_runs=3):
    """Run performance test."""
    print(f"\n{'='*80}")
    print(f"Test: {name}")
    print(f"{'='*80}")

    # Warmup
    print("Warming up...")
    _ = llm.generate(prompts, params)

    # Measure
    times = []
    token_counts = []

    print(f"Running {num_runs} iterations...")
    for i in range(num_runs):
        cpu_before = get_cpu_memory_gb()
        gpu_before = get_gpu_memory_gb()

        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - start

        cpu_after = get_cpu_memory_gb()
        gpu_after = get_gpu_memory_gb()

        times.append(elapsed)

        # Count tokens
        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        token_counts.append(total_tokens)

        print(f"  Run {i+1}: {elapsed:.3f}s, {total_tokens} tokens, "
              f"{total_tokens/elapsed:.1f} tok/s")

        # Memory info (first run only)
        if i == 0:
            cpu_delta = cpu_after - cpu_before
            print(f"  CPU memory delta: {cpu_delta:.3f} GB")

            for key in gpu_after:
                if key in gpu_before:
                    delta = gpu_after[key] - gpu_before[key]
                    print(f"  {key} delta: {delta:.3f} GB")

            # Check intermediate outputs
            sample = outputs[0].outputs[0]
            if sample.hidden_states is not None:
                print(f"  Hidden states shape: {sample.hidden_states.shape}")
                hidden_size_mb = (sample.hidden_states.nelement() *
                                sample.hidden_states.element_size() / (1024**2))
                print(f"  Hidden states size: {hidden_size_mb:.2f} MB")

            if sample.logits is not None:
                print(f"  Logits shape: {sample.logits.shape}")
                logits_size_mb = (sample.logits.nelement() *
                                sample.logits.element_size() / (1024**2))
                print(f"  Logits size: {logits_size_mb:.2f} MB")

    # Statistics
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    throughput = avg_tokens / avg_time

    print(f"\nSummary:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Average tokens: {avg_tokens:.1f}")
    print(f"  Throughput: {throughput:.2f} tokens/sec")

    return {
        "name": name,
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=1,
                       help="Tensor parallel size")
    args = parser.parse_args()

    print("="*80)
    print("Qwen3-VL-8B-Instruct Performance Test")
    print("="*80)
    print(f"Configuration:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Runs per test: {args.num_runs}")
    print(f"  Tensor parallel: {args.tp_size}")

    # Load model
    print(f"\nLoading Qwen3-VL-8B-Instruct...")
    start = time.time()
    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_model_len=2048,
        tensor_parallel_size=args.tp_size,
        enforce_eager=True,
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Memory after loading
    print(f"\nMemory after loading:")
    print(f"  CPU: {get_cpu_memory_gb():.2f} GB")
    for key, val in get_gpu_memory_gb().items():
        print(f"  {key}: {val:.2f} GB")

    # Prompts
    prompts = [
        "What is artificial intelligence? Explain briefly.",
        "Describe the concept of machine learning.",
        "What are neural networks?",
        "Explain deep learning in simple terms.",
    ][:args.batch_size]

    print(f"\nPrompts ({len(prompts)}):")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")

    # Test scenarios
    scenarios = [
        ("Baseline", SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            output_hidden_states=False,
            output_logits=False,
        )),
        ("Hidden states", SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            output_hidden_states=True,
            output_logits=False,
        )),
        ("Logits", SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            output_hidden_states=False,
            output_logits=True,
        )),
        ("Both", SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            output_hidden_states=True,
            output_logits=True,
        )),
    ]

    # Run tests
    results = []
    for name, params in scenarios:
        result = run_test(llm, prompts, params, name, args.num_runs)
        results.append(result)

    # Comparison
    baseline = results[0]

    print(f"\n{'='*80}")
    print("OVERHEAD COMPARISON")
    print("="*80)
    print(f"{'Scenario':<20} {'Time (s)':<12} {'Tokens/s':<12} {'Overhead':<12}")
    print("-"*80)

    for result in results:
        overhead = ""
        if result != baseline:
            overhead_pct = (result["avg_time"] / baseline["avg_time"] - 1) * 100
            overhead = f"{overhead_pct:+.1f}%"

        print(f"{result['name']:<20} "
              f"{result['avg_time']:<12.3f} "
              f"{result['throughput']:<12.2f} "
              f"{overhead:<12}")

    print(f"\n{'='*80}")
    print("Performance Summary:")
    print("="*80)

    for result in results[1:]:
        time_overhead = (result["avg_time"] / baseline["avg_time"] - 1) * 100
        throughput_loss = (1 - result["throughput"] / baseline["throughput"]) * 100

        print(f"\n{result['name']}:")
        print(f"  Time overhead: {time_overhead:+.2f}%")
        print(f"  Throughput impact: {throughput_loss:+.2f}%")
        print(f"  Baseline: {baseline['throughput']:.2f} tok/s")
        print(f"  Current:  {result['throughput']:.2f} tok/s")


if __name__ == "__main__":
    main()
