# Intermediate Outputs Performance Benchmarks

This directory contains performance benchmarks for the intermediate outputs feature.

## Quick Start

### 1. Quick Test (30 seconds, small model)

```bash
python3 benchmarks/quick_perf_test.py
```

Uses `facebook/opt-125m` for fast testing. Shows overhead of:
- Hidden states only
- Logits only
- Both hidden states and logits

### 2. Qwen3-VL Test (requires GPU)

```bash
python3 benchmarks/qwen3_vl_perf_test.py
```

Tests `Qwen/Qwen3-VL-8B-Instruct` specifically.

**Options:**
```bash
python3 benchmarks/qwen3_vl_perf_test.py \
    --max-tokens 50 \
    --batch-size 2 \
    --num-runs 3 \
    --tp-size 1
```

### 3. Comprehensive Benchmark

```bash
python3 benchmarks/benchmark_intermediate_outputs.py \
    --model facebook/opt-125m \
    --max-tokens 50 \
    --batch-size 4 \
    --num-iterations 5
```

For Qwen3-VL:
```bash
python3 benchmarks/benchmark_intermediate_outputs.py \
    --use-qwen \
    --max-tokens 50 \
    --batch-size 2 \
    --num-iterations 5 \
    --output qwen_results.json
```

## Benchmark Scripts

### `quick_perf_test.py`
- **Purpose**: Fast sanity check
- **Model**: facebook/opt-125m (tiny, fast)
- **Time**: ~30 seconds
- **Use when**: Quick validation of overhead

### `qwen3_vl_perf_test.py`
- **Purpose**: Test Qwen3-VL specifically
- **Model**: Qwen/Qwen3-VL-8B-Instruct
- **Time**: ~5-10 minutes
- **Use when**: Testing production model
- **GPU Required**: 24GB+ VRAM

### `benchmark_intermediate_outputs.py`
- **Purpose**: Comprehensive benchmarking
- **Model**: Configurable
- **Output**: JSON results file
- **Use when**: Detailed performance analysis

## Expected Performance Impact

Based on preliminary testing:

| Configuration | Time Overhead | Throughput Impact |
|--------------|---------------|-------------------|
| Baseline (disabled) | 0% | 0% |
| Hidden states only | +2-5% | -2-5% |
| Logits only | +5-10% | -5-10% |
| Both | +10-15% | -10-15% |

**Note**: Actual numbers vary by:
- Model size
- Sequence length
- Batch size
- Hardware
- GPU utilization

## Memory Impact

### Per Token (Approximate)

**Qwen3-VL-8B-Instruct:**
- Hidden states: ~16 KB (4096 dim × 4 bytes)
- Logits: ~608 KB (152K vocab × 4 bytes)
- Total both: ~624 KB per token

**For 50 tokens:**
- Hidden states: ~800 KB
- Logits: ~30 MB
- Total: ~31 MB per request

**Batch of 4 requests with 50 tokens:**
- Hidden states: ~3.2 MB
- Logits: ~120 MB
- Total: ~124 MB additional CPU memory

## Detailed Analysis

### Run Full Benchmark with Analysis

```bash
# Small model (fast)
python3 benchmarks/benchmark_intermediate_outputs.py \
    --model facebook/opt-125m \
    --batch-size 4 \
    --max-tokens 50 \
    --num-iterations 10 \
    --output results_opt125m.json

# Qwen3-VL (production)
python3 benchmarks/benchmark_intermediate_outputs.py \
    --use-qwen \
    --batch-size 2 \
    --max-tokens 50 \
    --num-iterations 5 \
    --output results_qwen3vl.json
```

### Analyze Results

```python
import json

with open("results_qwen3vl.json") as f:
    data = json.load(f)

baseline = data["results"][0]
for result in data["results"][1:]:
    overhead = (result["avg_time_sec"] / baseline["avg_time_sec"] - 1) * 100
    print(f"{result['scenario']}: {overhead:+.2f}% overhead")
```

## Command Line Options

### Common Options (all scripts)

- `--max-tokens N` - Maximum tokens to generate (default: 50)
- `--batch-size N` - Number of prompts per batch (default: varies)
- `--num-runs N` / `--num-iterations N` - Number of test iterations

### qwen3_vl_perf_test.py

- `--tp-size N` - Tensor parallel size for multi-GPU (default: 1)

### benchmark_intermediate_outputs.py

- `--model MODEL` - Model name (default: facebook/opt-125m)
- `--use-qwen` - Use Qwen3-VL-8B-Instruct
- `--num-warmup N` - Warmup iterations (default: 2)
- `--output FILE` - Output JSON file (default: benchmark_results.json)

## Interpreting Results

### Time Overhead
Percentage increase in generation time:
- **<5%**: Negligible impact
- **5-10%**: Acceptable for most use cases
- **>10%**: Consider if intermediate outputs are necessary

### Throughput Impact
Percentage decrease in tokens/second:
- Inverse of time overhead
- More intuitive for production planning

### Memory Delta
Additional memory used:
- CPU memory: Where intermediate outputs are stored
- GPU memory: Should be minimal (only during transfer)

## Tips for Benchmarking

1. **Warmup is important**: First run includes compilation overhead
2. **Multiple iterations**: Average over 5+ runs for stability
3. **Consistent load**: Close other applications
4. **GPU monitoring**: Watch `nvidia-smi` during tests
5. **Batch size matters**: Overhead percentage changes with batch size

## Troubleshooting

### OOM (Out of Memory)

**GPU OOM:**
- Reduce `--batch-size`
- Reduce `--max-tokens`
- Use `--tp-size` for tensor parallelism

**CPU OOM:**
- Disable logits (they're large)
- Reduce batch size
- Process shorter sequences

### Slow Performance

- Increase warmup iterations
- Check GPU utilization with `nvidia-smi`
- Ensure no CPU throttling

### Inconsistent Results

- Run more iterations (`--num-iterations 10`)
- Check background processes
- Monitor thermal throttling

## Example: Production Sizing

To estimate production performance:

```bash
# Test with your expected workload
python3 benchmarks/qwen3_vl_perf_test.py \
    --max-tokens 100 \      # Your typical output length
    --batch-size 8 \        # Your batch size
    --num-runs 10           # More runs for stability
```

Calculate capacity:
```
baseline_throughput = 50 tokens/sec
overhead = 10%  # from benchmark
effective_throughput = 50 * 0.90 = 45 tokens/sec

requests_per_second = effective_throughput / avg_tokens_per_request
```

## Continuous Benchmarking

For regression testing:

```bash
# Run benchmark
python3 benchmarks/benchmark_intermediate_outputs.py \
    --output baseline.json

# After code changes
python3 benchmarks/benchmark_intermediate_outputs.py \
    --output new_version.json

# Compare
python3 -c "
import json
baseline = json.load(open('baseline.json'))
new = json.load(open('new_version.json'))
print(f'Baseline: {baseline[\"baseline_throughput\"]:.2f} tok/s')
print(f'New:      {new[\"baseline_throughput\"]:.2f} tok/s')
"
```

## Contributing

When adding new features, please:
1. Run benchmarks before and after
2. Document any performance changes
3. Update expected impact table if needed
