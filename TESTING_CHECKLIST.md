# Testing Checklist - Intermediate Outputs

## Prerequisites

### Install Dependencies
```bash
# Option 1: Using pip
pip install torch vllm transformers pillow

# Option 2: Using uv (if available)
uv pip install torch vllm transformers pillow

# Option 3: Build from source
cd /Users/yuto.imai/study/vllm_fork
pip install -e .
```

---

## Phase 1: Syntax & Import Tests ✅ DONE

```bash
# All files compile successfully
python3 -m py_compile vllm/sampling_params.py
python3 -m py_compile vllm/outputs.py
python3 -m py_compile vllm/v1/outputs.py
python3 -m py_compile vllm/v1/worker/gpu_model_runner.py
python3 -m py_compile vllm/v1/core/sched/scheduler.py
python3 -m py_compile vllm/v1/engine/__init__.py
python3 -m py_compile vllm/v1/engine/output_processor.py
```

---

## Phase 2: Unit Tests ⏳ TODO

### Test 1: Parameter Validation
```bash
pytest tests/v1/sample/test_sampling_params_e2e.py::test_intermediate_outputs_validation -v
```

**Expected:**
- ✅ Valid values accepted (False, True, "final", "all")
- ✅ Invalid values rejected with ValueError

**Check:**
- [ ] Test passes
- [ ] Error messages are clear

---

### Test 2: Hidden States Capture
```bash
pytest tests/v1/sample/test_sampling_params_e2e.py::test_output_hidden_states -v
```

**Expected:**
- ✅ `output_hidden_states=True` returns hidden states
- ✅ `output_hidden_states="final"` returns hidden states
- ✅ `output_hidden_states=False` returns None
- ✅ Hidden states have correct shape: [num_tokens, hidden_dim]

**Check:**
- [ ] Test passes
- [ ] Shapes are correct
- [ ] Tensors are on CPU

---

### Test 3: Logits Capture
```bash
pytest tests/v1/sample/test_sampling_params_e2e.py::test_output_logits -v
```

**Expected:**
- ✅ `output_logits=True` returns logits
- ✅ `output_logits=False` returns None
- ✅ Logits have correct shape: [num_tokens, vocab_size]

**Check:**
- [ ] Test passes
- [ ] Shapes are correct
- [ ] Tensors are on CPU

---

## Phase 3: Integration Tests ⏳ TODO

### Test 4: Simple Example
```bash
python3 examples/qwen3_vl_simple.py
```

**Expected Output:**
```
Loading Qwen3-VL-8B-Instruct...
Model loaded in X.XXs

Prompt: What is machine learning? Explain in one sentence.

Generating...

================================================================================
RESULTS
================================================================================

Generated text:
[Generated response here]

================================================================================
INTERMEDIATE OUTPUTS
================================================================================

✓ Hidden States captured!
  Shape: torch.Size([N, 4096])
  - Number of tokens: N
  - Hidden dimension: 4096
  - Dtype: torch.float32
  - Device: cpu

  Statistics:
  - Mean: X.XXXX
  - Std: X.XXXX
  - Min: X.XXXX
  - Max: X.XXXX

✓ Logits captured!
  Shape: torch.Size([N, VOCAB_SIZE])
  - Number of tokens: N
  - Vocabulary size: ~152K
  - Dtype: torch.float32
  - Device: cpu

  Top-5 predictions for last token:
    1. Token ID XXXX: XX.XX%
    2. Token ID XXXX: XX.XX%
    ...

✓ Done!
```

**Check:**
- [ ] Model loads successfully
- [ ] Hidden states are present and correct shape
- [ ] Logits are present and correct shape
- [ ] Tensors are on CPU
- [ ] Statistics look reasonable

---

### Test 5: All Examples
```bash
# Run all 5 examples
python3 examples/qwen3_vl_intermediate_outputs.py

# Or run individually
python3 examples/qwen3_vl_intermediate_outputs.py 1  # Text-only
python3 examples/qwen3_vl_intermediate_outputs.py 2  # With image
python3 examples/qwen3_vl_intermediate_outputs.py 3  # Batch
python3 examples/qwen3_vl_intermediate_outputs.py 4  # Analysis
python3 examples/qwen3_vl_intermediate_outputs.py 5  # Save
```

**Check for each example:**
- [ ] No errors
- [ ] Outputs look reasonable
- [ ] Memory usage acceptable

---

## Phase 4: Performance Tests ⏳ TODO

### Test 6: Quick Performance Test
```bash
python3 benchmarks/quick_perf_test.py
```

**Expected:**
```
Quick Performance Test - Intermediate Outputs
================================================================================

Loading model (facebook/opt-125m)...
Model loaded in X.XXs
Prompt: What is machine learning? Explain in one sentence.
Max tokens: 30

Running tests...

Baseline             X.XXXs  XXX.X tok/s  hidden=False  logits=False
Hidden states        X.XXXs  XXX.X tok/s  hidden=True   logits=False
Logits               X.XXXs  XXX.X tok/s  hidden=False  logits=True
Both                 X.XXXs  XXX.X tok/s  hidden=True   logits=True

================================================================================
Overhead Analysis
================================================================================

Hidden states:
  Time overhead:     +X.XX%
  Throughput impact: -X.XX%

Logits:
  Time overhead:     +X.XX%
  Throughput impact: -X.XX%

Both:
  Time overhead:     +XX.XX%
  Throughput impact: -XX.XX%

================================================================================
Test complete!
================================================================================
```

**Check:**
- [ ] Baseline runs successfully
- [ ] Hidden states overhead < 10%
- [ ] Logits overhead < 15%
- [ ] Both overhead < 20%

---

### Test 7: Qwen3-VL Performance
```bash
python3 benchmarks/qwen3_vl_perf_test.py --batch-size 2 --num-runs 3
```

**Expected:**
- Similar overhead percentages
- Memory usage tracking shows reasonable increases
- No OOM errors

**Check:**
- [ ] All scenarios complete
- [ ] Overhead percentages reasonable
- [ ] Memory deltas make sense
- [ ] GPU memory not significantly impacted

---

### Test 8: Comprehensive Benchmark
```bash
# Small model (fast)
python3 benchmarks/benchmark_intermediate_outputs.py \
    --model facebook/opt-125m \
    --batch-size 4 \
    --num-iterations 5 \
    --output results_test.json

# Check results
cat results_test.json
```

**Check:**
- [ ] JSON output created
- [ ] All scenarios completed
- [ ] Results consistent with quick test

---

## Phase 5: Edge Cases ⏳ TODO

### Test 9: Large Batch
```python
from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m")
params = SamplingParams(output_hidden_states=True, output_logits=True)

# Large batch
prompts = [f"Question {i}" for i in range(32)]
outputs = llm.generate(prompts, params)

# Verify all have intermediate outputs
for i, output in enumerate(outputs):
    out = output.outputs[0]
    assert out.hidden_states is not None, f"Missing hidden states for {i}"
    assert out.logits is not None, f"Missing logits for {i}"
    print(f"{i}: hidden={out.hidden_states.shape}, logits={out.logits.shape}")
```

**Check:**
- [ ] All requests get intermediate outputs
- [ ] No memory errors
- [ ] Shapes are consistent

---

### Test 10: Long Sequence
```python
from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m")
params = SamplingParams(
    output_hidden_states=True,
    output_logits=True,
    max_tokens=200  # Longer sequence
)

outputs = llm.generate("Write a long story about", params)
out = outputs[0].outputs[0]

print(f"Tokens: {len(out.token_ids)}")
print(f"Hidden: {out.hidden_states.shape}")
print(f"Logits: {out.logits.shape}")

# Calculate memory
hidden_mb = out.hidden_states.nelement() * 4 / (1024**2)
logits_mb = out.logits.nelement() * 4 / (1024**2)
print(f"Hidden states: {hidden_mb:.2f} MB")
print(f"Logits: {logits_mb:.2f} MB")
```

**Check:**
- [ ] Works with longer sequences
- [ ] Memory scales linearly
- [ ] No performance degradation

---

### Test 11: Mixed Configuration Batch
```python
from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m")

prompts = ["Q1", "Q2", "Q3", "Q4"]
params_list = [
    SamplingParams(max_tokens=20),  # No intermediate outputs
    SamplingParams(max_tokens=20, output_hidden_states=True),
    SamplingParams(max_tokens=20, output_logits=True),
    SamplingParams(max_tokens=20, output_hidden_states=True, output_logits=True),
]

outputs = llm.generate(prompts, sampling_params=params_list)

# Verify each has correct outputs
assert outputs[0].outputs[0].hidden_states is None
assert outputs[0].outputs[0].logits is None

assert outputs[1].outputs[0].hidden_states is not None
assert outputs[1].outputs[0].logits is None

assert outputs[2].outputs[0].hidden_states is None
assert outputs[2].outputs[0].logits is not None

assert outputs[3].outputs[0].hidden_states is not None
assert outputs[3].outputs[0].logits is not None

print("✓ Mixed configuration works correctly!")
```

**Check:**
- [ ] Each request gets correct outputs
- [ ] No interference between requests

---

## Phase 6: Regression Tests ⏳ TODO

### Test 12: Backward Compatibility
```python
# Old code should still work
from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m")
params = SamplingParams(temperature=0.0, max_tokens=20)

outputs = llm.generate("Hello", params)
out = outputs[0].outputs[0]

# These should all be None (default behavior)
assert out.hidden_states is None
assert out.logits is None
assert out.all_hidden_states is None
assert out.attention_weights is None

print("✓ Backward compatibility maintained!")
```

**Check:**
- [ ] Default behavior unchanged
- [ ] No performance impact when disabled

---

## Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| 1. Parameter Validation | ⏳ | |
| 2. Hidden States Capture | ⏳ | |
| 3. Logits Capture | ⏳ | |
| 4. Simple Example | ⏳ | |
| 5. All Examples | ⏳ | |
| 6. Quick Performance | ⏳ | |
| 7. Qwen3-VL Performance | ⏳ | |
| 8. Comprehensive Benchmark | ⏳ | |
| 9. Large Batch | ⏳ | |
| 10. Long Sequence | ⏳ | |
| 11. Mixed Configuration | ⏳ | |
| 12. Backward Compatibility | ⏳ | |

**Legend:**
- ✅ Passed
- ❌ Failed
- ⏳ Not yet run
- ⚠️ Passed with warnings

---

## Troubleshooting

### Issue: Import Errors
```
ModuleNotFoundError: No module named 'vllm'
```
**Solution:** Install dependencies or build from source

### Issue: CUDA OOM
```
torch.cuda.OutOfMemoryError
```
**Solution:**
- Reduce batch size
- Reduce max_tokens
- Use smaller model for testing
- Use tensor parallelism

### Issue: Test Timeout
**Solution:**
- Increase timeout in pytest
- Use smaller model
- Reduce number of iterations

### Issue: Inconsistent Results
**Solution:**
- Increase warmup iterations
- Check for background processes
- Monitor thermal throttling

---

## Next Steps After Testing

1. ✅ All tests pass → Ready for production
2. ⚠️ Performance issues → Optimize bottlenecks
3. ❌ Failures → Debug and fix
4. 📝 Document findings
5. 🚀 Create PR or deploy

---

## Contact

For issues or questions:
- Check `IMPLEMENTATION_SUMMARY.md`
- Review `examples/INTERMEDIATE_OUTPUTS_README.md`
- See benchmarks in `benchmarks/BENCHMARK_README.md`
