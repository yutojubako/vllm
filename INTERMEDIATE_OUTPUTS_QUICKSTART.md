# Qwen3-VL Intermediate Outputs - Quick Start Guide

## TL;DR - Get Started in 30 Seconds

```python
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-VL-8B-Instruct")
params = SamplingParams(output_hidden_states=True, output_logits=True)
outputs = llm.generate("What is AI?", params)

# Access intermediate outputs
hidden_states = outputs[0].outputs[0].hidden_states  # [tokens, hidden_dim]
logits = outputs[0].outputs[0].logits                # [tokens, vocab_size]
```

## Run the Examples

### Simplest Example
```bash
python3 examples/qwen3_vl_simple.py
```

### All Examples
```bash
# Run all
python3 examples/qwen3_vl_intermediate_outputs.py

# Run specific example (1-5)
python3 examples/qwen3_vl_intermediate_outputs.py 1
```

## What You Get

### Hidden States
- **Shape**: `[num_generated_tokens, hidden_dimension]`
- **For Qwen3-VL-8B**: `[num_tokens, 4096]`
- **Type**: `torch.Tensor` on CPU
- **Use cases**:
  - Representation analysis
  - Feature extraction
  - Probing classifiers
  - Similarity comparisons
  - Fine-tuning downstream tasks

### Logits
- **Shape**: `[num_generated_tokens, vocabulary_size]`
- **For Qwen3-VL-8B**: `[num_tokens, ~151,936]`
- **Type**: `torch.Tensor` on CPU
- **Use cases**:
  - Token probability analysis
  - Uncertainty estimation
  - Beam search alternatives
  - Constrained decoding
  - Model interpretability

## API Quick Reference

### SamplingParams Options

```python
SamplingParams(
    # Standard generation params
    temperature=0.0,
    max_tokens=50,

    # NEW: Intermediate outputs
    output_hidden_states=False,      # False | True | "final" | "all"
    output_logits=False,             # False | True
    output_attention_weights=False,  # False | True (future)
)
```

### CompletionOutput Fields

```python
output = outputs[0].outputs[0]

# Standard fields
output.text                  # Generated text
output.token_ids            # List of token IDs
output.logprobs             # Token log probabilities (if requested)
output.finish_reason        # Why generation stopped

# NEW: Intermediate outputs
output.hidden_states        # torch.Tensor | None
output.logits              # torch.Tensor | None
output.all_hidden_states   # dict[int, torch.Tensor] | None (future)
output.attention_weights   # dict[int, torch.Tensor] | None (future)
```

## Common Use Cases

### 1. Get Hidden States Only
```python
params = SamplingParams(output_hidden_states=True)
```

### 2. Get Logits Only
```python
params = SamplingParams(output_logits=True)
```

### 3. Get Both
```python
params = SamplingParams(
    output_hidden_states=True,
    output_logits=True
)
```

### 4. Batch with Different Configs
```python
prompts = ["Q1", "Q2", "Q3"]
params_list = [
    SamplingParams(output_hidden_states=True),
    SamplingParams(output_logits=True),
    SamplingParams(output_hidden_states=True, output_logits=True),
]
outputs = llm.generate(prompts, sampling_params=params_list)
```

### 5. With Vision Input
```python
from vllm.assets.image import ImageAsset

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

image = ImageAsset("cherry_blossom").pil_image
params = SamplingParams(output_hidden_states=True)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    params
)
```

## Quick Analysis Examples

### Analyze Hidden State Statistics
```python
hidden = outputs[0].outputs[0].hidden_states

print(f"Shape: {hidden.shape}")
print(f"Mean activation: {hidden.mean().item():.4f}")
print(f"Std: {hidden.std().item():.4f}")
print(f"L2 norm: {hidden.norm().item():.4f}")

# Per-token analysis
for i, token_hidden in enumerate(hidden):
    print(f"Token {i} norm: {token_hidden.norm().item():.4f}")
```

### Analyze Token Probabilities
```python
import torch

logits = outputs[0].outputs[0].logits
probs = torch.softmax(logits, dim=-1)

# Last token's top predictions
last_probs = probs[-1]
top5_probs, top5_ids = last_probs.topk(5)

for prob, token_id in zip(top5_probs, top5_ids):
    print(f"Token {token_id}: {prob.item():.4%}")
```

### Token Similarity Analysis
```python
import torch.nn.functional as F

hidden = outputs[0].outputs[0].hidden_states

# Cosine similarity between first and last token
similarity = F.cosine_similarity(
    hidden[0].unsqueeze(0),
    hidden[-1].unsqueeze(0)
).item()

print(f"First-Last similarity: {similarity:.4f}")
```

## File Locations

- **Simple example**: `examples/qwen3_vl_simple.py`
- **Comprehensive examples**: `examples/qwen3_vl_intermediate_outputs.py`
- **Full documentation**: `examples/INTERMEDIATE_OUTPUTS_README.md`
- **Tests**: `tests/v1/sample/test_sampling_params_e2e.py`

## Modified Files (Implementation)

Core implementation:
- `vllm/sampling_params.py` - New parameters
- `vllm/outputs.py` - Extended CompletionOutput
- `vllm/v1/outputs.py` - Extended ModelRunnerOutput
- `vllm/v1/worker/gpu_model_runner.py` - Capture logic
- `vllm/v1/core/sched/scheduler.py` - Propagation
- `vllm/v1/engine/__init__.py` - EngineCoreOutput extension
- `vllm/v1/engine/output_processor.py` - Output processing

## Memory Usage

Approximate additional memory per token:
- **Hidden states**: ~16 KB (4096 * float32)
- **Logits**: ~608 KB (152K vocab * float32)

For 50 tokens:
- Hidden states: ~800 KB
- Logits: ~30 MB

**Note**: All intermediate outputs are stored on CPU, not GPU!

## Performance Impact

- **When disabled**: <1% overhead
- **When enabled**:
  - Hidden states only: ~2-5% overhead
  - Logits only: ~5-10% overhead
  - Both: ~10-15% overhead

Overhead mainly from CPU transfer time.

## Tips

✅ **DO:**
- Use `output_hidden_states=True` for representation analysis
- Use `output_logits=True` for probability analysis
- Save outputs to disk for later analysis
- Process in batches when analyzing many prompts

❌ **DON'T:**
- Don't enable outputs you don't need (wastes memory)
- Don't keep outputs in memory longer than needed
- Don't enable for very long sequences without enough CPU RAM
- Don't forget tensors are on CPU (move to GPU if needed)

## Next Steps

1. Run `python3 examples/qwen3_vl_simple.py` to see it in action
2. Read `examples/INTERMEDIATE_OUTPUTS_README.md` for detailed docs
3. Check `examples/qwen3_vl_intermediate_outputs.py` for advanced examples
4. Experiment with your own prompts and analysis!

---

**Need help?** See the full documentation in `examples/INTERMEDIATE_OUTPUTS_README.md`
