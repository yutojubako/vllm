# Getting Intermediate Outputs from Qwen3-VL-8B-Instruct

This guide shows how to capture intermediate outputs (hidden states and logits) from the Qwen3-VL-8B-Instruct model using vLLM.

## Quick Start

### 1. Simple Example (Recommended for testing)

```bash
python3 examples/qwen3_vl_simple.py
```

This will:
- Load the Qwen3-VL-8B-Instruct model
- Generate text with hidden states and logits capture enabled
- Display the captured intermediate outputs

### 2. Comprehensive Examples

```bash
# Run all examples
python3 examples/qwen3_vl_intermediate_outputs.py

# Run specific example
python3 examples/qwen3_vl_intermediate_outputs.py 1  # Text-only
python3 examples/qwen3_vl_intermediate_outputs.py 2  # With image
python3 examples/qwen3_vl_intermediate_outputs.py 3  # Batch processing
python3 examples/qwen3_vl_intermediate_outputs.py 4  # Analysis
python3 examples/qwen3_vl_intermediate_outputs.py 5  # Save outputs
```

## Usage in Your Code

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM("Qwen/Qwen3-VL-8B-Instruct")

# Create sampling params with intermediate outputs
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=50,
    output_hidden_states=True,  # Get final layer hidden states
    output_logits=True,          # Get logits before sampling
)

# Generate
outputs = llm.generate("Your prompt here", sampling_params)

# Access intermediate outputs
output = outputs[0].outputs[0]
hidden_states = output.hidden_states  # Shape: [num_tokens, hidden_dim]
logits = output.logits                # Shape: [num_tokens, vocab_size]
```

### Parameters

#### `output_hidden_states`
- `False` (default): No hidden states
- `True` or `"final"`: Return final layer hidden states
- `"all"`: Return all layer hidden states (future implementation)

#### `output_logits`
- `False` (default): No logits
- `True`: Return pre-sampling logits for each token

#### `output_attention_weights`
- `False` (default): No attention weights
- `True`: Return attention weights (future implementation)

## Vision-Language Example

For multimodal inputs with images:

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(
    output_hidden_states=True,
    output_logits=True,
    max_tokens=100,
)

# Create prompt with image placeholder
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in this image?<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Load image
image = ImageAsset("cherry_blossom").pil_image

# Generate
outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    sampling_params
)

# Access outputs
output = outputs[0].outputs[0]
print(f"Hidden states: {output.hidden_states.shape}")
print(f"Logits: {output.logits.shape}")
```

## Batch Processing with Different Configs

You can request different intermediate outputs for each request:

```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

sampling_params_list = [
    SamplingParams(output_hidden_states=True, output_logits=False),   # Only hidden states
    SamplingParams(output_hidden_states=False, output_logits=True),   # Only logits
    SamplingParams(output_hidden_states=True, output_logits=True),    # Both
]

outputs = llm.generate(prompts, sampling_params=sampling_params_list)
```

## Analyzing Hidden States

```python
import torch

# Get hidden states
hidden_states = output.hidden_states  # [num_tokens, hidden_dim]

# Statistics per token
for i, hidden in enumerate(hidden_states):
    print(f"Token {i}:")
    print(f"  Mean: {hidden.mean().item():.4f}")
    print(f"  Std: {hidden.std().item():.4f}")
    print(f"  Norm: {hidden.norm().item():.4f}")

# Similarity between tokens
similarity = torch.nn.functional.cosine_similarity(
    hidden_states[0].unsqueeze(0),
    hidden_states[-1].unsqueeze(0)
)
print(f"First-Last token similarity: {similarity.item():.4f}")
```

## Analyzing Logits

```python
import torch

# Get logits
logits = output.logits  # [num_tokens, vocab_size]

# Get probabilities
probs = torch.softmax(logits, dim=-1)

# Top-k predictions per token
for i, token_logits in enumerate(logits):
    token_probs = torch.softmax(token_logits, dim=-1)
    top5_probs, top5_indices = token_probs.topk(5)

    print(f"\nToken {i} top-5 predictions:")
    for prob, idx in zip(top5_probs, top5_indices):
        print(f"  Token {idx.item()}: {prob.item():.4%}")
```

## Saving Outputs

```python
import torch

# Save hidden states
torch.save(output.hidden_states, "hidden_states.pt")
torch.save(output.logits, "logits.pt")

# Load later
hidden_states = torch.load("hidden_states.pt")
logits = torch.load("logits.pt")
```

## Memory Considerations

- Intermediate outputs are stored on CPU to minimize GPU memory usage
- Hidden states: `[num_tokens, hidden_dim]` typically `~4KB per token` for 8B models
- Logits: `[num_tokens, vocab_size]` typically `~600KB per token` for 150K vocab
- For long sequences, consider:
  - Reducing `max_tokens`
  - Processing in smaller batches
  - Saving outputs incrementally

## Requirements

- vLLM with intermediate outputs support (this branch)
- Sufficient GPU memory for model weights
- Additional CPU memory for intermediate outputs storage

## GPU Requirements

For Qwen3-VL-8B-Instruct:
- Minimum: 1x GPU with 24GB VRAM (e.g., RTX 4090, A5000)
- Recommended: 1x GPU with 40GB+ VRAM (e.g., A100)
- With tensor parallelism: Multiple GPUs with `tensor_parallel_size` parameter

```python
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
)
```

## Troubleshooting

### "CUDA out of memory"
- Reduce `max_model_len`
- Reduce `max_tokens`
- Use tensor parallelism across multiple GPUs
- Disable intermediate outputs you don't need

### "Intermediate outputs are None"
- Ensure you're using the correct branch with intermediate outputs support
- Verify `output_hidden_states=True` or `output_logits=True` in SamplingParams
- Check that the model execution completed successfully

### Performance Issues
- Intermediate outputs add minimal overhead (<5%) when disabled
- When enabled, CPU transfer time is typically 10-50ms per request
- For maximum throughput, only enable outputs you actually need

## Next Steps

- See `qwen3_vl_intermediate_outputs.py` for more advanced examples
- Check the test files in `tests/v1/sample/test_sampling_params_e2e.py`
- Refer to vLLM documentation for model-specific configurations
