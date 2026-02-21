---
name: ai-leaner
description: Feynman-method AI learning tool. When the user describes an AI topic (e.g. attention, softmax, flash attention, LayerNorm, GEMM, quantization), generate a hands-on test scaffold in CUDA (.cu) and/or PyTorch (Python) that the user must implement themselves to pass. The scaffold includes: clear TODO stubs, correctness tests against reference implementations, and basic performance checks. Use when the user wants to learn an AI concept by building it, mentions "动手学", "费曼学习", "learn by doing", "implement from scratch", or asks to practice a specific AI kernel/algorithm.
---

# AI 动手学 (Learn by Doing)

**核心原则**: "I cannot understand what I cannot build." — Feynman

## Workflow

When user describes an AI topic, generate a learning scaffold:

1. **Analyze topic** → identify the core algorithmic primitives
2. **Choose scaffold type** based on topic:
   - Low-level (CUDA kernel) → `.cu` scaffold
   - High-level (module/layer) → Python/PyTorch scaffold
   - Both if the topic spans levels
3. **Write the scaffold** (see templates below)
4. **Tell the user**: "Fill in the TODOs, then run the test to verify."

---

## Scaffold Rules

### What to INCLUDE in scaffolds
- `// TODO:` stubs at every implementation point — never fill them in
- A **reference implementation** (NumPy / naive PyTorch / host CPU) used only for correctness comparison — always provided and complete
- Correctness test: `assert allclose(yours, reference)` or CUDA kernel output check
- Shape/dtype hints in comments
- One sentence describing the math at each TODO
- A `# HINT:` comment where the concept is non-obvious (formula, trick, or paper reference)

### What to NEVER do
- Never implement the user's TODO for them
- Never use placeholder variable names without explaining what they represent
- Never write tests that trivially pass with zeros or identity

---

## CUDA Scaffold Template

```cuda
// Topic: <TOPIC>
// Goal: implement <FUNCTION> from scratch
// Build: nvcc -O2 -arch=sm_120 <file>.cu -o <file> && ./<file>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ─── Reference (CPU, already implemented for you) ────────────────────────────
void <function>_reference(/* args */) {
    // complete reference — do not modify
}

// ─── Your CUDA kernel ─────────────────────────────────────────────────────────
__global__ void <function>_kernel(/* args */) {
    // TODO: compute thread index
    // TODO: <math description>
    // HINT: <formula or paper reference>
}

// ─── Host launcher ────────────────────────────────────────────────────────────
void <function>_cuda(/* args */) {
    // TODO: allocate device memory
    // TODO: copy inputs H→D
    // TODO: launch kernel with appropriate grid/block dims
    // TODO: copy output D→H
    // TODO: free device memory
}

// ─── Test harness (do not modify) ────────────────────────────────────────────
int main() {
    // setup inputs, call reference, call your impl
    // compare outputs with tolerance
    // print PASS / FAIL + max error
    // CUDA event timing for throughput
}
```

---

## PyTorch Scaffold Template

```python
# Topic: <TOPIC>
# Goal: implement <Class/Function> from scratch using only torch primitives
# Run: python <file>.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Reference (already implemented for you) ─────────────────────────────────
def reference_<function>(x, ...):
    return F.<builtin>(x, ...)   # or nn.Module equivalent

# ─── Your implementation ──────────────────────────────────────────────────────
def my_<function>(x, ...):
    # TODO: <step 1 — describe the math>
    # TODO: <step 2>
    # HINT: <formula / trick>
    raise NotImplementedError

class My<Module>(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # TODO: define learnable parameters (nn.Parameter)

    def forward(self, x):
        # TODO: implement forward pass
        # HINT: see <paper> eq. <N>
        raise NotImplementedError

# ─── Tests (do not modify) ───────────────────────────────────────────────────
def test_correctness():
    torch.manual_seed(42)
    x = torch.randn(...)
    ref = reference_<function>(x)
    out = my_<function>(x)
    assert torch.allclose(out, ref, atol=1e-5), f"max err: {(out-ref).abs().max():.2e}"
    print("✓ correctness")

def test_gradient():
    x = torch.randn(..., requires_grad=True)
    out = my_<function>(x)
    out.sum().backward()
    assert x.grad is not None
    print("✓ gradient flows")

def benchmark():
    import time
    x = torch.randn(...).cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        my_<function>(x)
    torch.cuda.synchronize()
    print(f"⏱  {(time.perf_counter()-t0)*1e3:.2f} ms / 1000 iters")

if __name__ == "__main__":
    test_correctness()
    test_gradient()
    benchmark()
```

---

## Topic → Scaffold Mapping

| Topic | CUDA scaffold | PyTorch scaffold |
|---|---|---|
| Softmax | vectorized row softmax kernel | `my_softmax` with stability trick |
| LayerNorm | online mean/var kernel | `MyLayerNorm(nn.Module)` |
| Attention (MHA) | QK^T matmul + softmax + AV | `MyAttention` with mask support |
| Flash Attention | tiled online softmax kernel | triton fallback or pure PyTorch |
| RoPE | in-place rotation kernel | `apply_rope(q, k, cos, sin)` |
| GEMM | tiled shared-mem matmul | `my_linear` + weight/bias |
| Quantization (INT8) | absmax scale kernel + dequant | `quantize / dequantize` funcs |
| Dropout | random mask generation kernel | `my_dropout(x, p, training)` |
| Cross-Entropy | log-sum-exp stable kernel | `my_cross_entropy(logits, targets)` |
| Conv2D | im2col or direct kernel | `MyConv2d(nn.Module)` |

---

## Output Format

Always structure your response as:

```
## 主题: <topic>

### 你需要掌握的核心概念
- <concept 1>
- <concept 2>

### 数学公式
<LaTeX or ASCII math>

### 测试文件: <filename>
<full scaffold code>

### 通过测试即意味着你真正理解了这个概念
运行方法: <build/run command>
```

Do NOT explain the implementation — let the user discover it. Only explain the *goal* and *math*.

---

## Example Trigger Phrases

- "动手学 softmax"
- "我想理解 Flash Attention，给我测试框架"
- "费曼学习 LayerNorm"
- "learn by doing: rotary position embedding"
- "给我一个 GEMM 的练习"
