---
name: kernel-analysis
description: Profile and analyze NVIDIA GPU kernels using ncu (Nsight Compute) with --set full. Use when the user asks to profile, benchmark, analyze, or optimize a CUDA kernel, or when investigating GPU performance bottlenecks, occupancy, memory throughput, or compute utilization.
---

# CUDA Kernel Performance Analysis

## Overview

This skill profiles CUDA kernels with `ncu --set full`, interprets the results, cross-references with source code, and delivers actionable optimization recommendations.

## Core Principle: Teach While Analyzing

**Every analysis must be a learning experience for the user.** Do NOT just dump metrics and conclusions. Instead:

1. **逐步拆解 (Step-by-step breakdown)**: Walk through each metric section one at a time. For every metric, explain:
   - What this metric measures at the hardware level (e.g., "L2 hit rate tells us how often data requested by the SM was already cached in the 40MB L2 — a miss means a ~600-cycle round trip to HBM")
   - What the observed value means for THIS specific kernel
   - Why it matters for performance

2. **代码与数据对照 (Code-data cross-reference)**: When a metric reveals an issue, immediately show the corresponding source code and explain the causal chain:
   - "The 45% global load efficiency comes from line 37: `A[row * N + col]` — when threads in a warp have different `row` values, each thread hits a different 128B cache line, wasting 55% of each memory transaction"

3. **思维过程外化 (Externalize reasoning)**: Share the analytical thought process:
   - "I see SM% is 23% and Memory% is 71%, so this kernel is clearly memory-bound. This means compute optimizations won't help — we need to either reduce memory traffic or improve access patterns. Let me check the memory workload section to pinpoint where..."

4. **经验法则传授 (Teach rules of thumb)**: When relevant, share practical heuristics:
   - "A useful rule: if your kernel does fewer than ~10 FLOPs per byte loaded from global memory, it's almost certainly memory-bound on modern GPUs (arithmetic intensity < 10)"

5. **对比式教学 (Teach by comparison)**: When proposing optimizations, show before/after:
   - Original code + expected hardware behavior
   - Optimized code + why it will be faster
   - Which metric should improve and by roughly how much

6. **Ask and answer "why" recursively**: Don't stop at surface-level observations. Dig deeper:
   - Surface: "Occupancy is 50%"
   - Level 1: "Because the kernel uses 48 registers per thread"
   - Level 2: "Because the unrolled inner loop at line 52-68 keeps 12 intermediate float variables alive simultaneously"
   - Level 3: "We can reduce this to 8 by splitting the accumulation into two passes, or use `__launch_bounds__(256, 4)` to force 32 registers with register spilling to local memory — but spilling would hurt more than low occupancy here because..."

## Workflow

### Step 1: Build the Target

Compile with debug + line info for source correlation:

```bash
nvcc -O3 -lineinfo -arch=sm_120 -o <binary> <source.cu>
```

- Use `-lineinfo` (NOT `-G`) to keep optimizations while enabling source mapping.
- Detect compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`

### Step 2: Profile with NCU

Run the full metric set:

```bash
ncu --set full --target-processes all -o <profile_output> ./<binary>
```

**Common variants:**

| Scenario | Command |
|----------|---------|
| Profile specific kernel | `ncu --set full --kernel-name <regex> ./<binary>` |
| Skip warmup launches | `ncu --set full --launch-skip <N> --launch-count <M> ./<binary>` |
| CSV export for parsing | `ncu --set full --csv ./<binary>` |
| Page-by-page stdout | `ncu --set full --page raw ./<binary>` |
| Import saved report | `ncu -i <profile_output>.ncu-rep --page raw` |

**Important:** If the binary requires arguments, append them after the binary path. If profiling hangs, add `--target-processes all` or reduce launch count.

### Step 3: Analyze Key Metrics (逐步详解)

Read the ncu output and walk the user through each section **in priority order**. For each section:
- **先读数据**: Quote the exact metric values from the profiling output
- **再解释含义**: Explain what the hardware is doing behind these numbers
- **然后定位代码**: Point to the specific source lines causing this behavior
- **最后给出判断**: State whether this is a problem, how severe it is, and what to do

#### 3.1 GPU Speed of Light (SOL)

The most critical section. Check these first:

| Metric | What It Tells You |
|--------|-------------------|
| SM [%] | Compute utilization vs peak (target: >60%) |
| Memory [%] | Memory bandwidth utilization vs peak (target: >60%) |
| Roofline position | Whether kernel is compute-bound or memory-bound |

**Classification:**
- **Compute-bound**: SM% >> Memory% → optimize arithmetic, use tensor cores, reduce instruction count
- **Memory-bound**: Memory% >> SM% → optimize memory access patterns, increase arithmetic intensity
- **Latency-bound**: Both SM% and Memory% are low → increase parallelism, reduce stalls, improve occupancy

#### 3.2 Memory Workload Analysis

| Metric | Target | Issue If Bad |
|--------|--------|--------------|
| Global Load/Store Efficiency | >50% | Uncoalesced access, strided patterns |
| L1 Hit Rate | Context-dependent | Thrashing or poor locality |
| L2 Hit Rate | >50% for reused data | Working set too large, poor tiling |
| Shared Memory Bank Conflicts | 0 | Padding needed or access pattern change |
| DRAM Throughput | Close to SOL% | Underutilized bandwidth |

**Key questions to answer:**
- Are global memory accesses coalesced? (check sector efficiency)
- Is shared memory used effectively? (bank conflicts, occupancy impact)
- Is L2 cache being leveraged? (tile sizes vs L2 capacity)

#### 3.3 Compute Workload Analysis

| Metric | What to Check |
|--------|---------------|
| Warp Execution Efficiency | <100% indicates branch divergence |
| Eligible Warps Per Scheduler | <1 means starvation, latency-bound |
| Instruction Mix (FP32/FP16/INT) | Match to algorithm's actual needs |
| Pipe Utilization | Identify bottleneck pipe (ALU, FMA, SFU, LSU, TEX) |
| Tensor Core Utilization | >0% if using WMMA/MMA (otherwise wasted opportunity) |

#### 3.4 Occupancy Analysis

| Metric | Action |
|--------|--------|
| Achieved Occupancy | Compare with theoretical; low achieved = latency issues |
| Theoretical Occupancy | Limited by registers, shared memory, or block size |
| Limiting Factor | Address the specific limiter: registers → `__launch_bounds__`, smem → reduce usage, block size → adjust |

#### 3.5 Scheduler Statistics

| Metric | Interpretation |
|--------|---------------|
| Stall reasons breakdown | Identifies WHY warps are stalled |
| Long Scoreboard | Waiting on global memory → prefetch, increase ILP |
| Short Scoreboard | Waiting on shared memory / math pipe → pipeline better |
| Not Selected | Enough warps but scheduler is busy → ok if occupancy is high |
| Wait | Barrier synchronization → reduce sync points, balance work |

#### 3.6 Warp State Statistics

Focus on dominant stall reasons:
- `stall_long_scoreboard` → memory latency is the bottleneck
- `stall_math_pipe_throttle` → compute saturated (good if compute-bound)
- `stall_barrier` → excessive `__syncthreads()` or unbalanced workload
- `stall_mio_throttle` → shared memory or special function unit congestion
- `stall_short_scoreboard` → data dependency, low ILP

### Step 4: Deep Source Code Analysis (代码深度分析)

After collecting metrics, **read the kernel source code thoroughly** and perform a line-by-line correlation with the profiling data. This is the most educational step — explain the causal relationship between code patterns and hardware behavior.

For each issue found, use this analysis template:

```
📍 Source location: file.cu:line_number
📊 Related metric: [metric_name] = [observed_value] (target: [ideal_value])
🔗 Causal chain: [code pattern] → [hardware behavior] → [performance impact]
💡 Fix: [specific code change with before/after]
🎓 Takeaway: [general lesson the user can apply to future kernels]
```

**Analysis checklist:**

1. **Memory access patterns** — Check indexing math against coalescing requirements
   - Walk through the address calculation for threads 0-31 (one warp)
   - Show which 128B cache lines they touch — are they contiguous or scattered?
   - Explain: "Thread k accesses address `base + k * stride`. When stride > 1 element, threads hit different cache lines → uncoalesced"
   - Teach: consecutive threads (`threadIdx.x`) must map to consecutive memory addresses

2. **Shared memory usage** — Verify tile sizes, bank conflict potential
   - Calculate which bank each thread accesses: `bank = (address / 4) % 32`
   - Show the conflict pattern visually if conflicts exist
   - Explain why powers-of-2 strides are dangerous and how +1 padding fixes it

3. **Register pressure** — Count live variables at the hottest point
   - Identify the loop body or computation block with the most simultaneously live variables
   - Explain: "Each `float` costs 1 register. At line 55, variables a, b, c, d, sum, temp are all live = 6 registers just for this block"
   - Show the occupancy trade-off: more registers → fewer concurrent warps → less latency hiding

4. **Divergent branches** — Trace `if/else` paths across a warp
   - Show which threads take which branch and explain serialization cost
   - Teach: "Within a 32-thread warp, ALL threads must execute BOTH paths if any thread diverges"

5. **Math operations** — Identify precision and intrinsic opportunities
   - Flag `double` operations that could be `float`, `exp()` that could be `__expf()`
   - Explain the throughput difference: "FP64 throughput is 1/32 of FP32 on consumer GPUs"

6. **Synchronization** — Audit every `__syncthreads()` call
   - Ask: "Is this sync strictly necessary? Could warp-level `__shfl_sync` replace it?"
   - Explain the cost: "Every `__syncthreads()` stalls all warps in the block until the slowest one catches up"

### Step 5: Generate Optimization Report (with Learning Summary)

Structure the report as follows:

```markdown
# Kernel Analysis Report: <kernel_name>

## Executive Summary
- Kernel classification: [compute-bound / memory-bound / latency-bound]
- SM Throughput: X% | Memory Throughput: Y%
- Achieved Occupancy: Z%
- Primary bottleneck: [specific stall reason or resource]

## Step-by-Step Profiling Walkthrough

### 1. Speed of Light Analysis
[Quote SOL metrics → Explain what they mean → Classify the kernel]
"SM throughput is 23%, memory throughput is 71%. This tells us that the memory
system is working much harder than the compute units — the kernel spends most
of its time waiting for data rather than computing. This is a classic
memory-bound pattern..."

### 2. Memory Behavior Deep Dive
[Quote memory metrics → Trace to source code → Explain hardware behavior]
"Global load efficiency is only 25%. Looking at line 42: `C[row][col] = ...`,
where row = blockIdx.y * blockDim.y + threadIdx.y. Adjacent threads have
adjacent threadIdx.x values but the innermost index is `col` which maps to
threadIdx.x — wait, actually row varies with threadIdx.y. Let me trace this:
Thread (tx=0,ty=0) accesses row=0, Thread (tx=1,ty=0) accesses row=0 — OK
they access the same row. So the access IS coalesced in x. The issue must be
elsewhere... Let me check the input loads..."

### 3. Compute & Scheduling Analysis
[Similar depth: metrics → source code → hardware explanation]

### 4. Occupancy Analysis
[Show the limiter, trace to source code, explain the trade-off]

## Key Findings (with Code Correlation)

### Finding 1: [Title]
- **Metric evidence**: [specific ncu metric and value]
- **Source code location**: [file:line or code snippet]
- **Causal chain**: [code] → [hardware behavior] → [performance impact]
- **Impact**: [estimated performance impact: high/medium/low]
- **Recommendation**: [specific code change with before/after example]

### Finding 2: [Title]
...

## Optimization Roadmap (Priority Order)
1. [Highest impact change] — Expected improvement: ~X%
2. [Second highest] — Expected improvement: ~X%
3. [Third] — Expected improvement: ~X%

## Hardware Utilization Summary
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| SM Compute | ... | ... | ...% |
| Memory BW | ... | ... | ...% |
| L2 Cache | ... | ... | ...% |
| Shared Memory | ... | ... | ...% |
| Registers/Thread | ... | 255 | ... |
| Occupancy | ... | ... | ...% |

## 学到的经验 (Lessons Learned)
Summarize 3-5 reusable lessons from this analysis that apply beyond this kernel:
1. [General principle learned from Finding 1, applicable to other kernels]
2. [General principle learned from Finding 2]
3. [A "rule of thumb" reinforced by this analysis]
```

## Common Optimization Patterns

Reference these patterns when making recommendations:

### Memory-Bound Optimizations
- **Vectorized loads**: Use `float4`/`int4` to issue 128-bit transactions
- **Coalescing**: Ensure warp threads access contiguous 128B segments
- **Shared memory tiling**: Load tiles cooperatively, compute from smem
- **L2 persistence**: Use `cudaAccessPropertyPersisting` for reused data
- **Prefetching**: Use `cp.async` (sm80+) or manual double-buffering

### Compute-Bound Optimizations
- **Tensor Cores**: Use WMMA/MMA for matrix ops (FP16/BF16/TF32/INT8)
- **Instruction-level parallelism**: Unroll loops, interleave independent ops
- **Fast math**: `__fmaf_rn`, `__expf`, `rsqrtf` instead of double-precision equivalents
- **Reduce instructions**: Strength reduction, precompute constants

### Latency-Bound Optimizations
- **Increase occupancy**: Reduce registers (`__launch_bounds__`), tune block size
- **Hide latency**: Double-buffered shared memory, software pipelining
- **Reduce synchronization**: Warp-level primitives (`__shfl_sync`) instead of smem + `__syncthreads`
- **Persistent kernels**: Keep threads alive to amortize launch overhead

### Occupancy Tuning
- `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` to hint register usage
- Shared memory: dynamic allocation to enable runtime tuning
- Block size: try multiples of 32, typically 128 or 256 threads

## Analysis Style Guidelines

- **Use natural language, not just tables**: After every metrics table, write a paragraph explaining the story the data tells.
- **Show your reasoning chain**: "I notice X metric is Y% → this means Z → looking at the code, line N does W → this causes the hardware to... → therefore we should..."
- **Never skip "why"**: If you recommend a change, always explain the hardware mechanism behind it. Don't say "use float4" without explaining that a single 128-bit load instruction replaces four 32-bit loads, reducing instruction count and maximizing bus utilization.
- **Use analogies for complex concepts**: E.g., "Bank conflicts are like 32 people trying to use 32 doors, but several people picked the same door — they have to go through one at a time."
- **Encourage experimentation**: After proposing changes, suggest the user re-profile to verify the improvement and explain which metrics should change and by how much.
- **Be honest about trade-offs**: Many optimizations have costs (e.g., more shared memory → lower occupancy). Always discuss both sides.

## Notes

- Always run ncu with **sudo** or ensure perf counters are accessible (`/proc/sys/kernel/perf_event_paranoid` ≤ 2).
- For multi-kernel programs, use `--kernel-name` regex to isolate the target kernel.
- Compare against cuBLAS/cuDNN baselines when optimizing GEMM/conv kernels.
- NCU overhead is significant (~100-1000x slowdown); use small problem sizes for profiling.
- After optimization, always suggest re-profiling to validate improvements and continue the learning loop.
