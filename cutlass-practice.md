# CUTLASS 从入门到精通：20道进阶练习题


## 第一部分：基础概念 (1-5题)

### 题目1：理解CUTLASS的核心抽象
**目标**: 掌握Tile、Thread Block、Warp的层次结构

编写一个简单的GEMM kernel，使用CUTLASS的基本组件：
- 实现一个 128x128x8 的tile配置
- 手动管理shared memory的分配
- 理解 `ThreadblockSwizzle` 的作用

**进阶要求**:
- 分析不同tile size对occupancy的影响
- 测量shared memory bank conflict
- 对比cutlass::gemm::threadblock命名空间下的预定义配置

---

### 题目2：Layout转换深度理解
**目标**: 掌握RowMajor、ColumnMajor、TensorOp layout

实现以下功能：
- 编写kernel将RowMajor矩阵转换为ColumnMajor
- 使用CUTLASS的 `layout::PitchLinear` 和 `layout::ColumnMajor`
- 实现zero-copy的layout view转换

**关键知识点**:
- `cutlass::MatrixCoord` 的使用
- `TensorRef` vs `TensorView` 的区别
- Strided layout的内存访问模式

---

### 题目3：Tensor Core基础操作
**目标**: 理解WMMA/MMA指令的CUTLASS封装

任务：
- 使用 `cutlass::arch::Mma` 实现16x8x16的矩阵乘法
- 对比FP16、BF16、INT8的Tensor Core吞吐
- 分析fragment的寄存器分布

**实验设计**:
- 测量不同数据类型的TFLOPS
- 理解 `cutlass::gemm::warp::MmaTensorOp` 的实现
- 可视化warp-level的数据flow

---

### 题目4：Epilogue定制化
**目标**: 掌握fusion操作的实现

实现自定义Epilogue：
- GEMM + ReLU + Scaling
- GEMM + Bias + GELU
- GEMM + Residual connection

**技术要点**:
- 继承 `cutlass::epilogue::thread::LinearCombination`
- 使用 `cutlass::epilogue::threadblock::Epilogue`
- 理解output tile iterator的工作原理

---

### 题目5：Predicate和边界处理
**目标**: 处理非规则矩阵尺寸

挑战：
- 实现支持任意M、N、K维度的GEMM
- 优化边界检查的性能开销
- 使用 `PredicateVector` 和 `Predicate Iterator`

**优化目标**:
- 最小化分支divergence
- 合理使用padding策略
- 对比residue处理的不同方法

---

## 第二部分：性能优化 (6-10题)

### 题目6：Double Buffering与软件流水
**目标**: 隐藏global memory访问延迟

实现技术：
- 使用 `cutlass::gemm::threadblock::MmaPipelined`
- 配置multi-stage pipeline (2-stage, 3-stage)
- 分析async copy的收益

**性能分析**:
- 使用Nsight Compute分析memory-bound vs compute-bound
- 对比cp.async与传统load的latency hiding
- 测量不同stage数的性能trade-off

---

### 题目7：Split-K并行化
**目标**: 优化小batch size GEMM

实现：
- 使用 `cutlass::gemm::kernel::GemmSplitKParallel`
- 实现atomic-based和reduction-based两种方案
- 分析split-k的scalability

**实验设计**:
- 测试M=N=4096, K=16384场景
- 对比不同split数量的效率
- 分析reduction overhead

---

### 题目8：Group GEMM优化
**目标**: 高效处理多个小矩阵乘法

任务：
- 实现 `cutlass::gemm::kernel::DefaultGemmGrouped`
- 优化不同大小矩阵的调度策略
- 使用persistent kernel减少launch overhead

**应用场景**:
- Transformer中的multi-head attention
- 不同专家网络的MoE计算
- 动态shape的batch推理

---

### 题目9：数值精度与量化
**目标**: 实现混合精度训练/推理

实现以下配置：
- FP16 accumulation with FP32 output
- INT8 GEMM with per-channel quantization
- 动态量化范围调整

**关键技术**:
- `cutlass::NumericConverter` 的使用
- Scale/Zero-point的fusion
- 分析量化误差的累积

---

### 题目10：Strided Batched GEMM
**目标**: 优化批量矩阵运算

实现：
- 使用 `cutlass::gemm::kernel::DefaultGemmUniversal`
- 支持可变stride的batch GEMM
- 对比array-of-pointers方案

**性能考量**:
- Batch内的负载均衡
- 不同batch size的kernel效率
- 与cuBLAS batched API对比

---

## 第三部分：高级特性 (11-15题)

### 题目11：自定义Warp-level MMA
**目标**: 深入理解Tensor Core编程

挑战：
- 手写warp-level的MMA accumulator
- 实现非标准tile shape (如24x32x16)
- 优化cross-warp的数据重用

**深度分析**:
- PTX级别的mma.sync指令使用
- Fragment layout的手动管理
- 寄存器压力优化

---

### 题目12：Sparse GEMM实现
**目标**: 利用结构化稀疏性

任务：
- 实现2:4 structured sparsity GEMM
- 使用 `cutlass::gemm::warp::MmaSparseTensorOp`
- 分析稀疏度对性能的影响

**技术细节**:
- Metadata tensor的生成和使用
- 对比dense vs sparse的实际加速比
- 不同稀疏模式的支持

---

### 题目13：自动调优系统设计
**目标**: 构建kernel性能预测模型

项目：
- 遍历CUTLASS的template参数空间
- 收集不同配置的性能数据
- 使用机器学习预测最优配置

**参数空间**:
- ThreadblockShape: 64x64, 128x128, 256x128
- WarpShape: 32x32, 64x32
- Stages: 2, 3, 4, 5
- InstructionShape: 根据架构选择

---

### 题目14：Flash Attention实现
**目标**: 融合softmax的高效attention

实现：
- Online softmax计算
- Tiling策略优化Q、K、V矩阵
- 使用CUTLASS构建完整的fused attention kernel

**核心挑战**:
- Block-wise softmax的数值稳定性
- Shared memory的复用策略
- 对比标准attention的memory footprint

---

### 题目15：CUTLASS 3.x新特性探索
**目标**: 掌握CuTe编程模型

学习内容：
- 使用CuTe的Layout代数系统
- 理解cute::Tensor的抽象
- 实现一个CuTe风格的GEMM kernel

**范式转换**:
- 从iterator到Layout transformation
- Collective操作的compose
- TMA (Tensor Memory Accelerator)的使用

---

## 第四部分：系统集成 (16-20题)

### 题目16：PyTorch自定义算子
**目标**: 将CUTLASS集成到深度学习框架

任务：
- 编写PyTorch C++ extension
- 实现forward和backward的CUTLASS kernel
- 处理autograd的正确性

**工程实践**:
- CUDA stream管理
- 异常处理和错误检查
- 性能profiling集成

---

### 题目17：多GPU GEMM调度
**目标**: 实现分布式矩阵乘法

设计：
- 使用NCCL进行all-reduce
- 实现tensor parallel的GEMM partition
- 优化通信与计算的overlap

**场景**:
- Large language model的FFN层
- Pipeline parallel中的boundary处理
- 动态batch size的负载均衡

---

### 题目18：Kernel Fusion大规模实践
**目标**: 构建复杂的fused operator

项目：
- LayerNorm + Linear + GELU fusion
- Multi-head attention完全fusion
- Transformer block端到端优化

**优化技术**:
- 寄存器复用策略
- 多阶段fusion的trade-off
- 与TensorRT/XLA的对比

---

### 题目19：动态shape适配
**目标**: 支持运行时动态尺寸

实现：
- 运行时kernel选择策略
- JIT compilation集成（optional）
- Autotuning cache系统

**关键问题**:
- Kernel launch overhead
- Warmup策略
- Shape bucketing优化

---

### 题目20：端到端模型优化案例
**目标**: 完整的生产环境部署

综合项目：
- 选择一个Transformer模型（如BERT/GPT）
- 使用CUTLASS优化所有线性层
- 实现INT8量化推理
- 性能benchmark与分析

**交付物**:
- 完整的优化代码
- 详细的性能报告（latency、throughput、memory）
- 与vendor库（cuBLAS、cuDNN）的对比
- 可复现的benchmark脚本

**评估指标**:
- End-to-end latency提升
- GPU利用率
- 数值精度loss
- 可维护性和可扩展性

---

## 学习路径建议

**入门阶段 (1-5题)**: 2-3周
- 理解CUTLASS的基本概念
- 熟悉template编程范式
- 掌握基础性能分析工具

**进阶阶段 (6-10题)**: 3-4周
- 深入优化技巧
- 理解硬件特性
- 建立性能intuition

**高级阶段 (11-15题)**: 4-6周
- 探索前沿技术
- 自定义复杂kernel
- 系统性能建模

**实战阶段 (16-20题)**: 持续实践
- 工程化能力
- 生产环境部署
- 持续优化迭代

## 额外资源

- **必读**: CUTLASS官方文档和examples
- **工具**: Nsight Compute, Nsight Systems
- **参考**: NVIDIA GTC talks, CUTLASS GitHub issues
- **社区**: NVIDIA Developer Forums
