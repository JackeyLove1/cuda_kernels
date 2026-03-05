一、岗位到底是做什么的？

AMD AI Infra 工程师，核心就是：
在 AMD GPU（MI25 / MI300 / Instinct 系列）上把大模型训好、推快、稳住。
 
主要工作：

• PyTorch/TensorRT 适配 AMD GPU

• 算子开发、性能优化、混合精度训练

• 分布式训练：ROCm + NCCL 替代方案（RCCL）

• 推理优化、量化、vLLM / TGI 移植

• 多机多卡、RDMA、OCI/K8s 调度

• 稳定性、OOM、卡死、掉卡、超时问题排查

一句话：CUDA 那套 Infra 能力，平移到 AMD ROCm 生态。
二、核心技能要求（面试官打分表）

1. 必须会（不会基本挂）

• C++ / Python 熟练

• Linux、GDB、CMake、Makefile

• 计算机体系：缓存、并发、SIMD、内存模型

• PyTorch 底层：计算图、算子、DDP、amp

• 分布式：DP/TP/PP、AllReduce、Ring AllReduce

• 推理：KV Cache、Batch、量化、算子融合

2. AMD 专属硬技能（必考）

• ROCm 栈：ROCm Runtime、HIP、HIPBLAS、ROCBLAS

• RCCL：AMD 版 NCCL，分布式通信

• HIP 语法：global、device、hipMalloc/hipMemcpy

• Instinct 架构：CU → 对标 CUDA Core

• ROCm Profiler：rocprof、omnitrace

3. 加分项（直接升档）

• MI25/MI300 训推落地经验

• vLLM / TGI / TensorRT-LLM 移植 AMD

• 大模型千卡级稳定性经验

• HIP 算子开发、自定义 Kernel
三、笔试高频考点（命中率极高）

1. 选择 / 判断（基础关）

1. AMD AI 训练的核心软件栈是？
A. CUDA  B. ROCm  C. CANN  D. BANG
答案：B

2. AMD 分布式通信库是？
A. NCCL  B. RCCL  C. HCCL  D. MPI
答案：B

3. HIP 对应 CUDA 中的什么？
A. 指令集  B. 编程语言/编译工具链  C. 推理引擎  D. 调度框架
答案：B

4. 大模型推理瓶颈通常是？
A. 算力  B. 显存 / 显存带宽  C. CPU  D. 网络
答案：B

5. ROCm 中查看 GPU 占用的工具是？
A. nvidia-smi  B. rocm-smi  C. npu-smi  D. cnmon
答案：B
2. 填空题（高频）

1. HIP 内核函数用 __global__ 标识。

2. 设备内存分配：hipMalloc，数据拷贝：hipMemcpy。

3. 张量并行 TP 核心通信：AllGather、ReduceScatter。

4. OOM 三大来源：参数、激活、KV Cache。

5. PagedAttention 核心思想：分页式 KV Cache。
3. 笔试题（手撕高频）

（1）HIP 向量加法（模拟题）
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < n) {
       c[idx] = a[idx] + b[idx];
   }
}
面试官必追问：

• block/thread 怎么设置？

• 如何避免越界？

• 如何向量化优化？
（2）计算 LLaMA 7B 单卡显存

参考答案思路：

• 权重：7B × 2Byte（BF16）≈ 14GB

• KV Cache、梯度、优化器、碎片

• 实际至少 20GB+ 显存
四、面试高频题（最值钱部分）

模块1：AMD / ROCm 基础（必问）

1. ROCm 和 CUDA 核心区别是什么？

• CUDA 闭源，ROCm 开源

• 内核语言：CUDA C → HIP C++

• 通信库：NCCL → RCCL

• 工具链：nvcc → hipcc

2. HIP 是什么？和 CUDA C 怎么对应？
HIP 是 AMD 异构编程模型，语法几乎和 CUDA 一一对应，
__global__、__device__、线程层级模型完全一致。

3. RCCL 和 NCCL 区别？
接口高度兼容，底层是 AMD 自研实现，
支持 AllReduce、AllGather、Broadcast 等集合通信。

4. CUDA 代码移植 AMD 最容易踩什么坑？

• 算子缺失 / 精度不对

• warp 大小、原子操作差异

• 同步语义、profiler 不同

• RCCL 拓扑、多机稳定性
模块2：分布式训练（高频）

1. DP / DDP / TP / PP 分别是什么？

• DP：数据并行，单进程多线程

• DDP：多进程数据并行，工业标准

• TP：张量并行，切层内张量

• PP：流水线并行，切层间

2. Ring AllReduce 流程？

• Scatter-Reduce

• AllGather
把带宽压力从 O(N) 压到 O(1)。

3. 训练不收敛怎么排查？

• 学习率、精度、loss scale

• 梯度是否爆炸/消失

• RCCL 通信是否正常

• 检查 HIP 同步/异步错误
模块3：推理优化（2026 最火）

1. PagedAttention 为什么能提速？
把 KV Cache 分成固定大小块，物理不连续、逻辑连续，减少显存碎片，大幅提升 Batch 吞吐。

2. vLLM 移植 AMD 要改什么？

• CUDA API → HIP API

• CUTLASS 算子 → ROCBLAS / 自定义 HIP 算子

• NCCL → RCCL

• 内存管理、Event、Stream 替换

3. INT8 量化误差来源？

• 校准方式（KL / 百分位）

• Outlier 通道

• 舍入误差 & 裁剪误差
模块4：系统 & 稳定性（社招必问）

1. OOM 怎么快速定位？

• 减小 batch / seq len

• 激活重计算

• TP / ZeRO 分片

• KV Cache 量化 / 分页

2. 多机训练卡死怎么排查？

• 看 RCCL log

• 网络 RDMA/RoCE 连通性

• 显卡掉卡、温度、功耗

• 时钟同步、网卡队列、防火墙

3. rocm-smi / rocprof 看什么？

• 显存占用

• 算力利用率

• 温度、功耗、时钟

• kernel 耗时、带宽
五、7 天快速上岸路线（最强版）

1. 第1–2天：基础
C++、Python、Linux、PyTorch 底层

2. 第3天：AMD 生态
ROCm 安装、HIP 语法、rocm-smi、rocprof

3. 第4天：分布式
DDP、RCCL、AllReduce、TP/PP

4. 第5天：推理
KV Cache、PagedAttention、vLLM

5. 第6天：刷题
向量加、矩阵乘、线程池、显存计算

6. 第7天：项目 + 面试模拟
训推优化、故障案例、性能数据
六、总结

AMD AI Infra 本质 = CUDA Infra 能力 + ROCm 生态知识。
笔面试重点：

• 基础扎实

• 分布式理解深

• 推理优化懂

• 能排障
