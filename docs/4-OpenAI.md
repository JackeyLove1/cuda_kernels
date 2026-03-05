设计一个支持 100B+ 参数模型的分布式训练系统（跨 2000+ GPU）。如何分区参数、同步梯度、处理故障、检查点？ （最经典题，常追问 OOM 恢复、异步调度器、通信拓扑如 fat-tree vs dragonfly。）
如何在万卡集群上实现高 MFU（Model FLOPs Utilization）？诊断低 MFU / hang / OOM 的流程？ （追问 Nsight、PyTorch Profiler、NCCL 日志、通信-计算 overlap、混合精度精度损失控制。）
ZeRO-3 vs FSDP vs Megatron-LM 3D Parallelism 在极端规模下的选择与 tradeoff？如何缓解 All-Gather/Reduce-Scatter 通信瓶颈？
训练失败监控与调试：loss spike、梯度爆炸/消失、节点掉线，如何快速定位并恢复？
设计 streaming training pipeline：数据摄入、checkpointing、参数分片、OOM 恢复、异步 job scheduler。
高频推理系统面试题（Inference / Serving at Scale）
KV Cache 导致推理成本 10x 高于预期，如何诊断和修复？（经典陷阱题：不要说“加内存”，而是讲内存碎片、PagedAttention、连续批处理、Radix Tree 前缀复用。）
如何设计高 QPS、低延迟的 LLM 推理服务（百万 QPS 规模）？包括负载均衡、动态批处理、GPU 调度、成本优化。
推理加速：量化（GPTQ/AWQ/FP8）、MoE、稀疏性、缓存、checkpointing、Speculative Decoding 如何组合？lossless 2x 加速方案？
部署 70B+ 模型的软件/硬件栈：低延迟高吞吐？（vLLM/Triton/TensorRT-LLM、H100/A100 配置、异构调度。）
推理栈优化：如何处理长上下文、混合 batch、多租户隔离、安全（prompt injection 防御）？
其他相关高频题（常出现在交叉面试）
如何设计 GPU 调度系统（credit-based、公平隔离、多队列、低延迟决策）？
设计实时模型 A/B 测试基础设施（推理优化、负载均衡、成本监控）。
多模态/视频训练基础设施：数据 pipeline、token 处理、跨 shot 连续性。
极端规模监控：训练/推理失败自动恢复、指标体系（MFU、tokens/s、tail latency）。
这些题在公开面经中零散出现，但模式一致：强调极端规模下的 tradeoff（吞吐 vs 可靠性 vs 成本 vs 安全）、实际生产级调试经验、AI-specific 基础设施思维（而非通用 SWE）。OpenAI 更看重你能否“从第一性原理”推理，而不是背工具名。