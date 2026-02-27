
# DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference

## 革新

DualPath 是 DeepSeek 与北京大学、清华大学联合提出的 LLM 推理系统，解决了**智能体场景下 KV-Cache 存储 I/O 瓶颈**这一关键问题。在当前主流的 Prefill-Decode 分离架构中，Prefill 引擎需要从远程存储加载大量 KV-Cache，导致其存储网卡带宽严重饱和，而 Decode 引擎的存储网卡却处于闲置状态。DualPath 的核心创新在于引入了**双路径 KV-Cache 加载机制**——除传统的"存储→Prefill"路径外，新增"存储→Decode→Prefill"路径，通过高速 RDMA 计算网络将 KV-Cache 从 Decode 引擎传输到 Prefill 引擎。在生产级智能体工作负载上，DualPath 将离线推理吞吐提升高达 **1.87×**，在线服务吞吐平均提升 **1.96×**。

## 论文解析

### 背景：智能体推理的 I/O 瓶颈

#### 从单轮对话到多轮智能体

![智能体轨迹示例](./assets/images/Figure%202%20Agent%20trajectory%20example..png)

LLM 正从单轮对话演变为**自主式智能体系统**，通过多轮交互与外部环境持续互动。典型的智能体运行包含数十甚至数百轮交互，上下文可增长至百万 Token 级别。每轮交互中，绝大部分上下文（通常 ≥95%）来自前轮复用，仅有少量新增 Token 需要计算。这使得推理性能的决定因素从**计算**转变为 **KV-Cache 的 I/O 加载效率**。

#### 三重因素加剧瓶颈

**因素一：智能体工作负载的高 Cache 命中率**

| 指标 | 数值 |
|------|------|
| 平均交互轮数 | 157 |
| 平均上下文长度 | 32.7K Token |
| 平均新增长度 | 429 Token |
| KV-Cache 命中率 | 98.7% |

在 DeepSeek-V3.2 上，Cache-Compute 比值约为 22 GB/PFLOP，对存储带宽构成巨大压力。对于 KV-Cache 更大的模型（如 Qwen2.5-32B），情况更为严峻：

| 模型 | GB/PFLOP (16K–64K) |
|------|---------------------|
| Qwen2.5-32B (FP16) | 117–267 |
| GPT-OSS-120B | 47–95 |
| Qwen3-235B-A22B | 39–60 |
| DeepSeek-V3.2 660B | 13–36 |
| DeepSeek-V3 660B | 4.8–5.8 |

**因素二：硬件演进趋势不利**

从 NVIDIA Ampere 到 Blackwell，I/O-Compute 比下降了 **14.4×**。网络带宽和 HBM 容量的增速远落后于 GPU 算力增长，导致在智能体工作负载下严重的通信和内存瓶颈。

**因素三：存储网络利用率严重失衡**

在 PD 分离架构中，KV-Cache 仅由 Prefill 引擎从远程存储加载，所有存储 I/O 压力集中于 Prefill 侧网卡（SNIC），而 Decode 侧网卡几乎完全闲置。

### 核心创新：双路径 KV-Cache 加载

![现有瓶颈与 DualPath 对比](./assets/images/Figure%201%20Existing%20bottleneck%20(left)%20and%20DualPath%20(right)..png)

#### 设计哲学

DualPath 的核心洞察：**KV-Cache 加载不必以 Prefill 为中心**。现有系统总是将 KV-Cache 从存储直接加载到 Prefill 引擎，无法利用 Decode 引擎的远程存储带宽。DualPath 打破了这一限制，将存储 I/O 从单一瓶颈资源转变为**全局可调度的带宽池**。

#### 双路径数据流

![双路径加载示意图](./assets/images/Figure%204%20Dual-path%20loading%20illustration.%20The%20scheduler%20dynamically%20distributes%20data%20traffic%20between%20the%20two%20path.png)

DualPath 在每个 PE 和 DE 上分配少量 DRAM 作为缓冲区（PE Buffer 和 DE Buffer）。

**PE 读取路径（存储→Prefill）**：
1. KV-Cache 从持久存储读入 PE Buffer
2. 在每层注意力计算前，将该层 KV-Cache 传入 PE HBM
3. 计算 Cache-miss Token 的 KV-Cache
4. 所有 KV-Cache（命中 + 未命中）传输到 DE Buffer
5. 上述过程重复 n_layer 次，传输与计算重叠执行

**DE 读取路径（存储→Decode→Prefill）**：
1. KV-Cache 先读入 DE Buffer
2. 在 PE Prefill 期间，对应层的 KV-Cache 从 DE Buffer 读取，通过**高速 RDMA 计算网络**传输到 PE
3. 层计算完成后，仅将 miss Token 的 KV-Cache 传回 DE Buffer 与已有命中缓存合并

**关键设计**：DE Buffer 的引入虽然增加了一次 H2D 拷贝，但由于智能体场景中生成长度短、TTFT 占端到端时间比例大，减少 GPU 内存使用的收益更为显著。

#### 无瓶颈分析

在合理的 P/D 比例下，DualPath 可以完全饱和所有存储网卡而不引入计算网卡或 DRAM 瓶颈。令 P、D 为 Prefill 和 Decode 节点数，g 为每节点 GPU 数，B 为计算网卡带宽，s×B 为每节点存储带宽，M 为每节点内存带宽，无瓶颈条件为：

```
s/(g-s) ≤ P/D ≤ min{(g-2s)/s, (g-s)/(2s), (M/(Bs)-3)/2}
```

对于典型配置 (g=8, s=1, M≈500 GB/s, Bs≈50 GB/s)，无瓶颈范围为 **1/7 ≤ P/D ≤ 7/2**，覆盖绝大多数实际部署场景。

### CNIC 中心化流量管理

#### 为何需要流量隔离？

双路径架构在计算网络和 PCIe 链路上引入了额外的 KV-Cache 传输流量。关键挑战是：这些流量可能干扰模型推理中**时延敏感的集合通信操作**（如专家并行的 AllToAll、张量并行的 ReduceScatter/AllGather），这些通信以亚毫秒级脉冲式发生，软件流量整形难以介入。

#### 解决方案：CNIC 中心化数据传输

**核心原则**：所有进出 GPU 的数据流量（包括本地 H2D/D2H 拷贝）都必须通过 GPU 配对的计算网卡（CNIC），使用 GPUDirect RDMA 数据路径。

**流量隔离实现**：
- 利用 InfiniBand 的**虚拟通道（Virtual Lanes）**进行隔离
- 模型推理通信 → 高优先级 VL（~99% 带宽预留）
- KV-Cache 传输 → 低优先级 VL（~1% 带宽防饥饿）
- 模型执行几乎不受 KV-Cache 传输影响
- KV-Cache 流量可机会性利用计算网络空闲带宽

**CNIC 辅助拷贝的额外优势**：当处理大量小数据块时，CNIC 辅助 H2D/D2H 优于 CUDA copy engine。单次 `cudaMemcpyAsync` 开销约 5-7μs，而一次 RDMA Write work request 仅需 ~1μs，且可通过 doorbell batching 进一步摊薄。

### 自适应请求调度器

#### 调度的两个维度

DualPath 需要同时平衡两个维度：
1. **NIC 流量**平衡
2. **GPU 利用率**平衡

调度分为两级：**引擎间调度**（将请求分配给 PE-DE 对并选择读取路径）和**引擎内调度**（决定前向批次中包含哪些请求）。

#### 引擎间调度

![引擎间 PE 调度示意图](./assets/images/Figure%205%20An%20illustration%20of%20Inter-Engine%20PE%20Scheduling..png)

**PE 调度**：每个引擎报告三个指标——未完成请求数 seq_e、Token 总量 tok_e、所在节点磁盘读取队列长度 read_q。引擎被分为三类：

| 类别 | 条件 | 调度优先级 |
|------|------|-----------|
| 过载引擎 | tok_e > β | 不分配新请求 |
| 短磁盘队列引擎 | read_q ≤ α 且 tok_e ≤ β | **最高**（防止存储 NIC 闲置） |
| 长磁盘队列引擎 | read_q > α 且 tok_e ≤ β | 次高 |

在同类别内，选择 tok_e 最小的引擎（FIFO 顺序分配请求）。

**DE 调度（两级）**：
- **跨组调度**：将全局队列中的请求分配给 tok_e 总量最小的 DE 组
- **组内调度**：设定高 Token 阈值 Z = 1.05 × 平均值，优先选择非高负载 DE（类别 2），在同类中选 seq_e 最小的；若所有 DE 都高负载，选 tok_e 最小的

**KV-Cache 读取路径选择**：选定 PE 和 DE 后，选择读取队列较短的一侧。

#### 引擎内调度

![引擎内调度示意图](./assets/images/Figure%206%20Intra-Engine%20Schedule.%20Left%20compute-quota-based.png)

仅 PE 需要引擎内调度（DE 始终将所有请求放入前向批次）。

**核心问题**：数据并行配置下，不同 GPU 服务不同请求集合。注意力层执行时间不均会导致同步等待气泡。

**解决方案：Compute Quota**

- 每个请求描述为 (cached, bsz) 对，估算注意力层执行时间
- 按 FIFO 顺序添加请求，直到预测执行时间达到预设上限（compute quota = 300ms）
- 若添加某请求会超限，通过二分搜索找到更小的 bsz'，执行分块预填充（chunked prefill）

**效果**：最小化 GPU 间等待气泡，Max/Avg 比率低至 1.06。

### 评估结果：全面提升

#### 实验设置

**硬件**：NVIDIA Hopper GPU 集群，InfiniBand 互连，每节点 8 GPU + 8×400Gbps RDMA NIC + 1 个存储 NIC，3FS 分布式存储后端。

**模型**：
- DeepSeek V3.2 660B（MoE + DeepSeek Sparse Attention）
- DS 27B（660B 的缩小版，内部实验模型）
- Qwen2.5-32B（Dense + GQA）

**数据集**：三个来自生产智能体 RL 训练的轨迹数据集：

| 最大长度 | 平均轮数 | 平均新增 Token | 平均生成 Token | 平均总 Token | 平均上下文 |
|---------|---------|--------------|--------------|------------|-----------|
| 32K | 60 | 608 | 148 | 28,639 | 17,183 |
| 48K | 106 | 474 | 172 | 42,607 | 25,120 |
| 64K | 157 | 429 | 176 | 55,958 | 32,721 |

#### 离线批量推理

![离线推理性能](./assets/images/Figure%207%20Offline%20inference%20performance%20under%20varying%20numbers%20of%20agents%20and%20maximum%20agent%20context%20lengths.png)

**核心结果**：
- DS 660B：DualPath 相比 Basic 提升高达 **1.87×**，性能接近 Oracle（零 I/O 开销理论上限）
- DS 27B：提升高达 **1.78×**
- Qwen 32B：趋势与 DS 27B 类似

**DualPath 在以下场景收益更大**：
- 更大的 Agent 批量大小
- 更长的最大上下文长度
- 更短的 Append 和 Generation Token

#### P/D 比例影响

![P/D 比例影响](./assets/images/Figure%208%20Impact%20of%20prefill-decode%20ratio%20on%20offline%20inference%20performance.png)

在 DS 27B 上测试 1P1D、2P1D、1P2D 三种配置：

- DualPath 在所有配置下平均加速 **1.64×**（最高 **2.46×**）
- Basic 1P1D ≈ Basic 1P2D；DualPath 1P1D ≈ Basic 2P1D；DualPath 2P1D ≈ DualPath 1P2D
- **验证了存储带宽是智能体场景的主导瓶颈**：每对系统恰好具有等量可用存储带宽

#### 在线服务

![在线服务延迟指标](./assets/images/Figure%2010%20TTFT,%20TTST,%20and%20TPOT%20as%20functions%20of%20agent%20arrival%20rate%20(APS).%20Shadow%20means%20the%20fluctuation%20in%20the%20last%20150s.png)

SLO 约束：TTFT ≤ 4s，TPOT ≤ 50ms

**关键结果**：
- DS 27B：DualPath APS 容量为 Basic 的 **1.67×**
- DS 660B：DualPath APS 容量为 Basic 的 **2.25×**
- TTST 和 TPOT 无额外开销
- DualPath 保持稳定的 TTFT 组成，而 Basic 的排队时间随 APS 增长急剧上升

#### 消融研究

三项技术的逐步叠加效果（DS 660B, 64K）：

| 技术 | 平均 JCT 降低 |
|------|-------------|
| +Layerwise Prefill | 17.21% |
| +Dual-Path Loading | 38.19% |
| +Scheduling Algorithm | **45.62%** |

**负载均衡效果**：
- 存储 NIC 流量：Max/Avg 从轮询调度的 1.53 降至 **1.18**
- 注意力执行时间：Max/Avg 低至 **1.06**

#### 大规模可扩展性

| 配置 | JCT | TTFT | TTST | TPOT |
|------|-----|------|------|------|
| 离线 2P4D (2K agents) | 3,167s | – | – | – |
| 离线 48P96D (48K agents) | 3,201s | – | – | – |
| 在线 2P4D (0.4 APS) | – | 1.739s | 0.228s | 0.039s |
| 在线 44P88D (8.8 APS) | – | 1.847s | 0.194s | 0.036s |

从 2P4D 扩展到 48P96D（1152 GPU），实现**近线性扩展**，JCT 仅从 3167s 增至 3201s。在线服务从 0.4 APS 扩展到 8.8 APS（22×），延迟基本不变。调度器 CPU 使用量始终低于 10 核。

## 技术亮点深度解读

### 1. 为何存储带宽成为瓶颈而非计算？

**传统认知**：LLM 推理是计算密集型任务，GPU 算力是瓶颈。

**智能体场景的颠覆**：
- 98.7% 的 Token 命中 KV-Cache → 几乎不需要计算
- 仅 1.3% 的 Token 需要真正 Prefill → GPU 大部分时间在等 I/O
- 每个 PFLOP 计算需加载 22GB KV-Cache → I/O 远比计算慢

**根本原因**：智能体的"长上下文、短新增、多轮次"模式，将工作负载从 compute-bound 彻底转变为 I/O-bound。

### 2. 双路径设计的精妙之处

**为何不简单增加 Prefill 侧带宽？**

- 通用集群中成本高且不切实际
- Decode 侧的存储带宽已经存在但被浪费
- 增加一侧带宽无法解决架构层面的不对称性

**DualPath 的巧妙之处**：

不是增加资源，而是**重新分配已有资源**：
- Decode 引擎的 SNIC 闲置 → 让它加载 KV-Cache
- 计算网络（CNIC）有大量空闲带宽（通信呈脉冲式） → 让它搬运 KV-Cache
- 两条路径动态切换 → 存储带宽全局池化

**类比**：就像高速公路，一条车道（Prefill SNIC）严重拥堵，另一条车道（Decode SNIC）空空如也。DualPath 不是拓宽拥堵车道，而是打通一条绕行路线，让车辆先走空闲车道再通过立交桥（RDMA）合流。

### 3. 流量隔离为何至关重要？

**问题本质**：模型推理的集合通信（AllToAll 等）是亚毫秒级脉冲，对延迟极其敏感。如果 KV-Cache 传输流量与之竞争 PCIe 和网络带宽，推理性能将严重退化。

**为何选择 CNIC 中心化而非 GPUDirect Storage？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| GPUDirect Storage | 直接从存储到 GPU HBM，路径最短 | 无法与推理通信隔离 |
| CUDA Copy Engine | 从 Host DRAM 直接拷贝 | GPU PCIe 无 QoS 支持，无法隔离 |
| **CNIC 中心化** | **利用计算网络原生 QoS 能力** | 看似绕路 |

**关键洞察**：看似绕路的 CNIC 方案，是目前**唯一能确保 KV-Cache 传输不影响推理通信**的实用方法。VL 机制让推理流量获得 99% 的带宽保障，KV-Cache 传输只使用剩余空闲带宽。

### 4. 调度算法的设计哲学

**挑战**：需要同时平衡三个维度——存储 NIC 负载、GPU 计算负载、请求公平性。

**PE 调度的三类分级**：

这是一个精心设计的**优先级反转策略**：
- 短磁盘队列的引擎优先（而非低 Token 量引擎）
- 原因：磁盘队列短意味着存储 NIC 即将空闲，若不及时补充请求，带宽会白白浪费
- 体现了"带宽利用率第一"的设计原则

**Compute Quota 的巧思**：
- 并行注意力计算中，最慢的 GPU 决定整体速度
- 固定上限（300ms）确保各 GPU 执行时间相近
- 分块预填充（chunked prefill）处理边界情况，避免浪费

### 5. 为何 Working Set 分析意义重大？

论文给出了一个简洁的 Working Set 估算公式：

```
Working Set ≈ λ × T̄ × total_len_avg / 2
```

在 DS 660B 在线服务中，Working Set 从 69GB (APS=0.1) 增长到 681GB (APS=0.45)。

**深层意义**：
- 生产环境中实际 Working Set 远大于实验值（工具调用延迟和到达间隔会使 JCT 增加 r 倍）
- Working Set 以 r² 增长，所需存储以 r² 扩展，实验成本以 r³ 扩展
- 这解释了**为何 DRAM 缓存方案不够用**：Working Set 可轻易超过可用内存，必须依赖 SSD 级外部存储

## 架构与系统洞察

### 现代 AI 数据中心的网络分离

DualPath 的设计深度利用了现代 AI 数据中心的双网络架构：

- **计算网络（CNIC）**：每 GPU 配一个 400Gbps NIC，用于 GPU 间通信
- **存储网络（SNIC）**：每节点一个 400Gbps NIC，用于访问存储
- 两个网络**物理隔离**，互不干扰

DualPath 巧妙地在这两个隔离网络之间架起了桥梁：让存储数据可以"绕道"计算网络，打破了物理隔离带来的带宽利用不均。

### 与相关工作的差异化定位

| 方案 | 思路 | 局限 |
|------|------|------|
| Mooncake | DRAM 池分布式缓存 KV-Cache | 内存受限场景不适用；大 Working Set 下性价比差 |
| HCache | 减少 KV-Cache 数据量 | 不解决带宽不对称问题 |
| Strata | GPU 辅助 I/O + 分层存储 | 仍是单路径，未利用 Decode 侧带宽 |
| KVPR/TailorKV | 重计算重叠 / 层级量化减少带宽需求 | 优化单路径效率，未解决根本不对称 |
| **DualPath** | **双路径加载 + 全局带宽池化** | **从架构层面消除不对称** |

**DualPath 的独特贡献**：不是在单条路径上做优化，而是**开辟第二条路径**，从根本上重塑数据流动方式。

### 块布局设计：Layer Block 与 Full Block

Layerwise Prefill 将 KV-Cache 块大小缩小为 1/layer，块数量增加 layer 倍，对传输和存储性能构成挑战。

**解决方案**：两种块类型
- **Layer Block** `[1, tokens, bytes]`：单层 KV-Cache，用于 PE/DE 间的层级流式传输
- **Full Block** `[layer, tokens, bytes]`：全层 KV-Cache，用于与存储交互

n 个 Layer Block 可直接拼接为一个 Full Block，**无需手动内存布局转换**。

## 对大规模推理系统的启示

### 1. I/O 将成为智能体推理的核心瓶颈

随着智能体工作负载成为主流，传统以 GPU 算力为中心的推理优化思路需要根本性转变：

**传统优化焦点**：
```
更快的 GPU → 更大的批量 → 更高的吞吐
```

**智能体时代的优化焦点**：
```
更高的存储带宽 → 更快的 KV-Cache 加载 → 更高的吞吐
```

### 2. 系统级全局优化优于局部优化

DualPath 不是在某个组件上做极致优化，而是**重新审视整个数据流动**，发现并利用了全局资源的不对称性。

**启示**：
- 单一组件优化（如更快的存储、更大的缓存）收益有限
- 全局视角下发现"闲置资源"并加以利用，往往事半功倍
- 系统设计应追求**资源利用率的全局均衡**

### 3. 网络 QoS 是混合流量系统的基石

DualPath 成功的关键前提是流量隔离。没有可靠的 QoS 机制，引入额外数据路径只会带来干扰而非收益。

**启示**：
- 未来的推理系统将承载越来越多异构流量（推理通信、KV-Cache 传输、检查点保存等）
- 硬件级 QoS 支持（VL、DSCP、TC）是系统设计的必要基础设施
- PCIe QoS 的缺失是当前的重要瓶颈，期待未来硬件改进

### 4. 智能体 RL 训练需要专门的推理基础设施

论文揭示了一个重要趋势：智能体 RL 训练的 rollout 阶段本质上是大规模多轮推理，且 DRAM 被训练状态占用，进一步加剧了存储带宽压力。

**启示**：
- RL 训练中的 rollout 推理 ≠ 普通推理，需要专门优化
- 训练和推理的基础设施正在融合
- DualPath 式的全局带宽池化对 RL 训练场景尤为重要

### 5. 动态调度是充分发挥硬件潜力的关键

即使双路径硬件架构就位，没有好的调度策略也无法充分发挥其优势。消融实验表明：
- Layerwise Prefill 贡献 17.21% 提升
- Dual-Path Loading 贡献 20.98% 额外提升
- Scheduling 额外贡献 7.43% 提升

调度算法将存储 NIC 负载不均衡从 1.53 降至 1.18，看似简单的百分比改进，在大规模部署中转化为显著的吞吐收益。

## 局限性与未来方向

### 当前局限

1. **KV-Cache 读取路径未分割**：当前一个请求只能选择 PE 或 DE 一侧读取，未来可将请求拆分为两部分同时从双路径读取

2. **小模型场景下 P-D 传输开销**：DS 27B 的 TPOT 显著高于 Oracle，说明基础的 P-D KV-Cache 传输在小模型场景中开销不可忽略

3. **大规模部署的 P/D 比例调优**：由于实验预算限制，大规模实验未展示超越等量小规模单元的额外收益

4. **工作负载高度动态**：智能体 RL 任务中，前半段 Prefill 压力远高于后半段，静态配置难以最优应对

### 未来研究方向

1. **自适应 P/D 比例和并行配置**：开发模拟器或在线调整机制，动态适应变化的工作负载

2. **更细粒度的读取路径调度**：将单个请求的 KV-Cache 拆分到双路径同时加载

3. **大规模部署下的尾延迟优化**：更多调度机会可缓解突发请求的排队延迟

4. **与 DRAM 缓存层的协同**：DualPath 可与中间 DRAM 缓存层结合，但论文指出边际收益有限，需进一步探索

5. **跨代硬件适配**：随着 Ultra Ethernet 等新互连技术的发展，DualPath 的 QoS 机制需要相应演进

## 相关资源

- **论文**：[arXiv:2602.21548](https://arxiv.org/abs/2602.21548)
- **3FS 分布式存储**：[GitHub - deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)
- **FlashMLA**：[GitHub - deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- **DeepGEMM**：[GitHub - deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- **DeepEP**：[GitHub - deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)

## 论文信息

- **标题**：DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
- **作者**：Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang
- **机构**：北京大学、清华大学、DeepSeek-AI
- **发表时间**：2026 年 2 月
- **评估模型**：
  - DeepSeek V3.2 660B（MoE，DeepSeek Sparse Attention）
  - DS 27B（内部实验模型，660B 缩小版）
  - Qwen2.5-32B（Dense，GQA）
- **核心技术**：
  - 双路径 KV-Cache 加载（Dual-Path Loading）
  - CNIC 中心化流量管理与 VL 隔离
  - Layerwise Prefill
  - 自适应请求调度器（引擎间 + 引擎内）
  - Compute Quota 机制
- **关键词**：Agentic LLM Inference, KV-Cache, Storage Bandwidth, PD Disaggregation, Dual-Path Loading, Traffic Isolation, Request Scheduling

---

**总结**：DualPath 敏锐地捕捉到了智能体推理从 compute-bound 向 I/O-bound 转变的关键趋势，并以一种优雅的架构创新——双路径 KV-Cache 加载——从根本上打破了 PD 分离架构中存储带宽的不对称瓶颈。其设计哲学不是简单地"加资源"，而是"重新分配已有资源"，通过全局带宽池化、精细流量隔离和智能调度，实现了接近理论上限的推理吞吐。这一工作不仅是对当前智能体推理系统的重要优化，更为我们理解和设计下一代 AI 基础设施提供了深刻的系统级洞察：**当工作负载模式发生根本性转变时，系统架构也必须相应重构，而非在旧架构上修修补补**。
