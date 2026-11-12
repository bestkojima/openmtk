# OpenMTK

面向 MediaTek NPU 的独立算子后端与端侧 LLM 推理引擎研究。

> **一句话路线：先把最小 Matmul 做成“语义契约 → CPU/官方参考 → 运行轨迹 → 数据布局与命令生成 → 独立执行 → 验证调优”的闭环，再把同一套流程复用于其他算子，由自研 Runtime 组合成 Qwen2.5、Qwen3.5 等模型，而不是为每个模型重新包装一套 `.dla`。**

**状态：设计与实验规范阶段，尚无本仓库独立 NPU 算子、整模运行或性能提升的实机验证。** 本 README 是计划，不是已实现功能清单；接口名称和目录草案不代表已有代码。设计更新：2026-11-12。完整设计入口：[docs/index.md](docs/index.md)。

## 1. 最终目标与“通用”的含义

最终使用路径是：受支持模型的 `config.json + safetensors + tokenizer`，经过 **OpenMTK 自己的量化/打包** 形成模型包或首次加载缓存，再由自有引擎执行。最终 NPU 推理路径不依赖厂商 LLM SDK、逐模型 `.dla` 或闭源 `libneuron` 的加载/准备/执行，但仍使用目标板内核驱动与固件。研发阶段保留官方工具链作为参考答案生成器，不能把这种研发依赖混入最终运行依赖。

这里的**通用是接入方法、执行接口和测试设施可复用，不是已经支持任意 PyTorch 算子，也不是所有算子都必须在 NPU 上跑**。新算子按能力选择：原生 kernel、已有 primitive 的组合/融合、经过验证的近似，或显式 CPU 回退。未知能力必须报不支持，不得静默使用 Neuron 后仍宣称“独立后端成功”。

```text
模型配置 + safetensors
       |
架构适配器：权重名称、shape、bias、位置规则、状态类型
       |
自有转换：量化元数据 + 可移植 codes/scales + 设备 packing cache
       |
执行计划：逻辑算子 → 分解/融合 → 能力匹配 → 内存/状态/调度
       |                                  |
CPU/NEON 后端                         MTK 原生后端
                                布局、分块、任务生成、提交
                                          |
                                 目标 BSP 驱动 + 固件
```

`safetensors` 只提供张量，模型结构仍由架构适配器解释。相同的 Matmul 可服务不同模型；模型出现新数学或新状态时，仍需要增加对应适配。初版采用明确的 Qwen2.5 执行图，不先开发任意 Python 动态图编译器。详见 [架构设计](docs/01-architecture-and-roadmap.md) 与 [safetensors 项目][safetensors]。

## 2. 所有算子共用的一条接入路径

**不做“抓一次 ioctl，就保存成一个算子”的黑盒集合。** 要把观测结果转成有适用范围、有数值规则、能重建资源的实现。`ioctl` 是提交边界，一次提交可能对应融合子图；`.dla` 文件也不能直接等同于硬件命令缓冲。

| 步骤 | 每个算子都要做的事 | 交付/通过条件 |
|---|---|---|
| ① 定义语义 | 明确数学、shape、广播、stride、dtype、量化、是否修改状态、错误行为 | 一份算子契约；不能只写 PyTorch 名称 |
| ② 建立参考 | CPU 高精度参考；目标精度/量化模拟；官方路径作为可用时的设备对照 | 固定 seed 和输入资产；区分数学误差、量化误差和后端错误 |
| ③ 最小化与观测 | 最小合法图；核对目标 BSP；记录调用前后、buffer、子任务及完成事件 | 可解释 trace；实际执行引擎有证据；未知字段明确标记 |
| ④ 控制变量实验 | 同形状改输入/权重、同输入改分配地址；再单独改尺寸、stride、精度或融合 | 地址变化与计算语义变化分开；记录布局、分块、精度及固件边界 |
| ⑤ 独立实现 | 自有 pack/unpack、plan、地址重定位、命令生成和同步；或分解成已有 kernel | 新进程重建资源仍正确；没有借用厂商 runtime/HAL 执行 |
| ⑥ 参数化与能力登记 | 声明合法尺寸、对齐、尾块、动态操作数、累加精度和状态约束 | 在声明范围内生成任务；包含未被录制过的形状或组合测试；不是无限泛化承诺 |
| ⑦ 模型集成与调优 | 先独立对齐，再融合、常驻、复用、流水线与 autotune | 算子、Block、长序列和端到端均验证；完整调用有收益才启用优化 |

每个算子最终交付一套 `spec + reference + fixtures + capability + implementation + tests + benchmark`，共用 device、buffer、同步、trace 和结果报告设施。对于 view、CPU 小算子、由已有 primitive 组合的算子，不需要重复逆向底层命令；只有遇到新的硬件原语或执行模式时才新增探测。

最小 Matmul 是第一份“完整样板”：先验证原始整数乘加，再接量化 Linear；不要同时引入 W4、LUT 和全模型。候选形状从目标板允许的 `M=1,K=64/128,N=64/128` 起步，合法范围由实测确定。完整实验规程见 [MTK 探测](docs/05-mtk-probing-and-native-backend.md)。

### 算子契约必须覆盖的字段

| 契约 | 必须记录什么 |
|---|---|
| 数学与输入 | 操作语义、常量/动态输入、转置、广播、边界条件；Attention 的 K/V 不是静态 Linear 权重 |
| 数值 | storage/compute/accumulator dtype，scale 的轴，零点，round/clip，bias，允许误差 |
| 布局 | 逻辑 shape/stride、设备 layout、padding、对齐；view 不可表达时明确产生复制 |
| 资源 | 持久权重、每次调用 scratch、状态 buffer 的生命周期；读写范围、alias/in-place 规则 |
| 计划 | shape bucket、tiling、任务依赖、重定位；持久缓存不能保存过期 fd/VA/内核对象 ID |
| 状态 | 无状态或有状态；初始化、append/update、reset、失败后的有效性；按层、会话隔离 |
| 能力与回退 | 支持哪些设备/形状/精度；不支持时明确失败或显式 CPU 路径，分别报告耗时 |
| 证据 | CPU/板端测试、编译器与固件指纹、源版本、实际设备执行与性能条件 |

建议的统一接口是 `supports(op, tensors, precision, state) → make_plan → prepare → run → wait → release`；这是待实现接口。`wait` 完成并完成必要同步前，输出不可当成可读结果。缓存键至少包括算子/shape/布局/量化/近似版本、权重 hash、SoC 和 ABI/固件兼容标识。

## 3. 优先建立可复用的 primitive，而非逐一照搬 ATen

高层算子可以分解，也可以融合。`Linear` 通常是 Matmul 加 epilogue；`SwiGLU` 是两路投影、SiLU、乘法；Attention/GDN 是带状态与调度约束的复合操作。第一版无需收集 PyTorch 的全部算子。

| 类别 | 主要覆盖 | 首版与后续方向 | RK 参照 |
|---|---|---|---|
| 静态权重 GEMV/GEMM | Q/K/V/O、MLP、LM Head 的线性投影 | 独立 INT8 MAC → W8A8；权重一次 pack、反复使用 | [typed W8A8][rk-api]、`rk_npu_quant_matmul.cpp` |
| 动态 B 的 MM/BMM | QKᵀ、PV、GDN 的部分矩阵计算 | 单独验证浮点/混合精度及输入布局；不能拿静态 INT8 Linear 的通过结果替代 | [MM][rk-mm]、[Attention][rk-attn] |
| 元素级与广播 | Add/Mul、残差、bias、门控、缩放 | 先 CPU/NEON；再与相邻计算融合，避免为小算子单独提交 | `rk_npu_cpu_kernels.cpp`、融合 SwiGLU |
| 归约与归一化 | sum/max/amax、RMSNorm、L2Norm、Softmax | FP32 参考；归约顺序、epsilon、mask 和溢出单独验证 | [CPU 算子][rk-cpu]、[Attention][rk-attn] |
| 数据访问与转换 | gather、split、reshape、transpose、cast、Q/DQ、pack | 尽量用 view；必须复制的操作计入字节数、同步和端到端时延 | packing、native KV |
| 局部卷积 | causal/depthwise Conv1D、短卷积 | 先直接 CPU 实现；避免为了复用 GEMM 产生过大的 im2col/packing 开销 | [GDN projected-input 路径][rk-gdn] |
| 持久状态 | KV append、Conv history、GDN recurrent state | 独立状态管理，不当成普通临时 Tensor；设备布局由 backend 决定 | native KV、GDN state/history |

NPU 上可执行不等于更快。比较时计入 quant/pack、复制、cache 同步、提交、等待、归约和反量化；同时报告冷启动与热执行。不能用单个 MAC 时间替代 token 延迟，也不能把重叠的 CPU/NPU 区间直接求和。

## 4. LUT 策略：区分预计算常量表和函数近似表

**RoPE 的位置 sin/cos 表值得优先采用；其他函数只是 LUT 候选，不默认查表一定更快。** CPU 的向量数学、多项式近似、NPU 原生算子、硬件 LUT 的收益不同。RK 有可编程 DPU LUT 的实现不代表 MTK 存在或开放同样能力。[RK Attention][rk-attn]

| 对象 | 建议处理 | 必须注意 |
|---|---|---|
| RoPE sin/cos | 按位置与频率预计算/按需扩展 FP32 表，再执行旋转乘加 | 表本身不是整个 RoPE；缓存键包括 theta、rotary dim、scaling、位置布局；长位置不能简单先截成低精度 |
| Softmax 的 exp | 先稳定 FP32；后比较 exp LUT/多项式/NPU 近似 | 先减 row-max，仍要做 mask、求和、归一化；不能用 LUT 取代整个 Softmax；处理全 mask 行 |
| sigmoid、SiLU、tanh、GELU | 有限输入区间的分段表或近似，只有实测有效才启用 | 选定实际模型函数及近似版本；SiLU 可由 sigmoid 表配合乘法实现，但仍需评估误差与带宽 |
| GDN softplus、动态 exp(g) | 先保留稳定 FP32，后独立评估 LUT | 门控误差会跨 token 积累；必须测试长序列状态漂移，不能只检查单步 cosine |
| GDN 的 exp(A_log) | 对固定权重可在加载时计算一次 | 这是常量折叠，不必做 LUT；保留精度与参数来源 |
| rsqrt / reciprocal | CPU/原生指令优先；必要时近似加迭代修正 | Norm 仍需动态归约；接近零的范围和 epsilon 特别敏感，不先上粗糙统一表 |
| Embedding | token ID 查权重的一行 | 是精确索引，不是函数近似；量化 embedding 的反量化另算 |
| LM Head | 大词表矩阵投影 | **不是 LUT**；即使与 embedding 共享权重，计算方式也不同 |

**每个近似都必须登记**：输入范围、采样方式、表大小、表/输出 dtype、插值方式、范围外策略、NaN/Inf 行为、最大绝对/相对误差，以及模块/整模误差预算。不能偷偷把超范围输入 clamp 成另一个函数；需要约定的稳定尾部公式或安全回退。近似版本进入模型/plan 缓存键。

RoPE 的离散位置表主要消除重复三角函数计算，并非粗粒度输入量化。注意 split-half 与 adjacent-pair 两种旋转布局、partial RoPE、scaling 和 M-RoPE 轴选择；位置编码由模型配置决定，不把 token 序号、cache 槽位和多模态位置混为一个变量。RK dense 路径的 [sin/cos 预计算][rk-dense] 与 [CPU 旋转][rk-cpu] 可作参照。

## 5. 量化、packing、LM Head 是全流程的一部分

### W8A8 首版数值边界

拟议 Linear 契约为 `Y = X @ W^T + b`，先使用对称 INT8：权重 per-output-channel、激活 per-token，FP32 scales；CPU INT64 golden 检查累加范围，目标 accumulator 语义须实测确认。CPU 先保留稳定的 dequant/bias epilogue。

```text
加载/转换：W → quantize → codes + scales → device pack → 常驻权重
每次执行：X → row scale → quantize/pack → INT8 MAC
          → Split-K reduction → dequantize → bias → 输出
```

quant 改数值，pack 改布局；两者分别测试。动态 activation scale 下，不能把一份静态 INT32 bias 用于所有 token。Split-K 只在 scale 与分块关系允许时先合并整数，bias 最后加一次。非零零点的 padding 应填“数值零对应的码”，不是无条件填 byte 0。具体 rounding、饱和、零行与非有限值契约见 [04 量化规范](docs/04-quantization-and-packing.md)。

始终区分：**FP32 数学参考、同精度 quant-sim、NPU 输出**。NPU 对 quant-sim 的误差用于找后端问题；quant-sim 对 FP32 的误差用于找量化损失。W8A8 不意味着 residual、Norm、RoPE、Attention、KV 或 GDN state 都转 INT8。第一版 KV 存储采用 FP16 并保留 FP32 对照；GDN state/门控采用 FP32 基线。W4、KV 量化和 state 量化分别后置，不能一并打开。

### LM Head 的专门处理

LM Head 计算 `logits[T,V] = hidden[T,H] @ W_head[V,H]^T`，可以复用 Matmul，但必须增加大 N/词表分块、最后有效 token 选择和采样接口，而不是再设计一种查表算子。RK 的 [output_head.rs][rk-head] 分别提供量化 Linear 和浮点路径。

| 问题 | 设计要求 |
|---|---|
| 大词表 | 沿词表 N 分块，复用输入量化结果，避免多份完整临时 logits 和重复权重 packing；尾块正确 |
| prefill | 普通生成只算最后有效位置的 head；PPL、teacher-forcing、投机验证需要的多行 logits 不可省略 |
| greedy / top-k | greedy 可逐块归并全局 argmax，无需 Softmax；top-k 必须全局合并，并按规定先应用 logit processors |
| top-p / 概率采样 | 需要正确的全局概率分布；不能每块独立归一化或截断后冒充全词表采样；首版可保留完整一行 logits |
| 共享权重 | 依配置处理 tied embedding/head；缺少独立 head 权重不一定是损坏；逻辑共享不保证设备布局共享 |
| 量化 head | 单独测 logits、token 排序和质量；量化派生 packed copy 的额外内存必须统计，不能假装仍与浮点 embedding 逐值一致 |

Qwen2.5-0.5B 的真实投影形状、QKV bias、GQA 以及 shared head 处理见 [03 对照表](docs/03-qwen25-operator-matrix.md)。模型 pack/cache 必须绑定权重 hash、量化和布局版本，不能仅按文件名复用。

## 6. Qwen3.5：线性注意力必须增加状态与执行模式契约

Qwen3.5 是 **Gated DeltaNet 线性注意力与 gated full attention 的混合架构**，不能把所有层替换为同一个 Attention，也不能把“线性注意力”理解为一个 `nn.Linear`。参考实现按 `layer_types` 分派；本项目也必须读取 checkpoint 的实际配置，不硬编码层比例、head 数、卷积宽度或 rotary dim。[官方实现说明][q35-doc]、[模型源码][q35-code]

第一阶段扩展仅针对选定的 dense 文本骨干；视觉、MoE 和投机解码另开阶段。Qwen2.5 的算子与数据管理可以复用，但以下数学/状态要单独实现和验证。

| 模块 | 可复用部分 | Qwen3.5 必须新增或核对的部分 |
|---|---|---|
| QKV/Z/a/b 与输出投影 | Linear、weight pack、量化 epilogue | head 分组、切分顺序与门控维度；投影拼接不等于随意重排 |
| Causal depthwise Conv1D + SiLU | 逐元素/局部卷积基础算子 | 保存原始投影的历史；tap 顺序、padding、prefill/单步 update 一致；有效 token 才推进状态 |
| Q/K L2 normalization | FP32 reduction、rsqrt、mul | 是 L2 的 sum，不是 RMSNorm 的 mean；epsilon、Q 缩放次数、head 映射单独核对 |
| beta 和 log decay | sigmoid、softplus、exp 候选 | 参考形式 `beta=sigmoid(b)`、`g=-exp(A_log)*softplus(a+dt_bias)`；先 FP32，不能照搬 W8 激活范围 |
| Gated Delta Rule | MM/BMM、归约、外积、逐元素更新 | 持久 recurrent state；单步 recurrence 与 chunk prefill；读写顺序、累计精度和状态布局 |
| Gated RMSNorm | norm、SiLU gate、multiply | Norm-before-gate 顺序与自身权重语义；不能简单用普通 Norm 替换 |
| Full attention 层 | GQA、KV、RoPE、Softmax | Q/K Norm、Q 投影同时产生 gate、Attention 输出的 sigmoid gate，以及 partial/M-RoPE 规则 |
| 模型普通 RMSNorm | norm primitive | 当前参照实现用 `1 + weight`，与 Qwen2.5 的直接 gamma 不同；Gated RMSNorm 也不能一律套用这个偏移 |

以上差异直接对照锁定的 [Transformers Qwen3.5 源码][q35-code]；不把 RK 的固定 C7 形状当作所有 Qwen3.5 checkpoint 的形状。

### 两种 Attention、三类持久状态

```text
full_attention 层：K cache + V cache，随有效上下文长度增长
linear_attention 层：recurrent matrix state + causal Conv history
```

Runtime 因此需要 `AttentionKVState`、`GatedDeltaState`、`ConvHistory`，不能把后一种强行塞进“KV append”。它们按层和会话拥有独立生命周期。线性层状态大小在固定模型配置下不随序列长度线性增长，但其数值会依赖整个历史；混合模型仍有 full-attention KV，因此整模内存并非完全与上下文长度无关。

采用参考实现方向 `S[Dk,Dv]` 时，单个 head 的递推可理解为下式；这是数学示意，不是可直接提交的 NPU 程序：

```text
# q/k 已按目标实现完成 L2 norm；q 的 1/sqrt(Dk) 只应用一次
S_decay = exp(g_t) * S_prev
prediction = k_t^T @ S_decay
correction = beta_t * (v_t - prediction)
S_new = S_decay + outer(k_t, correction)
o_t = q_t^T @ S_new
```

RK 文档使用转置的 `[Dv,Dk]` canonical state；公式等价但内存布局、归约轴和 head 映射必须转换，不能原样复制。[RK GDN 契约][rk-gdn]

**Prefill 与 decode 分开计划。** 先做 CPU FP32 单步递推对照，逐 token 验证 `output + recurrent state + Conv history`；再做 chunk prefill 的矩阵化，核对完整序列、分块、`8+1+1` 在同精度下的结果。chunk 实现可能包含累计 log decay、Gram/BMM 和三角系统求解，不能粗暴变成一批互相独立的 Matmul。它是新的复合算子实现，而不是只换一层循环。

RK 的参照实现将部分 prefill 矩阵运算放到 NPU，并由 CPU 处理部分输入变换、衰减系数和三角系统；文档明确单 token decode 仍是独立 CPU 操作。OpenMTK 也先测完整调用，再决定小型递推是否值得下沉 NPU；不能从 RK 的 TF32 路径推断 MTK 支持 TF32。[RK GDN][rk-gdn]

**状态验证比单次输出更重要。** 测试 chunk 边界、断点续跑、reset、长序列、多会话隔离；padding token 不得悄悄推动 Conv history 或衰减有效状态。零输入并不自动等于无状态更新。发生执行失败时，状态要么未提交、要么标记失效后重建，不能继续使用半更新结果。未来投机回滚必须恢复 recurrent state 和 Conv history，不能像普通 KV 一样只缩短 length。

## 7. 分阶段计划：先做出可复用样板，再扩模型

下表沿用详细文档的 P0–P7，新增 P8 作为后续模型扩展。全部是待完成里程碑，不承诺硬件能力或加速倍数。

| 阶段 | 主要工作 | 关键验收 |
|---|---|---|
| P0 | 冻结唯一目标板、BSP/UAPI、固件与官方参考；建立 CPU 测试工程 | 官方路径复现、数据和版本可追溯 |
| P1 | 最小 INT8 Matmul，确定原始累加/重定量化输出契约 | CPU golden 与已知设备结果可比较，确认实际执行引擎 |
| P2 | Recorder、buffer 关系与同步；A/B/A、换权重、换地址、形状 diff | 看清一次任务而不是只得到顶层十六进制 |
| P3 | 独立固定形状 Matmul | 无 libneuron/HAL 代执行；新进程、重新分配、不同 A/B 正确 |
| P4 | 参数化 Matmul、W8A8 Linear、通用能力注册与 plan/cache | 支持范围内的未录制形状、真实 Qwen 非方阵/尾块、quant/pack/dequant 全链通过 |
| P5 | Qwen2.5 原宽单层；其余数学先 CPU | 逐节点比较，`8+1+1` 和历史 KV 干预通过 |
| P6 | 两层 → 完整 24 层 → Embedding/LM Head/Tokenizer/生成 | 固定 token teacher-forcing、质量、状态隔离、内存稳定通过 |
| P7 | 动态 B MM/BMM、原生 KV/Attention、融合、autotune；再评估 W4/LUT | 完整调用与整模加速有证据，误差和热稳定达标 |
| P8 | Qwen3.5 dense 文本：CPU GDN → 单个 GDN/full-attention block → 混合层 → 整模 | chunk/recurrent、Conv/GDN/KV 状态、门控/Norm/RoPE 与长序列全部对齐 |

CPU 模型参考与设备探测可以并行；模型主线不能被误报为原生后端进展。**第一份独立 Matmul 未通过时，不把更换模型、增加 LUT 或封装 `.dla` 当成替代验收。** 官方 `.dla` 与模板可以用于研究，但最终能力等级必须如实区分：观察、录制/受控复用、独立固定形状、参数化算子、模型集成。

## 8. 必须从第一天补上的工程能力

| 项目 | 为什么必须有 |
|---|---|
| Shape/stride 与 padding 契约 | 算子从 64×64 扩到真实非方阵、GQA、长上下文时，错误常在尾块或布局而非乘法 |
| 静态权重和动态输入分开 | Linear、Attention、GDN 的数据生命周期和 packing 成本不同，不共用错误的缓存假设 |
| 统一 tensor/buffer/workspace 管理 | 明确 owner、alias、sync、设备可见性和回收；预分配可复用 scratch；避免每 token 分配 |
| 有状态算子接口 | KV、卷积和递推矩阵必须支持初始化/更新/reset；统一事务边界，为恢复/回滚留接口 |
| 数值与近似版本 | FP32/quant-sim/NPU 三路对照；LUT、多项式、FP16 scale、state 低精度各自独立实验 |
| 可验证的 fallback | 同时报告不支持原因、实际后端与成本；“混合推理”不写成“全 NPU” |
| 计划与 tuning cache | 只缓存规则/布局与可重建资源；计入 SoC、ABI、shape、精度、CPU/NPU 资源配置，变化后失效 |
| CPU/mock/板端/模型四层测试 | 能在没有板子的环境测试数学和契约，但 CPU CI 通过不能冒充 NPU 测试通过 |
| 端侧持续性能 | 无插桩基线；cold/warm、TTFT、prefill/decode、内存、复制量、时钟/温度/功耗条件分别记录 |

发布算子能力时，必须附支持矩阵与失败样例。避免“所有差异小于一个统一 1e-5”或“能输出中文就是正确”；原始整数算术、量化模拟、浮点近似和模型质量采用不同标准，详见 [07 验证规范](docs/07-validation-and-work-items.md)。

## 9. 参考项目：学边界与方法，不搬硬件常量

| 来源 | 主要参考内容 | 不应推导出的结论 |
|---|---|---|
| [happyme531/rknpu2_matmul_open][rk-root] | 自有 kernel、typed quant API、packing/tiling、native KV、LUT exp、GDN prefill | MTK 使用相同寄存器、INT4/TF32、cache 或 LUT 单元 |
| [happyme531/open_rknpu2_lm_infer][rk-engine] | 权重加载、CPU/NPU 分工、Block、LM Head、状态与生成调度 | 更换设备名就能支持 MTK；所有模型的 bias/Norm/位置规则相同 |
| [allbilly/rk3588][allbilly] | 微算子、寄存器编程和对照实验方法 | ioctl 等于通用 ISA，或记录到字节就已实现可移植 kernel |
| [Peterc3-dev/phone2a-apu-unlock][phone2a] | MT6886 探测、ABI 与初始 recorder 的参考 | 其顶层 hook 已完整捕获/回放任务，或 MT6897/MT8792 可照用结构体 |
| [公开 MT6897 APUSYS 源码][mt6897] | driver 对指针/句柄、命令和内存的真实解释 | 与自己的 BSP 必然二进制兼容；最终仍需匹配本机源码 |
| [Transformers Qwen2][qwen2-code] / [Qwen3.5][q35-code] | 模型数学、权重/维度、状态转移的参考 | 高层算子和硬件命令必须一对一 |

本项目借鉴 RK 的提速方向是：权重常驻、减少重复 quant/pack 和中间落地、合适 tiling、融合、CPU/NPU 重叠与布局复用。是否在 MTK 有收益必须做 ablation；不套用 RK 的倍数，也不把“绕过厂商”本身当作性能来源。

参考阅读不等于代码复用授权。RK 算子库注明 AGPL-3.0-only，且其 [第三方声明][rk-notices]记录了部分上游授权待澄清；复用前核对文件来源和许可证。不要上传厂商 SDK/固件、模型权重、密钥或未经审查的原始设备 trace。本仓库未在本次 README 更新中选择项目许可证。

## 10. 详细文档与当前工作入口

| 文档 | 用途 |
|---|---|
| [文档总目录](docs/index.md) | 当前证据和完整阅读顺序 |
| [01 架构与分阶段计划](docs/01-architecture-and-roadmap.md) | P0–P7、接口职责与三条验证路径 |
| [02 参考项目与 RK 加速机制](docs/02-references-and-rk-design.md) | 仓库、源码入口、优化机制及引用版本 |
| [03 Qwen2.5 算子对照表](docs/03-qwen25-operator-matrix.md) | 实际形状、bias、CPU/NPU 分工和优先级 |
| [04 量化与 packing](docs/04-quantization-and-packing.md) | scales、zero-point、Split-K、bias、共享 head |
| [05 MTK 探测与原生后端](docs/05-mtk-probing-and-native-backend.md) | Recorder、扰动实验、独立性及重定位验证 |
| [06 Transformer / KV / 生成](docs/06-transformer-kv-and-generation.md) | 原宽单层、两层、整模与 KV 状态机 |
| [07 验证与任务清单](docs/07-validation-and-work-items.md) | 测试层次、性能规范与首批工作 |
| [08 实验模板](docs/08-experiment-templates.md) | 设备、算子、trace 与结果记录 |

**当前先做：设备契约、最小 Matmul golden、官方微算子轨迹、原宽单层 CPU 基线。目标是把第一个算子的接入流程做扎实，让下一种算子复用同一套设施；特殊数学写进独立 recipe，而不是散落在模型执行代码中的临时补丁。**

[rk-root]: https://github.com/happyme531/rknpu2_matmul_open/tree/3be24de319b140f95d94634f00c0d6f8df5e96b3
[rk-api]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/API.md
[rk-mm]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/MM.md
[rk-attn]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/ATTENTION.md
[rk-gdn]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/GDN.md
[rk-notices]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/THIRD_PARTY_NOTICES.md
[rk-engine]: https://github.com/happyme531/open_rknpu2_lm_infer/tree/e1331947c00d438470e48b585f7b34d0f783f40b
[rk-dense]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/dense.rs
[rk-cpu]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/cpu.rs
[rk-head]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/output_head.rs
[qwen2-code]: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py
[q35-code]: https://github.com/huggingface/transformers/blob/57583c28b04ddc6b6c1e9390ad86181d1033c660/src/transformers/models/qwen3_5/modeling_qwen3_5.py
[q35-doc]: https://github.com/huggingface/transformers/blob/57583c28b04ddc6b6c1e9390ad86181d1033c660/docs/source/en/model_doc/qwen3_5.md
[allbilly]: https://github.com/allbilly/rk3588
[phone2a]: https://github.com/Peterc3-dev/phone2a-apu-unlock
[mt6897]: https://github.com/oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897/tree/e92259a0119b33a6d7b1676614b6e812ad605989/kernel/kernel_device_modules-6.1/drivers/misc/mediatek/apusys
[safetensors]: https://github.com/huggingface/safetensors
