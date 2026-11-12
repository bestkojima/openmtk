# 06 · 从原宽单层 Transformer 到完整 Qwen2.5

[返回目录](index.md) · 本文是拟议实施流程。数学基准为 [Qwen2 实现][qwen] 和 [Qwen2.5-0.5B 配置][config]；硬件能力以本项目实测为准。

## 1. Tiny 版本到底缩哪里

用于模型语义验证的 tiny 版本：**只取真实 checkpoint 的第 0 个 decoder block，保留 H=896、F=4864、14 个 Q heads、2 个 KV heads、D=64、真实权重和 bias**。不先缩宽，不删 RoPE/GQA/残差，不给它加 Qwen3 的 QK Norm。

这样保留了真实 Linear 形状、量化分布和状态接口。但单层通过并不能证明 24 层的误差累积、内存、语言质量和性能通过。另建很小的随机 Matmul 用于驱动/布局实验；两种 tiny 各有用途，不互相替代。

微算子跑通后不必等所有 NPU 算子完成：先让七类 Linear 使用原生 NPU Matmul，Norm、RoPE、Attention、激活与残差留在 CPU，得到可解释的混合执行基线。后续根据真实转换/同步成本选择融合。

## 2. 一个 Block 的数据流

下面是语义伪代码，不是已实现的 OpenMTK API。`linear` 在 CPU quant-sim 与 mtk-native 两个后端使用相同量化契约：

```python
# x: [T,896]，positions: 本次各 token 的绝对位置
# cache 是这一层专属对象；global position 由整个模型事务管理
u = rmsnorm_fp32(x, input_norm_gamma, eps=1e-6)
q = linear(u, Wq, bq).reshape(T, 14, 64)
k = linear(u, Wk, bk).reshape(T, 2, 64)
v = linear(u, Wv, bv).reshape(T, 2, 64)
q, k = rope_split_half(q, k, positions, theta=1e6)
cache.append(k, v)  # K 已旋转，V 未旋转；第一版保存 FP16
z = gqa_stable_fp32(q, cache, positions, scale=1/8)
a = x + linear(z.reshape(T, 896), Wo, None)
u2 = rmsnorm_fp32(a, post_norm_gamma, eps=1e-6)
gate = linear(u2, Wgate, None)
up = linear(u2, Wup, None)
middle = silu_fp32(gate) * up
y = a + linear(middle, Wdown, None)
```

每个节点有稳定名称与可选 snapshot：`norm1/q/k/v/q_rope/k_rope/attn/o/residual1/norm2/gate/up/swiglu/down/output`。默认性能运行不 dump；诊断模式记录张量 shape、dtype、min/max、误差与必要的测试数据。RK 的 [dense/batch.rs][rkblock]提供类似组织方式，但其模型支持和 bias 条件与本模型不同。

## 3. RoPE、GQA 和 mask 的正确约定

**RoPE**采用 Qwen2 的 split-half：前半 a 与后半 b 配对，`a'=a*cos-b*sin`、`b'=b*cos+a*sin`。不能用 adjacent-pair 旋转替换。每个 head 的三角频率由 D=64 和 theta 得到；绝对位置决定取哪行 sin/cos。第一版 CPU FP32 生成/使用表，再做 NEON；表长度按实际配置 context，而不是无限预分配。[Qwen2 源码][qwen]

**GQA**中 query head `hq` 使用 KV head `floor(hq/7)`。缓存仅保存两个 KV heads；不要为 14 个 Q heads 复制整份 KV，否则把本来节省的内存和带宽消耗掉。CPU reference 可以借助广播表达，生产布局应按组复用。

**Mask**必须按全局位置定义：假设 prefix 长度 p，本次 T 行，第 r 行只看位置 `0..p+r` 的有效 keys。decode 时 T=1，它应看全部合法历史和当前 token，而不是矩形矩阵的左上角一项。使用 PyTorch SDPA 时也要显式核对非方形 causal mask 的对齐方式，不能只设置 `is_causal=True` 就假定满足已有 prefix 的语义。

仅对有效区间 softmax；padding 和未来槽不能改变输出。稳定计算先减 row max，再 exp/sum；fully-masked 行的返回规则要固定并测试。第一版不做 fixed-shift softmax 或未经验证的 exp LUT。

## 4. KV Cache 是有状态内存，不是导出权重

每层一个 cache，逻辑 K/V 形状均为 `[2,capacity,64]`；batch=1。每层记录 valid length、capacity、dtype、logical/native layout、generation/session ID 和资源状态。存储 FP16，比较时先明确是否把 CPU 参考同样舍入到 FP16。

| 接口/状态 | 必须满足的语义 |
|---|---|
| `reserve(capacity)` | 仅保证空间，不能增加 valid length；不得把新增未初始化区域当历史 |
| `append(k,v)` | 仅写新有效位置；K 已完成 RoPE；提交失败不得继续使用可能部分更新的状态 |
| `valid_length` | 是已经成功处理并可供后续 Attention 使用的 token 数，不是分配容量 |
| `reset()` | valid length/position 归零，旧值不可见；容量可复用；跨请求隔离必须测试 |
| `truncate()` | 延后到 prefix reuse/投机解码需要时再做；长度回退后无效后缀不能参与计算 |
| native layout | K 和 V 可以不同；query/append 明确布局版本和 stride；不能把 BF16 bits 当 FP16 |
| workspace | 临时计算内存与持久 KV 分离；串行层可复用 workspace，但每层 cache 不共用数据 |

实际读写先遵循目标 BSP 的 CPU/device 所有权和同步契约，再优化同步范围。RK 的 [Attention 文档][rkattn]明确把 native KV 放在 DDR GEM buffers，按新 token 追加，并区分 capacity 与 valid length；这不是把 KV 常驻片上 SRAM。MTK 的 buffer/同步方式需独立实现。

不要为了每步输出完整 cache 而把整个前缀反量化、读回、重新上传。第一版 CPU Attention 允许存在已知搬运，但必须记录字节数和耗时；后续 NPU Attention 的验收要确认只更新需要的数据，不能只看 API 名叫 zero-copy。

## 5. 必须完成的 8+1+1 实验

用 10 个固定 token IDs；参考与板端必须同权重、同 Norm/RoPE、同 KV 存储精度、同量化规则。先 teacher forcing 固定输入，避免生成 token 分叉掩盖根因。

1. CPU reference 一次处理 10 个 token，保存每位置中间张量、输出和各层 KV。
2. 候选实现 prefill 前 8 个 token，验证输出与前 8 个有效 KV。
3. 显式输入第 9 个 token，位置 8；比较输出与第 9 个 KV，valid length 应为 9。
4. 显式输入第 10 个 token，位置 9；比较输出与第 10 个 KV，valid length 应为 10。
5. 重置到同一已知状态，只改一个被关注的有效历史 V 槽，预测/比较 Attention 输出变化；恢复后结果复现。若该位置概率极小，选择有明显非零 Attention 权重的合成用例，不能要求所有扰动必然有可见变化。
6. 单独改 capacity 内但 valid length 外的槽，输出不得改变；再测 chunk size 1/2/8 的等价性与跨边界位置。

至少用真实 checkpoint 激活和合成可辨识输入两类数据。只在随机权重 CPU 程序上通过，不能算目标板的真实模型验证。

## 6. 从一层走到整模的具体顺序

| 步骤 | 工作 | 关键验收 |
|---|---|---|
| M0 | 读取配置、权重名称、分片、bias；CPU FP32 原宽第 0 层 | 与冻结的 HF eager reference 对齐；拒绝缺失或错 shape |
| M1 | CPU W8A8 quant-sim，第 0 层仍使用 FP32/FP16 明确边界 | 区分量化损失与实现误差，确定临时误差预算 |
| M2 | 只把 q_proj 替换为独立 NPU Linear | 小范围替换通过，再换 k/v；先查 bias 和 native layout |
| M3 | Q/K/V/O 全部 NPU，Attention CPU | 8+1+1 和 mask/KV 干预通过 |
| M4 | Gate/Up/Down NPU，得到完整单层 | 七种 Linear shape 都通过；Down scale 使用真实 SwiGLU 输出 |
| M5 | 两层，缓存按 layer 独立 | 抓“第一层正确、第二层出错”的 state/weight/workspace 别名问题 |
| M6 | 24 层 teacher forcing；初期短 context | 每层边界误差、峰值内存、资源清理通过 |
| M7 | Embedding、final norm、共享浮点 LM head | 能计算正确的完整 logits；不要用可读文本代替对照 |
| M8 | tokenizer、chat template（适用时）、greedy/EOS | 固定输入生成及 tokenizer roundtrip；位置计数正确 |
| M9 | 量化/分块 LM head、扩展上下文、性能/质量评测 | 纳入额外 head packed copy、长期 KV 和工作区；冻结语料评测通过 |
| M10 | 设备 Attention/native KV、融合及异步优化 | 同质量/同工作量的完整调用与端到端提升，逐项 ablation |

Base checkpoint 与 Instruct checkpoint 的 tokenizer/template/EOS 配置要分别固定；不要把模型名称相近当作相同测试对象。原宽单层是算子/状态验证，不是一个经过重新训练、可以用回答质量评估的小语言模型。

## 7. 生成循环与常见 off-by-one

正常生成先把 prompt 喂入所有层并得到最后有效位置 logits，从中采样第一个新 token。这个 token 只有在下一次 `decode(token)` 被模型处理之后，才被写进各层 KV。**已发给用户的 token 数与 cache 中已处理 token 数不总相同。** EOS/取消时不应凭显示长度调整 cache。

模型的全局 position 只在所有层本次执行都成功后推进。任一层失败时，将该 session 标记为失效并 reset/重建；不要一半层已追加、另一半未追加后继续 decode。第一版单 session、单线程提交；请求并发留到资源所有权与工作区隔离验证之后。

先用 greedy 做可重复测试；top-k/top-p、重复惩罚和随机采样后置。采样不是 NPU Matmul 的一部分。若出现首个 token 分歧，回到同一组 teacher-forced 输入比较 logits，不能继续比较已经走向不同文本的两个轨迹。

## 8. 端侧内存预算：先计算再分配

下列为按配置推导的**逻辑 payload 估算**，不含 allocator、padding、alignment、driver、runtime、复制与临时 workspace，不是实测 RSS。

KV 字节数：`2(K/V) × L × Hkv × S × D × bytes_per_element × batch`。本基线 FP16、batch=1 时每个 token 为 12288 bytes，即 12 KiB：

| 有效/预留长度 S | 24 层 FP16 KV 逻辑容量 |
|---|---:|
| 512 | 6 MiB |
| 2048 | 24 MiB |
| 4096 | 48 MiB |
| 32768 | 384 MiB |

Embedding 逻辑矩阵为 `151936×896`，BF16 约 259.66 MiB。24 层七类 Linear 的纯 INT8 codes 共约 341.25 MiB，per-channel FP32 scales 约 1.16 MiB，另有 norm/bias 等。若保留浮点 embedding 并增加 INT8 head packed copy，head 的原始 codes 就另需约 129.83 MiB，尚未算 packing。tie weights 不保证不同 native layout 可以零额外内存。

Attention 若为 4096 个 token 一次物化全部 FP32 scores，`14×4096×4096×4` 已约 896 MiB；而 chunk=128 对同一 KV 长度仅 scores 仍需约 28 MiB。因此 prefill 要有 chunk/workspace 策略，后期研究分块稳定 Softmax，不能只预算权重和 KV。

128 行 FP32 全词表 logits 约 74.19 MiB；普通生成只保留最后有效行约 0.58 MiB。评测所需多行 logits 可按块消费，而非永久保存整个上下文。

设计要求：持久权重/KV 与 scratch 分离；预分配或有界增长；steady decode 尽量不重新 alloc/map/pack 权重；释放取消/错误路径资源；首加载 mmap 与 device packed copy 分开统计。双缓冲会增加临时内存，只在实际有可重叠阶段时引入。

下一步：[验证、性能与首批任务](07-validation-and-work-items.md)。

[qwen]: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py
[config]: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
[rkblock]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/dense/batch.rs
[rkattn]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/ATTENTION.md
