# 03 · Qwen2.5 算子、量化步骤与端侧特殊处理对照表

[返回目录](index.md) · 本文区分模型事实与 OpenMTK 的拟议实现。优先级：P0=先建立正确性/后端契约；P1=单层与整模必须；P2=正确后优化；P3=延后。

## 1. 模型基线与统一记号

以 [Qwen/Qwen2.5-0.5B 官方配置][config]、[Qwen2 数学实现][qwen] 为准。实际实验须冻结 checkpoint revision、配置和权重 hash。

| 项目 | 本基线 |
|---|---|
| model_type / 层数 | `qwen2` / 24 |
| hidden H / FFN F | 896 / 4864 |
| Q heads / KV heads / head dim | 14 / 2 / 64；每个 KV head 服务 7 个 Q heads |
| vocab / checkpoint dtype | 151936 / BF16 |
| RMSNorm epsilon / RoPE theta | `1e-6` / `1e6` |
| attention scale | `1/sqrt(64)=1/8`，由 head dim 推导 |
| Q/K/V projection bias | **有**；O 与三个 MLP Linear 无 bias |
| Q/K RMSNorm | **没有**；不能照搬 Qwen3 的 Q/K Norm |
| embedding / LM head | `tie_word_embeddings=true`；逻辑上共享权重 |
| context | 配置上限 32768；第一阶段只测试小 context，再扩到 512/2048/4096，不预先承诺最大长度 |
| sliding window | `use_sliding_window=false`；不能仅看到 sliding_window 数值就打开滑窗 |

以下假设 batch=1；`T` 是本次 token 数，`S` 是历史加本次有效 KV 长度。Linear 统一为 `X[T,K] @ W[N,K]^T + bias[N]`。HF 文件中的 `W[N,K]` 与 NPU 物理 packing 是两层概念，不能逐 token 做完整权重转置。

## 2. 七类 Linear 的确切形状

| 投影 | K → N | HF 权重形状 | bias | 可合并关系 |
|---|---|---|---|---|
| Q | 896 → 896 | `[896,896]` | `[896]` | 与 K/V 共享同一 X，可合成 N=1152 的投影，但需保留三段 bias/scale |
| K | 896 → 128 | `[128,896]` | `[128]` | 同上 |
| V | 896 → 128 | `[128,896]` | `[128]` | 同上 |
| O | 896 → 896 | `[896,896]` | 无 | 输入是 Attention 输出，不可复用 QKV 的输入量化结果 |
| Gate | 896 → 4864 | `[4864,896]` | 无 | 与 Up 合为 N=9728，仅在布局与精度契约允许时 |
| Up | 896 → 4864 | `[4864,896]` | 无 | 同上 |
| Down | 4864 → 896 | `[896,4864]` | 无 | 输入为 SwiGLU 输出，必须重新得到适合该输入的 scale |
| LM head，Block 外 | 896 → 151936 | 逻辑 `[151936,896]` | 无 | 共享 embedding 的逻辑权重，但设备 packed copy 可能另外占内存 |

最小 Matmul 后优先覆盖：`(M,K,N)=(1,896,896),(1,896,128),(1,896,4864),(1,4864,896)`，随后把 M 扩至 8/32/128。LM head 是单独的大 N 分块测试，不能只测方阵就宣布整模支持。

## 3. Block 与模型外围算子表

| 优先级 / 操作 | 数学与张量边界 | OpenMTK 第一版路线 | RK 参照 | 端侧重点与错误检查 |
|---|---|---|---|---|
| P1 Embedding / gather | token IDs → `X[T,896]` | CPU；BF16 权重按需转 FP32 | [RK loader][rkload] 与 [CPU][rkcpu] | 是取行，不是 Matmul；检查 token 范围、BF16 解码、共享权重别名 |
| P1 input RMSNorm | `X / sqrt(mean(X²)+eps) * gamma` | CPU FP32 归约，先不做低位 norm | [RK CPU][rkcpu]；库另有固定形状 [Add+RMSNorm][rknorm] | gamma 长度 896；eps 不能遗漏；RK 库的 D=4096 特例不能直接用于本模型 |
| P0/P1 QKV Linear | `U[T,896] → Q[T,896],K/V[T,128]` | 自研 W8A8 Matmul + FP32 epilogue | [typed W8A8][rkquant]、[Block][rkblock] | **先反量化再加 FP32 bias**是基线；动态 activation scale 下不能固定一份 INT32 bias 给所有 token |
| P1 reshape / split / permute | Q→`[14,T,64]`，K/V→`[2,T,64]` | 优先 view/stride；必须复制时显式计数 | [Block][rkblock] | 不默认每个 PyTorch view 都是硬件 kernel；注意 contiguous 与 native layout 区别 |
| P1 RoPE | Q、K 的 split-half 旋转 | CPU FP32 sin/cos 表 + 乘加；可做 NEON | [RK loader][rkload]、[CPU][rkcpu] | 取当前绝对 position；不是对 V 旋转；不是 adjacent-pair 版本；cos/sin 表不是 NPU DPU LUT |
| P1 KV append | 新 K（RoPE 后）与 V → 本层 cache | FP16 存储、FP32 参考；容量/长度分离 | [RK Attention/KV][rkattn] | 只追加有效 token；history 不能每步重新 RoPE；GQA 不应物理复制成 14 个 KV heads |
| P1/P2 QKᵀ | 每个 Q head 对其所属 KV head，得 `[14,T,S]` | 先 CPU FP32；后做动态 B 的浮点/专用 Attention kernel | [RK FP16 MM][rkmm]、[Attention][rkattn] | K cache 不是静态权重；W8A8 静态 Linear 成功不代表此路径成功；归约范围与 FP16 overflow 单独验证 |
| P1 scale + causal mask | scores/8，禁止未来或无效位置 | CPU FP32 | [RK Attention][rkattn] | 带 prefix 的 prefill 使用绝对位置；decode 的 Q 长 1 不意味着只能看 KV 第 0 项 |
| P1/P2 Softmax | 减 row-max → exp → sum → normalize | CPU 稳定 FP32；硬件 LUT 后置 | [RK Attention][rkattn] | 明确 fully-masked 行策略；LUT 只近似函数局部，不替代 Attention；检查 NaN/Inf 和极端 logits |
| P1/P2 P@V | `[14,T,S] × [2,S,64]`，按 GQA head 映射 | 先 CPU FP32；后浮点/专用 NPU kernel | [RK Attention][rkattn] | V 同样为动态数据；避免完整 scores/概率在 DDR 来回搬运；后续采用分块稳定累加 |
| P1 O Linear + residual | 合并 heads→`[T,896]`，O 投影后加原始 X | W8A8 + CPU FP32 add | [RK Block][rkblock] | 残差引用的是本层 attention 前输入；不得直接相加不同 scale 的 INT8 tensor |
| P1 post RMSNorm | 对第一残差结果归一化 | CPU FP32 | [RK CPU][rkcpu] | 与 input norm 是不同 gamma；不要共享错权重 |
| P1 Gate/Up Linear | `U[T,896] → 两个[T,4864]` | W8A8，两路可共享输入 quant/pack | [RK Block][rkblock]、[W4A8 API][rkw4] | 输出各自有 weight scale；初版先分开可观测，融合后保留同样语义 |
| P1/P2 SiLU × Up | `silu(gate) * up` | CPU FP32，后融合 epilogue | [RK CPU][rkcpu]、[融合实现][rkswiglu] | 使用稳定实现；不能随意换 polynomial/LUT 后只看 cosine，极端值和最终质量也要测 |
| P1 Down Linear + residual | `[T,4864]→[T,896]`，加第一残差结果 | W8A8 + CPU FP32 | [RK Block][rkblock] | Down 的新 row scale 来自实际 SwiGLU 值；不能用 Gate/Up 输入 scale 代替 |
| P1 final RMSNorm | 所有 decoder 层之后 | CPU FP32 | [RK head 调用][rkheadcall] | 不属于每一层 FFN 后的额外操作，只有模型末尾一次 |
| P1 LM head | hidden→151936 个 logits | 先 CPU 浮点共享权重基线，后独立量化 NPU head | [RK output head][rkhead] | **不是 LUT**；是否量化、是否复制 tied 权重要显式设计；普通生成 prefill 只需最后有效位置的 logits |
| P1 sampler / tokenizer | logits→token ID→文本 | CPU；先 greedy，后 top-k/top-p | RK 引擎 `src/sampling.rs` / `src/text/` | greedy 不要求先对整词表做 softmax；EOS、chat template 和 tokenizer 必须与 checkpoint 一致 |
| P3 QK Norm / MoE / GDN | 不属于本基线的额外模型结构 | 本阶段不实现 | RK 其他模型模块 | Qwen3、Qwen3.5 与 Qwen2.5 必须分开能力声明 |

## 4. 量化和数据流也必须作为一等算子管理

不仅需要 Matmul。下面这些操作经常占用端侧时间或造成数值错误；即使融合进一个 C++ 函数，也要能分项验证和统计。

| 优先级 / 操作 | 发生时点与数值 | RK 参照 | OpenMTK 首要处理 |
|---|---|---|---|
| P0 weight quantization | 离线/首次加载，按输出通道的 INT8 codes + FP32 scales | [API][rkquant]，引擎 checkpoint loader | 与 checkpoint source hash 绑定；真实模型质量不能靠随机数据校准证明 |
| P0 weight packing | 离线/加载时把逻辑 W 重排、padding 到物理布局 | [CPU kernels][rkpack]、[plan][rkplan] | 与量化分开；pack→unpack bit-exact；尾块填数值零对应的整数码 |
| P0 activation scale / quantize | 每次 Linear，按 token 行 amax→scale→round/clip | [API][rkquant]、[CPU kernels][rkpack] | FP32 输入；零行、极值、非有限值有明确行为；记录 scale 的坐标轴 |
| P0 activation packing | 每次调用按 kernel 输入布局写 scratch | [CPU kernels][rkpack] | 尽可能与 quantize 一次遍历；先有非融合 golden 再优化 |
| P0 integer MAC | INT8×INT8→已验证的 accumulator | [typed API][rkquant] | CPU INT64 参考检查溢出；硬件是否 INT32 原始输出必须实测 |
| P0 Split-K reduction | 多个 K tile 的 partial 合并 | [plan][rkplan] | 全行 scale 不变时可先合并整数；per-group scales 时应逐组缩放后相加 |
| P0 dequant + bias | accumulator×row scale×channel scale + bias | [API][rkquant] | FP32 epilogue 基线；检查符号、广播、bias 加一次；不可静默使用 FP16-rounded scales |
| P1 dtype conversion | BF16/FP16/FP32/int8 之间 | [CPU][rkcpu] 与 [API][rkquant] | BF16/FP16 不是同格式；cast、dequant、requant 必须区分，不能看 bit 数猜 dtype |
| P1 layout transform / cache sync | CPU↔NPU 或两种 kernel 之间 | [core][rkcore]、[Attention][rkattn] | 每个转换记录字节和同步；mmap 不是自动一致性/零复制的证明 |
| P2 W4A8 | 新量化格式、解包/分解、scale 和可能的校正项 | [W4A8 API][rkw4] | 不按 RK 的 INT16 partial 限制推断 MTK；验证原生指令或等价分解成本 |
| P3 KV quantize / dequantize | 运行时新 K/V 写入，外加 scale 元数据 | RK 当前参考路径是 FP16 KV | 与 weight quant 独立设计；第一版不做 KV INT8/INT4 |

## 5. 最先考虑的方向

第一优先级不是实现所有 PyTorch 算子，而是建立 **整数 Matmul + weight pack + activation quant + dequant/bias + buffer/同步** 的闭环。随后先用 CPU 完成 Norm/RoPE/GQA/Softmax/SiLU，把单层状态跑对；最后依据完整 trace 决定哪些边界值得移到 NPU 或融合。算子在 NPU 上“可执行”与整个 Block “更快”是两项不同验收。

继续阅读：[量化数值契约](04-quantization-and-packing.md)、[单层到整模](06-transformer-kv-and-generation.md)。

[config]: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
[qwen]: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py
[rkquant]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/API.md
[rkpack]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_cpu_kernels.cpp
[rkplan]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_kn_plan.cpp
[rkcore]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_core.cpp
[rkattn]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/ATTENTION.md
[rkmm]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/MM.md
[rknorm]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/include/rk_npu_add_rmsnorm_f16.h
[rkw4]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/include/rk_npu_w4a8.h
[rkswiglu]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_w4a8_swiglu.cpp
[rkload]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/dense.rs
[rkblock]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/dense/batch.rs
[rkcpu]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/cpu.rs
[rkhead]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/output_head.rs
[rkheadcall]: https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/dense/head.rs
