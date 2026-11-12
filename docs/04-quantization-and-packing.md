# 04 · 量化、反量化、bias 与权重打包契约

[返回目录](index.md) · 本文是 OpenMTK 第一版的**设计选择**，不是对 MTK 原生数值语义的断言。硬件不同于契约时，要实现明确适配或报告不支持，不能静默改变数学。

## 1. 三种参考结果必须同时保留

| 参考 | 内容 | 用来定位什么 |
|---|---|---|
| R0 FP32 | 原始权重解码成 FP32 的 CPU 前向 | 模型数学和未量化质量基准 |
| R1 quant-sim | 在 CPU 执行与目标一致的 round/clip/scale、整数乘加、反量化及 bias | 量化损失；与 R0 的差异不是后端 bug 的直接证据 |
| R2 mtk-native | 与 R1 完全相同 codes/scales 的设备执行 | 后端、packing、溢出、布局和同步错误 |

若 R2 与 R1 不一致，先修执行契约；若二者一致但都偏离 R0，才调整量化算法、校准和混合精度。不要把 cosine similarity 当绝对误差，也不要对所有 tensor 统一套 `1e-5`。

RK 的 [typed W8A8 API][rkapi]分别暴露整数与浮点输入输出，并把 CPU quantize/pack、NPU MAC、CPU reduction/dequantize 区分开，适合作为这个边界的参照。

## 2. 第一版 W8A8 定义

逻辑运算：`Y[M,N] = X[M,K] @ W[N,K]^T + bias[N]`。内部使用对称 INT8，数值范围选择 `[-127,127]`、零点 0、round-to-nearest ties-to-even。这个范围是我们为可解释性做的选择，不证明目标硬件自动采用同样范围。

权重按输出通道 n：

```text
sw[n] = max(abs(W[n,:])) / 127
Wq[n,k] = clip(round_even(W[n,k] / sw[n]), -127, 127)
```

输入按 token/行 m，每个 Linear 独立计算：

```text
sx[m] = max(abs(X[m,:])) / 127
Xq[m,k] = clip(round_even(X[m,k] / sx[m]), -127, 127)
I[m,n] = sum_k int32(Xq[m,k]) * int32(Wq[n,k])
Yhat[m,n] = float32(I[m,n]) * sx[m] * sw[n] + bias[n]
```

全零行/全零权重通道约定 scale=1、codes 全零。非有限输入在提交前返回错误；不得悄悄 cast 成任意整数。scale 保存为 FP32，先做 FP32 epilogue；任何 FP16 scale 优化都要成为另一个数值版本并重新验证。CPU golden 用 INT64 累加，先证明结果落在 INT32 可表示范围，再比较目标输出。

硬件若要求非对称激活、融合 requantization 或其他整数精度，需要另立 kernel contract。**官方模型导出为 INT8 I/O 不代表其输出就是 INT32 原始累加值。** 如果暂时只能观测 requantized 输出，必须按照相同 rounding/saturation 比较，不伪造 raw accumulator 验证。

## 3. bias 是 Qwen2.5 最容易遗漏的一步

Q/K/V 都有 bias，O 和 MLP 的 Linear 没有。[Qwen2 实现][qwen]。第一版保留 FP32 bias，在累加值反量化后加一次。

动态 `sx[m]` 意味着整数 bias 理论上应为 `round(bias[n]/(sx[m]*sw[n]))`，随 token 行变化。不能在导出时固化一份 INT32 bias，然后对所有 token 使用。除非 kernel 显式支持对应的动态缩放/浮点 epilogue，否则不要强行把 bias 塞入静态整数命令。

合并 QKV 时，逻辑列为 `[Q(896), K(128), V(128)]`；weight scale 与 bias 必须按同一顺序合并。拆分各输出时保留正确偏移。Split-K 不得每个 K tile 都加 bias，最终输出才加一次。

## 4. Split-K、padding、零点与累加溢出

若整个 K 上使用同一 `sx[m]` 和 `sw[n]`，可先精确累加各 K tile 的整数 partial，再统一缩放。若不同 K group 使用不同 scale，必须采用：

```text
Y[m,n] = sum_g (sx[m,g] * sw[n,g] * I_g[m,n]) + bias[n]
```

只有 scale 确实与 g 无关时，才能把缩放提到求和外。这个条件直接影响未来 group-wise W4 的归约实现。

在所选 `[-127,127]` 域上，最坏整数幅度上界为 `K*127*127`。还必须检查硬件**内部 partial**是否比最终输出更窄；输出 INT32 不代表内部一定不会饱和。RK W4A4 路径专门约束 INT16 partial，再扩为 INT32；该限制是 RK 的实现证据，不是 MTK 规格。[RK API][rkapi]

如果采用非零零点，数学为 `sum((qa-za)*(qw-zw))`，等价展开含两个行/列和校正与 `K*za*zw` 项。padding 要填“数值零对应的整数码”，即相应零点，不能一律填 byte 0。K 的有效与物理范围必须保持一致。第一版选零点 0，就是为了减少这类混杂变量。

验证至少覆盖：全零、全一、小范围随机、正负交替、单个非零基向量、不同输出通道图样、极值和 K 尾块。不要用所有列相同的权重作为唯一布局测试，因为错误转置也可能得到看似合理的结果。

## 5. Quant 与 pack 是两个独立步骤

量化改变数值表示；packing 改变物理存储顺序和对齐。必须能独立测试：

```text
原始 W[N,K]
  → quantize → Wq[N,K] + sw[N]
  → pack(target layout) → W_native + sw_native + metadata
  → unpack → Wq[N,K]              # codes 必须逐位一致
```

scale 的顺序跟随输出通道，不一定与 packed 字节连续分布。任何通道置换、转置、K 分块都要保存可重建关系。权重只在转换/加载时打包并尽可能常驻；激活每次调用 quantize+pack，可在正确后融合一次遍历。优先参考 [RK CPU kernels][rkpack] 和 [K/N plan][rkplan] 的分层，而不是复制其布局常量。

模型包采用两层表示：可移植的量化 codes/scales 与 target-specific packed cache。用户可以只保留所需 target 包，但 manifest 必须可识别。cache key 至少含模型权重 hash、模型架构、量化版本、packing 版本、SoC/后端能力与 ABI/固件兼容标识。不能把上一进程的 fd、VA、内核对象 ID 写进持久 cache 并直接复用。

下面是拟议元数据片段，不是现成文件格式解析器：

```json
{
  "format": "openmtk.model",
  "version": 1,
  "model_type": "qwen2",
  "source_revision": "TO_FILL",
  "quantization": {
    "weight_bits": 8,
    "activation_bits": 8,
    "weight_axis": "output_channel",
    "activation_axis": "token_row",
    "signed_range": [-127, 127],
    "zero_point": 0,
    "rounding": "nearest_even",
    "scale_dtype": "fp32",
    "accumulator": "int32_contract_to_verify"
  },
  "storage": {
    "logical_weight_layout": "N,K",
    "packing_version": "TO_FILL",
    "target_device": "TO_FILL",
    "content_hashes": {}
  },
  "kv": {"dtype": "fp16", "layout": "backend_defined_v1"},
  "lm_head_mode": "shared_floating_baseline"
}
```

Loader 检查 shape 乘法溢出、offset/length 边界、分片索引、dtype、字节序、文件 hash、scale 个数以及是否有必要 bias；不能依赖文件名猜张量。safetensors 容器与 compressed-tensors 量化元数据是不同层级；不是任意 INT4 safetensors 都能直接被某个内核使用。[safetensors 项目](https://github.com/huggingface/safetensors)

## 6. 哪些 tensor 先保持浮点

| 对象 | 第一版 | 理由/条件 |
|---|---|---|
| 七类 Linear 的权重 | per-output-channel INT8 | 建立统一、易验证的 NPU 数学边界 |
| Linear 输入 | FP32 → 动态 per-token INT8 | 每次基于真实输入重新量化；QKV 或 Gate/Up 可共享同一输入的 quant/pack |
| Linear 输出与 bias | FP32 | 便于 Norm/残差/激活和逐节点对齐；后续再研究 FP16 中间表示 |
| Norm gamma、归约、residual | FP32 基线 | norm epsilon、归约与残差对误差敏感；CPU 实现成本也要测 |
| sin/cos | FP32 表 | 控制长位置精度；不要将常量表看成 KV 或权重量化 |
| Softmax max/exp/sum | 稳定 FP32 | 先正确，后 LUT/近似；要验证负大值、mask、长上下文 |
| KV 存储 | FP16，另有 FP32 对照 | 在运行时产生并追加；与 W8A8 不同，不自动变成 INT8 |
| Embedding / LM head | 先共享原浮点逻辑权重，后设计量化 head | 防止 shared weight 被意外拆开造成语义和内存变化 |

## 7. tied embedding/head 的三种明确策略

[Qwen2.5-0.5B 配置][config]设定共享权重。不能仅因缺少 `lm_head.weight` 就报模型损坏，也不能补一份随机 head。

A. **正确性首版：** 保留同一份 BF16/FP16/FP32 逻辑矩阵，embedding 取行，LM head 做浮点投影。慢但最容易定位问题。

B. **量化派生 head：** embedding 保留原浮点，LM head 由相同原权重生成 INT8 packed copy。需要在 manifest 标记“共享来源、不同数值表示”，算额外内存，并单独评估量化误差；不能声称两条路径仍使用逐值相同的权重。

C. **共同量化表示：** embedding 与 head 共用量化 codes/scales，embedding 每次取行并反量化，head 使用适合 NPU 的布局。可保留逻辑别名，但物理布局可能仍需要第二份 copy；质量必须重新评估。

普通生成的 prefill 只需要最后有效位置的 head 投影；评测 perplexity/teacher-forcing 则需要指定目标位置，不能用省略 logits 的路径偷换计时。

## 8. W4 和 KV INT8 什么时候做

W4 放在 W8 全模型闭环之后。依次验证 nibble 顺序、符号扩展、group size、scale、转置/padding、尾部 K/N、partial 累加、输出误差。RK 的 W4A8 会借助 W4A4 路径对激活分解并重建结果，含校正和安全 tile 边界；**不能把 W4A8 名称当作 MTK 原生单条算子**。[RK API][rkapi]

KV 量化另开任务：确定 K 在 RoPE 前/后量化、K/V 独立 granularity、每 token/head/group 的 scale、残留浮点区、读取路径以及长上下文质量。导出阶段可以确定 Q/DQ 策略和静态校准参数，但真实 cache 内容在推理时才产生。使用量化权重，不自动获得量化 KV。RK 当前参照的 [native KV][rkattn] 为 FP16。

校准数据必须来自代表性文本和真实激活；随机微算子数据只验证算术/布局。先冻结 baseline，再逐项尝试 clipping、混合精度、通道平滑或旋转等方法，确认对应逆变换/权重更新的数学一致性，不把所有技巧同时加入。

[rkapi]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/API.md
[rkpack]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_cpu_kernels.cpp
[rkplan]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_kn_plan.cpp
[rkattn]: https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/ATTENTION.md
[qwen]: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py
[config]: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
