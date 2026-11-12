# 02 · 参考项目、源码入口与 RK 提速机制

[返回目录](index.md) · 核对日期：2026-11-12。引用源码证明其实现，不代表 OpenMTK 已移植、运行或复现性能。

## 1. 参考仓库及应当借鉴的边界

| 优先级 | 项目 | 已有内容与阅读入口 | OpenMTK 借鉴什么 | 不能直接照搬什么 |
|---|---|---|---|---|
| P0 | [happyme531/rknpu2_matmul_open](https://github.com/happyme531/rknpu2_matmul_open) | 独立 RKNPU2 算子库；`src/rk_npu_core.cpp`、`src/rk_npu_internal.h`、`src/rk_npu_matmul_i8.cpp`、`docs/API.md` | device/buffer/plan/submit 分层；整数乘加、packing、split-K、复用与完整调用计时 | RK 寄存器、GEM、IOMMU domain、CBUF 容量和 dtype 限制 |
| P0 | [allbilly/rk3588](https://github.com/allbilly/rk3588) | Python 寄存器编程；`examples/gemm.py`、`experimental/gemm_int8.py`、README 的 task/PC-chain 解释 | 从最小算子理解数据布局、任务链和受控实验；用简单输入识别错误 | RK 的 1×1 Conv 映射细节不等于 MTK 的映射；多核/驱动注意事项不能直接套用 |
| P0 | [Peterc3-dev/phone2a-apu-unlock](https://github.com/Peterc3-dev/phone2a-apu-unlock) | `frida/apu_ioctl_tap.js`、`frida/decode.py`、`docs/ABI.md`、`docs/HANDSHAKE.md` | Hook 入口、ioctl 分类和 ABI 记录方式 | 不是完整 replay/指令解码器；MT6886 字段、操作码、内存语义不一定适用于目标 BSP |
| P0 | [MT6897 公开 OEM 内核](https://github.com/oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897) | `kernel/kernel_device_modules-6.1/drivers/misc/mediatek/apusys/midware/2.0/` 中的 `mdw_ioctl.h`、`mdw_cmd.c`、`mdw_mem.c` | 从真正消费参数的代码判定 pointer/handle、命令 ID、同步和 copy-in/out | 该分支不是发起者目标板的已验证 BSP，只能比对，不能据此写死 ioctl |
| P1 | [happyme531/open_rknpu2_lm_infer](https://github.com/happyme531/open_rknpu2_lm_infer) | `src/model/dense/batch.rs`、`src/model/output_head.rs`、`src/model/cpu.rs`、`src/npu.rs` | 如何组合 Linear、RoPE、GQA、KV、FFN、输出头，以及 CPU/NPU 协作 | 该快照 dense loader 限定 Qwen3/Llama/Granite 且无 projection bias；不是现成 Qwen2.5 loader |
| P1 | [huggingface/transformers](https://github.com/huggingface/transformers) | `src/transformers/models/qwen2/modeling_qwen2.py` | Qwen2.5 的数学和状态对照，特别是 QKV bias、split-half RoPE、GQA、mask | 不把框架中的每一个 view/reshape 都变成硬件 kernel |
| P1 | [huggingface/safetensors](https://github.com/huggingface/safetensors) | 安全张量容器、名称/shape/dtype/offset 解析 | 有界加载、分片、元数据校验 | safetensors 本身不是模型执行图，也不保证特定量化格式能直接执行 |

本轮不把未经核对的“RKLLM 必然基于某个上层项目”“整个 MTK 社区无人做过某事”作为设计前提。我们只需要知道参考库公开实现了什么，以及目标板上自己验证了什么。

## 2. 主要引用版本

| 标识 | 固定快照或读取基线 |
|---|---|
| RK-OPS | `happyme531/rknpu2_matmul_open@3be24de319b140f95d94634f00c0d6f8df5e96b3` |
| RK-ENGINE | `happyme531/open_rknpu2_lm_infer@e1331947c00d438470e48b585f7b34d0f783f40b` |
| MTK-REF | `oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897@e92259a0119b33a6d7b1676614b6e812ad605989` |
| HF-MATH | Transformers `v4.45.2` 的 `modeling_qwen2.py`，作为明确的历史数学参考；实际测试要固定安装版本和 attention 实现 |
| QWEN-CONFIG | `Qwen/Qwen2.5-0.5B/config.json`，2026-11-12 读取；实验开始时另记录 Hugging Face revision 和文件 hash |
| Phone2a / allbilly | 本轮读取公开 main 文档/源码；实验引用时再冻结 commit，不能用漂移的 main 判定 ABI |

### 源码入口索引

| 标识 | 文件与用途 |
|---|---|
| RK-QUANT | [docs/API.md](https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/API.md)：typed W8A8、整数输出、CPU 量化/归约、W4A8 分解 |
| RK-CORE | [src/rk_npu_core.cpp](https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_core.cpp)：内存、同步、提交和任务链 |
| RK-PACK | [src/rk_npu_cpu_kernels.cpp](https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_cpu_kernels.cpp)：CPU packing、quantize、reduction、dequantize |
| RK-PLAN | [src/rk_npu_kn_plan.cpp](https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/src/rk_npu_kn_plan.cpp)：K 分块和 N 方向分片 |
| RK-ATTN | [docs/ATTENTION.md](https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/ATTENTION.md)：FP16 GQA、稳定 Softmax、native KV、CPU/NPU pipeline |
| RK-BLOCK | [src/model/dense/batch.rs](https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/dense/batch.rs)：一个 Block 的完整执行顺序 |
| RK-CPU | [src/model/cpu.rs](https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/cpu.rs)：Norm、RoPE、NEON、激活 |
| RK-HEAD | [src/model/output_head.rs](https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/output_head.rs)：量化 Linear 与浮点 head 的分流 |
| RK-LOAD | [src/model/dense.rs](https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/src/model/dense.rs)：模型支持条件、cos/sin 预计算、head/embedding |
| QWEN-MATH | [modeling_qwen2.py](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py)：QKV bias、RoPE、GQA、FFN 与残差 |

## 3. 为什么 RK 引擎更快：机制、收益和代价一起看

| 优化 | RK 可见实现/文档 | 有效原因 | OpenMTK 的采用时机与检查 |
|---|---|---|---|
| 常驻打包权重 | RK-QUANT、RK-PLAN | 避免每 token 重量化/重排权重；decode 减少准备成本 | P4；分别记录冷启动、热执行，权重打包不能偷偷移出一个对照组的计时 |
| 按形状 autotune | RK-QUANT 与 `rk_npu_w4a8_tune*` | 不同 M/K/N 需要不同 tile、线程数、core mask、pipeline | P7；以完整算子时延和正确性筛选，不只看 MAC 时间 |
| Split-K 与布局优化 | RK-PLAN、RK-PACK | 满足片上存储/累加限制，减少不必要转置 | P4；先保证整数 partial 无溢出；scale 按块变化时不能直接合并整数 partial |
| QKV/Gate-Up 合并与 SwiGLU epilogue | RK-BLOCK、`rk_npu_w4a8_swiglu.cpp` | 减少调度及完整中间张量落内存 | P7；Qwen2.5 必须保留每个投影的 bias 和独立权重 scale |
| 原生 FP16 KV | RK-ATTN | 追加新 token，历史 KV 以 Attention 所需布局常驻；不反复重排完整前缀 | P7；这是 DDR buffer，不是把 KV 全塞进 NPU SRAM；必须实测复制字节数 |
| CPU/NPU pipeline | RK-ATTN、RK-QUANT | CPU 准备/量化/归约与 NPU 计算重叠 | P7；两份 scratch、明确完成事件；同一 buffer 未完成不可覆盖 |
| 双 ubatch prefill | RK 引擎 README | 层级流水提高硬件利用率 | 最后；增加 activation/workspace 内存且并非所有模型支持 |
| W4/W8 混合 | RK-QUANT | 减少权重流量和容量，decode 可能受益 | W8 正确后；W4A8 不是保证原生单条指令，RK 有 W4A4 分解/校正路径 |
| mmap 按需加载 | RK 引擎 README | 改善加载和主机 RSS 行为 | 主机权重管理优化；不等于 NPU 可直接访问文件映射，仍可能需要 packed device copy |
| W8 无损硬件压缩 | RK README / DCOMP | 进一步减少权重存储/流量 | MTK 是否支持完全未知；无损压缩与低位量化不是同一件事，也可能变慢 |
| 投机解码 | RK 引擎 README | 用一次验证处理多个候选 token | 不列入第一版；需 draft、acceptance 与回滚，不能算作单 token kernel 的纯加速 |

来源：[RK 引擎 README 快照](https://github.com/happyme531/open_rknpu2_lm_infer/blob/e1331947c00d438470e48b585f7b34d0f783f40b/README.md)、RK-QUANT、RK-ATTN。这里列的是代码中的机制，不声称每项对最终加速比的贡献已经独立量化。

作者公布的 Qwen3-4B 特定测试中，4096-token prefill 从约 50.133 tok/s 到 163.069 tok/s（双 ubatch，约 3.25×）；带 3800-token history 的 decode128 从约 2.310 到 8.544 tok/s（约 3.70×）。这是作者的模型、量化、SDK、硬件与计时配置下的端到端结果，不是本项目实测；两边量化和执行策略也并不完全相同。不能用它承诺 MTK 的加速比，不能据此把收益全归于“绕过厂商库”。

尤其注意 RK-ATTN 对某次 KV append 优化单独报告的完整模型增益很小且存在漂移，decode 没有显著收益。这正说明必须做 ablation，不能看到零复制/布局优化名称就认定必然提速。

## 4. 会直接影响 OpenMTK 设计的差异

**RoPE 查表与 NPU LUT 是两件事。** RK dense loader 预计算 sin/cos，CPU/NEON 做旋转；某些其他模型路径会直接算三角函数。Attention 的 exp 近似则可用 RK DPU LUT。LM head 是矩阵投影，不是 token 查表；embedding 才是按 token 取行。详见 RK-CPU、RK-HEAD、RK-ATTN。

**Qwen2.5 不能照抄 RK dense loader。** 该快照接受 qwen3/llama/granite，要求无投影 bias；Qwen2.5 的 Q/K/V 各有 bias，没有 Qwen3 式 Q/K RMSNorm。Qwen2.5-0.5B 还共享 embedding/head 权重，需要明确量化后共享还是保留两种布局。[QWEN-MATH](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py)、[官方配置](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)。

**Phone2a 的状态必须按证据描述。** `docs/HANDSHAKE.md` 明确写 MEM/CMD 在该记录里尚未实际验证；不能把“头文件有四类 ioctl”写成“四类都已经实测跑通”。现有 Hook 只取顶层入口快照，并不完整记录嵌套 buffer，也不能证明命令可回放。

**MTK 枚举存在不等于实现可用。** MTK-REF 的 `mdw_mem_ioctl()` 对 FREE/FLUSH/INVALIDATE 分支返回错误；目标设备必须查实际代码和运行调用，不硬编码通用流程。[源码](https://github.com/oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897/blob/e92259a0119b33a6d7b1676614b6e812ad605989/kernel/kernel_device_modules-6.1/drivers/misc/mediatek/apusys/midware/2.0/mdw_mem.c#L842-L870)。

## 5. 代码与资料使用边界

本轮只上传原创设计说明与公开来源链接。RK-OPS 的原始贡献标为 AGPL-3.0-only，其 THIRD_PARTY_NOTICES 还记录了部分 allbilly 来源的许可未澄清；不能仅因 GitHub 可见就直接复制并给 OpenMTK 套另一个许可证。[许可与第三方声明](https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/THIRD_PARTY_NOTICES.md)。

实现前记录每段复用代码来源和许可。厂商 SDK、固件、库、真实权重、带内存内容的 trace 默认保留本地；公开库只提交自造输入、经审查元数据、结果摘要及重现步骤。暂不替项目选根 LICENSE，不因文档引用而复制依赖代码。
