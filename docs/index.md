# OpenMTK 工程设计文档

版本：0.1 · 2026-11-12 · 状态：设计与验收规范，尚不代表独立 MTK NPU 后端已实现。

> **一句话路线：用官方工具链生成可核对的计算参考，通过目标 BSP 源码与运行轨迹重建可重定位、可换权重的 Matmul 后端，再由 OpenMTK 自己完成权重量化/打包、Transformer、KV Cache 和生成调度，逐步实现 safetensors → 自有模型包 → MTK NPU 推理。**

## 阅读入口

| 文档 | 回答的问题 |
|---|---|
| [01 架构与分阶段计划](01-architecture-and-roadmap.md) | 最终要做什么；从最小 Matmul 到完整模型的依赖关系和阶段门槛 |
| [02 参考项目与 RK 加速机制](02-references-and-rk-design.md) | 哪些 GitHub 项目值得看；具体源码入口；哪些优化可借鉴、哪些不能照搬 |
| [03 Qwen2.5 算子与端侧处理对照表](03-qwen25-operator-matrix.md) | 每个算子对应哪段数学、张量形状、精度、CPU/NPU 路径及 RK 实现 |
| [04 量化、反量化与权重打包](04-quantization-and-packing.md) | W8A8、bias、零点、Split-K、W4、KV 量化以及模型包契约 |
| [05 MTK 探测与独立后端](05-mtk-probing-and-native-backend.md) | Recorder 如何补齐、怎样做扰动实验、怎样证明不依赖 libneuron |
| [06 Transformer、KV 与完整生成](06-transformer-kv-and-generation.md) | 原宽单层如何跑通；prefill/decode、GQA、位置、共享权重与内存预算 |
| [07 验证、性能与首批任务](07-validation-and-work-items.md) | 测什么、如何判断通过、第一批工作怎样交给开发者或 Codex |
| [设备与实验记录模板](08-experiment-templates.md) | 设备指纹、数值契约、结果和证据如何记录 |

## 本项目的边界

**目标产品**是架构感知的 LLM 引擎，不是 PyTorch 全算子执行器，也不是给每个模型包一层 Neuron API。第一版支持明确锁定的 Qwen2.5-0.5B；CPU 执行 Norm、RoPE 等辅助运算是允许且需要公开报告的设计。最终 NPU 路径仍依赖目标板内核驱动及固件，但不能要求用户使用厂商 LLM SDK，不能依赖逐模型 `.dla` 或闭源 `libneuron` 来准备/执行任务。

研发阶段允许使用官方编译器与 runtime 作为参考答案生成器；这些依赖与最终推理依赖必须分开记录。由官方编译产物提炼出的固定形状模板只能标记为过渡性实验，不自动等于自主命令生成器。

## 当前证据状态

| 项目 | 当前状态 |
|---|---|
| 官方 safetensors → `.dla` → 设备 NPU 路径 | 项目发起者报告已跑通；具体 checkpoint、SDK/BSP 指纹和测试日志需录入实验记录，不把它当成本仓库复测结果 |
| OEM 工程调试条件 | 作为项目输入；只在授权、可恢复工程机上开展实验 |
| OpenMTK 自主 Matmul、权重重绑定、单层及整模 | **尚待实现和实机验证** |
| RK 开源项目能力 | 有公开源码可参照；不能作为 MTK 已支持同样硬件命令或精度的证据 |
| 文档中的目录/API/测试名称 | 拟议接口和交付物；除明确说明外，不是现有可执行命令 |

首块验证设备应选已经成功运行官方模型的那一块，不同时混用 MT6897、MT8792 或 MT6886 的头文件、固件及运行库。首次实验前填写[设备模板](08-experiment-templates.md)。

## 先统一五个概念

1. `safetensors` 记录张量，不包含完整执行图；模型结构由配置和架构实现确定。[格式说明](https://huggingface.co/docs/safetensors/metadata_parsing)
2. `ioctl` 是提交边界，不是算子或 ISA；一次提交可能包含多个融合任务。
3. 录制/回放、独立固定形状执行、可参数化算子生成，是不同完成度。
4. W8A8 Linear 的 INT8 激活，不意味着整个 Block、Attention 或 KV 都是 INT8。
5. Qwen2.5 的公开小模型是 **0.5B**，Qwen3 有 **0.6B**。本设计以 [Qwen2.5-0.5B 配置](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json) 为数值基线；不据此改写发起者既有实验的型号。

## 当前最重要的方向

先完成 **目标设备契约 → 最小 Matmul 参考 → 可解释提交记录 → 可换 A/B、可重定位的独立 Matmul**。同时在主机完成 Qwen2.5 原宽单层 CPU 基准，以免后续把数学错误和驱动错误混在一起。通过独立 Matmul 门槛后，才接 W8A8 Linear、原宽单层、两层和完整 24 层模型。

所有阶段均以[验收标准](07-validation-and-work-items.md)判断推进，不以“编译成功”“有 ioctl”“能输出中文”单独证明正确或加速。

本次文档不修改或创建根目录 README，不上传厂商 SDK、固件、模型权重或未经审查的原始 trace。来源和代码复用边界见 [02](02-references-and-rk-design.md)。
