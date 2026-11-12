# 01 · 目标架构与分阶段计划

[返回目录](index.md) · 设计版本 0.1，2026-11-12。本文中的 API、目录和命令是待实现规格，不表示代码已经存在。

## 1. 最终产品，而不是临时验证方式

用户最终应能把受支持的 Hugging Face 权重转换成 OpenMTK 自有量化/布局包，再由引擎生成文本。转换可以离线完成，也可以在首次加载时完成并缓存；**不要求每个模型再经过厂商 LLM SDK、ncc 或 `.dla` 编译**。模型仍需匹配已实现的架构、精度、形状和算子能力，不能仅因文件扩展名是 safetensors 就承诺兼容。

```text
config.json + safetensors + tokenizer
                |
OpenMTK 权重转换：名称/形状验证、量化、权重打包、版本化 manifest
                |
OpenMTK Engine：Qwen2.5 Block、KV Cache、prefill/decode、采样
          /                           \
CPU/NEON 后端                   MTK 原生算子后端
                              布局/分块/任务生成/同步
                                        |
                             目标 BSP 驱动 + 固件 + NPU
```

目标用法示意，当前不可直接执行：

```sh
openmtk-convert --input ./Qwen2.5-0.5B --output ./qwen25-openmtk --quant w8a8
openmtk-run --model ./qwen25-openmtk --backend mtk-native --prompt "你好"
```

这里“原生”指推理进程自行准备和提交任务，不调用 `libneuron`；不表示替换内核驱动、固件或硬件安全机制。允许 CPU 执行 Norm、RoPE、Softmax、采样；设备分工必须明确报告，不把混合执行写成全 NPU。

## 2. 三条路径分开命名

| 路径 | 用途 | 可以证明什么 | 不能证明什么 |
|---|---|---|---|
| `cpu-reference` | FP32 与量化模拟、单层/整模正确性 | 数学、KV、量化契约正确 | NPU 能运行 |
| `neuron-oracle` | 官方编译器与运行时生成对照结果/执行轨迹 | 已知参考任务可以在目标设备运行 | OpenMTK 独立执行 |
| `mtk-native` | 自有内存、数据布局、命令生成与提交 | 在声明能力范围内的独立后端 | 未测试的 dtype/形状/设备也支持 |

运行日志必须记录实际后端，不允许原生后端失败后静默调用 Neuron 或 CPU，再打印“原生 NPU 成功”。CPU 回退只能由显式策略启用，并计入结果报告。

[RK 算子库的 API 说明](https://github.com/happyme531/rknpu2_matmul_open/blob/3be24de319b140f95d94634f00c0d6f8df5e96b3/docs/API.md)描述了独立寄存器编程及 CPU/NPU 分工；借鉴的是后端边界，不是把 RK 命令交给 MTK 执行。

## 3. 两条可并行主线与合流点

**设备主线：** 设备契约 → 官方微算子 → 结构化 Recorder → 资源重建 → 独立固定形状 Matmul → 有边界的参数化 Matmul。

**模型主线：** 权重/配置检查 → CPU FP32 原宽单层 → W8A8 CPU 模拟 → 逐层测试及 KV 状态机。

两条主线在“用独立 NPU Matmul 替换单层的 Linear”合流。设备后端尚未成功时，CPU 工作仍然有效，但不能提前宣布完成原生引擎。

| 阶段 | 输入与工作 | 交付物 | 进入下一阶段的门槛 |
|---|---|---|---|
| P0 环境冻结 | 选择已经跑通官方模型的唯一工程板；记录芯片、BSP、固件、库版本与 hash；固定官方基线 | 设备 manifest、参考命令、输入/输出 hash、源码对应关系 | 相同输入官方路径稳定复现；UAPI 来源与目标板相匹配或差异已标记 |
| P1 微算子基线 | 最小合法 INT8 Matmul；CPU INT64 累加检查；识别官方输出是原始 INT32、浮点还是重定量化整数 | 微算子生成器规格、数值契约、参考结果 | 确认真实执行引擎和可观测边界；不把 requantized 输出当 raw accumulator |
| P2 Recorder 与解码 | 关联加载/准备/执行、ioctl 前后参数、buffer 生命周期、完成事件；同形状 A/B/A 测试 | ABI 描述、规范化 trace、资源关系图、扰动报告 | 能解释已观察任务的输入/输出来源、同步、错误；未知 payload 继续标记未知 |
| P3 独立固定形状 Matmul | 重新创建资源和合法任务，权重可替换；新进程/不同分配地址执行 | `mtk-native` 微算子原型与失败记录 | 无 libneuron/HAL 代执行；多组 A/B 正确；fresh process 通过；无未知重定位 |
| P4 参数化算子与 W8A8 Linear | 明确 M/K/N、对齐、尾块、累加范围；权重常驻；动态行 scale、反量化、bias | 算子 capability、packer、workspace、CPU 位精确模拟 | 至少一种未捕获过的合法尺寸由规则/tiling 生成并正确；生产 Qwen 形状族通过 |
| P5 原宽单层 | 真实第 0 层；七类投影 NPU，Norm/RoPE/Attention/激活先 CPU | 中间张量对照、8+1+1 与整段计算对照 | 同精度 CPU/NPU 对齐；历史 KV 干预有效；有效长度/位置正确 |
| P6 两层及完整模型 | 两层验证逐层状态，再 24 层；embedding、final norm、tied head、tokenizer、greedy | 原生引擎测试报告和模型包 | 固定 token teacher-forcing 对齐、冻结语料质量门槛通过、状态清理通过 |
| P7 优化 | 动态 B Attention、设备原生 KV、融合、双缓冲、autotune、W4 候选 | A/B ablation、数值和性能报告 | 完整调用/端到端有收益，质量不回退，持续运行和热稳定通过 |

以上是依赖顺序，不承诺日历工期。P3/P4 未通过时，记录卡点并继续受控实验；不能用逐模型 `.dla` 包装器代替本项目的最终验收。

## 4. 第一版必须具备的接口契约

推荐核心使用 C++17 + CMake，测试/权重转换/分析使用 Python；第一版不同时引入第二套 Rust/C++ Runtime。RK 的 Rust 上层可以作为结构参考，而不是强制迁移语言。以下均为拟议职责：

| 接口 | 必须描述的内容 |
|---|---|
| `Capabilities` | 设备/BSP 标识、支持 dtype、M/K/N 范围、对齐、accumulator、bias、同步、动态 B 能力；未知能力返回不支持 |
| `PackedWeight` | 原权重 hash、逻辑 `[N,K]`、量化 scale、物理布局、padding、packing 版本、设备 buffer、驻留状态 |
| `TensorView` | shape、stride、dtype、layout、有效范围、量化元数据、host/device 可见性；不把所有 uint64 都解释成地址 |
| `MatmulPlan` | M/K/N、tile/split-K、输入输出布局、scratch 大小、重定位规则、数值语义；不得含过期 fd/VA |
| `ExecutionWorkspace` | 临时激活/partial/命令 buffer，固定容量优先；不得与持久权重或 KV 混用生命周期 |
| `Completion` | 提交状态、完成/错误、允许 CPU 读取的时点；提交成功不等于运算完成 |
| `KVCache` | layer、capacity、valid length、位置语义、K/V layout、dtype、append/reset；不负责重新计算模型权重 |
| `Profiler` | 预处理、转换/复制字节数、提交/等待、设备区间和墙钟；时间重叠不能直接相加 |

模块草案：

```text
tools/convert/        自有权重转换与 manifest
reference/            CPU 数学与量化对照，不含厂商二进制
src/model/qwen2/      配置、权重名称、Block、LM head
src/ops/              数学接口与 CPU 实现
src/backend/mtk/      BSP 适配、buffer、plan、submit、Matmul
src/runtime/          tensor arena、KV、prefill/decode、sampler
research/recorder/    独立研究工具；不得成为生产推理依赖
specs/                数值与 buffer 契约
 tests/               CPU、mock-driver、板端、模型集成测试
```

## 5. 第一版明确不做

不做任意 PyTorch 动态图编译；不支持所有 MTK 芯片；不做 MoE、投机解码、KV INT4、任意 RoPE scaling 或 Android App UI。W4、Attention NPU、异步并发放到正确性基线之后。不能为了“所有算子上 NPU”而在每个小算子间增加往返复制。

最终独立性证明至少包括：运行进程的依赖与已加载库、实际发起驱动提交的进程、没有通过 Binder/子进程借用厂商执行器、重新创建资源后仍运行、改变权重/输入后输出相应改变。只看 `ldd` 或一次 `CMD ret=0` 不足够。

下一步阅读：[RK 参照](02-references-and-rk-design.md)、[具体探测](05-mtk-probing-and-native-backend.md)、[验收](07-validation-and-work-items.md)。
