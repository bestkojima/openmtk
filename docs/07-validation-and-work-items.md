# 07 · 验证、性能测量与第一批工程任务

[返回目录](index.md) · 下面是待执行的测试与验收规格。本次只建立文档，不代表这些测试已经实现或通过。

## 1. 五层测试体系

| 层次 | 运行环境 | 核心用例 | 证据与通过标准 |
|---|---|---|---|
| T-CPU | 主机 | safetensors 边界、BF16 转换、quant/pack/unpack、bias、RoPE、RMSNorm、mask、GQA、KV | 保存固定 seed/config/hash；pack/unpack 与 codes 逐位一致；浮点使用登记的数值契约 |
| T-ABI | 主机交叉编译 + 工程板查询 | sizeof/alignof/offsetof、ioctl 编码、指针/句柄类型、mock 资源生命周期 | 与选定 BSP 头文件比对；结构断言不替代真实运行语义验证 |
| T-KERNEL | 工程板 | 原生 Matmul、多组 A/B、重定位、尾块、重复/错误恢复 | 实际 NPU 输出与相同契约 CPU golden 对齐；提交/设备状态均成功 |
| T-BLOCK | 主机 + 工程板 | 单投影→QKV/O→FFN→原宽单层→两层；8+1+1 与整段 | 每个中间 tensor 与状态可定位；有效 KV 干预和无效区隔离通过 |
| T-MODEL | 工程板 | 24 层 teacher forcing、logits、greedy、固定语料 NLL、context 增长、取消/重置 | 同精度后端一致性、量化质量和资源/持续运行三项分别达标 |

不能用 CPU-only 单元测试通过代表板端通过；不能用硬件时间非零代表输出正确；不能用生成文本可读代表数学正确。

## 2. 数值验收要比较正确的对象

**整数层：** 在无溢出、原始 INT32 输出且约定完全相同的条件下，所有元素必须逐值精确一致。CPU 用 INT64 累加做范围检查。如果硬件输出已 requantize，则模拟相同 round/clip 后比较；不允许把 INT8 输出当作 INT32 的低精度近似来糊弄通过。

**浮点层：** 同时报告 max-abs、RMSE/相对 L2、非有限值数、数值范围；相似度只作补充。每个路径注明 accumulator、scale dtype、舍入、近似函数和计算顺序。先登记容差再测试。初始可用 FP32 epilogue `atol=1e-5, rtol=1e-4` 作为候选检查，FP16 边界另按舍入误差设计；它不是所有层、所有模型的通用通过值。失败时必须定位，不允许只放宽阈值。

**模型层：** 先比较 CPU quant-sim 与 NPU，即排查实现误差；再比较 quant-sim 与原 FP32，即评估量化代价。用固定输入 token 的 teacher forcing 找第一处分歧，避免自由生成分叉之后还把不同输入的输出作比较。top-1 在近似并列 logits 时可能变化，应同时看 top-k、margin、NLL 和 KL，而不是要求所有量化模型逐 token 完全相同。

初始质量研发目标建议冻结为：在独立于校准集的固定测试集上，平均 NLL 相对 FP32 增量不超过 0.05 nat/token，并报告完整条件；这约等于 perplexity 比例不超过 exp(0.05)。**这是拟议项目目标，不是已通过结果或硬件保证。** 小模型/领域用例可登记其他阈值，但不得看完结果后不留记录地改标准。

## 3. 微算子与状态覆盖矩阵

| 维度 | 初始覆盖建议 | 检查目的 |
|---|---|---|
| M | 1、2、7、8、31、32、127、128 中 capability 支持的子集 | decode、小 batch、tile 前后边界和 prefill |
| K/N | 先 64/128，再 Qwen 的 896、128、4864，以及 LM head N=151936 | 不只测方阵；K/N 尾块和大 N 分片 |
| 不合法 shape | 超范围/负值/溢出、违反必要对齐 | 提交前明确拒绝；不得产生任意设备访问 |
| 输入 | 零、基向量、交替符号、不同列图样、固定随机、边界值 | 识别转置/错位/符号扩展/饱和 |
| 权重 | 固定 shape B1/B2/B1，不重新调用官方编译器 | 证明权重可替换且无陈旧缓存 |
| 资源 | 不同分配顺序、重新打开上下文、新进程 | 重定位和资源依赖不是录下来的旧 ID |
| 生命周期 | 重复 create/run/free，失败路径、cancel/reset | 无 fd/mapping/driver buffer 泄漏；错误后状态可恢复 |
| KV | 0/1/8/9/10、chunk 和容量边界；future slot 与有效 slot 干预 | position、valid length、mask、历史复用 |

这不是全尺寸笛卡尔积；按 capability 和成本选择有意义的覆盖。先串行稳定，再做并发。初始板端稳定性回归建议重复 1000 次微算子和 100 次完整 create/destroy 循环，计数是测试计划，不是现有实测。

## 4. 如何证明原生 NPU，而不是“看起来跑了”

每份 P3/P4 成果必须包含：实际进程和已加载库清单；没有调用 libneuron 的代码路径证据；没有借助 Binder 服务/子进程代执行；自有上下文、资源和命令；device engine trace/执行状态；两组权重/输入的输出；fresh-process 结果。只检查 ELF 直接依赖不足以发现动态加载或远端执行，也不能只凭官方 profiler 的 delegate 名字判断内部不存在 CPU 路径。

对于“完全不依赖厂商编译器”的声明，还要证明目标用户换 checkpoint 权重后仅用 OpenMTK quant/pack，不需要新编译 `.dla`。若内核仍由厂商离线工具按形状生成，报告为模板依赖阶段，不用同一句话宣称自主 codegen。

## 5. 性能协议

将三种运行模式严格分开：`capture-deep` 用来理解 payload；`trace-light` 记录少量时序；`benchmark` 不注入 Frida，不逐 tensor dump，不开高开销 profiler。三者不能混报。

| 指标 | 定义/记录方式 |
|---|---|
| cold load | 读模型、量化/打包、上传、分配与计划构造，单独计时 |
| warm operator | 包含 activation scale、quant/pack、同步、submit/wait、输出反量化/解布局；不含可复用一次性 weight pack |
| device interval | 由驱动/固件提供的执行区间及其定义；不自动等于纯 MAC 时间 |
| 完整调用墙钟 | 单调时钟测量；若 CPU/NPU 异步重叠，各阶段时长不能简单相加重建墙钟 |
| prefill | 实际有效 prompt token 数 / prompt 处理墙钟；披露是否包括最后位置 head、padding、tokenizer |
| decode | 固定历史长度下、相同有效生成步数；披露采样、head、输出文本开销；首 token 与后续分开 |
| transfer | host↔device、layout conversion、cache sync、KV append 的字节数与耗时分别报告 |
| memory | RSS/PSS、device/dma-buf、workspace、KV、packed 权重；说明共享映射，避免重复统计 |
| sustained | 冷热态、温度、CPU/NPU/DDR 策略、频率、线程 affinity、持续时长与降频情况 |

初始热跑协议：至少 10 次预热、50 次记录，报告 median/p95、样本数、最小/最大及原始序列；慢模型可调整但必须记录。比较采用交替 A/B 或 ABBA，排除热漂移。记录所有可观察的时钟、QoS 与线程策略，不对 OEM 散热保护进行关闭或规避。

先比较完全相同的计算量、精度、cache 状态与时钟，再讨论加速。RK 作者的总体性能不能作为 MTK 的目标保证；机制参见 [02](02-references-and-rk-design.md)。

优化采用以下独立实验顺序：权重/plan 常驻 → quant+pack 融合 → 减少重复 layout/sync → QKV/Gate-Up 合并 → Attention/native KV → 双缓冲 → autotune → W4。每次只改变一项，保存数值验证结果；组合优化另开一行，不冒充单项收益。

## 6. 首批任务清单，可直接转成 Issues

下列均为待办，不自动创建 GitHub Issues。前四项先做；不等底层指令全部理解才写 CPU 对照。

| ID | 任务 / 拟议产物 | 依赖 | 完成证据与边界 |
|---|---|---|---|
| OM-001 | 设备与工具链 manifest，`specs/device-contract.json` | 无 | OEM BSP 路径/版本与运行板对应；输入输出 hash；不公开私有二进制 |
| OM-002 | 微 Matmul CPU golden 与数值契约，`reference/matmul` | 无 | INT64 golden、round/clip、输出 dtype、边界用例；CPU-only 可完成 |
| OM-003 | Qwen2.5 config/权重检查及原宽单层 CPU reference | 无 | bias、tied head、RoPE/GQA、8+1+1 对照；冻结软件/权重 revision |
| OM-004 | 目标 BSP ABI schema 与静态断言 | OM-001 | 字段来源与 pointer/handle 消费者；不硬编码 Phone2a 常量 |
| OM-005 | 官方微算子 runner 与输出/profile 记录 | OM-001/002 | 实际 NPU 执行和算术/量化输出对齐；只作 oracle |
| OM-006 | Recorder 前后快照、资源追踪、完成关联 | OM-004/005 | 一次调用完整可解释；正确 errno；有丢事件/不可读取标记 |
| OM-007 | E0–E6 规范化 diff 报告 | OM-006 | 排除地址/时序噪声；权重、shape、重定位假设分别有证据 |
| OM-008 | 独立固定形状 Matmul | OM-007 | 不调用 libneuron/HAL；A/B/A、换 B、新进程与资源生命周期通过 |
| OM-009 | 有边界参数化 plan + weight pack/unpack | OM-008 | 未捕获合法尺寸由规则/tiling生成；未知形状拒绝；尾块测试 |
| OM-010 | W8A8 Linear wrapper，FP32 bias/epilogue | OM-002/009 | 与 CPU quant-sim 对齐；Qwen 四种基本 K/N 覆盖 |
| OM-011 | 单层七投影替换与 FP16 KV | OM-003/010 | 中间张量与 8+1+1、有效/无效 KV 干预通过 |
| OM-012 | 两层→24 层→head/tokenizer/greedy | OM-011 | teacher forcing、质量、memory、取消/重置与独立性证据 |
| OM-013 | profile 驱动的 P7 优化 | OM-012 | 逐项 ablation，质量守门，不把 Hook 开销算优化收益 |

每个任务提交：变更说明、测试方法、通过/失败/未执行三态、日志摘要、当前限制。硬件任务只能由能访问目标板的执行者验收；Codex 可以写 CPU 测试、schema、分析器和 mock，不能凭生成代码宣称 NPU 实测通过。

## 7. 首次开发的执行提示

```text
先阅读 docs/index.md、03、04、05、07。保持 README 不变。
本轮只实施 OM-001 至 OM-004 中能够验证的内容，不预写未知 NPU 指令。
所有目标设备信息缺失处标注 UNKNOWN，不能从 MT6886 或 RK 示例补猜。
CPU reference 与 quant-sim 独立；补齐 Qwen2.5 的 QKV bias 和 tied head。
输出实际测试命令、结果与未执行项。未经目标 BSP 和实机证据，不提交未知
ioctl/payload，不把空壳 backend 标为支持 NPU，不用 CPU 回退伪造成功。
```

## 8. 风险优先级

最高风险是命令/资源语义无法独立重建；其次是权重无法参数化、quant/packing 合同错误、KV 状态错误。性能问题应在这些正确性门槛之后处理。任何 native 路径卡住都要记录可复现的最小失败案例，同时保留 CPU 和官方参考，不能通过更换项目目标掩盖阻塞。

使用[实验模板](08-experiment-templates.md)登记每次结果；本文件中的阈值和次数若调整，必须在开始新一轮实验前提交原因与新版本。
