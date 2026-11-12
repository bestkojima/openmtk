# 05 · MTK 探测、Recorder 与独立 Matmul 后端

[返回目录](index.md) · 目标：把“官方能执行”变成“知道怎样准备任务，并由自己的进程独立执行”。本文是实验规程，不是已经验证的 MTK 指令编码手册。

## 1. 第一个板端实验怎样选

先选**已在目标工具链确认支持的最小矩阵乘法**。候选逻辑形状为 `(M,K,N)=(1,64,64)`；若编译器/硬件有其他合法粒度，调整到满足约束的最小形状并记录原因，不能把 64 当作 MTK 固有对齐。

第一轮无 bias、无激活、无残差、单次提交、一个计算上下文，先用小整数图样，再测试完整整数范围。数学定义始终为 `C=A[M,K] @ B[K,N]`。CPU 用 INT64 求和并验证输出范围。

优先研究 INT8×INT8→INT32 路径，但官方通用图可能只暴露经过 scale/round/clip 的 INT8 输出或浮点输出。必须通过模型 I/O 元数据、编译计划和数值实验确认实际语义；无法看到原始累加时，先按实际输出契约校验，不把量化差异当驱动错误。不为了等待理想接口而跳过已有的可观察成功路径。

如果 Matmul frontend 不可用，可把同一数学运算表达为 1×1 Conv，先明确 N/H/W/C 映射、权重转置、量化参数和 padding，再检查实际落点。这是数学替代路线，不是宣称 MTK 硬件和 RK 的卷积映射相同。RK 的简单示例见 [allbilly/rk3588](https://github.com/allbilly/rk3588)。

## 2. 先锁定目标 BSP，不能照抄 Phone2a ABI

已有公开参考能够说明问题，但不能替代目标板源码：

| 观察证据 | 设计后果 |
|---|---|
| [Phone2a Handshake 文档](https://github.com/Peterc3-dev/phone2a-apu-unlock/blob/main/docs/HANDSHAKE.md)记录握手，MEM/CMD 在该记录里未验证 | 不写“四类 ioctl 都跑通”；只能当探测起点 |
| [MT6897 UAPI](https://github.com/oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897/blob/e92259a0119b33a6d7b1676614b6e812ad605989/kernel/kernel_device_modules-6.1/drivers/misc/mediatek/apusys/midware/2.0/mdw_ioctl.h)有额外字段和 ENQ | 同名结构不保证 sizeof/offset 相同；操作枚举也可能扩展 |
| [mdw_cmd_create](https://github.com/oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897/blob/e92259a0119b33a6d7b1676614b6e812ad605989/kernel/kernel_device_modules-6.1/drivers/misc/mediatek/apusys/midware/2.0/mdw_cmd.c#L1460-L1567)对 subcmd_infos 用 copy_from_user，对 exec_infos 用 mdw_mem_get | 一个 uint64 字段可能是用户指针，也可能是该版本内存接口接受的 handle；必须追到消费者，不凭名字递归读 |
| [mdw_mem_ioctl](https://github.com/oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897/blob/e92259a0119b33a6d7b1676614b6e812ad605989/kernel/kernel_device_modules-6.1/drivers/misc/mediatek/apusys/midware/2.0/mdw_mem.c#L842-L870)不支持部分枚举中的 MEM 操作 | release/cache sync 可能走其他途径，按目标实现和 trace 建模；不要猜 FREE/FLUSH 一定可用 |
| [ENQ 分支](https://github.com/oppo-source/android_kernel_modules_and_devicetree_oppo_mt6897/blob/e92259a0119b33a6d7b1676614b6e812ad605989/kernel/kernel_device_modules-6.1/drivers/misc/mediatek/apusys/midware/2.0/mdw_cmd.c#L1695-L1714)创建命令后可直接返回 | 成功返回 ID 不证明已执行，必须等实际完成并验证输出 |

在 OEM 源码中定位 ioctl dispatch，再逐项记录 `sizeof`、`alignof`、`offsetof`、union/op、指针/handle/设备 VA 类型及 copy-in/out 方向。交叉编译结构断言和板端能力查询。Handshake 的 meta_size 相同只是一项一致性线索，不能证明所有命令结构和语义正确。

## 3. Recorder 必须分三层采集

**A：应用/Neuron 边界。** 给加载、prepare、set input、execute、等待完成、读取结果、释放资源加同一 run ID，标记哪个实验输入正在执行。把 runner 在首次设备初始化前设置可控等待点，Hook 完毕后才继续；不要等初始化结束才 attach。

**B：实际系统调用边界。** 从目标进程和驱动 trace 找真正提交者。直接 Neuron 调用可能在本进程，Binder 路线可能在服务进程；必须按实测确认。只 Hook `libc.ioctl` 可能漏掉直接 syscall 或其他入口，可用 OEM 内核 tracepoints/受控驱动插桩交叉验证。不要因为没有抓到日志就断言没有用 NPU。

**C：内核消费与完成边界。** 既然有 OEM 源码，优先利用已有 trace，必要时只给工程内核加带开关的元数据采集：copy-in 后的结构、buffer 大小/角色、任务 ID、后端执行器、排队/开始/结束、错误。采用有界缓冲和丢事件计数，初版不在每个算子上 printk 巨量数据。涉及 payload 的采集只使用自造测试数据并留在本地。

| 对象 | 最小记录内容 | 常见盲区 |
|---|---|---|
| ioctl | pid/tid、序号、monotonic 时间、fd/path、request、op、入口与出口快照、ret、真实 errno | `_IOC_SIZE` 只是线索，不描述嵌套内存；调用返回不一定是设备完成 |
| 资源 | 分配/导入/映射/关闭；fd/handle/host VA/device VA 的不同命名空间；大小、offset、别名与生命周期 | 只过滤 `/dev/apusys` 会漏掉 dma_heap、DMA-BUF sync、mmap、dup/close 或 Binder 传递 |
| 命令结构 | subcmd 数组、依赖、cmdbuf 描述及方向、链接、exec info、frame/run 对应 | 一个 subcmd 不一定等于一个 PyTorch op；融合和 firmware 协议可能隐藏更细任务 |
| payload | 根据目标 ABI 获取的合法、有界 snapshot；checksum、capture phase、可读性 | 未映射/设备专属 buffer 不能盲读；同步前读到旧数据不等于设备错误 |
| 完成事件 | fence/等待、任务完成状态、实际 device engine、错误码 | submit 通过、enqueue 返回或有 fence fd 都不等于输出已完成 |

[现有 Frida 脚本](https://github.com/Peterc3-dev/phone2a-apu-unlock/blob/main/frida/apu_ioctl_tap.js)需要补齐：出口快照、正确 errno、分层资源追踪、ABI-aware 深度采集、单调高精度时间、主机接收及批量写日志。原脚本强制至少读 32 字节并不能保证安全；新实现对 size=0/未知 opcode 只记录元数据，已知结构才按边界读取。读取嵌套结构前先检查数量、乘法溢出、最大长度及内存可读范围。

libc ioctl 错误通常是 ret=-1 加线程 errno，不能将 `-ret` 当 errno。Frida 在回调里暴露的 errno 要在做其他可能修改它的操作前保存。脚本 API 和 CLI 参数按安装版本验证；官方入口见 [Frida JavaScript API](https://frida.re/docs/javascript-api/)。

Recorder 的深度 dump 模式只用于理解行为；性能模式只记录必要事件；正式 benchmark 另跑无 Hook 基线，不能从逐次 hexdump/flush 的测量推断真实速度。

## 4. 扰动实验矩阵：一次只改一个变量

| 顺序 | 改什么 | 固定什么 | 预期回答 |
|---|---|---|---|
| E0 | 什么也不改，重复三次并跨进程重启 | 模型、输入、权重、工具版本 | 哪些差异只是地址、ID、时间戳、冷启动与缓存 |
| E1 | A1→A2→A1 | 同一个参考产物、B、shape、资源计划 | 输入绑定是否正确；第三次能复现第一次；不是读到旧输出 |
| E2 | B1→B2→B1 | 同 shape、A、量化规则 | 权重是否是独立可更换的数据；是否存在内部缓存或重新 packing |
| E3 | 分配顺序/地址变化 | A/B/shape | 哪些字段是真正重定位；任务能否在新资源上执行 |
| E4 | M=1→8→32→128 | K/N/B 与编译选项 | batch、工作区、分块和调度如何变化 |
| E5 | 只改 K，或只改 N | 其他逻辑因素 | packed tensor 长度、tile/stride/输出布局如何变化 |
| E6 | 加 bias，之后才加其他融合 | 同 shape/输入/权重 | epilogue 与其他任务边界，避免把全部变化混成一次 diff |

B 为运行时输入时，先验证编译器真正在 NPU 执行动态 B，而不是 CPU fallback。如果工具链只接受静态权重，可以分别编译 B1/B2 作为分析样本，但**每换 B 就重新编译官方模型不能算可复用算子**；后续必须识别权重编码和重绑定规则。

首次对比先把已确认的地址与 ID 归一化为资源 ID+offset，再分析剩余字段。不要根据一个二进制数刚好等于已知地址，就全局搜索替换所有相同数值。重定位项必须有字段宽度、字节序、目标 buffer、offset、alignment、range 和证据来源。

## 5. 回放/模板/原生生成的不同等级

| 等级 | 能力与试验 | 可对外宣称 |
|---|---|---|
| L0 | 读到 ioctl 与资源信息 | 观察工具 |
| L1 | 在同一 live context 复用原任务；仅在 BSP 确认语义后试 RUN_STALE 等路径 | 会话内重复执行，不是脱离 Neuron |
| L2 | 自己创建上下文、重新分配/绑定资源，重新提交捕获的合法固定任务 | 独立固定任务执行，仍有模板来源依赖 |
| L3 | 同 shape 下可任意更换合法 A/B，正确做 packing 与重定位；fresh process 通过 | 独立固定形状算子 |
| L4 | 根据声明的 shape/layout/dtype 规则或 tiling 生成未捕获过的合法任务 | 可参数化后端，是第一版正式目标 |
| L5 | 从自己的模型包组装已支持架构，prefill/decode/质量/资源验收通过 | 原生 LLM 引擎，仍需披露 CPU/NPU 分工 |

RUN_STALE 是否重新读取输入、是否复用旧 cmdbuf 内容、绑定何种上下文、何时能释放资源，都必须查看目标实现并做 A/B/A 实验。不能把记录的旧 cmd_id 放进新进程就调用，也不能把会话内成功当成 L3/L4。

`.dla` 是厂商编译产物，实际驱动 payload 经过 runtime 解析、资源准备与绑定；**dump 的 CMD 不等于原始 `.dla` 文件**。不需要为了 L3 理解文件每个字节，但必须理解所有涉及当前任务执行的外部状态与重定位。固定模板只覆盖有限算子/形状，不能许诺能处理任意新架构。

## 6. 最小独立后端执行流程

```text
打开/验证目标设备及 ABI
  → 创建自有上下文
  → 分配/导入 A、packed B、C、scratch、command 和必要状态
  → 建立真实设备可见地址/句柄
  → CPU 写数据，按该 BSP 的机制完成可见性处理
  → 根据 MatmulPlan 构造描述符、payload、依赖和重定位
  → 提交并等待真实完成，检查设备执行状态
  → 按实际同步契约让 CPU 读取 C
  → 解布局，与整数 golden / quant-sim 对齐
  → 释放/复用所有资源；错误路径也完整清理
```

所有 unsupported dtype/shape/ABI 必须在提交前返回明确错误。禁止在生产板随机 fuzz 寄存器、随机 patch payload 或猜物理地址；从已知合法任务单变量推进。默认串行，先确认资源所有权、超时与错误恢复，再开展并发。

## 7. 卡点与下一步分支

| 卡点 | 收集什么 | 下一步，而不是强行宣布成功 |
|---|---|---|
| 只有顶层描述符，没有完整 payload | 哪个 buffer 不可见、哪个边界在 firmware 内 | 使用 OEM driver trace/公开执行器代码缩小边界；标为未知，不盲目回放 |
| B 不能换 | 固定图权重来源、packed layout、scale、校正项和缓存 | 对照静态权重样本建立 packer；没有 packer 就不能成立通用 Linear |
| 新进程失败 | 上下文状态、driver-managed 对象、初始化、地址重定位 | 逐项重建 session，禁止依赖上一进程内核 ID |
| 存在验证/签名/受保护资源边界 | 实际错误与合法接口说明 | 使用受支持的 OEM 开发接口；不把绕过安全机制设为项目任务 |
| 算术正确但慢 | prepare/pack/sync/submit/device/finish 的完整计时与字节数 | 先减少每 token 重建和搬运，再调 tile；不要先追整模 tokens/s |

交付的公开报告应只保留必要的结构描述、合成测试和结果，真实地址、原始内存、SDK/固件/私有模型权重留在受控环境。下一步按 [验收规范](07-validation-and-work-items.md)进入 W8A8 与单层集成。
