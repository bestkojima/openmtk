# 08 · 设备、算子、实验与结果记录模板

[返回目录](index.md) · 复制到本地实验目录后填写。以下示例全部是空模板，`UNKNOWN`、`null`、`NOT_RUN` 不能当作实测值。公开前按末尾清单审查。

## 1. 设备与工具链 manifest

```yaml
schema_version: 1
record_id: UNKNOWN
recorded_at: UNKNOWN
status: NOT_RUN
owner_role: authorized_engineering_device

device:
  soc: UNKNOWN
  board: UNKNOWN
  android_release: UNKNOWN
  build_fingerprint: UNKNOWN
  kernel_release: UNKNOWN
  bsp_commit: UNKNOWN
  apusys_driver_commit: UNKNOWN
  uapi_header_path: UNKNOWN
  uapi_header_sha256: UNKNOWN
  uapi_version_observed: null
  firmware_version_or_hash: UNKNOWN
  npu_device_nodes: []
  access_domain: UNKNOWN
  recovery_procedure_verified: false

toolchain:
  ndk: UNKNOWN
  cxx_compiler: UNKNOWN
  llm_sdk: UNKNOWN
  model_converter: UNKNOWN
  neuron_compiler: UNKNOWN
  neuron_runtime: UNKNOWN
  loaded_vendor_library_hashes: {}
  compile_flags: []

model_reference:
  repo_id: UNKNOWN
  revision: UNKNOWN
  config_sha256: UNKNOWN
  weight_manifest_sha256: UNKNOWN
  tokenizer_sha256: UNKNOWN
  quant_recipe: UNKNOWN
  input_sha256: UNKNOWN
  output_sha256: UNKNOWN
  official_success_log: UNKNOWN

execution:
  submitted_by_pid_role: UNKNOWN
  process_path: UNKNOWN
  uses_binder_service: null
  engine_evidence: UNKNOWN
  cpu_affinity: UNKNOWN
  omp_threads: null
  cpu_npu_ddr_policy: UNKNOWN
  thermal_state: UNKNOWN
```

工程板记录一次完整指纹后，各实验引用它，不要跨设备拼接 compiler/runtime/ABI。实际运行库版本与 `.dla` 生成版本分别记录；不能只写“都是 NP9”就认为完整兼容。

## 2. Matmul 算子契约

```yaml
schema_version: 1
contract_id: matmul_i8_v1_proposed
status: NOT_RUN
logical:
  equation: C[M,N] = A[M,K] * B[K,N]
  M: null
  K: null
  N: null
  weight_checkpoint_layout: N_K
  transpose_semantics: explicit
numeric:
  input_dtype: int8
  weight_dtype: int8
  input_range: [-127, 127]
  weight_range: [-127, 127]
  input_zero_point: 0
  weight_zero_point: 0
  accumulator_observed: UNKNOWN
  output_dtype_observed: UNKNOWN
  fused_requantization: null
  fused_activation: null
  bias_mode: none
  rounding: UNKNOWN
  saturation: UNKNOWN
  partial_accumulator_width: null
layout:
  A_layout: UNKNOWN
  B_layout: UNKNOWN
  C_layout: UNKNOWN
  alignment: UNKNOWN
  padding_value: UNKNOWN
  tile_limits: UNKNOWN
  packing_version: UNKNOWN
runtime:
  persistent_weight: null
  dynamic_weight_rebinding: null
  dynamic_M_supported: null
  synchronization_contract: UNKNOWN
  completion_contract: UNKNOWN
```

`numeric` 中的拟议范围与硬件观测不同要分别保留，不覆盖原始观察。W8A8 wrapper 另写 activation/weight scales 轴、dtype、零行规则、FP32 bias 和 epilogue，不混入 raw integer kernel 契约。

## 3. 一次实验记录

```yaml
schema_version: 1
experiment_id: UNKNOWN
work_item: OM-002
hypothesis: UNKNOWN
changed_variable: UNKNOWN
controlled_variables: []
device_record: UNKNOWN
model_record: UNKNOWN
source_commit: UNKNOWN
backend_requested: UNKNOWN
backend_observed: UNKNOWN
mode: capture-deep_or_trace-light_or_benchmark
seed: null
input_hashes: {}
weight_hashes: {}
compile_artifact_hashes: {}
commands_file: UNKNOWN
status: NOT_RUN
results:
  submitted: null
  completed: null
  device_engine_verified: null
  numerical_contract_passed: null
  allclose_policy: UNKNOWN
  max_abs: null
  rmse: null
  relative_l2: null
  nan_inf_count: null
  exact_integer_mismatches: null
  num_samples: null
  median_us: null
  p95_us: null
  process_rss_peak_bytes: null
  device_memory_bytes: null
  transfer_bytes: null
  trace_dropped_events: null
independence:
  libneuron_loaded: null
  delegated_vendor_process: null
  fresh_process_passed: null
  different_weights_passed: null
  relocated_resources_passed: null
artifacts:
  private_trace_directory: UNKNOWN
  public_summary: UNKNOWN
limitations: []
next_action: UNKNOWN
```

状态仅使用 `NOT_RUN`、`PASS`、`FAIL`、`BLOCKED`；`PASS` 必须附通过的具体门槛和输入范围。工具不能访问板子时写 `NOT_RUN`，不能把主机模拟结果标成 NPU 成果。

## 4. Recorder 事件与资源引用规范草案

事件应分开记录 `enter`、`return` 和必要的 `completion`，在消费时关联，而不是假设 ioctl return 时设备一定完成。示意：

```json
{
  "schema_version": 1,
  "run_id": "synthetic-example",
  "event_id": 1,
  "phase": "enter",
  "clock_domain": "host_process_monotonic",
  "timestamp_ns": null,
  "process_role": "test_runner",
  "thread_id": null,
  "device_resource_id": "dev-1",
  "request": null,
  "operation": "UNKNOWN_UNTIL_ABI_DECODED",
  "abi_id": "UNKNOWN",
  "snapshot_ref": null,
  "resource_refs": [],
  "capture_status": "NOT_RUN"
}
```

资源关系至少含 `resource_id`、类型、大小、有效期、owner context、host/device 可见性、父 buffer+offset、别名/dup 关系。fd、用户 VA、device VA、kernel command ID 和 firmware opaque handle 分别命名，不能统称 pointer。无法证明的引用保持 UNKNOWN，不自动重写或解引用。

归一化 diff 仅消去已证实的非语义差异，保留原始本地记录和规则版本；同样 payload hash 不证明地址/状态可重用，不同 hash 也不自动证明算子不同。时间线关联使用 run/event IDs 和时钟域，跨进程/内核时间戳先确认可比。

## 5. PR/任务交付模板

```markdown
## 目标与范围
对应 OM-xxx；实现什么，不实现什么。

## 来源与数值契约
引用的公开源码/论文/SDK 文档版本；自有代码与复用代码区分。
输入输出 shape/dtype/layout、quant scales、bias、同步和生命周期。

## 已执行验证
主机/模拟/工程板分开写，附命令、commit、数据 hash 和结果。

## 未执行项
板子不可访问、未知 ABI、未验证 shape 等必须列明。

## 变更风险与回退
错误路径、资源清理、取消/重置、回退的显式策略。

## 性能
同条件 baseline、warmup、样本数、完整调用 median/p95、内存和转换字节；
不提供数据就写未测，不由 kernel 理论吞吐推断模型加速比。
```

## 6. 公开提交前审查

本仓库当前为公开仓库，本轮只提交设计说明。后续上传前检查：无 token/密钥/账号凭据；无厂商 `.so`/SDK/固件/私有权重；无未经许可的 `.dla`/命令 blob；无含业务输入、内存内容或原始地址的 trace；来源许可已记录；真实硬件数据已区分报告值和推断值。

可以公开合成输入生成代码、CPU reference、经核验的结构说明、归一化元数据、测试摘要和重现步骤。模型文件通过使用者合法获取，文档保存 repo/revision/hash 而非权重副本。

模板填写完后，按 [07 验证与任务](07-validation-and-work-items.md)执行；没有证据的能力仍保持未知。
