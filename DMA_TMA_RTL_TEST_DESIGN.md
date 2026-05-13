# DMA/TMA RTL Directed Test Design

## 背景

本文档记录在 `testcases/` 中补完整 DMA/TMA RTL 定向测试的设计和当前实现状态。目标不是替代
`gpgpu/ventus/tests/src/DmaTest/` 中的 Chisel unit test，而是在 OpenCL application
层补端到端测试，让 case 真实经过编译器、POCL/OpenCL runtime、RTL pipeline、DMA core、
L2/cache、shared memory 和 fence scheduler。

当前已经按本文档初版落地了新增 testcase，并完成了 `spike -> gvm` 顺序验证；验证结果见
后文“验证记录”。

## 现有测试入口

当前相关 testcase 都已经在 `testcases/_get_case/` 下：

- `dma_test/`：单 workgroup、单 warp leader 发 `CP_ASYNC_BULK + CP_ASYNC_FENCE`。
- `copysize_test/`：固定 `CP_ASYNC_COPYSIZE` 16B 冒烟测试。
- `tensor_dma_test/`：固定 2D FP32 4x4 `CP_ASYNC_TENSOR` 冒烟测试。
- `multi_wg_dma_test/`：多个 workgroup 各自发 bulk DMA，但不是同一 workgroup 内多 warp。
- `tma_matrix_test/`：最适合扩展的 table-driven TMA 矩阵测试，host 端已有 descriptor
  构造、source pattern、expected replay 和逐 case 运行机制。
- `bulk_dma_matrix_test/`：新增，覆盖 bulk 跨 128B cacheline 和 shared dst offset。
- `multi_warp_dma_fence_test/`：新增，覆盖同 workgroup 多 warp、多 DMA、per-warp fence。
- `dma_shared_routing_conflict_test/`：新增，覆盖 shared bank conflict 与 DMA shared response
  routing 交叠。

`regression-test.py` 和 `testcases/_get_case/cases.csv` 目前没有把这些 DMA/TMA case
纳入常规回归。当前已新增专用入口 `testcases/_get_case/cases_dma_tma.csv`。

## RTL/模型事实

关键参数和行为：

- `l2cacheline = dcache_BlockWords * BytesOfWord = 32 * 4 = 128B`。
- DMA shared 写入粒度是 `dma_aligned_bulk = 4B`。
- `CP_ASYNC_BULK` 和 `CP_ASYNC_COPYSIZE` 在 Spike 中是同步 byte copy。
- `CP_ASYNC_FENCE` 在 Spike 中是 NOP；RTL 中会通过 scheduler 的 per-warp inflight
  计数阻塞/释放 warp。
- `CP_ASYNC_TENSOR` descriptor 的 dataType/rank/shape/stride 语义以
  `spike/riscv/insns/cp_async_tensor.h` 为参考。
- shared memory response routing 依赖 `sourceTag`：pipe shared request 标记为 false，
  DMA shared request 标记为 true；该 tag 必须穿过 bank-conflict replay 后再路由回
  `pipe.io.shared_rsp` 或 `pipe.io.dma_shared_rsp`。
- no-cache RTL 移除了 L1 DCache/L2 cache，但仍保留 SMEM。`GPGPU_top_nocache.scala`
  现在将 DMA cacheline `Get` 转换成现有 C++ dcache bypass 能处理的 `DCacheCoreReq_np`，
  并用 response route FIFO 将外部 dcache response 分回 LSU 或 DMA；DMA shared request
  与 pipe shared request 共用 SMEM 仲裁和 `sourceTag` 回包路由。

## 当前覆盖状态

| 需求 | 当前状态 | 对应用例 |
| --- | --- | --- |
| bulk 跨 cacheline | 已有 app-level 覆盖 | `bulk_dma_matrix_test` 覆盖 120/124B 边界、192B、多 shared dst offset |
| 2D/3D tensor | 已有 app-level 覆盖 | `tma_matrix_test` 覆盖 1D、2D、3D full/subbox/partial |
| elementStride | 已有 app-level 覆盖 | `tma_matrix_test` 覆盖 dim0 stride、dim1 stride、dim0+dim1 stride |
| subbox | 已有 app-level 覆盖 | `FP32_2D_subbox_8x8_at_2_2`、`FP32_3D_subbox_6x6x6_at_1_1_1` |
| OOB | 已有 app-level 覆盖 | dim0、dim1、dim2、subbox+OOB、elementStride+OOB：`FP32_2D_oob_zero_dim0`、`FP16_2D_oob_fill`、`FP32_2D_oob_zero_dim1`、`FP32_3D_oob_zero_dim2`、`FP16_2D_oob_subbox_dim0_dim1_fill`、`FP32_2D_oob_estride_dim1` |
| 多 warp 多 DMA + fence | 已有 app-level 覆盖 | `multi_warp_dma_fence_test` 4 个 case |
| shared bank conflict 下 DMA response routing | 已有 app-level 覆盖 | `dma_shared_routing_conflict_test` 2 个 pressure case |

## 实现状态

已完成的 testcase 改动：

- 扩展 `testcases/_get_case/tma_matrix_test/tma_matrix_test.c`：
  - 新增 `expect_oob`、`oobfill`、`rtl_only` 字段。
  - host expected model 支持 dim>0 `elementStrides` 的有效输出维度。
  - host expected model 支持 OOB fill：整数类型补 0；浮点类 `oobfill=1` 补全 1 bit。
  - 新增 `FP32_3D_subbox_6x6x6_at_1_1_1`、`FP32_3D_estride2_dim1`、
    `FP32_2D_estride2_rows_cols`、`FP32_2D_oob_zero_dim0`、`FP16_2D_oob_fill`、
    `FP32_2D_oob_zero_dim1`、`FP32_3D_oob_zero_dim2`、
    `FP16_2D_oob_subbox_dim0_dim1_fill`、`FP32_2D_oob_estride_dim1`。
  - OOB-directed case 的 source buffer 会覆盖逻辑 tensor footprint 和物理 box walk footprint，
    避免错误 RTL 在 host verdict 前先因越界读失效。
  - `rtl_only` case 默认在非 RTL-directed 路径跳过，可用 `VENTUS_TMA_RUN_RTL_ONLY=1`
    强制运行。
- 新增 `testcases/_get_case/bulk_dma_matrix_test/`：
  - `bulk_32B_at_120`、`bulk_8B_at_124`、`bulk_192B_aligned`、`bulk_64B_dst_offset`。
- 新增 `testcases/_get_case/multi_warp_dma_fence_test/`：
  - `2warp_1dma_each_fence`、`2warp_2dma_each_single_fence`、
    `4warp_1dma_each_fence`、`2warp_cross_cacheline_each`。
  - kernel 在 asm 内读 `CSR_WID=0x805` 计算 per-warp segment，避免 vector-to-scalar
    operand 搬运错误。
  - 1-DMA 和 2-DMA 路径拆成两个 kernel，避免 GVM 对运行时分支包住 custom DMA/fence
    序列时出现 reference PC fatal。
- 新增 `testcases/_get_case/dma_shared_routing_conflict_test/`：
  - `routing_64B_conflict64`、`routing_192B_crossline_conflict64`。
  - 使用单个 `scratch` local buffer 手动分区，避免多个静态 `__local` 对象在当前 LDS
    布局下别名导致 DMA payload 被 conflict buffer 覆盖。
- 新增 `testcases/_get_case/cases_dma_tma.csv`。

## 修复记录

本轮调试除补齐 testcase 外，还修复了 RTL TMA OOB/subbox 交互问题：

- `gpgpu/ventus/src/pipeline/DMA_core.scala`
  - TMA OOB fill 不再只按 box row bounds 判断；现在同时检查当前元素是否落在 full tensor
    的 dim0 row `[row_start, row_start + globalDim0 * datawidth)` 内。
  - tag 中记录的 `tensor_dim0_start` 改为“当前 box row 对应的 full tensor dim0 row 起点”。
    计算方式是从 `BoxAddress - globalAddress` 中剥掉 dim0 offset，保留 dim1/dim2/... 的
    subbox 固定偏移；这样 `FP32_2D_oob_zero_dim0` 不再泄漏下一行数据，同时 2D/3D subbox
    不会被误判为 OOB 后填 0。
  - tag 新增 `tensor_high_dim_valid`。AddrCalc 阶段从 `BoxAddress - globalAddress` 按
    `globalStrides` 解出 subbox 的 dim1..dim4 起点，再结合当前 `tensor_dim_step` 和
    `elementStrides` 判断高维坐标是否仍小于 `globalDim`。
  - Temp_mem 的 OOB fill 现在同时使用 dim0 byte valid 和 `tensor_high_dim_valid`。高维 OOB
    行仍保留 box row mask 写回 shared，但 payload 会按 dataType/oobfill 写成 fill value，
    不再泄漏 backing buffer 中的后续 pattern。

新增 OOB test 曾在旧 GVM RTL 上复现缺口：

- `FP32_2D_oob_zero_dim1` 失败于 byte 64：expected `0x00`，got `0x87`，说明 dim1 OOB
  行读取了源 pattern 而不是 zero-fill。

本轮 no-cache 检查修复了 no-cache RTL 的 DMA 端到端通路：

- `gpgpu/ventus/src/top/GPGPU_top_nocache.scala`
  - 删除原先对 `pipe.io.dma_cache_req` / `pipe.io.dma_shared_req` 的 tie-off 和
    “DMA instructions not supported” 断言。
  - 新增 no-cache DMA dcache adapter：把 DMA `DCacheMemReq_p` 的 128B cacheline `Get`
    映射成 `DCacheCoreReq_np`，走已有 no-cache C++ physical-memory bypass；response route FIFO
    保存 DMA `a_source`/`a_addr`，在外部 dcache response 返回时恢复成 `DCacheMemRsp` 给
    `DMA_core`。
  - SMEM 侧复用 with-cache 路径的二路仲裁：pipe shared request 与 DMA shared request
    共用 `SharedMemory`，并按 `sourceTag` 分别回到 `pipe.io.shared_rsp` 或
    `pipe.io.dma_shared_rsp`。

no-cache 旧 RTL 复现：

- `bulk_dma_matrix_test`、`multi_warp_dma_fence_test`、`dma_shared_routing_conflict_test`
  均在首个 case 触发 `GPGPU_top_nocache.scala:147` 断言：
  `DMA L2 request reached nocache build - DMA instructions not supported in this configuration`。
- `tma_matrix_test` 在 20 个 RTL-directed case 中均因同一断言失败。

## 验证记录

验证顺序按 `spike -> gvm` 执行。

Spike 结果：

| testcase | 结果 |
| --- | --- |
| `tma_matrix_test` | 12 pass / 0 fail / 8 skip；skip 为 RTL-only OOB 和非 dim0 stride |
| `bulk_dma_matrix_test` | 4 pass / 0 fail |
| `multi_warp_dma_fence_test` | 4 pass / 0 fail |
| `dma_shared_routing_conflict_test` | 2 pass / 0 fail |

GVM 结果：

| testcase | 结果 |
| --- | --- |
| `tma_matrix_test` with `VENTUS_TMA_RUN_RTL_ONLY=1` | 20 pass / 0 fail / 0 skip |
| `bulk_dma_matrix_test` | 4 pass / 0 fail |
| `multi_warp_dma_fence_test` | 4 pass / 0 fail |
| `dma_shared_routing_conflict_test` | 2 pass / 0 fail |

GVM no-cache 结果：

| testcase | 结果 |
| --- | --- |
| `tma_matrix_test` with `VENTUS_TMA_RUN_RTL_ONLY=1` | 20 pass / 0 fail / 0 skip |
| `bulk_dma_matrix_test` | 4 pass / 0 fail |
| `multi_warp_dma_fence_test` | 4 pass / 0 fail |
| `dma_shared_routing_conflict_test` | 2 pass / 0 fail |

备注：GVM 日志中仍会打印已知 reference checker 噪声，例如 PC `0x80000014` 的 XREG mismatch
和 TMA OOB fill 场景下的 VREG mismatch；这些未导致 host verdict 失败，本轮以 testcase
最终 `OK`/`FAILED` 为准。

本轮主要日志：

- baseline：`/tmp/codex-oob-baseline-20260512-221511`
- 新增 case 旧 RTL 复现：`/tmp/codex-oob-prefail-20260512-222816.log`
- 修复后完整验证：`/tmp/codex-oob-full-verify-20260512-224252`
- no-cache 旧 RTL 复现：`/tmp/codex-nocache-dma-check-cwd-20260512-231159`
- no-cache 修复后完整验证：`/tmp/codex-nocache-dma-verify-20260512-234104`

## 设计方案

### 1. 扩展 `tma_matrix_test`

优先从 `testcases/_get_case/tma_matrix_test/tma_matrix_test.c` 入手。该文件已经有
`g_cases[]`、`build_descriptor()`、`compute_expected()`、`run_one()`，新增 case 的改动集中，
维护成本最低。

建议新增字段：

- `expect_oob`：是否按 globalDim 做 OOB 判断。
- `oobfill`：写入 descriptor 的 `VRS2[14]`，即 `desc[46]`。
- `rtl_only` 或 `expected_backend_note`：若某 case 只用于 RTL 行为确认，避免 Spike 同步模型
  与 RTL OOB 语义不一致时误判。

建议新增 case：

- `FP32_3D_subbox_4x4x4_at_1_1_1`：验证 3D offset、slice stride 和 packed dst。
- `FP32_3D_estride2_dim1`：验证非 dim0 的 elementStride 计数和地址 carry。
- `FP32_2D_estride2_rows_cols`：在 dim0/dim1 同时 stride，验证 source gather 与 dst packed。
- `FP32_2D_oob_zero_dim0`：`globalDim[0] < boxDim[0]`，先覆盖 RTL 当前注释里提到的 dim0 OOB。
- `FP16_2D_oob_fill`：如果 RTL 期望 floating OOB fill 支持 NaN/全 1，再单独覆盖 `oobfill=1`。
- `FP32_2D_oob_zero_dim1`：验证 dim1 越界行 zero-fill。
- `FP32_3D_oob_zero_dim2`：验证 3D slice 越界 zero-fill。
- `FP16_2D_oob_subbox_dim0_dim1_fill`：验证 subbox 同时触发 dim0/dim1 OOB 且 FP16 all-one fill。
- `FP32_2D_oob_estride_dim1`：验证 `elementStrides[1]` 参与高维 OOB 判定。

`compute_expected()` 需要从“总是从 src 读”升级为：

1. 计算每个输出元素对应的 global index。
2. 若任一维越过 `globalDim[d]`，按 dataType 和 `oobfill` 写 fill value。
3. 否则按现有 stride 公式从 `src` 拷贝元素。

当前已覆盖 dim0 和高维 OOB。RTL `DMA_core.scala` 的 OOB fill 判定需要同时保留 box row mask
和 full tensor validity：前者决定哪些 requested bytes 写入 shared，后者决定这些 bytes 是
真实 payload 还是 fill value。

### 2. 新增 bulk 跨 cacheline matrix

建议新增目录：

`testcases/_get_case/bulk_dma_matrix_test/`

测试结构沿用现有 Makefile 和 `../common/ventus_opencl_test.h`。host 传入一个 case table，
kernel 接收 `src_offset_bytes`、`copy_bytes`、`dst_offset_bytes`，在 asm 中对从 kernel args
取到的 src base 加 offset 后发 `CP_ASYNC_BULK`。

建议 case：

- `bulk_32B_at_120`：`src_offset=120`、`copy_bytes=32`，跨 128B cacheline。
- `bulk_8B_at_124`：尾部 4B + 下一行 4B，检验最小跨线。
- `bulk_192B_aligned`：覆盖 1.5 个 cacheline，对应 unit test 中的 multi-cacheline exact payload。
- `bulk_64B_dst_offset`：验证 shared dst 非 0 offset。

host 端用 byte pattern 比较，不要限制 `count <= wg_size`；shared buffer 大小按最大 copy bytes
预留，输出也按 byte 数回读。

### 3. 新增多 warp 多 DMA + fence

建议新增目录：

`testcases/_get_case/multi_warp_dma_fence_test/`

设计：

- `local_size = num_warp * 32`，建议先用 2 warp，再扩到 4 warp。
- 每个 warp 的 `lane_id == 0` 发起 DMA。
- 每个 warp 发两条 bulk DMA 到不同 shared 区域，然后执行一次 `CP_ASYNC_FENCE`。
- fence 后 workgroup barrier，再由所有 lanes 把 shared 中的数据写回 global。
- host 验证每个 warp、每条 DMA 的 segment 都正确，且没有互相覆盖。

建议 case：

- `2warp_1dma_each_fence`
- `2warp_2dma_each_single_fence`
- `4warp_1dma_each_fence`
- `2warp_cross_cacheline_each`

这个测试补的是 OpenCL/RTL 端到端路径；scheduler 的 per-warp inflight 细节已有
`DMA_fence_scheduler_test` 覆盖。

### 4. 新增 shared bank conflict 下 DMA response routing

建议新增目录：

`testcases/_get_case/dma_shared_routing_conflict_test/`

目标是让普通 pipe shared memory request 和 DMA shared write 交叠，并且普通 shared request
触发 bank conflict replay，从而验证 `sourceTag` 在 shared memory pipeline 和 replay 过程中
不会丢失。

bank conflict 构造：

- bank index 来自 word block offset 的低位。
- 让多个 lane 访问 `shared_conflict[lid * 32]`，这些地址在 32-lane 配置下落到同一 bank，
  但 block offset/addr 不同，会触发 replay。

kernel 结构建议：

1. warp0 leader 发 DMA 到 `shared_dma[]`。
2. warp1 或所有 lanes 做多轮 `shared_conflict[lid * 32]` 写/读，制造 shared bank conflict。
3. warp0 执行 `CP_ASYNC_FENCE`。
4. barrier 后把 `shared_dma[]` 和 `shared_conflict[]` 的 checksum 写回 global。
5. host 同时验证 DMA payload 和 conflict checksum。

为了提高交叠概率，可以：

- 在 DMA 前后都做 bank-conflict loop。
- 多次发 DMA，或者增加 copy bytes 让 DMA response 延迟更长。
- 让 host 支持 `repeat` 参数，RTL 下重复跑同一 kernel 多次。

失败信号：

- DMA output 错误：DMA response 被错误路由或 shared 写入损坏。
- conflict checksum 错误：pipe shared response 被错误路由或 replay 损坏。
- kernel hang：`sharedmem.io.coreRsp.ready` routing 或 DMA fence completion 路径卡住。

## 接入建议

短期不要直接塞进默认 `regression-test.py` 的全部 checklist，因为 RTL case 可能慢。建议先
保留专用入口：

- `testcases/_get_case/cases_dma_tma.csv`
- 或 `testcases/_get_case/run_dma_tma_rtl.sh`

推荐命令形态：

```sh
source ../../../env.sh
export VENTUS_BACKEND=${VENTUS_BACKEND:-rtl-withcache}
make
./tma_matrix_test.out
```

no-cache GVM 也可用于 DMA/TMA directed suite：

```sh
source ../../../env.sh
export VENTUS_BACKEND=gvm-nocache
export VENTUS_TMA_RUN_RTL_ONLY=1  # only needed for tma_matrix_test full RTL-directed set
make
./tma_matrix_test.out
```

case 输出必须保持短小：只打印 case name、PASS/FAIL、首个 mismatch。RTL/stdout 日志可能极长，
不要在测试程序里主动 dump 大量中间状态。

## 优先级

1. 扩 `tma_matrix_test`：OOB、3D subbox、non-dim0 elementStride。
2. 新增 `bulk_dma_matrix_test`：补 bulk 跨 128B cacheline 的 app-level RTL case。
3. 新增 `multi_warp_dma_fence_test`：补多 warp、多 DMA、per-warp fence 端到端。
4. 新增 `dma_shared_routing_conflict_test`：最后做 shared bank conflict + DMA response routing 压力。

这个顺序从最小改动到最高风险，便于每一步都能定位问题来源。

## 验收标准

- 每个新增 testcase 都有 Makefile，可直接 `make && ./<case>.out`。
- 验证顺序固定为先 `VENTUS_BACKEND=spike`、后 `VENTUS_BACKEND=gvm`；Spike 用于排除
  testcase 自身问题，GVM/RTL 用于暴露 DUT 行为差异。
- RTL directed run 下每个 case 输出 `OK` 或 `FAILED`，失败时给出首个 mismatch。
- 覆盖矩阵中的七项需求都至少有一个 application-level testcase 对应。
- 若某项已由 unit test 覆盖但 app-level 暂不覆盖，必须在本文档中保留说明，避免误认为已完成。
