# DMA/TMA Project Structure and Test Objects

本文档是 `testcases/_get_case/` 目录下 DMA/TMA OpenCL testcase 的总览，同时索引相关 RTL、
Spike/GVM model 和仿真项目。详细设计、修复记录和验证结果仍以
`../DMA_TMA_RTL_TEST_DESIGN.md` 和仓库根目录 `DMA_TMA_PREFETCH_DESCRIPTOR_RESEARCH.md`
为准。

## 总体分层

```text
ventus-env-copilot-test/
├── DMA_TMA_PREFETCH_DESCRIPTOR_RESEARCH.md      # prefetch/descriptor RTL 设计与验证主文档
├── testcases/
│   ├── DMA_TMA_RTL_TEST_DESIGN.md               # app-level directed testcase 设计与验证记录
│   └── _get_case/
│       ├── DMA_TMA_PROJECT_STRUCTURE.md         # 本文档
│       ├── cases_dma_tma.csv                    # DMA/TMA testcase 登记表
│       ├── run_dma_tma_rtl.sh                   # Spike/GVM directed suite runner
│       ├── common/                              # OpenCL host helper 和通用 Makefile 片段
│       ├── dma_test/                            # legacy bulk DMA smoke
│       ├── copysize_test/                       # legacy copysize smoke
│       ├── tensor_dma_test/                     # legacy VGPR tensor DMA smoke
│       ├── tma_descriptor_test/                 # descriptor-addressed TMA + prefetch smoke
│       ├── multi_wg_dma_test/                   # legacy multi-workgroup bulk DMA smoke
│       ├── tma_matrix_test/                     # table-driven TMA matrix
│       ├── bulk_dma_matrix_test/                # table-driven bulk DMA matrix
│       ├── multi_warp_dma_fence_test/           # same-workgroup multi-warp DMA/fence matrix
│       ├── dma_shared_routing_conflict_test/    # DMA shared response routing pressure
│       └── tma/                                 # older standalone C++ TMA sample
├── gpgpu/
│   ├── ventus/src/pipeline/DMA_core.scala       # DMA/TMA RTL 主实现
│   ├── ventus/src/top/parameters.scala          # DMA/TMA 相关参数
│   ├── ventus/src/top/GPGPU_top_nocache.scala   # no-cache GVM 的 DMA adapter/SMEM route
│   ├── ventus/tests/src/DmaTest/                # Chisel RTL unit tests
│   ├── sim-verilator/                           # with-cache GVM/Verilator project
│   └── sim-verilator-nocache/                   # no-cache GVM/Verilator project
└── spike/
    ├── riscv/insns/cp_async_*.h                 # Spike DMA/TMA instruction model
    └── gvmref/                                  # GVM reference interface/workgroup model
```

## 运行入口

DMA/TMA app-level directed suite 的入口是：

```sh
source ./env.sh
VENTUS_BACKEND=spike ./testcases/_get_case/run_dma_tma_rtl.sh /tmp/<spike-log-dir>
./testcases/_get_case/run_dma_tma_rtl.sh /tmp/<gvm-log-dir>
```

规则：

- `.out` 必须从各 testcase 目录内启动。host 会按当前目录读取同名 `.cl`，从仓库根目录直接跑
  `testcases/_get_case/<case>/<case>.out` 会导致 OpenCL build error `-44`。
- `run_dma_tma_rtl.sh` 默认 `VENTUS_BACKEND=gvm`，Spike 需要显式设置 `VENTUS_BACKEND=spike`。
- `VENTUS_TMA_RUN_RTL_ONLY=1` 只应影响 `tma_matrix_test` 的 GVM run；当前 runner 会自动处理。
- 不要在 runner 中设置 `ulimit -s unlimited`。该设置会让 GVM 下 `tma_descriptor_test` 在 RTL trace
  前 host segfault。
- `cases_dma_tma.csv` 是登记表；runner 当前只执行已验证的 directed 子集。

如果使用通用批量 runner：

```sh
cd /home/liyb/ventus-env-copilot-test/testcases/_get_case
python3 run_ventus_tests.py --cases cases_dma_tma.csv --backends spike,gvm --rtol 1e-2 --atol 1e-2 --jobs 4 --timeout 7200
```

`cases_dma_tma.csv` 中的 DMA/TMA 项目使用 `check_mode=verdict`。这类 testcase 不生成
`VENTUS_RESULT_FILE`/`final.hex`，也不能使用 NVIDIA baseline；runner 应跳过 NVIDIA baseline，
保存 `results/<case>/<env>/run.log`，并按程序退出码和 stdout verdict 判定。
`run_dma_tma_rtl.sh` 仍是 RTL directed 子集的推荐入口。

## Directed App-Level 测试对象

这些 case 会被 `run_dma_tma_rtl.sh` 执行，覆盖当前 DMA/TMA RTL 验证主路径。

| 项目 | 文件 | 内部测试对象 | 覆盖点 |
| --- | --- | --- | --- |
| `tma_descriptor_test/` | `tma_descriptor_test.c`、`tma_descriptor_test.cl` | `use_prefetch=0`、`use_prefetch=1` | descriptor-addressed `CP_ASYNC_TENSOR_G2S`；`PREFETCH_TENSORMAP` 后 descriptor cache 命中路径的端到端功能 |
| `tma_matrix_test/` | `tma_matrix_test.c`、`tma_matrix_test.cl`、`run_tma_cases.sh`、`run_tma_quiet.sh` | 23 个 table-driven case | legacy VGPR `CP_ASYNC_TENSOR` 的 1D/2D/3D、dataType、subbox、elementStride、swizzle、OOB fill |
| `bulk_dma_matrix_test/` | `bulk_dma_matrix_test.c`、`bulk_dma_matrix_test.cl` | 4 个 bulk case | `CP_ASYNC_BULK` 跨 128B cacheline、multi-cacheline、shared dst offset |
| `multi_warp_dma_fence_test/` | `multi_warp_dma_fence_test.c`、`multi_warp_dma_fence_test.cl` | 4 个 multi-warp case | 同一 workgroup 内多个 warp 发 DMA；per-warp `CP_ASYNC_FENCE` inflight 释放 |
| `dma_shared_routing_conflict_test/` | `dma_shared_routing_conflict_test.c`、`dma_shared_routing_conflict_test.cl` | 2 个 conflict case | DMA shared response 与普通 shared bank conflict replay 交叠时的 `sourceTag` 路由 |

### `tma_descriptor_test`

目录结构：

```text
tma_descriptor_test/
├── Makefile
├── tma_descriptor_test.c
├── tma_descriptor_test.cl
├── tma_descriptor_test.out      # build output
├── object0.cl                   # generated/copied kernel source
├── object0.riscv                # generated kernel binary
└── object0.riscv.log            # compiler log
```

测试对象：

- `use_prefetch=0`：直接用 descriptor pointer 和 dynamic coords 发 descriptor-addressed TMA。
- `use_prefetch=1`：先发 `PREFETCH_TENSORMAP`，再发 descriptor-addressed TMA。

验证内容：

- descriptor 在 global memory 中构造，rank=2、FP32、8x8 global tensor、4x4 box、坐标 `[2,2]`。
- kernel patch runtime src pointer 后发 TMA，结果应等于 host replay 的 4x4 subbox。
- Spike 中 prefetch 是 no-op；GVM 中 prefetch 走 RTL metadata L2 path 并填充 DMA-local descriptor cache。
- 该 testcase 证明 prefetch/no-prefetch 两条 app-level 功能路径正确；prefetch issue path 提前释放和
  descriptor cache skip L2 的微结构断言由 Chisel `TMA_T28/TMA_T29` 覆盖。

### `tma_matrix_test`

目录结构：

```text
tma_matrix_test/
├── Makefile
├── tma_matrix_test.c
├── tma_matrix_test.cl
├── run_tma_cases.sh
├── run_tma_quiet.sh
├── tma_matrix_test.out
├── object0.cl
├── object0.riscv
└── object0.riscv.log
```

测试对象：

| case | 属性 | 覆盖点 |
| --- | --- | --- |
| `FP32_2D_4x4_full` | rank=2 FP32 | 2D full tensor baseline |
| `FP32_1D_16` | rank=1 FP32 | 1D tensor path |
| `FP32_2D_subbox_8x8_at_2_2` | rank=2 FP32 | 2D subbox offset |
| `FP32_2D_padded_rows_4x4_stride64` | rank=2 FP32 | row stride 大于 packed width |
| `FP32_3D_2x2x2` | rank=3 FP32 | 3D carry 和 slice stride |
| `FP16_2D_8x4` | rank=2 FP16 | 2B datawidth |
| `I32_2D_4x4` | rank=2 INT32 | integer 4B dataType |
| `I8_2D_16x4` | rank=2 INT8 | byte-granular payload |
| `I16_1D_32` | rank=1 INT16 | 2B integer payload |
| `FP32_2D_partial_box_8x8_at_0_0` | rank=2 FP32 | global tensor 大于 box |
| `FP32_2D_estride2_cols` | rank=2 FP32 | dim0 elementStride gather |
| `FP32_2D_swizzle32_rows` | rank=2 FP32 | 32B swizzle rows |
| `FP32_2D_swizzle64_rows` | rank=2 FP32 | 64B swizzle rows |
| `FP32_2D_swizzle128_subbox_row1` | rank=2 FP32 | 128B swizzle + row subbox |
| `FP32_3D_subbox_6x6x6_at_1_1_1` | rank=3 FP32 | 3D subbox offset |
| `FP32_3D_estride2_dim1` | RTL-only | high-dim elementStride |
| `FP32_2D_estride2_rows_cols` | RTL-only | dim0 + dim1 elementStride |
| `FP32_2D_oob_zero_dim0` | RTL-only | dim0 OOB zero-fill |
| `FP16_2D_oob_fill` | RTL-only | FP16 OOB all-one fill |
| `FP32_2D_oob_zero_dim1` | RTL-only | dim1 OOB zero-fill |
| `FP32_3D_oob_zero_dim2` | RTL-only | dim2 OOB zero-fill |
| `FP16_2D_oob_subbox_dim0_dim1_fill` | RTL-only | subbox + dim0/dim1 OOB + FP16 fill |
| `FP32_2D_oob_estride_dim1` | RTL-only | elementStride 参与 high-dim OOB 判定 |

运行说明：

- Spike 默认跳过 RTL-only case，当前快照为 15 pass / 0 fail / 8 skip。
- GVM 需要 `VENTUS_TMA_RUN_RTL_ONLY=1` 跑满矩阵，当前快照为 23 pass / 0 fail / 0 skip。

### `bulk_dma_matrix_test`

目录结构：

```text
bulk_dma_matrix_test/
├── Makefile
├── bulk_dma_matrix_test.c
├── bulk_dma_matrix_test.cl
├── bulk_dma_matrix_test.out
├── object0.cl
├── object0.riscv
└── object0.riscv.log
```

测试对象：

| case | `src_offset` | `copy_bytes` | `dst_offset` | 覆盖点 |
| --- | ---: | ---: | ---: | --- |
| `bulk_32B_at_120` | 120 | 32 | 0 | 从 128B cacheline 尾部跨线 |
| `bulk_8B_at_124` | 124 | 8 | 0 | 最小跨线窗口 |
| `bulk_192B_aligned` | 0 | 192 | 0 | 1.5 个 cacheline |
| `bulk_64B_dst_offset` | 64 | 64 | 16 | shared 目的地址非 0 |

### `multi_warp_dma_fence_test`

目录结构：

```text
multi_warp_dma_fence_test/
├── Makefile
├── multi_warp_dma_fence_test.c
├── multi_warp_dma_fence_test.cl
├── multi_warp_dma_fence_test.out
├── object0.cl
├── object0.riscv
└── object0.riscv.log
```

测试对象：

| case | `num_warps` | `dmas_per_warp` | `copy_bytes` | `src_base_offset` | `src_stride` | 覆盖点 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `2warp_1dma_each_fence` | 2 | 1 | 64 | 0 | 64 | 2 warp 各 1 个 DMA 后 fence |
| `2warp_2dma_each_single_fence` | 2 | 2 | 64 | 0 | 64 | 每 warp 连发 2 个 DMA，再单次 fence |
| `4warp_1dma_each_fence` | 4 | 1 | 64 | 0 | 64 | 4 warp 并发 inflight |
| `2warp_cross_cacheline_each` | 2 | 1 | 32 | 120 | 256 | 每 warp 的 DMA 都跨 cacheline |

维护说明：

- kernel 使用 `CSR_WID=0x805` 计算每个 warp 的 segment。
- 1-DMA 和 2-DMA 使用两个 kernel，避免 GVM reference 对运行时分支包住 custom DMA/fence 序列时误判。

### `dma_shared_routing_conflict_test`

目录结构：

```text
dma_shared_routing_conflict_test/
├── Makefile
├── dma_shared_routing_conflict_test.c
├── dma_shared_routing_conflict_test.cl
├── dma_shared_routing_conflict_test.out
├── object0.cl
├── object0.riscv
└── object0.riscv.log
```

测试对象：

| case | `local_size` | `copy_bytes` | `rounds` | `src_offset` | 覆盖点 |
| --- | ---: | ---: | ---: | ---: | --- |
| `routing_64B_conflict64` | 64 | 64 | 32 | 0 | DMA shared write 与 64 lane shared pressure 交叠 |
| `routing_192B_crossline_conflict64` | 64 | 192 | 64 | 120 | 跨 cacheline DMA response 与更长 conflict pressure 交叠 |

维护说明：

- kernel 使用单个 local scratch 手动分区，避免多个静态 `__local` 对象在当前 LDS layout 下别名。
- host 同时检查 DMA payload 和 conflict lane 输出。

## 登记但非 Directed Runner 主集的测试对象

这些项目保留在 `cases_dma_tma.csv` 或当前目录中，适合作为 legacy smoke/手工复现入口。它们不是
`run_dma_tma_rtl.sh` 当前默认执行的 directed 子集。

| 项目 | 文件 | 测试对象 | 覆盖点 |
| --- | --- | --- | --- |
| `dma_test/` | `dma_test.c`、`dma_test.cl` | 默认 `count=16`、`wg_size=32`；可用命令行覆盖 | 单 workgroup、warp leader 发 `CP_ASYNC_BULK + CP_ASYNC_FENCE` |
| `copysize_test/` | `copysize_test.c`、`copysize_test.cl` | 固定 `copysize=2`，16B/4 int | `CP_ASYNC_COPYSIZE` smoke |
| `tensor_dma_test/` | `tensor_dma_test.c`、`tensor_dma_test.cl` | 固定 4x4 FP32 | legacy VGPR `CP_ASYNC_TENSOR` smoke |
| `multi_wg_dma_test/` | `multi_wg_dma_test.c`、`multi_wg_dma_test.cl` | 默认 2 WG、每 WG 16 int；可用命令行覆盖 | 多 workgroup bulk DMA，各 WG 独立 shared memory |
| `tma/` | `main.cc`、`kernel.cl` | 旧 C++ `dma_3` sample | 早期 TMA smoke，未纳入 CSV 和 directed runner |

## 共用基础设施

```text
common/
├── common.mk
├── make.config
├── ventus_opencl_test.h
└── ventus_result_io.h
```

作用：

- `ventus_opencl_test.h`：封装默认 platform/device/context/queue 获取、从当前工作目录读取 `.cl`、
  `clCreateProgramWithSource + clBuildProgram`、build log 打印和 OpenCL error 跳转。
- `common.mk` / `make.config`：各 testcase 的 host build 和 kernel compile 规则。
- 由于 `.cl` 是按相对路径读取，所有 app-level testcase 都必须在自己的目录内执行。

## RTL 设计和 Unit Test 项目

相关 RTL 源码：

| 路径 | 责任 |
| --- | --- |
| `gpgpu/ventus/src/pipeline/DMA_core.scala` | DMA/TMA 主状态机；bulk、copysize、legacy tensor、descriptor-addressed TMA、prefetch metadata path、descriptor cache、OOB fill |
| `gpgpu/ventus/src/top/parameters.scala` | `l2cacheline=128B`、`dma_aligned_bulk=4B`、`tma_desc_cache_entries=2`、`tma_prefetch_slots=2` 等参数 |
| `gpgpu/ventus/src/top/GPGPU_top_nocache.scala` | no-cache GVM 的 DMA cache request adapter 和 DMA/pipe shared response route |

Chisel unit test 目录：

```text
gpgpu/ventus/tests/src/DmaTest/
├── DMA_core_test.scala
├── DMA_fence_scheduler_test.scala
└── TMA_core_test.scala
```

`DMA_core_test.scala` 测试对象：

- `S1_single_bulk_copy`
- `S2_cross_cacheline_bulk_copy`
- `S3_copysize_min_transfer`
- `S4_multi_dma_inflight`
- `I1_copysize_4B_exact_payload`
- `I4_bulk_32B_cross_cacheline_exact_payload`
- `I3_bulk_128B_aligned_exact_payload`
- `I5_bulk_192B_multi_cacheline_exact_payload`
- `I6_bulk_64B_dst_offset_exact_payload`
- `T1_L2_delayed_response_bulk_128B`
- `T1b_L2_delayed_response_cross_cacheline`
- `T_TLB_1_basic_tlb_path`
- `T_TLB_2_cross_page_dma`
- `T_TLB_3_high_latency_tlb`
- `T_TLB_4_backpressure`
- `T_MMU_DMA_E2E_1_multipage_bulk`
- `T_MMU_DMA_E2E_2_concurrent_cmds_distinct_pages`

`DMA_fence_scheduler_test.scala` 测试对象：

- `DMA fence blocks until completion`
- `DMA issue allow deasserts at inflight limit and recovers on completion`
- `Multi-warp DMA + fence interleave: fence on warp0 does not block warp1`
- `Fence with zero inflight is a no-op`
- `All warps concurrent DMA then fence: all block, then release one by one`

`TMA_core_test.scala` 测试对象：

- `TMA_T1_2d_aligned`
- `TMA_T4_3d_carry`
- `TMA_T2_2d_oob_zero`
- `TMA_T10_back2back_B2`
- `TMA_T12_5d`
- `TMA_T20_2d_datawidth2_row_attribution`
- `TMA_T21_2d_datawidth1_row_attribution`
- `TMA_T23_2d_datawidth2_batched_response`
- `TMA_T22_2d_subbox_row_attribution`
- `TMA_T24_2d_estride2_dim0_gather`
- `TMA_T25_2d_padded_rows_row_attribution`
- `TMA_T26_descriptor_g2s_fetches_descriptor_and_coords`
- `TMA_T27_prefetch_tensormap_drops_payload_and_completes`
- `TMA_T28_prefetch_releases_issue_path_before_response`
- `TMA_T29_descriptor_cache_reuses_fetched_descriptor`

说明：

- app-level GVM testcase 验证端到端功能。
- `TMA_T28` 明确验证 prefetch 发出后 issue path 在 L2 response 前释放。
- `TMA_T29` 明确验证 descriptor cache 命中后 descriptor L2 fetch 被跳过。

## Spike/GVM Model 和仿真项目

Spike 指令模型：

```text
spike/riscv/insns/
├── cp_async_bulk.h
├── cp_async_copysize.h
├── cp_async_fence.h
├── cp_async_tensor.h
└── cp_async_tensor_g2s.h
```

GVM reference：

```text
spike/gvmref/
├── gvmref.cc
├── gvmref.h
├── gvmref_interface.cc
├── gvmref_interface.h
├── workgroup.cc
└── workgroup.h
```

GVM/Verilator 项目：

```text
gpgpu/sim-verilator/
├── gvm.mk
├── gvm.cpp
├── gvm.hpp
├── gvm_dpic.cpp
├── gvm_dpic.hpp
├── gvm_care_insns.cpp
├── gvm_global_var.cpp
├── gvm_global_var.hpp
├── gvm_macro.h
├── gvm_structs.hpp
└── gvm-log-script.py

gpgpu/sim-verilator-nocache/
├── gvm.mk
├── gvm.cpp
├── gvm.hpp
├── gvm_dpic.cpp
├── gvm_dpic.hpp
├── gvm_care_insns.cpp
├── gvm_global_var.cpp
├── gvm_global_var.hpp
├── gvm_macro.h
├── gvm_structs.hpp
└── gvm-log-script.py
```

`pocl/build/examples/{dma_test,copysize_test,multi_wg_dma_test,tensor_dma_test,tma_matrix_test}/`
是 CMake build tree 中已有的生成/构建输出目录，不是当前 DMA/TMA directed testcase 的源码权威位置。
当前维护的 app-level testcase 源码以 `testcases/_get_case/` 为准。

## 当前验证快照

最近一次 directed suite 结果：

| backend | summary | log dir |
| --- | --- | --- |
| Spike | `backend=spike pass=5 fail=0` | `/tmp/codex-dma-tma-directed-spike-20260513-233547` |
| GVM | `backend=gvm pass=5 fail=0` | `/tmp/codex-dma-tma-directed-gvm-20260513-235608` |

逻辑子对象快照：

- Spike：`tma_descriptor_test` 2 pass；`tma_matrix_test` 15 pass / 0 fail / 8 skip；
  `bulk_dma_matrix_test` 4 pass；`multi_warp_dma_fence_test` 4 pass；
  `dma_shared_routing_conflict_test` 2 pass。
- GVM：`tma_descriptor_test` 2 pass；`tma_matrix_test` 23 pass / 0 fail / 0 skip；
  `bulk_dma_matrix_test` 4 pass；`multi_warp_dma_fence_test` 4 pass；
  `dma_shared_routing_conflict_test` 2 pass。

case 数量会随覆盖扩展变化。判断回归时以当前运行输出的 `[n/m]`、`PASS/FAIL/SKIP` 和最终 summary 为准。
