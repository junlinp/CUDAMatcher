# CUDAMatcher

CUDA experiments for descriptor matching and GEMM.

## Matcher

The matcher takes two vectors of 128-float descriptors and returns the nearest
right-hand descriptor for every left-hand descriptor:

```cpp
bool Match(const std::vector<Descriptor>& lhs,
           const std::vector<Descriptor>& rhs,
           std::vector<std::pair<int, int>>& match_result);
```

The v1 implementation is also exposed directly for benchmarking:

```cpp
bool MatchV1(...);
bool MatchV2(...);
bool MatchV3(...);
bool BenchmarkKernelV1(...);
bool BenchmarkKernelV2(...);
bool BenchmarkKernelV3(...);
```

`Descriptor` is `std::array<float, 128>`. The current CUDA matcher expects
`lhs.size() == rhs.size()`. The default v3 matcher expects a descriptor count
divisible by 1024; v1 expects divisibility by 256, and v2 expects divisibility
by 32.

## Version

- `v1`: computes the full pairwise distance matrix, then runs a second CUDA
  kernel to find each nearest neighbor.
- `v2`: fuses distance computation and nearest-neighbor reduction. It scans
  right-hand descriptor tiles and updates each left descriptor's best
  score/index online, avoiding the full `N x N` distance matrix.
- `v3`: tiled cuBLAS SGEMM path. It computes
  `dist(a, b) = ||a||^2 + ||b||^2 - 2ab^T`, calls cuBLAS for
  `A * B_tile^T`, reduces each dot-product tile online into top-1
  score/index, and discards the tile.

On an NVIDIA GeForce RTX 3050 Ti Laptop GPU, the matcher benchmark for
`16,384 x 16,384` descriptors produced this sample run.

Device theoretical peaks used for context:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
CUDA cores: 2560
Max graphics clock reported by nvidia-smi: 2100 MHz
Max memory clock reported by nvidia-smi: 6001 MHz
Memory bus: 128-bit GDDR6
FP32 CUDA peak: ~10.75 TFLOP/s
FP16 Tensor Core dense peak: ~43.01 TFLOP/s
FP16 Tensor Core sparse peak: ~86.02 TFLOP/s
Memory bandwidth: ~192.03 GB/s
```

The executable still prints CUDA-event and wall-clock timing as a correctness
and regression sanity check. Nsight Compute is used as the authoritative kernel
performance measurement because it reports kernel duration and hardware
counters.

Application timing from `./build/CUDA_MATCH`:

```text
version  wall best  wall avg  device best  algorithmic compute  logical bytes
v1        79.973 ms   89.610 ms   79.978 ms  1288.93 GFLOP/s      80.66 GB/s
v2        62.513 ms   65.505 ms   62.526 ms  1648.91 GFLOP/s      68.84 GB/s
v3        45.457 ms   47.986 ms   45.462 ms  2267.61 GFLOP/s     141.91 GB/s
```

Nsight Compute kernel timing:

```text
kernel                   duration   DRAM read   DRAM write  DRAM peak  L2 peak  SM peak  active warps
ComputeDistanceMatrixV1   56.57 ms  126.03 MB     1.07 GB     11.06%     9.67%   51.20%       33.24%
CloseElementV1             5.71 ms    1.07 GB   195.71 KB     98.13%    31.80%   36.20%       66.42%
V1 total                  62.28 ms    1.20 GB     1.07 GB          -         -        -            -
ComputeNearestNeighborV2  63.15 ms  123.35 MB   965.50 KB      1.03%     5.75%   52.05%       33.24%
ampere_sgemm_128x64_tn     1.20 ms    8.95 MB    66.08 MB     32.60%    37.43%   73.85%       32.84%
ReduceDotTileKernel        0.37 ms   67.29 MB   148.99 KB     94.63%    31.44%   28.53%       32.44%
```

The `ncu` result shows that v2 removes the full-matrix write/read traffic, but
the custom fused FP32 kernel is only roughly equal to v1 at the kernel level.
V3 is faster because cuBLAS raises the GEMM tile's SM throughput to about 74%.
For a 1024-descriptor B tile, one cuBLAS SGEMM costs about `1.20 ms` and the
online reduction over that dot tile costs about `0.37 ms`. With 16 B tiles, the
measured v3 kernel-only loop is about `25 ms`.

## Nsight Profile

Historical v9 WMMA profile, before the v2-v10 code was removed:

```text
ComputeNearestNeighborV9WMMA:
dram__throughput.avg.pct_of_peak_sustained_elapsed: 4.9% - 5.9%
dram__bytes_read.sum:                              240 MB - 290 MB
dram__bytes_write.sum:                             200 KB - 410 KB
lts__throughput.avg.pct_of_peak_sustained_elapsed: ~40.8%
sm__throughput.avg.pct_of_peak_sustained_elapsed:  ~30.1%
sm__warps_active.avg.pct_of_peak_sustained_active: ~93.3%
smsp__inst_executed_pipe_tensor.sum:               16,777,216
smsp__pipe_tensor_cycles_active...:                ~10.9%
```

The v9 result showed that the Tensor Core path was not limited by physical DRAM
bandwidth. It had high active-warps but low Tensor Core pipe activity, so the
bottleneck was the tiled online top-1 structure, shared-memory staging,
reductions, synchronization, and bookkeeping.

Current v1 Nsight Compute profile:

```text
ComputeDistanceMatrixV1:
dram__bytes_read.sum:                               118.25 MB
dram__bytes_write.sum:                                1.07 GB
dram__throughput.avg.pct_of_peak_sustained_elapsed:  10.98%
lts__throughput.avg.pct_of_peak_sustained_elapsed:    9.64%
sm__throughput.avg.pct_of_peak_sustained_elapsed:    51.20%
sm__warps_active.avg.pct_of_peak_sustained_active:   33.33%

CloseElementV1:
dram__bytes_read.sum:                                 1.07 GB
dram__bytes_write.sum:                              187.52 KB
dram__throughput.avg.pct_of_peak_sustained_elapsed:  98.14%
lts__throughput.avg.pct_of_peak_sustained_elapsed:   31.60%
sm__throughput.avg.pct_of_peak_sustained_elapsed:    36.20%
sm__warps_active.avg.pct_of_peak_sustained_active:   66.42%
```

V1 writes the full `N x N` distance matrix in `ComputeDistanceMatrixV1`, then
reads it back in `CloseElementV1`. For `16,384 x 16,384`, that matrix is about
1.07 GB. The second kernel is DRAM-bandwidth bound, reaching about 98% of peak
DRAM throughput, while the first kernel spends more time on FP32 arithmetic and
shared-memory data movement.

V2 changes the bottleneck. It has much lower DRAM traffic and only reaches
`0.97%` of peak DRAM throughput, so it is not memory-bandwidth bound. Its SM
throughput is close to v1's distance kernel, but active warps are still only
about `33.24%` and the kernel still uses the original shared-memory transpose
and score-buffer staging pattern.

Optimization direction:

- V2 already fuses distance computation and nearest-neighbor reduction so the
  full matrix is never written to or reread from DRAM.
- Keep only per-left-descriptor best score/index in registers or shared memory
  while scanning right-hand tiles.
- Preserve the existing `float4` descriptor loads, but remove the
  `distance_matrix` allocation and the `CloseElementV1` pass.
- Use warp-level reductions for each row's tile-local top-1 result instead of
  writing tile scores through shared-memory transpose buffers.
- After the fused FP32 path is correct, consider `half` storage or WMMA dot
  products only if descriptor precision can change.

The latest v2 optimization replaced the tile-score shared-memory scan with
warp-level reductions and then moved per-row best-score state from shared memory
to registers, reducing `ncu` duration from `69.37 ms` to `63.15 ms`. The next
optimization should remove more of the shared-memory `score_buffer` and
`rotation_score` staging path.

V3 changes the bottleneck again. The cuBLAS GEMM tile is compute-efficient, but
the reduction kernel is memory-bandwidth bound because it streams the
`N x tile` dot-product tile from global memory. The next useful v3 optimization
is increasing tile size or fusing more of the distance/top-1 update with GEMM
output when practical.

## Build

This project uses CMake with CUDA and GTest:

```bash
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j
```

The policy override is needed with newer CMake versions because this repository
still uses the legacy `FindCUDA` flow.

## Run

Run the matcher benchmark:

```bash
./build/CUDA_MATCH
```

Run the GEMM tests:

```bash
./build/gemm_test
```

The matcher requires a visible CUDA-capable NVIDIA GPU at runtime. If CUDA
cannot access the GPU, `Match` returns `false` instead of continuing after the
runtime error.
