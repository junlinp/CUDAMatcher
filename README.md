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

Named implementations are also available for benchmarking:

```cpp
bool MatchV1(...);
bool MatchV2(...);
bool MatchV3(...);
bool MatchV4(...);
bool MatchV5(...);
bool MatchV5a(...);
bool MatchV5b(...);
bool MatchV5c(...);
bool MatchV6(...);
bool MatchV7(...);
bool MatchV8(...);
bool MatchV9(...);
bool MatchV10(...);
```

`v10` also exposes a persistent matcher context for repeated matching without
rebuilding FP16 descriptors and device buffers every call:

```cpp
struct MatchV10Context;
bool CreateMatchV10Context(...);
bool RunMatchV10Context(...);
void DestroyMatchV10Context(...);
```

`Descriptor` is `std::array<float, 128>`. The current CUDA matcher expects
`lhs.size() == rhs.size()` and a descriptor count divisible by 32.

## Versions

- `v1`: computes the full pairwise distance matrix, then runs a second CUDA
  kernel to find each nearest neighbor.
- `v2`: fuses tiled distance computation with nearest-neighbor reduction, so it
  avoids materializing the full `N x N` distance matrix.
- `v3`: keeps the v2 fused tiled computation and replaces the per-tile shared
  memory scan with warp-level nearest-neighbor reduction.
- `v4`: warp-per-descriptor nearest-neighbor scan with shared-memory tiles for
  the 8 left descriptors owned by a block and each 32-descriptor right tile. It
  does not allocate a full pairwise distance matrix or a per-tile score matrix;
  each warp keeps only the current best distance/index for one left descriptor.
- `v5`: alias of `v5c` for current default matcher.
- `v5a`: based on `ComputeNearestNeighborV3` and preserves the original
  shared-matrix transposition flow.
- `v5b`: keeps explicit `32 × 32` tiling and the shared-matrix flow,
  but is separated as its own kernel variant.
- `v5c`: a register-first rewrite of the v5 path, with direct `float4` loads
  into registers, simplified lane-per-row reduction logic, and reduced shared
  transposition.
- `v6`: tiled top-1 matcher. For each block, shared-memory tiles for right
  descriptors are loaded one-by-one and reused across a `32 x 32` left/right
  tile. The kernel updates the current best index/distance online, still
  without building an `N x N` matrix.
- `v7`: keeps the v6 `32 x 32` tiled top-1 layout and moves per-row best
  score/index tracking from shared memory into registers while preserving the
  bank-friendly shared-memory rotation step.
- `v8`: FP16 storage matcher. Input descriptors are converted to `half` on host,
  loaded as `half2` in the CUDA kernel, accumulated in FP32, and reduced online
  to top-1 without building an `N x N` matrix.
- `v9`: custom WMMA/Tensor Core online top-1 path. Descriptors are converted to
  `half`, each CTA owns a 16-row A tile and uses four warps to compute four
  `16 x 16` B subtiles with WMMA per scan step. Distances are recovered from
  `||a||² + ||b||² - 2ABᵀ`, and top-1 is updated immediately without
  materializing the full `N x N` dot matrix.
- `v10`: persistent v9 matcher context. FP16 descriptor conversion, norm
  computation, device allocation, and H2D copies are done once during context
  creation. Repeated runs launch the WMMA top-1 kernel and copy back only the
  best-index output.

On an NVIDIA GeForce RTX 3050 Ti Laptop GPU, the matcher benchmark for
`16,384 x 16,384` descriptors produced these sample runs:

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

```text
version  wall best  wall avg  device best  estimated compute  estimated bandwidth
v1        85.034 ms   89.621 ms   85.037 ms  1212.22 GFLOP/s    75.86 GB/s
v2        67.525 ms   71.835 ms   67.532 ms  1526.53 GFLOP/s    63.73 GB/s
v3        65.854 ms   74.328 ms   65.856 ms  1565.28 GFLOP/s    65.35 GB/s
v4       308.759 ms  314.538 ms  308.767 ms   333.85 GFLOP/s    55.67 GB/s
v5       104.893 ms  106.797 ms  104.901 ms   982.71 GFLOP/s    41.03 GB/s
v5a       61.298 ms   66.410 ms   61.303 ms  1681.60 GFLOP/s    70.20 GB/s
v5b       66.539 ms   67.639 ms   66.544 ms  1549.15 GFLOP/s    64.67 GB/s
v5c      105.513 ms  108.305 ms  105.517 ms   976.94 GFLOP/s    40.79 GB/s
v6        66.290 ms   67.780 ms   66.294 ms  1554.98 GFLOP/s    64.92 GB/s
v7        62.045 ms   64.096 ms   62.049 ms  1661.37 GFLOP/s    69.36 GB/s
v8       113.970 ms  116.848 ms  113.974 ms   904.44 GFLOP/s    18.88 GB/s
v9       128.293 ms  132.160 ms  128.297 ms   541.92 GFLOP/s    34.03 GB/s
v10      129.199 ms  135.307 ms  129.203 ms   538.12 GFLOP/s    33.80 GB/s
```

Each version is measured with 1 warmup run and 5 measured runs. All runs
reported 0 mismatches. The compute and bandwidth columns use wall-best time and
are benchmark estimates based on the algorithm's modeled FLOPs and memory
traffic, not Nsight hardware counters. The benchmark also prints CUDA event
device average/best time and device-side estimated compute/bandwidth for each
version.

The benchmark also includes kernel-only timing for all versions. In this mode
descriptors are allocated, converted, and copied once; then only repeated kernel
launches are timed with CUDA events:

```text
version  kernel best  kernel avg  estimated compute  estimated bandwidth
v1        57.584 ms    60.237 ms  1790.08 GFLOP/s   112.03 GB/s
v2        56.878 ms    61.422 ms  1812.28 GFLOP/s    75.66 GB/s
v3        54.403 ms    58.413 ms  1894.73 GFLOP/s    79.10 GB/s
v4       293.575 ms   302.951 ms   351.12 GFLOP/s    58.55 GB/s
v5        96.031 ms    98.830 ms  1073.40 GFLOP/s    44.81 GB/s
v5a       57.008 ms    59.526 ms  1808.15 GFLOP/s    75.49 GB/s
v5b       56.396 ms    59.699 ms  1827.78 GFLOP/s    76.31 GB/s
v5c       92.117 ms    99.334 ms  1119.00 GFLOP/s    46.72 GB/s
v6        55.529 ms    59.853 ms  1856.30 GFLOP/s    77.50 GB/s
v7        55.146 ms    58.132 ms  1869.19 GFLOP/s    78.03 GB/s
v8        51.559 ms    57.772 ms  1999.23 GFLOP/s    41.73 GB/s
v9        20.703 ms    24.220 ms  3358.16 GFLOP/s   210.90 GB/s
v10       20.256 ms    24.171 ms  3432.34 GFLOP/s   215.56 GB/s
```

Kernel-only timing shows the WMMA/Tensor Core v9 kernel is the fastest path once
host conversion, allocation, H2D copies, D2H copies, and per-call setup are
removed. The remaining end-to-end bottleneck for v9 is the preparation path, not
the matching kernel itself.

The v10 persistent context removes that repeated preparation from steady-state
matching:

```text
v10 persistent create time: 121.157 ms
v10 persistent run best: 21.605 ms
v10 persistent run avg: 23.884 ms
v10 persistent estimated compute: 3218.00 GFLOP/s
v10 persistent estimated bandwidth: 202.10 GB/s
```

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

The benchmark generates one descriptor set and compares v1 through v9 (including
v5a/v5b/v5c) on the same data.

Run the GEMM tests:

```bash
./build/gemm_test
```

The matcher requires a visible CUDA-capable NVIDIA GPU at runtime. If CUDA
cannot access the GPU, `Match` returns `false` instead of continuing after the
runtime error.
