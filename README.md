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
version  time  estimated compute  estimated bandwidth
v1        74 ms  1392.96 GFLOP/s    87.17 GB/s
v2        77 ms  1338.69 GFLOP/s    55.89 GB/s
v3        65 ms  1585.83 GFLOP/s    66.21 GB/s
v4       298 ms   345.90 GFLOP/s    57.68 GB/s
v5        98 ms  1051.83 GFLOP/s    43.91 GB/s
v5a       61 ms  1689.82 GFLOP/s    70.55 GB/s
v5b       63 ms  1636.18 GFLOP/s    68.31 GB/s
v5c      102 ms  1010.58 GFLOP/s    42.19 GB/s
v6        66 ms  1561.81 GFLOP/s    65.20 GB/s
v7        59 ms  1747.11 GFLOP/s    72.94 GB/s
v8       112 ms   920.35 GFLOP/s    19.21 GB/s
v9       135 ms   515.00 GFLOP/s    32.34 GB/s
```

All runs reported 0 mismatches. The compute and bandwidth columns are benchmark
estimates based on the algorithm's modeled FLOPs and memory traffic, not Nsight
hardware counters. The benchmark also prints CUDA event device elapsed time and
device-side estimated compute/bandwidth for each version.

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
