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
  `half`, each `16 x 16` dot tile is computed with WMMA, distances are recovered
  from `||a||² + ||b||² - 2ABᵀ`, and top-1 is updated immediately without
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
v1       223 ms   462.24 GFLOP/s    28.93 GB/s
v2        83 ms  1241.92 GFLOP/s    51.85 GB/s
v3        68 ms  1515.87 GFLOP/s    63.28 GB/s
v4       290 ms   355.45 GFLOP/s    59.27 GB/s
v5        98 ms  1051.83 GFLOP/s    43.91 GB/s
v5a       64 ms  1610.61 GFLOP/s    67.24 GB/s
v5b       66 ms  1561.81 GFLOP/s    65.20 GB/s
v5c      103 ms  1000.77 GFLOP/s    41.78 GB/s
v6        67 ms  1538.50 GFLOP/s    64.23 GB/s
v7        63 ms  1636.18 GFLOP/s    68.31 GB/s
v8       115 ms   896.34 GFLOP/s    18.71 GB/s
v9       127 ms   547.44 GFLOP/s    34.38 GB/s
```

All runs reported 0 mismatches. The compute and bandwidth columns are benchmark
estimates based on the algorithm's modeled FLOPs and memory traffic, not Nsight
hardware counters.

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
