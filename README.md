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
version  wall best  wall avg  device best  estimated compute  estimated bandwidth
v1        81.555 ms   87.658 ms   81.562 ms  1263.93 GFLOP/s    79.10 GB/s
v2        65.470 ms   69.913 ms   65.479 ms  1574.45 GFLOP/s    65.73 GB/s
v3        60.688 ms   66.499 ms   60.694 ms  1698.51 GFLOP/s    70.91 GB/s
v4       312.868 ms  316.664 ms  312.873 ms   329.47 GFLOP/s    54.94 GB/s
v5       107.369 ms  108.835 ms  107.374 ms   960.04 GFLOP/s    40.08 GB/s
v5a       62.874 ms   67.123 ms   62.883 ms  1639.47 GFLOP/s    68.44 GB/s
v5b       64.141 ms   66.917 ms   64.146 ms  1607.08 GFLOP/s    67.09 GB/s
v5c      107.348 ms  109.202 ms  107.355 ms   960.24 GFLOP/s    40.09 GB/s
v6        63.917 ms   68.260 ms   63.845 ms  1612.70 GFLOP/s    67.33 GB/s
v7        65.120 ms   66.384 ms   65.127 ms  1582.90 GFLOP/s    66.08 GB/s
v8       115.316 ms  118.791 ms  115.320 ms   893.88 GFLOP/s    18.66 GB/s
v9       125.417 ms  126.783 ms  125.422 ms   554.35 GFLOP/s    34.81 GB/s
```

Each version is measured with 1 warmup run and 5 measured runs. All runs
reported 0 mismatches. The compute and bandwidth columns use wall-best time and
are benchmark estimates based on the algorithm's modeled FLOPs and memory
traffic, not Nsight hardware counters. The benchmark also prints CUDA event
device average/best time and device-side estimated compute/bandwidth for each
version.

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
