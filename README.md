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
v1        87.207 ms   90.148 ms   87.215 ms  1182.01 GFLOP/s    73.97 GB/s
v2        65.358 ms   70.674 ms   65.365 ms  1577.16 GFLOP/s    65.84 GB/s
v3        66.277 ms   68.331 ms   66.284 ms  1555.28 GFLOP/s    64.93 GB/s
v4       318.585 ms  321.221 ms  318.592 ms   323.55 GFLOP/s    53.95 GB/s
v5       101.796 ms  107.189 ms  101.803 ms  1012.60 GFLOP/s    42.27 GB/s
v5a       67.537 ms   67.985 ms   67.545 ms  1526.26 GFLOP/s    63.72 GB/s
v5b       65.541 ms   68.874 ms   65.548 ms  1572.74 GFLOP/s    65.66 GB/s
v5c      106.434 ms  109.227 ms  106.440 ms   968.48 GFLOP/s    40.43 GB/s
v6        68.302 ms   71.027 ms   68.306 ms  1509.18 GFLOP/s    63.01 GB/s
v7        64.770 ms   67.044 ms   64.775 ms  1591.46 GFLOP/s    66.44 GB/s
v8       122.433 ms  124.483 ms  122.438 ms   841.92 GFLOP/s    17.57 GB/s
v9       124.043 ms  125.533 ms  124.046 ms   560.49 GFLOP/s    35.20 GB/s
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
v1        60.311 ms    63.436 ms  1709.14 GFLOP/s   106.96 GB/s
v2        58.186 ms    63.587 ms  1771.56 GFLOP/s    73.96 GB/s
v3        57.428 ms    60.955 ms  1794.93 GFLOP/s    74.93 GB/s
v4       306.674 ms   312.692 ms   336.12 GFLOP/s    56.05 GB/s
v5        99.880 ms   102.917 ms  1032.03 GFLOP/s    43.09 GB/s
v5a       56.109 ms    60.939 ms  1837.12 GFLOP/s    76.70 GB/s
v5b       56.189 ms    60.781 ms  1834.51 GFLOP/s    76.59 GB/s
v5c       94.597 ms    99.890 ms  1089.67 GFLOP/s    45.49 GB/s
v6        55.704 ms    59.456 ms  1850.50 GFLOP/s    77.25 GB/s
v7        52.630 ms    57.470 ms  1958.58 GFLOP/s    81.77 GB/s
v8        49.691 ms    56.410 ms  2074.41 GFLOP/s    43.30 GB/s
v9        19.715 ms    24.447 ms  3526.48 GFLOP/s   221.47 GB/s
```

Kernel-only timing shows the WMMA/Tensor Core v9 kernel is the fastest path once
host conversion, allocation, H2D copies, D2H copies, and per-call setup are
removed. The remaining end-to-end bottleneck for v9 is the preparation path, not
the matching kernel itself.

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
