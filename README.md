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
- `v5`: optimized no-distance-matrix matcher. It keeps the current best
  distance/index like v4, but uses the tiled warp-reduction path from v3 to
  increase useful work per shared-memory tile. `Match(...)` currently defaults
  to v5.
- `v6`: tiled top-1 matcher. For each block, shared-memory tiles for right
  descriptors are loaded one-by-one and each warp directly maintains the current
  best index/distance, still without building an `N x N` matrix.
- `v7`: each A-block scans right-side B blocks end-to-end and updates best in
  one pass. The descriptor row is loaded once per block into shared memory and
  reused across all B tiles.
- `v8`: memory-bandwidth-oriented matcher based on `uint8` descriptors. Input
  descriptors are converted to `uint8` on host before transfer and distance is
  computed with integer arithmetic and top-1 reduction.
- `v9`: Tensor Core/WMMA path using cuBLAS-GEMM. Descriptors are converted to
  `half`, dot-product matrix is computed in GEMM using tensor-core-capable
  operations, and nearest-neighbor top-1 is recovered from
  `||a||² + ||b||² - 2ABᵀ`; the closest index is reduced per row after GEMM.

On an NVIDIA GeForce RTX 3050 Ti Laptop GPU, the matcher benchmark for
`16,384 x 16,384` descriptors produced these sample runs:

```text
v1: 214 ms, 214 ms, 214 ms
v2: 93 ms, 93 ms, 93 ms
v3: 69 ms, 69 ms, 69 ms
v4: 319 ms, 319 ms, 319 ms
v5: 64 ms, 64 ms, 64 ms
v6: 327 ms, 327 ms, 327 ms
v7: 315 ms, 315 ms, 315 ms
v8: 1165 ms, 1165 ms, 1165 ms
v9: 327 ms, 327 ms, 327 ms
```

All runs reported 0 mismatches. On this device, v6–v9 are all correct.

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

The benchmark generates one descriptor set and compares v1 through v5 on the
same data.

Run the GEMM tests:

```bash
./build/gemm_test
```

The matcher requires a visible CUDA-capable NVIDIA GPU at runtime. If CUDA
cannot access the GPU, `Match` returns `false` instead of continuing after the
runtime error.
