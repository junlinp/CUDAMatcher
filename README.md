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
- `v4`: direct warp-per-descriptor nearest-neighbor scan. It does not allocate a
  full pairwise distance matrix or a per-tile score matrix; each warp keeps only
  the current best distance/index for one left descriptor. This version is a
  no-matrix baseline and is slower than v3 because it gives up shared-memory
  reuse of RHS descriptor tiles.

On an NVIDIA GeForce RTX 3050 Ti Laptop GPU, the matcher benchmark for
`16,384 x 16,384` descriptors produced these sample runs:

```text
v1: 285 ms, 184 ms, 180 ms
v2: 83 ms, 81 ms, 77 ms
v3: 66 ms, 63 ms, 63 ms
v4: 355 ms, 353 ms, 351 ms
```

All runs reported 0 mismatches. Median timings were 184 ms for v1, 81 ms for
v2, 63 ms for v3, and 353 ms for v4. On that run, v3 was 2.9x faster than v1
and 1.29x faster than v2. v4 confirms that avoiding every intermediate matrix
is not enough by itself; preserving tiled data reuse is more important for this
benchmark.

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

The benchmark generates one descriptor set and compares v1, v2, v3, and v4 on
the same data.

Run the GEMM tests:

```bash
./build/gemm_test
```

The matcher requires a visible CUDA-capable NVIDIA GPU at runtime. If CUDA
cannot access the GPU, `Match` returns `false` instead of continuing after the
runtime error.
