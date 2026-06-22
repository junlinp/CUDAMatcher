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

On an NVIDIA GeForce RTX 3050 Ti Laptop GPU, the matcher benchmark for
`16,384 x 16,384` descriptors produced these sample runs:

```text
v1: 285 ms, 184 ms, 180 ms
v2: 83 ms, 81 ms, 77 ms
v3: 66 ms, 63 ms, 63 ms
v4: 300 ms, 295 ms, 305 ms
v5: 64 ms, 63 ms, 63 ms
```

All runs reported 0 mismatches. Median timings were 184 ms for v1, 81 ms for
v2, 63 ms for v3, 300 ms for v4, and 63 ms for v5. v5 was about 4.6x faster
than v4. v4 confirms that avoiding every intermediate matrix is not enough by
itself; the tile layout and amount of work reused per block matter heavily for
this benchmark.

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
