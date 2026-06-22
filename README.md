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

On an NVIDIA GeForce RTX 3050 Ti Laptop GPU, the matcher benchmark for
`16,384 x 16,384` descriptors produced these sample runs:

```text
v1: 285 ms, 184 ms, 180 ms
v2: 83 ms, 81 ms, 77 ms
v3: 66 ms, 63 ms, 63 ms
```

All runs reported 0 mismatches. Median timings were 184 ms for v1, 81 ms for
v2, and 63 ms for v3. On that run, v3 was 2.9x faster than v1 and 1.29x faster
than v2.

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

The benchmark generates one descriptor set and compares v1, v2, and v3 on the
same data.

Run the GEMM tests:

```bash
./build/gemm_test
```

The matcher requires a visible CUDA-capable NVIDIA GPU at runtime. If CUDA
cannot access the GPU, `Match` returns `false` instead of continuing after the
runtime error.
