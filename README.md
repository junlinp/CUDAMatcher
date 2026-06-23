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
bool BenchmarkKernelV1(...);
```

`Descriptor` is `std::array<float, 128>`. The current CUDA matcher expects
`lhs.size() == rhs.size()` and a descriptor count divisible by 256.

## Version

- `v1`: computes the full pairwise distance matrix, then runs a second CUDA
  kernel to find each nearest neighbor.

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

```text
version  wall best  wall avg  device best  algorithmic compute  logical bytes
v1        81.347 ms   86.836 ms   81.355 ms  1267.15 GFLOP/s      79.30 GB/s
```

The end-to-end benchmark is measured with 1 warmup run and 5 measured runs. The
run above reported 0 mismatches. The algorithmic compute column uses modeled
matching FLOPs divided by wall-best time. The logical bytes column uses modeled
algorithm-level bytes divided by wall-best time. Logical bytes are not measured
DRAM bandwidth.

The benchmark also includes kernel-only timing for v1. In this mode descriptors
are allocated and copied once; then only repeated kernel launches are timed with
CUDA events:

```text
version  kernel best  kernel avg  algorithmic compute  logical bytes
v1        54.836 ms    57.505 ms  1879.78 GFLOP/s      117.64 GB/s
```

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

Optimization direction for v1:

- Fuse distance computation and nearest-neighbor reduction so the full matrix is
  never written to or reread from DRAM.
- Keep only per-left-descriptor best score/index in registers or shared memory
  while scanning right-hand tiles.
- Preserve the existing `float4` descriptor loads, but remove the
  `distance_matrix` allocation and the `CloseElementV1` pass.
- Use warp-level reductions for each row's tile-local top-1 result instead of
  writing tile scores through shared-memory transpose buffers.
- After the fused FP32 path is correct, consider `half` storage or WMMA dot
  products only if descriptor precision can change.

These steps remove roughly one full-matrix write plus one full-matrix read per
match and target the measured v1 bottleneck directly. The first optimization
should stay FP32 and exact relative to v1 so the effect of eliminating matrix
materialization can be measured cleanly.

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
