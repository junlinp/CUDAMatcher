#include "cuda_gemm.cuh"

void gemm_cuda(size_t m, size_t n, size_t k, float const *alpha, float const *A,
               size_t lda, float const *B, size_t ldb, float const *beta,
               float *C, size_t ldc) {
  GEMMCUDAWrap<float, GemmV00<float>>::gemm(m, n, k, alpha, A, k, B, n, beta,
                                            C, n);
}