
#include "gemm_cpu.h"
template<class T, class Impl>
struct GEMM {
  static void gemm(size_t m, size_t n, size_t k, T const* alpha, T const* A,
            size_t lda, T const* B, size_t ldb, T const* beta, T* C,
            size_t ldc) {
    Impl::gemm(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

template<class T>
using GEMMCPU = GEMM<T, GEMM_CPU<T>>;

using GEMMCPUFloat = GEMMCPU<float>;