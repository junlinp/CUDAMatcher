#include "iostream"
template<class T>
struct GEMM_CPU{
static void gemm(size_t m, size_t n, size_t k, T const* alpha, T const* A, size_t lda,
          T const* B, size_t ldb, T const* beta, T* C, size_t ldc) {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    T sum{0};
                    for (int l = 0; l < k; l++) {
                        sum += A[i * lda + l] * B[l * ldb + j];
                    }
                    C[i *ldc + j] = *alpha * sum  + *beta * C[i * ldc + j];
                }
            }
          }
};
