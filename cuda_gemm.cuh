#include <cuda_runtime.h>

template<class T, class Impl>
struct GEMMCUDAWrap {
  static void gemm(size_t m, size_t n, size_t k, T const* alpha, T const* A,
                   size_t lda, T const* B, size_t ldb, T const* beta, T* C,
                   size_t ldc) {
    // Allocate memory on device.
    T* A_device{nullptr};
    T* B_device{nullptr};
    T* C_device{nullptr};
    (cudaMalloc(&A_device, m * lda * sizeof(T)));
    (cudaMalloc(&B_device, k * ldb * sizeof(T)));
    (cudaMalloc(&C_device, m * ldc * sizeof(T)));

    (cudaMemcpy(A_device, A, m * lda * sizeof(T),
                                cudaMemcpyHostToDevice));
    (cudaMemcpy(B_device, B, k * ldb * sizeof(T),
                                cudaMemcpyHostToDevice));
    (cudaMemcpy(C_device, C, m * ldc * sizeof(T),
                                cudaMemcpyHostToDevice));

    Impl::GemmCuda(m, n, k, alpha, A_device, lda, B_device, ldb, beta, C_device, ldc);

    (cudaMemcpy(C, C_device, m * ldc * sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
  }
};

// GEMM kernel v00.
// Non-coalesced read and write from global memory.
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}
template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
   
}
template<class T>
struct GemmV00 {
  static void GemmCuda(size_t m, size_t n, size_t k, T const* alpha, T const* A,
                       size_t lda, T const* B, size_t ldb, T const* beta, T* C,
                       size_t ldc) {
    launch_gemm_kernel_v00<float>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};
