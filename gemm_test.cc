#include "gtest/gtest.h"
#include <random>
#include "gemm.h"
#include "cuda_gemm.h"
#include "random"

int m = 1024;
int n = 1024;
int k = 1024;

float* A = nullptr;
float* B = nullptr;
float* C = nullptr;
float* D = nullptr;
float* target_result = nullptr;

void InitializationData() {
    A = new float[m * k];
    B = new float[n * k];
    C = new float[m * n];
    D = new float[m * n];
    target_result = new float[m * n];

    std::default_random_engine seeds;
    std::uniform_real_distribution<float> distribute(0, 5);
    for (int i = 0; i < m * k; i++) {
        A[i] = distribute(seeds);
    }
    for (int i = 0; i < n * k; i++) {
        B[i] = distribute(seeds);
    }
    for (int i = 0; i < m * n; i++) {
        C[i] = distribute(seeds);
        D[i] = C[i];
        target_result[i] = C[i];
    }
    float alpha = 1.0;
    float beta = 1.0;
    GEMMCPUFloat::gemm(m, n, k, &alpha, A, k, B, n, &beta, target_result, n);
}

void RecoveryData() {
    for (int i = 0; i < m * n; i++) {
        C[i] = D[i];
    }
}

void CheckResult() {
    for (int i = 0; i < m * n; i++) {
      ASSERT_NEAR((target_result[i] - C[i]) / target_result[i], 0.0, 1e-6);
    }
}

TEST(GEMMCPU, Basic) {
    float A = 1.0;
    float B = 2.0;
    float C = 3.0;
    float alpha = 1.0;
    float beta = 1.0;
    GEMMCPUFloat::gemm(1, 1, 1, &alpha, &A, 1, &B, 1, &beta, &C, 1);
    EXPECT_NEAR(C, 5.0, 1e-6);
}


TEST(GEMMCPU, Random) {
    float alpha = 1.0;
    float beta = 1.0;
    RecoveryData();
    GEMMCPUFloat::gemm(m, n, k, &alpha, A, k, B, n, &beta, C, n);
    CheckResult();
}

TEST(GEMMCUDA, Random) {
    float alpha = 1.0;
    float beta = 1.0;
    RecoveryData();
    gemm_cuda(m, n, k, &alpha, A, k, B, n, &beta, C, n);
    CheckResult();
}
int main(int argc, char** argv) {
    InitializationData();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}