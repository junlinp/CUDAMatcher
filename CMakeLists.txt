cmake_minimum_required(VERSION 3.0)

project(CUDA_MATCH) 
set(CXX_STANDARD 11)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_86,code=sm_86)
find_package(GTest REQUIRED)

cuda_add_library(cuda_kernel match.cu)

add_executable(${PROJECT_NAME} main.cc)

target_link_libraries(${PROJECT_NAME} PUBLIC cuda_kernel)

add_executable(gemm_test gemm_test.cc)

cuda_add_library(cuda_gemm cuda_gemm.cuh cuda_gemm.cu)
target_link_libraries(gemm_test PUBLIC GTest::gtest cuda_gemm)
