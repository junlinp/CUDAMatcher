cmake_minimum_required(VERSION 3.0)

project(CUDA_MATCH) 
set(CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)
find_library(CUDA_CUBLAS_LIBRARY
  NAMES cublas
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
)
if (NOT CUDA_CUBLAS_LIBRARY)
  message(FATAL_ERROR "Could not find cuBLAS library")
endif()

cuda_add_library(cuda_kernel match.cu)

add_executable(${PROJECT_NAME} main.cc)

target_link_libraries(${PROJECT_NAME} PUBLIC cuda_kernel ${CUDA_CUBLAS_LIBRARY})

add_executable(gemm_test gemm_test.cc)

cuda_add_library(cuda_gemm cuda_gemm.cuh cuda_gemm.cu)
target_link_libraries(gemm_test PUBLIC GTest::gtest cuda_gemm)
