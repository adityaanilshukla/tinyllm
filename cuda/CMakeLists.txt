# CUDA kernels + benchmark. Enabled from the top level with -DTINYLLM_CUDA=ON.
cmake_minimum_required(VERSION 3.24)

# Must be decided before enable_language(CUDA), which otherwise picks the nvcc default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(tinyllm_cuda STATIC
  matmul.cu
  layernorm.cu
  softmax.cu
  attention.cu
)
target_include_directories(tinyllm_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tinyllm_cuda PUBLIC CUDA::cudart)
target_compile_options(tinyllm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(tinyllm_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF)

add_executable(tinyllm-bench-cuda ${CMAKE_SOURCE_DIR}/bench/bench_cuda.cpp)
target_link_libraries(tinyllm-bench-cuda PRIVATE tinyllm_cuda tinyllm CUDA::cublas CUDA::cudart)
