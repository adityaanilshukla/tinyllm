cmake_minimum_required(VERSION 3.20)
project(tinyllm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

option(TINYLLM_CUDA "Build CUDA kernels (needs nvcc)" OFF)
option(TINYLLM_BUILD_TESTS "Build the GoogleTest suite in tests/" ON)

find_package(PkgConfig REQUIRED)
# Homebrew sentencepiece needs abseil, which its .pc file does not list.
pkg_check_modules(SENTENCEPIECE REQUIRED IMPORTED_TARGET sentencepiece absl_status)

file(GLOB TINYLLM_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(tinyllm STATIC ${TINYLLM_SRC})
target_include_directories(tinyllm PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tinyllm PUBLIC PkgConfig::SENTENCEPIECE)

add_executable(tinyllm-cli tools/cli.cpp)
target_link_libraries(tinyllm-cli PRIVATE tinyllm)

add_executable(tinyllm-convert tools/convert.cpp)
target_link_libraries(tinyllm-convert PRIVATE tinyllm)

if(TINYLLM_CUDA)
  add_subdirectory(cuda)
endif()

if(TINYLLM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
