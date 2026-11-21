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
pkg_check_modules(SENTENCEPIECE REQUIRED IMPORTED_TARGET sentencepiece)
# Homebrew sentencepiece needs abseil, which its .pc file does not list. Distro
# packages (e.g. Ubuntu's libsentencepiece-dev) may not ship absl_status.pc at all.
pkg_check_modules(ABSL_STATUS IMPORTED_TARGET absl_status)

# The pkg-config imported target reliably supplies the include dirs, but some distro
# .pc files (notably Ubuntu's libsentencepiece-dev) do not put -lsentencepiece on the
# default, non-static link line, so the imported target alone leaves the symbols
# undefined at link time. Locate the library explicitly and link it as well.
find_library(SENTENCEPIECE_LIBRARY NAMES sentencepiece
  HINTS ${SENTENCEPIECE_LIBDIR} ${SENTENCEPIECE_LIBRARY_DIRS})
message(STATUS "sentencepiece: imported libs='${SENTENCEPIECE_LINK_LIBRARIES}' "
               "libdir='${SENTENCEPIECE_LIBDIR}' explicit lib='${SENTENCEPIECE_LIBRARY}'")

file(GLOB TINYLLM_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(tinyllm STATIC ${TINYLLM_SRC})
target_include_directories(tinyllm PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tinyllm PUBLIC PkgConfig::SENTENCEPIECE)
if(SENTENCEPIECE_LIBRARY)
  target_link_libraries(tinyllm PUBLIC ${SENTENCEPIECE_LIBRARY})
endif()
if(ABSL_STATUS_FOUND)
  target_link_libraries(tinyllm PUBLIC PkgConfig::ABSL_STATUS)
endif()

add_executable(tinyllm-cli tools/cli.cpp)
target_link_libraries(tinyllm-cli PRIVATE tinyllm)

add_executable(tinyllm-convert tools/convert.cpp)
target_link_libraries(tinyllm-convert PRIVATE tinyllm)

add_executable(tinyllm-bench-cpu bench/bench_cpu.cpp)
target_link_libraries(tinyllm-bench-cpu PRIVATE tinyllm)

if(TINYLLM_CUDA)
  add_subdirectory(cuda)
endif()

if(TINYLLM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
