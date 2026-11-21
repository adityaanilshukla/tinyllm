# GoogleTest suite. Prefers an installed GTest (brew install googletest) and falls back
# to fetching a pinned release.
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

file(GLOB TINYLLM_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(tinyllm-tests ${TINYLLM_TEST_SRC})
target_link_libraries(tinyllm-tests PRIVATE tinyllm GTest::gtest_main)
target_compile_options(tinyllm-tests PRIVATE -Wall -Wextra)
# Model-dependent tests skip when these files are missing. Both paths can be overridden
# at run time with TINYLLM_MODELS_DIR / TINYLLM_TEST_TMPDIR.
target_compile_definitions(tinyllm-tests PRIVATE
  TINYLLM_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
  TINYLLM_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}/tmp"
  TINYLLM_CONVERT_BIN="$<TARGET_FILE:tinyllm-convert>")
add_dependencies(tinyllm-tests tinyllm-convert)

gtest_discover_tests(tinyllm-tests DISCOVERY_TIMEOUT 30)
