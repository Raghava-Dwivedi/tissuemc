cmake_minimum_required(VERSION 3.24)
project(tissuemc VERSION 0.1.0 LANGUAGES CXX)

option(TISSUEMC_BUILD_TESTS "Build unit tests" ON)
option(TISSUEMC_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(TISSUEMC_TSAN "Build with ThreadSanitizer (exclusive with TISSUEMC_SANITIZE)" OFF)
option(TISSUEMC_ENABLE_CUDA "Build the CUDA engine (requires nvcc)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall$<SEMICOLON>-Wextra$<SEMICOLON>-Wpedantic$<SEMICOLON>-Wshadow$<SEMICOLON>-Wconversion$<SEMICOLON>-Wno-sign-conversion>)
if(TISSUEMC_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(TISSUEMC_TSAN)
  if(TISSUEMC_SANITIZE)
    message(FATAL_ERROR "TISSUEMC_TSAN and TISSUEMC_SANITIZE cannot be combined")
  endif()
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

find_package(Threads REQUIRED)

# Header-only physics shared by the CPU and CUDA engines.
add_library(tissuemc_core INTERFACE)
target_include_directories(tissuemc_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(tissuemc_core INTERFACE TISSUEMC_VERSION="${PROJECT_VERSION}")

add_library(tissuemc_cpu STATIC
  src/io/mci.cpp
  src/io/json_out.cpp
  src/cpu/engine.cpp
)
target_link_libraries(tissuemc_cpu PUBLIC tissuemc_core Threads::Threads)

add_executable(tissuemc src/cli/main.cpp)
target_link_libraries(tissuemc PRIVATE tissuemc_cpu)

if(TISSUEMC_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 75) # Turing: the free Colab and Kaggle T4
  endif()
  add_library(tissuemc_cuda STATIC src/cuda/engine.cu)
  set_target_properties(tissuemc_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                                                  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(tissuemc_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  target_link_libraries(tissuemc_cuda PUBLIC tissuemc_core CUDA::cudart)
  target_link_libraries(tissuemc PRIVATE tissuemc_cuda)
  target_compile_definitions(tissuemc PRIVATE TISSUEMC_WITH_CUDA=1)
endif()

if(TISSUEMC_BUILD_TESTS)
  include(FetchContent)
  # doctest is a single header; fetch only the header so its own CMake files are never run.
  FetchContent_Declare(doctest
    URL https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h
    DOWNLOAD_NO_EXTRACT TRUE
  )
  FetchContent_MakeAvailable(doctest)

  enable_testing()
  add_executable(tissuemc_tests
    tests/cpp/test_main.cpp
    tests/cpp/test_version.cpp
    tests/cpp/test_accumulate.cpp
    tests/cpp/test_philox.cpp
    tests/cpp/test_physics.cpp
    tests/cpp/test_mci.cpp
    tests/cpp/test_engine.cpp
    tests/cpp/test_gpu_emulation.cpp
  )
  target_include_directories(tissuemc_tests PRIVATE ${doctest_SOURCE_DIR})
  target_link_libraries(tissuemc_tests PRIVATE tissuemc_cpu)
  target_compile_definitions(tissuemc_tests PRIVATE
    TISSUEMC_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/configs")
  add_test(NAME tissuemc_tests COMMAND tissuemc_tests)
endif()
