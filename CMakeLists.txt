cmake_minimum_required(VERSION 3.20)
project(streamscore CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")

# Directory-level compile options only reach targets created AFTER them, so
# these must precede every add_executable below.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

if(NOT APPLE)
  add_compile_options(-march=native)
endif()
add_compile_options(-ffp-contract=off -fno-math-errno)

option(NO_VECTORIZE "Disable compiler auto-vectorization" OFF)
if(NO_VECTORIZE)
  add_compile_options(-fno-vectorize -fno-slp-vectorize)
endif()

set(APPLE_SILICON OFF)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  set(APPLE_SILICON ON)
endif()

# NeonBackend's vdotq_s32 needs the dotprod extension on any ARM target.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  add_compile_options(-march=armv8.2-a+dotprod)
endif()

add_executable(ingest src/ingest.cpp)
target_include_directories(ingest PRIVATE include)

add_executable(book src/book.cpp)
target_include_directories(book PRIVATE include)

add_executable(replay src/replay.cpp)
target_include_directories(replay PRIVATE include)

add_executable(infer_dump src/infer_dump.cpp)
target_include_directories(infer_dump PRIVATE include)

add_executable(infer_int8 src/infer_int8.cpp)
target_include_directories(infer_int8 PRIVATE include)

add_executable(schedule src/schedule.cpp)
target_include_directories(schedule PRIVATE include)

add_executable(replay_all src/replay_all.cpp)
target_include_directories(replay_all PRIVATE include)

# add_executable(bookprofile src/bookprofile.cpp)
# target_include_directories(bookprofile PRIVATE include)

# These call NEON backends directly and have no portable fallback.
if(APPLE_SILICON)
  add_executable(infer_neon src/infer_neon.cpp)
  target_include_directories(infer_neon PRIVATE include)

  add_executable(bench src/bench.cpp)
  target_include_directories(bench PRIVATE include)

  add_executable(batchtest src/batchtest.cpp)
  target_include_directories(batchtest PRIVATE include)
endif()

# Avx512Backend test/bench. Only if -march=native on this box turns on VNNI.
if(NOT APPLE)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-march=native")
  check_cxx_source_compiles("
    #if !(defined(__AVX512F__) && defined(__AVX512VNNI__))
    #error no vnni
    #endif
    int main() { return 0; }" HAVE_AVX512_VNNI)
  unset(CMAKE_REQUIRED_FLAGS)
  if(HAVE_AVX512_VNNI)
    add_executable(avx512test src/avx512test.cpp)
    target_include_directories(avx512test PRIVATE include)
  endif()
endif()

# CUDA backend. Optional: skipped cleanly on machines without nvcc.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 6000 Ada
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -lineinfo")

  add_library(model_cuda STATIC src/model_cuda.cu)
  target_include_directories(model_cuda PUBLIC include)

  add_executable(cudabench src/cudabench.cpp)
  target_link_libraries(cudabench PRIVATE model_cuda)
else()
  message(STATUS "No CUDA compiler found -- skipping model_cuda and cudabench")
endif()
