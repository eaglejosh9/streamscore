cmake_minimum_required(VERSION 3.20)
project(streamscore CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
add_compile_options(-Wall -Wextra)

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

add_executable(infer_neon src/infer_neon.cpp)
target_include_directories(infer_neon PRIVATE include)

add_executable(bench src/bench.cpp)
target_include_directories(bench PRIVATE include)

add_executable(schedule src/schedule.cpp)
target_include_directories(schedule PRIVATE include)

add_executable(replay_all src/replay_all.cpp)
target_include_directories(replay_all PRIVATE include)

# add_executable(bookprofile src/bookprofile.cpp)
# target_include_directories(bookprofile PRIVATE include)

option(NO_VECTORIZE "Disable compiler auto-vectorization" OFF)
if(NO_VECTORIZE)
  add_compile_options(-fno-vectorize -fno-slp-vectorize)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  add_compile_options(-march=armv8.2-a+dotprod)
endif()