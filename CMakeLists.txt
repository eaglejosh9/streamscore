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