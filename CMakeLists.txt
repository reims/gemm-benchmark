cmake_minimum_required(VERSION 3.11)

project(gemm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(gemm
  gemm.cpp
  )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(gemm PRIVATE Threads::Threads)

target_compile_options(gemm PRIVATE -O2 -mavx -Wpedantic -Werror -Wextra -Wall -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function)

find_package(benchmark REQUIRED)
target_link_libraries(gemm PRIVATE benchmark::benchmark)

