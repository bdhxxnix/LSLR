cmake_minimum_required(VERSION 3.16)
project(DynamicOptimalPLA VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CORE_SOURCES
    src/optimal_pla.cpp
    src/lslr_pla.cpp
    src/baselines.cpp
    src/data_generator.cpp
)

add_executable(experiments ${CORE_SOURCES} main.cpp)
add_executable(benchmark_full ${CORE_SOURCES} benchmark_full.cpp)
