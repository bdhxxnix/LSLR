cmake_minimum_required(VERSION 3.16)
project(DynamicOptimalPLA VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CORE_SOURCES
    src/lslr_pla.cpp
    src/baselines.cpp
    src/data_generator.cpp
)

add_executable(experiments ${CORE_SOURCES} main.cpp)
add_executable(benchmark_full ${CORE_SOURCES} benchmark_full.cpp)
add_executable(debug_precision ${CORE_SOURCES} debug_precision.cpp)

# Comparison: LSLR-PLA vs DynamicLearnedIndex baseline
add_executable(compare ${CORE_SOURCES} baseline_wrapper.cpp compare.cpp)
target_include_directories(compare PRIVATE ${CMAKE_SOURCE_DIR}/baseline)
target_compile_options(compare PRIVATE -Wno-unused-parameter)
add_executable(debug_baseline ${CORE_SOURCES} baseline_wrapper.cpp debug_baseline.cpp)
target_include_directories(debug_baseline PRIVATE ${CMAKE_SOURCE_DIR}/baseline)
target_compile_options(debug_baseline PRIVATE -Wno-unused-parameter)
