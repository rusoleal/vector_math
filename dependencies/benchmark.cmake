cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)

FetchContent_Declare(
    extern_benchmark

    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG        v1.9.4
)

FetchContent_GetProperties(extern_benchmark)

if(NOT extern_benchmark_POPULATED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    message(STATUS "Fetching benchmark...")
    FetchContent_Populate(extern_benchmark)
    include_directories(${extern_benchmark_SOURCE_DIR}/include)
    add_subdirectory(
        ${extern_benchmark_SOURCE_DIR}
        ${extern_benchmark_BINARY_DIR}
        EXCLUDE_FROM_ALL
    )
endif()
