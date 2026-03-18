cmake_minimum_required(VERSION 3.22.1)

project(vector_math VERSION 0.3.0)

configure_file(src/vector_math_config.h.in vector_math_config.h)

set(CMAKE_CXX_STANDARD 20)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

message(STATUS ${CMAKE_SYSTEM_PROCESSOR})

# enable avx simd extension for x86 processor family
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)")
    message(STATUS "Enabling AVX support")
    if(MSVC AND NOT MSVC_VERSION LESS 1600)
        message(STATUS "Enabling AVX2+FMA support for MSVC")
        set( CMAKE_CXX_FLAGS "/arch:AVX2")
    else()
        message(STATUS "Enabling AVX+FMA support for gcc/clang")
        set( CMAKE_CXX_FLAGS "-mavx -mfma")
    endif()
else ()
endif ()

option(VECTOR_MATH_BUILD_TEST "Enable building tests" OFF)

option(VECTOR_MATH_BUILD_BENCHMARK "Enable building benchmark" OFF)

include_directories(inc)

message(STATUS "Building ${PROJECT_NAME}...")

add_library(${PROJECT_NAME}
    src/vector_math.cpp
)

target_include_directories(vector_math PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS vector_math
    EXPORT vector_mathTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY inc/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vector_math_config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT vector_mathTargets
    FILE vector_mathTargets.cmake
    NAMESPACE vector_math::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vector_math
)

configure_package_config_file(
    cmake/vector_mathConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vector_mathConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vector_math
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vector_mathConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vector_mathConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vector_mathConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vector_math
)

if (VECTOR_MATH_BUILD_TEST)
    include(CTest)

    SET(BUILD_GMOCK OFF CACHE BOOL "DISABLE GMOCK")
    SET(INSTALL_GTEST OFF CACHE BOOL "DISABLE INSTALL_GTEST")
    include(dependencies/googletest.cmake)

    add_executable(${PROJECT_NAME}_test 
        test/main.cpp
    )
    target_link_libraries(${PROJECT_NAME}_test 
        ${PROJECT_NAME}
        gtest
    )
    add_test(NAME vector_math_test COMMAND ${PROJECT_NAME}_test)
    #add_test(NAME gltf_test2 COMMAND ${PROJECT_NAME}_test)
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
endif()

if (VECTOR_MATH_BUILD_BENCHMARK)
    SET(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "enable download dependencies")
    include(dependencies/benchmark.cmake)

    # comparison against glm
    include(dependencies/glm.cmake)

    add_executable(${PROJECT_NAME}_benchmark
        benchmark/main.cpp
    )
    target_link_libraries(${PROJECT_NAME}_benchmark
        ${PROJECT_NAME}
        benchmark::benchmark
        glm::glm
    )
endif()
