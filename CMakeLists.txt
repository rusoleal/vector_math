cmake_minimum_required(VERSION 3.22.1)

project(vector_math VERSION 0.0.1)

configure_file(src/vector_math_config.h.in vector_math_config.h)

set(CMAKE_CXX_STANDARD 17)

option(VECTOR_MATH_BUILD_TEST "Enable building tests" OFF)

include_directories(inc)

message(STATUS ${PROJECT_NAME})

add_library(${PROJECT_NAME}
    src/vector_math.cpp
)

target_include_directories (vector_math PUBLIC 
                            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
                            )

target_include_directories(vector_math PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

if (VECTOR_MATH_BUILD_TEST)
    include(CTest)

    add_executable(${PROJECT_NAME}_test 
        test/main.cpp
    )
    target_link_libraries(${PROJECT_NAME}_test 
        ${PROJECT_NAME}
    )
    add_test(NAME vector_math_test COMMAND ${PROJECT_NAME}_test)
    #add_test(NAME gltf_test2 COMMAND ${PROJECT_NAME}_test)
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
endif()
