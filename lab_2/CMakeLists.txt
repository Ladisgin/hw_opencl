cmake_minimum_required(VERSION 3.15)
project(lab_2 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_C_FLAGS "-framework OpenCL")
else (OPENCL_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})
endif ()

add_executable(firstopencl main.c)