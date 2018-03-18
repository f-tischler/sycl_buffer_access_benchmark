cmake_minimum_required(VERSION 2.8.12)
project(sycl_buffer_access_benchmark LANGUAGES CXX)

# cmake module directory
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        ${PROJECT_SOURCE_DIR}/cmake)

find_package(ComputeCpp REQUIRED)

include(CTest)
enable_testing(true)

add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)
