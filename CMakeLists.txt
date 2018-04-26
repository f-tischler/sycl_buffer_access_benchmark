cmake_minimum_required(VERSION 2.8.12)
project(sycl_buffer_access_benchmark LANGUAGES CXX)

# cmake module directory
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        ${PROJECT_SOURCE_DIR}/cmake)

find_package(ComputeCpp REQUIRED)

include(CTest)
include(ConfigGBench)

find_package(Threads REQUIRED)
find_path(BOOST_ALIGN_INCLUDE_DIR "boost/align/align.hpp")

enable_testing(true)
 
add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)
