# sycl_buffer_access_benchmark

This project aims to benchmark Codeplay's ComputeCpp implementation of SYCL in terms of accessing subranges of SYCL buffers. 

## Prerequisites

CMake >= 2.8.12
ComputeCpp >= 0.60
C++14 compliant compiler (when using MSVC make sure you have the v140 toolset installed)

## Generating build files

CMake you needs the path to the ComputeCpp root directory stored in the "COMPUTECPP_PACKAGE_ROOT_DIR" variable. 

On linux use `-DCOMPUTECPP_PACKAGE_ROOT_DIR=<path_to_your_installation>` to provide the required path to CMake.

When using the CMake integration of Visual Studio 2017 please use the supplied CMakeSettings.json and add a environment variable `COMPUTECPP_ROOT_DIR` which points to the required directory

If you want to use MSVC without the CMake integration, please refer to CMakeSettings.json for hints on how to generate solutions from the command line.

## Benchmarks

The suite consists of four benchmarks: 

1) **subr_device_from_host/[n]/***: measures the time of accessing the first `n` elements of a host buffer consisting of `N` elements from the device
1) **subr_host_from_device/[n]/***: measures the time of accessing the first `n` elements of a host buffer consisting of `N` elements, which were modified by the executed kernel, from the host
1) **full_device_from_host/[n]/***: measures the time of accessing all elements of a host buffer consisting of `n` elements, from the device
1) **full_host_from_device/[n]/***: measures the time of accessing all elements of a host buffer consisting of `n` elements, which were modified by the executed kernel, from the host

The benchmarks are run on both a dedicated gpu and a integrated gpu with unified memory. The reason is that we can preclude that the kernel is somehow compute bound which would defeat the purpose of the benchmark. This unintended behaviour would show by having the integrated gpu benchmarks scale at the same level as those on dedicated gpu.

## Outcome

The interesting benchmarks are the first two - they show if the benchmark times scale with the number of elements accessed or not. The latter case would suggest that the runtime copies the entire buffer to the target device regardless of the acccessed range which is not the expected behavior.
The last two benchmarks provide a reference on how the times of the first two benchmarks should scale in theory when implemented correctly. 

The takeaway of this benchmark suite is that if the SYCL runtime is implemented efficiently the times of benchmark 1 and 3 and the times of 2 and 4 will coincide.

Benchmarks with the message "validation failed" most likely exceeded the maximum buffer size of the selected device. Try to reduce buffer sizes by modifying the values of `min_num_accessed` and
`num_elements` in `access_benchmark.h`.
