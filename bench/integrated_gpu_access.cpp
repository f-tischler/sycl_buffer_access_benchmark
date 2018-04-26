#include "../integrated_gpu_selector.h"
#include "benchmarks.h"

BUFFER_ACCESS_BENCHMARK(subr_host_from_device_access, integrated_gpu_selector)
BUFFER_ACCESS_BENCHMARK(subr_device_from_host_access, integrated_gpu_selector)

BUFFER_ACCESS_BENCHMARK(full_host_from_device_access, integrated_gpu_selector)
BUFFER_ACCESS_BENCHMARK(full_device_from_host_access, integrated_gpu_selector)

BENCHMARK_MAIN();
