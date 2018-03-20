#include <SYCL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <iostream>

#include "../integrated_gpu_selector.h"
#include "device_access_benchmark.h"

static cl::sycl::device select_device() {
	try {
		return integrated_gpu_selector().select_device();
	} catch(...) {
		std::cout << "no (compatible) integrated gpu found" << std::endl;
		throw;
	}
}

static void full_device_access(benchmark::State& state) {
	full_device_access_impl(state, select_device());
}

static void subrange_device_access(benchmark::State& state) {
	subrange_device_access_impl(state, select_device());
}

BUFFER_ACCESS_BENCHMARK(full_device_access)
BUFFER_ACCESS_BENCHMARK(subrange_device_access)

BENCHMARK_MAIN()
