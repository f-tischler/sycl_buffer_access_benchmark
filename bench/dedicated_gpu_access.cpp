#include <SYCL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <iostream>

#include "../dedicated_gpu_selector.h"
#include "device_access_benchmark.h"
#include "host_access_benchmark.h"

static cl::sycl::device select_device() {
	try {
		static auto device = dedicated_gpu_selector().select_device();
		return device;
	} catch(...) {
		std::cout << "no (compatible) dedicated gpu found" << std::endl;
		exit(-1);
	}
}

static void full_device_access(benchmark::State& state) {
	full_device_access_impl(state, select_device());
}

static void subrange_device_access(benchmark::State& state) {
	subrange_device_access_impl(state, select_device());
}

static void full_host_access(benchmark::State& state) {
	full_host_access_impl(state, select_device());
}

static void subrange_host_access(benchmark::State& state) {
	subrange_host_access_impl(state, select_device());
}

BUFFER_ACCESS_BENCHMARK(full_device_access)
BUFFER_ACCESS_BENCHMARK(subrange_device_access)
BUFFER_ACCESS_BENCHMARK(full_host_access)
BUFFER_ACCESS_BENCHMARK(subrange_host_access)

BENCHMARK_MAIN();
