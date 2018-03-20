#include <SYCL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <iostream>

#include "../dedicated_gpu_selector.h"
#include "access_benchmark.h"

static void full_device_access(benchmark::State& state) {
	const auto my_device = [&]() {
		try {
			return dedicated_gpu_selector().select_device();
		} catch(...) {
			std::cout << "no (compatible) dedicated gpu found" << std::endl;
			throw;
		}
	}();

	full_device_access_impl(state, my_device);
}

static void subrange_device_access(benchmark::State& state) {
	const auto my_device = [&]() {
		try {
			return dedicated_gpu_selector().select_device();
		} catch(...) {
			std::cout << "no (compatible) dedicated gpu found" << std::endl;
			throw;
		}
	}();

	subrange_device_access_impl(state, my_device);
}

BUFFER_ACCESS_BENCHMARK(full_device_access)
BUFFER_ACCESS_BENCHMARK(subrange_device_access)

BENCHMARK_MAIN()
