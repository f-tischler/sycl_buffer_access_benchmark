#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "device_access_benchmark.h"
#include "host_access_benchmark.h"
#include <SYCL/device.h>

template <class Selector>
static class cl::sycl::device select_device(const Selector& s) {
	try {
		static auto device = [&s]() {
			auto d = s.select_device();
			std::cout << std::endl;
			std::cout << "using " << d.template get_info<cl::sycl::info::device::name>() << std::endl;
			std::cout << std::endl;
			return d;
		}();

		return device;
	} catch(...) {
		std::cout << Selector::error_message << std::endl;
		exit(-1);
	}
}

template <class DeviceSelector>
static void full_host_from_device_access(benchmark::State& state) {
	full_device_access_impl(state, select_device(DeviceSelector()));
}

template <class DeviceSelector>
static void subr_host_from_device_access(benchmark::State& state) {
	subrange_device_access_impl(state, select_device(DeviceSelector()));
}

template <class DeviceSelector>
static void full_device_from_host_access(benchmark::State& state) {
	full_host_access_impl(state, select_device(DeviceSelector()));
}

template <class DeviceSelector>
static void subr_device_from_host_access(benchmark::State& state) {
	subrange_host_access_impl(state, select_device(DeviceSelector()));
}

#endif // BENCHMARKS_H
