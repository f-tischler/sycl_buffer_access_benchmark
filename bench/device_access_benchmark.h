#ifndef DEVICE_ACCESS_BENCHMARK_H
#define DEVICE_ACCESS_BENCHMARK_H

#include "access_benchmark.h"
#include <SYCL/sycl.hpp>

/**
 * \brief benchmarks the access on the given data and range from the device
 * \param state
 * \param device
 * \param data
 * \param num_accessed_elements
 */
template <class T>
static void perform_device_access(benchmark::State& state, const cl::sycl::device& device, T& data, const int64_t num_accessed_elements) {
	using namespace cl::sycl;

	for(auto _ : state) {
		{
			// do not measure initialization
			state.PauseTiming();

			// reset data and create buffer/queue
			auto buf = reset(data);
			const auto my_queue = create_queue(device);

			// measure gpu work
			state.ResumeTiming();

			// perform work on gpu
			if(!submit_and_wait(my_queue, state, get_mutator<class device_access>(buf, num_accessed_elements))) break;

			// do not measure validation
			state.PauseTiming();

			// validate results on host
			if(!validate(state, buf, num_accessed_elements)) break;
		}

		state.ResumeTiming();
	}
}

/**
 * \brief performs a benchmark with a increasing number of elements where
 *		  all elements are accessed from the device. Used to assert that the
 *		  benchmark times are scaling with increasing number of elements
 * \param state
 * \param device
 */
static void full_device_access_impl(benchmark::State& state, const cl::sycl::device& device) {
	using namespace cl::sycl;

	const auto num_accessed_elements = state.range(0);

	// increasing number of elements
	std::vector<int64_t> data(num_accessed_elements);

	state.SetComplexityN(num_accessed_elements);

	// always access the entire buffer
	perform_device_access(state, device, data, num_accessed_elements);
}

/**
 * \brief performs a benchmark with a constant number of elements but with an
 *		 increasing amount of elements being access from the device. Used to assert
 *		 if the SYCL runtime supports partial access to memory without copying the entire
 *		 buffer
 * \param state
 * \param device
 */
static void subrange_device_access_impl(benchmark::State& state, const cl::sycl::device& device) {
	// fixed number of elements
	std::vector<int64_t> data(num_elements);

	// increasing number of accessed elements
	const auto num_accessed_elements = state.range(0);
	state.SetComplexityN(num_accessed_elements);

	perform_device_access(state, device, data, num_accessed_elements);
}

#endif // DEVICE_ACCESS_BENCHMARK_H
