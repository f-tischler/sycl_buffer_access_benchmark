#ifndef HOST_ACCESS_BENCHMARK_H
#define HOST_ACCESS_BENCHMARK_H

#include "access_benchmark.h"
#include <SYCL/sycl.hpp>
#include <iostream>

/**
 * \brief benchmarks the access on the given data and range from the device
 * \param state
 * \param device
 * \param data
 * \param num_accessed_elements
 */
template <class T>
static void perform_host_access(benchmark::State& state, const cl::sycl::device& device, T& data, const int64_t num_accessed_elements) {
	using namespace cl::sycl;

	for(auto _ : state) {
		{
			state.PauseTiming();

			std::fill(data.begin(), data.end(), 0);

			buffer<size_t, 1> buf(data.data(), range<1>(data.size()));
			buf.set_final_data(data.data());

			queue my_queue(device);

			state.ResumeTiming();

			my_queue.submit([&](handler& cgh) {
				auto ptr = buf.get_access<access::mode::read_write>(cgh);

				/* We create an nd_range to describe the work space that the kernel is
				 * to be executed across. Here we create a linear (one dimensional)
				 * nd_range, which creates a work item per element of the vector. The
				 * first parameter of the nd_range is the range of global work items
				 * and the second is the range of local work items (i.e. the work group
				 * range). */
				auto my_range = nd_range<1>(range<1>(data.size()), range<1>(64));

				/* We construct the lambda outside of the parallel_for function call,
				 * though it can be inline inside the function call too. For this
				 * parallel_for API the lambda is required to take a single parameter;
				 * an item<N> of the same dimensionality as the nd_range - in this
				 * case one. Other kernel dispatches might have different parameters -
				 * for example, the single_task takes no arguments. */
				auto my_kernel = ([=](nd_item<1> item) {
					/* Items have various methods to extract ids and ranges. The
					 * specification has full details of these. Here we use the
					 * item::get_global() to retrieve the global id as an id<1>.
					 * This particular kernel will set the ith element to the value
					 * of i. */
					ptr[item.get_global()] = item.get_global()[0];
				});

				/* We call the parallel_for() API with two parameters; the nd_range
				 * we constructed above and the lambda that we constructed. Because
				 * the kernel is a lambda we *must* specify a template parameter to
				 * use as a name. */
				cgh.parallel_for<class assign_elements_1>(my_range, my_kernel);

				auto host_access = buf.get_access<access::mode::read, access::target::host_buffer>(cgh, range<1>(num_accessed_elements));

				// validate result
				for(auto i = 0; i < num_accessed_elements; ++i) {
					if(data[i] == i) continue;

					std::cout << "validation failed";
					exit(-1);
				}
			});

			my_queue.wait_and_throw();
		}
	}
}

/**
 * \brief performs a benchmark with a increasing number of elements where
 *		  all elements are accessed from the device. Used to assert that the
 *		  benchmark times are scaling with increasing number of elements
 * \param state
 * \param device
 */
static void full_host_access_impl(benchmark::State& state, const cl::sycl::device& device) {
	using namespace cl::sycl;

	const auto num_accessed_elements = state.range(0);

	// increasing number of elements
	aligned_vector<size_t> data(num_accessed_elements);

	state.SetComplexityN(num_accessed_elements);

	// access the entire buffer
	perform_host_access(state, device, data, num_accessed_elements);
}

/**
 * \brief performs a benchmark with a constant number of elements but with an
 *		 increasing amount of elements being access from the device. Used to assert
 *		 if the SYCL runtime supports partial access to memory without copying the entire
 *		 buffer
 * \param state
 * \param device
 */
static void subrange_host_access_impl(benchmark::State& state, const cl::sycl::device& device) {
	// fixed number of elements
	aligned_vector<size_t> data(num_elements);

	// increasing number of accessed elements
	const auto num_accessed_elements = state.range(0);
	state.SetComplexityN(num_accessed_elements);

	perform_host_access(state, device, data, num_accessed_elements);
}

#endif // HOST_ACCESS_BENCHMARK_H