#ifndef ACCESS_BENCHMARK_H
#define ACCESS_BENCHMARK_H

#include <SYCL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <boost/align/aligned_allocator.hpp>

constexpr auto min_num_accessed = 100'000'000;
constexpr auto num_elements = 500'000'000;
constexpr auto multiplier = 2;

template <class T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

/**
 * \brief benchmarks the access on the given data and range from the device
 * \param state
 * \param device
 * \param data
 * \param num_accessed_elements
 */
template <class T>
static void perform_device_access(benchmark::State& state, const cl::sycl::device& device, T& data, int num_accessed_elements) {
	using namespace cl::sycl;

	for(auto _ : state) {
		{
			state.PauseTiming();

			std::fill(data.begin(), data.end(), 0);

			buffer<size_t, 1> buf(data.data(), range<1>(data.size()));
			// buf.set_final_data(nullptr);

			queue my_queue(device);

			state.ResumeTiming();

			my_queue.submit([&](handler& cgh) {
				// access the subrange started at 0 with size "num_accessed_elements"
				auto ptr = buf.get_access<access::mode::read_write>(cgh, 0, num_accessed_elements);

				/* We create an nd_range to describe the work space that the kernel is
				 * to be executed across. Here we create a linear (one dimensional)
				 * nd_range, which creates a work item per element of the vector. The
				 * first parameter of the nd_range is the range of global work items
				 * and the second is the range of local work items (i.e. the work group
				 * range). */
				auto my_range = nd_range<1>(range<1>(num_accessed_elements), range<1>(64));

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
			});

			my_queue.wait_and_throw();

			state.PauseTiming();
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
	aligned_vector<size_t> data(num_accessed_elements);

	state.SetComplexityN(num_accessed_elements);
	state.SetBytesProcessed(num_accessed_elements * sizeof(decltype(data)::value_type));

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
	aligned_vector<size_t> data(num_elements);

	// increasing number of accessed elements
	const auto num_accessed_elements = state.range(0);
	state.SetComplexityN(num_accessed_elements);
	state.SetBytesProcessed(num_accessed_elements * sizeof(decltype(data)::value_type));

	perform_device_access(state, device, data, num_accessed_elements);
}

#define BUFFER_ACCESS_BENCHMARK(f)                                                                                                                             \
	BENCHMARK(f)                                                                                                                                               \
	    ->RangeMultiplier(multiplier)                                                                                                                          \
	    ->Range(min_num_accessed, num_elements)                                                                                                                \
	    ->ReportAggregatesOnly()                                                                                                                               \
	    ->Repetitions(3)                                                                                                                                       \
	    ->Complexity()                                                                                                                                         \
	    ->Unit(benchmark::TimeUnit::kMillisecond);


#endif // ACCESS_BENCHMARK_H
