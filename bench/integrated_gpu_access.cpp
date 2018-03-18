#include <SYCL/sycl.hpp>
#include <array>
#include <benchmark/benchmark.h>
#include <iostream>

#include "../integrated_gpu_selector.h"

using namespace cl::sycl;

constexpr auto min_num_accessed = 50'000'000;
constexpr auto num_elements = 500'000'000;
constexpr auto multiplier = 2;

static void full_device_access(benchmark::State& state) {
	const auto num_accessed_elements = state.range(0);

	std::vector<int32_t> data(num_accessed_elements);

	state.SetComplexityN(num_accessed_elements);
	state.SetBytesProcessed(num_accessed_elements * sizeof(decltype(data)::value_type));

	const auto my_device = [&]() {
		try {
			return integrated_gpu_selector().select_device();
		} catch(...) {
			std::cout << "no (compatible) dedicated gpu found" << std::endl;
			throw;
		}
	}();

	for(auto _ : state) {
		{
			state.PauseTiming();

			std::fill(data.begin(), data.end(), 0);

			buffer<int32_t, 1> buf(data.data(), range<1>(data.size()));
			// buf.set_final_data(nullptr);

			queue my_queue(my_device);

			state.ResumeTiming();

			my_queue.submit([&](handler& cgh) {
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

static void subrange_device_access(benchmark::State& state) {
	const auto num_accessed_elements = state.range(0);

	std::vector<int32_t> data(num_elements);

	state.SetComplexityN(num_accessed_elements);
	state.SetBytesProcessed(num_accessed_elements * sizeof(decltype(data)::value_type));

	const auto my_device = [&]() {
		try {
			return integrated_gpu_selector().select_device();
		} catch(...) {
			std::cout << "no (compatible) dedicated gpu found" << std::endl;
			throw;
		}
	}();

	for(auto _ : state) {
		{
			state.PauseTiming();

			std::fill(data.begin(), data.end(), 0);

			buffer<int32_t, 1> buf(data.data(), range<1>(data.size()));
			// buf.set_final_data(nullptr);

			queue my_queue(my_device);

			state.ResumeTiming();

			my_queue.submit([&](handler& cgh) {
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
				cgh.parallel_for<class assign_elements_2>(my_range, my_kernel);
			});

			my_queue.wait_and_throw();

			state.PauseTiming();
		}

		state.ResumeTiming();
	}
}

BENCHMARK(full_device_access)->RangeMultiplier(multiplier)->Range(min_num_accessed, num_elements)->ReportAggregatesOnly()->Repetitions(5)->Complexity();
BENCHMARK(subrange_device_access)->RangeMultiplier(multiplier)->Range(min_num_accessed, num_elements)->ReportAggregatesOnly()->Repetitions(5)->Complexity();

BENCHMARK_MAIN()
