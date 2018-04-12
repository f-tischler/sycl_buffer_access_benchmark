#ifndef ACCESS_BENCHMARK_H
#define ACCESS_BENCHMARK_H

#include <benchmark/benchmark.h>
#include <boost/align/aligned_allocator.hpp>
#include <strstream>

constexpr auto min_num_accessed = 1 << 25;
constexpr auto num_elements = 1 << 28;
constexpr auto multiplier = 2;

template <class T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

#define BUFFER_ACCESS_BENCHMARK(f)                                                                                                                             \
	BENCHMARK(f)                                                                                                                                               \
	    ->RangeMultiplier(multiplier)                                                                                                                          \
	    ->Range(min_num_accessed, num_elements)                                                                                                                \
	    ->UseRealTime()                                                                                                                                        \
	    ->ReportAggregatesOnly()                                                                                                                               \
	    ->Repetitions(10)                                                                                                                                      \
	    ->Complexity()                                                                                                                                         \
	    ->Unit(benchmark::TimeUnit::kMillisecond);

template <class T>
bool validate(benchmark::State& state, cl::sycl::buffer<T, 1>& buf, int64_t num_accessed_elements) {
	auto host_access = buf.get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(num_accessed_elements));

	// validate result
	for(auto i = 0; i < num_accessed_elements; ++i) {
		if(host_access[cl::sycl::id<1>(i)] == i) continue;

		std::strstream str;
		str << "validation failed (data[" << i << "] == " << host_access[cl::sycl::id<1>(i)] << ")";

		state.SkipWithError(str.str());
		return false;
	}

	return true;
}
template <class KernelName, class T>
auto get_mutator(cl::sycl::buffer<T, 1>& buf, int64_t num_accessed_elements) {
	return [&buf, num_accessed_elements](cl::sycl::handler& cgh) {
		auto ptr = buf.get_access<cl::sycl::access::mode::read_write>(cgh, cl::sycl::range<1>(num_accessed_elements));
		auto my_range = cl::sycl::nd_range<1>(cl::sycl::range<1>(num_accessed_elements), cl::sycl::range<1>(64));

		// data[i] = i
		auto my_kernel = ([=](cl::sycl::nd_item<1> item) { ptr[item.get_global()] = item.get_global()[0]; });

		cgh.parallel_for<KernelName>(my_range, my_kernel);
	};
}

inline bool wait(cl::sycl::queue queue, benchmark::State& state) {
	try {
		queue.wait_and_throw();
	} catch(std::exception& ex) {
		state.SkipWithError(ex.what());
		return false;
	}

	return true;
}

template <class T>
bool submit_and_wait(cl::sycl::queue queue, benchmark::State& state, const T& cgf) {
	queue.submit(cgf);
	return wait(queue, state);
}

#endif // ACCESS_BENCHMARK_H
