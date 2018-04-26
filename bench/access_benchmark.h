#ifndef ACCESS_BENCHMARK_H
#define ACCESS_BENCHMARK_H

#include <SYCL/accessor.h>
#include <SYCL/buffer.h>
#include <SYCL/queue.h>
#include <benchmark/benchmark.h>
#include <boost/align/aligned_allocator.hpp>

constexpr auto min_num_accessed = 1 << 25;
constexpr auto num_elements = 1 << 28;
constexpr auto multiplier = 2;
constexpr auto repetitions = 1;
#include <iostream>
#include <vector>

template <class T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

#define BUFFER_ACCESS_BENCHMARK(f, t)                                                                                                                          \
	BENCHMARK_TEMPLATE(f, t)                                                                                                                                   \
	    ->RangeMultiplier(multiplier)                                                                                                                          \
	    ->Range(min_num_accessed, num_elements)                                                                                                                \
	    ->UseRealTime()                                                                                                                                        \
	    ->ReportAggregatesOnly()                                                                                                                               \
	    ->Repetitions(1)                                                                                                                                       \
	    ->Complexity()                                                                                                                                         \
	    ->Unit(benchmark::TimeUnit::kMillisecond);
class async_error_handler {
  public:
	void operator()(const cl::sycl::exception_list list) {
		for(const auto& ex : list) {
			if(!ex) continue;

			try {
				std::rethrow_exception(ex);
			} catch(std::exception& e) { exceptions_.push_back(e); }
		}
	}

	void print_and_throw() {
		if(exceptions_.empty()) return;

		std::cout << "async exceptions thrown:" << std::endl;

		for(const auto& ex : exceptions_) {
			std::cout << std::endl << ex.what() << std::endl;
		}

		exceptions_.clear();

		throw std::runtime_error("async errors");
	}

  private:
	std::vector<std::exception> exceptions_;
};

static async_error_handler handler;

static cl::sycl::async_handler async_error_proxy = [](auto&& list) { handler(std::forward<decltype(list)>(list)); };

inline cl::sycl::queue create_queue(const cl::sycl::device& device) {
	return cl::sycl::queue(device, async_error_proxy);
}

inline bool validate(benchmark::State& state, cl::sycl::buffer<int64_t, 1>& buf, const int64_t num_accessed_elements) {
	const auto host_access = buf.get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(num_accessed_elements));

	// validate result
	for(size_t i = 0; i < static_cast<size_t>(num_accessed_elements); ++i) {
		if(host_access[i] == i) continue;

		state.SkipWithError("validation failed");
		return false;
	}

	return true;
}

template <class KernelName>
auto get_mutator(cl::sycl::buffer<int64_t, 1>& buf, const int64_t num_accessed_elements) {
	return [&buf, num_accessed_elements](cl::sycl::handler& cgh) {
		const auto ptr = buf.get_access<cl::sycl::access::mode::read_write>(cgh, cl::sycl::range<1>(num_accessed_elements));
		auto my_range = cl::sycl::nd_range<1>(cl::sycl::range<1>(num_accessed_elements), cl::sycl::range<1>(64));

		// data[i] = i
		auto my_kernel = ([=](cl::sycl::nd_item<1> item) { ptr[item.get_global()] = item.get_global()[0]; });

		cgh.parallel_for<KernelName>(my_range, my_kernel);
	};
}

inline bool wait(cl::sycl::queue queue, benchmark::State& state) {
	try {
		queue.wait_and_throw();
		handler.print_and_throw();
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

template <class T>
auto reset(T& data) {
	std::fill(data.begin(), data.end(), 0);
	cl::sycl::buffer<typename T::value_type, 1> buf(data.data(), cl::sycl::range<1>(data.size()));
	buf.set_final_data(nullptr);

	return buf;
}

#endif // ACCESS_BENCHMARK_H
