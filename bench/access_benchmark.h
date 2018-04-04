#ifndef ACCESS_BENCHMARK_H
#define ACCESS_BENCHMARK_H

#include <benchmark/benchmark.h>
#include <boost/align/aligned_allocator.hpp>

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
	    ->Repetitions(3)                                                                                                                                       \
	    ->Complexity()                                                                                                                                         \
	    ->Unit(benchmark::TimeUnit::kMillisecond);

#endif // ACCESS_BENCHMARK_H
