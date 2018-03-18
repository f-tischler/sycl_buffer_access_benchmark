#ifndef WORKING_DEVICE_SELECTOR_H
#define WORKING_DEVICE_SELECTOR_H

#include <SYCL/sycl.hpp>

namespace celerity {
namespace detail {

	/* needs to be defined in namespace or free function scope to
	 * be processed by the sycl compiler
	 */
	static const auto test_kernel = [](cl::sycl::handler& cgh) { cgh.single_task<class test_kernel>([]() {}); };

} // namespace detail

/**
 * \brief Selects the first device which
 *        works in a sense that it successfully runs
 *        a sycl kernel. If there are multiple working
 *        devices, ComputeCpp will chose which one to use.
 */
class working_device_selector : public cl::sycl::device_selector {
  protected:
	int operator()(const cl::sycl::device& device) const override {
		constexpr auto valid_rank = 1;
		constexpr auto invalid_rank = -1;

		return is_supported(device) ? valid_rank    // mark device as suitable
		                            : invalid_rank; // reject device
	}

  public:
	static bool is_supported(const cl::sycl::device& device) {
		auto async_exception_thrown = false;
		const cl::sycl::async_handler async_error_handler = [&async_exception_thrown](auto&&) { async_exception_thrown = true; };

		try {
			// create queue and submit test kernel
			// to verify if the current device can
			// execute sycl kernels
			cl::sycl::queue queue(device, async_error_handler);
			queue.submit(detail::test_kernel);
			queue.wait_and_throw();
		} catch(...) {
			// consume any exception and return false
			// to indicate that the device is not supported
			return false;
		}

		// no synchronous exceptions
		// => check for asynchronous exceptions
		return !async_exception_thrown;
	}
};

} // namespace celerity

#endif // WORKING_DEVICE_SELECTOR_H
