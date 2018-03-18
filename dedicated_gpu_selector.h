#ifndef DEDICATED_GPU_SELECTOR_H
#define DEDICATED_GPU_SELECTOR_H

#include "working_device_selector.h"
#include <SYCL/device_selector.h>

class dedicated_gpu_selector : public cl::sycl::device_selector {
  protected:
	int operator()(const cl::sycl::device& device) const override {
		constexpr auto valid_rank = 1;
		constexpr auto invalid_rank = -1;

		return is_supported(device) ? valid_rank    // mark device as suitable
		                            : invalid_rank; // reject device
	}

  public:
	static bool is_supported(const cl::sycl::device& device) {
		using namespace cl::sycl::info;
		return device.get_info<device::host_unified_memory>() == false && device.get_info<device::device_type>() == device_type::gpu
		       && celerity::working_device_selector::is_supported(device);
	}
};


#endif // DEDICATED_GPU_SELECTOR_H
