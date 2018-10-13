#ifndef __cuda_context_hpp__
#define __cuda_context_hpp__

#include <memory>
#include <cuda_runtime.h>
#include "defines.hpp"

namespace aks {
	typedef cudaError_t status_type;

	status_type last_status_with_reset()
	{
		return cudaGetLastError();
	}

	status_type last_status()
	{
		return cudaPeekAtLastError();
	}

	struct cuda_device
	{
		cuda_device(int device_id) : m_device_id(device_id) {}
		int device_id() const { return m_device_id; }
	private:
		int m_device_id;
	};

	struct cuda_context {
		cuda_context(cuda_device const device) { cudaSetDevice(device.device_id()); }
		~cuda_context() { gpu_error_check(last_status()); cudaDeviceReset(); }
	};

	struct cuda_sync_context {
		cuda_sync_context() { gpu_error_check(last_status()); }
		~cuda_sync_context() {
			gpu_error_check(last_status()); cudaDeviceSynchronize(); gpu_error_check(last_status());
		}
	};
}

#endif // __cuda_context_hpp__