#ifndef __cuda_context_hpp__
#define __cuda_context_hpp__

#include <memory>
#include <cuda_runtime.h>

namespace aks {
	struct cuda_device
	{
		cuda_device(int device_id) : m_device_id(device_id) {}
		int device_id() const { return m_device_id; }
	private:
		int m_device_id;
	};

	struct cuda_context {
		cuda_context(cuda_device const device) { cudaSetDevice(device.device_id()); }
		~cuda_context() { cudaDeviceReset(); }
	};

	struct cuda_sync_context {
		~cuda_sync_context() { cudaDeviceSynchronize(); }
	};

	typedef cudaError_t status_type;

	status_type last_status()
	{
		return cudaGetLastError();
	}
}

#endif // __cuda_context_hpp__