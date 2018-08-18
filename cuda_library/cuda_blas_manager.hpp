#ifndef __cuda_blas_manager_hpp__
#define __cuda_blas_manager_hpp__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>

namespace aks
{
	template<typename F, typename... Ts>
	cublasStatus_t cuda_blas_monad(cublasStatus_t status, F func, Ts... ts)
	{
		if (status == CUBLAS_STATUS_SUCCESS) {
			return func(ts...);
		}
		else {
			return status;
		}
	}

	struct cuda_blas_manager
	{
		typedef cublasHandle_t handle_type;
		typedef cublasStatus_t status_type;
		static const cublasStatus_t success_value = CUBLAS_STATUS_SUCCESS;

		cuda_blas_manager() : m_handle(nullptr), m_status(success_value)
		{
			m_status = cuda_blas_monad(m_status, cublasCreate, &m_handle);
			assert(!has_error_occurred());
		}

		~cuda_blas_manager() {
			m_status = cuda_blas_monad(m_status, cublasDestroy, m_handle);
		}

		bool has_error_occurred() const { return m_status != success_value; }

		handle_type m_handle;
		mutable status_type m_status;
	};
}

#endif __cuda_blas_manager_hpp__