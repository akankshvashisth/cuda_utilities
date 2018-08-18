#ifndef __cuda_blas_level_2_hpp__
#define __cuda_blas_level_2_hpp__

#include "cuda_blas_manager.hpp"
#include "multi_dim_vector.hpp"
#include "cuda_pointer.hpp"
#include <cassert>

namespace aks
{
	namespace cuda_blas
	{
		cuda_vector<double> matrix_multiply(cuda_blas_manager const& mgr, cuda_matrix<double> const& A_matrix, cuda_vector<double> const& x_vec)
		{
			//assume mxn (row major) and nx1
			auto const A = A_matrix.cview();
			auto const x = x_vec.cview();
			assert(cols(A) == get_max_dim<0>(x));
			double const alpha = 1.0, beta = 0.0;

			cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
			cuda_vector<double> y_vec(rows(A));
			auto y = y_vec.view();
			status = cuda_blas_monad(status, cublasDgemv, mgr.m_handle, CUBLAS_OP_T, cols(A), rows(A), &alpha, A.data(), cols(A), x.data(), 1, &beta, y.data(), 1);
			assert(status == cuda_blas_manager::success_value);
			return y_vec;
		}
	}
}

#endif // !__cuda_blas_level_2_hpp__