#include "cuda_blas_level_3.hpp"

namespace aks
{
	namespace cuda_blas
	{
		cuda_blas_manager::status_type cublas_matrix_multiply(cuda_blas_manager const& mgr, matrix<double const> const& A, matrix<double const> const& B, matrix<double>& y)
		{
			double const alpha = 1.0, beta = 0.0;
			int const m = int(rows(A)), k = int(cols(A)), n = int(cols(B));
			return cuda_blas_monad(cuda_blas_manager::success_value, cublasDgemm, mgr.m_handle
				, CUBLAS_OP_N, CUBLAS_OP_N
				, n, m, k
				, &alpha
				, B.data(), n
				, A.data(), k
				, &beta
				, y.data(), n
			);
		}

		cuda_matrix<double> matrix_multiply(cuda_blas_manager const& mgr, cuda_matrix<double> const& A_matrix, cuda_matrix<double> const& B_matrix)
		{
			//auto rows = [](matrix<double const> const& M) {return rows(M);  };
			//auto cols = [](matrix<double const> const& M) {return cols(M);  };

			//assume mxk (row major) and kxn (row major)
			matrix<double const> const A = A_matrix.cview();
			matrix<double const> const B = B_matrix.cview();
			assert(cols(A) == rows(B));
			cuda_matrix<double> y_vec(rows(A), cols(B));
			matrix<double> y = y_vec.view();
			cuda_blas_manager::status_type status = cublas_matrix_multiply(mgr, A, B, y);
			assert(status == cuda_blas_manager::success_value);
			return y_vec;
		}

		cuda_blas_manager::status_type cublas_transpose(cuda_blas_manager const& mgr, matrix<double const> const& A, matrix<double>& y)
		{
			double const alpha = 1.0, beta = 0.0;
			return cuda_blas_monad(cuda_blas_manager::success_value, cublasDgeam, mgr.m_handle
				, CUBLAS_OP_T, CUBLAS_OP_N
				, rows(A), cols(A)
				, &alpha
				, A.data(), cols(A)
				, &beta
				, A.data(), cols(A)
				, y.data(), rows(A)
			);
		}

		cuda_matrix<double> transpose(cuda_blas_manager const& mgr, cuda_matrix<double> const& A_matrix)
		{
			matrix<double const> const A = A_matrix.cview();
			cuda_matrix<double> y_vec(cols(A), rows(A));

			matrix<double> y = y_vec.view();
			double const alpha = 1.0, beta = 0.0;
			cuda_blas_manager::status_type status = cublas_transpose(mgr, A, y);
			assert(status == cuda_blas_manager::success_value);
			return y_vec;
		}

		cuda_blas_manager::status_type cublas_alpha_A_plus_beta_B(cuda_blas_manager const& mgr, double const alpha, matrix<double const> const& A, double const beta, matrix<double const> const& B, matrix<double>& y)
		{
			return cuda_blas_monad(cuda_blas_manager::success_value, cublasDgeam, mgr.m_handle
				, CUBLAS_OP_N, CUBLAS_OP_N
				, cols(A), rows(A)
				, &alpha
				, A.data(), cols(A)
				, &beta
				, B.data(), cols(B)
				, y.data(), cols(y)
			);
		}

		cuda_matrix<double> alpha_A_plus_beta_B(cuda_blas_manager const& mgr, double const alpha, cuda_matrix<double> const& A_matrix, double const beta, cuda_matrix<double> const& B_matrix)
		{
			matrix<double const> const A = A_matrix.cview();
			matrix<double const> const B = B_matrix.cview();

			assert(rows(A) == rows(B) && cols(A) == cols(B));
			cuda_matrix<double> y_vec(rows(A), cols(A));
			matrix<double> y = y_vec.view();

			//double const alpha = 1.0, beta = 0.0;
			cuda_blas_manager::status_type status = cublas_alpha_A_plus_beta_B(mgr, alpha, A, beta, B, y);
			assert(status == cuda_blas_manager::success_value);
			return y_vec;
		}
	}
}