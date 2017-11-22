#ifndef __cuda_blas_level_3_hpp__
#define __cuda_blas_level_3_hpp__

#include "cuda_blas_manager.hpp"
#include "multi_dim_vector.hpp"
#include "cuda_blas_utilities.hpp"

namespace aks
{
    namespace cuda_blas
    {
        cuda_blas_manager::status_type cublas_matrix_multiply(cuda_blas_manager const& mgr, matrix<double const> const& A, matrix<double const> const& B, matrix<double>& y);
        cuda_matrix<double> matrix_multiply(cuda_blas_manager const& mgr, cuda_matrix<double> const& A_matrix, cuda_matrix<double> const& B_matrix);

        cuda_blas_manager::status_type cublas_transpose(cuda_blas_manager const& mgr, matrix<double const> const& A, matrix<double>& y);
        cuda_matrix<double> transpose(cuda_blas_manager const& mgr, cuda_matrix<double> const& A_matrix);

        cuda_blas_manager::status_type cublas_alpha_A_plus_beta_B(cuda_blas_manager const& mgr, double const alpha, matrix<double const> const& A, double const beta, matrix<double const> const& B, matrix<double>& y);
        cuda_matrix<double> alpha_A_plus_beta_B(cuda_blas_manager const& mgr, double const alpha, cuda_matrix<double> const& A_matrix, double const beta, cuda_matrix<double> const& B_matrix);
    }
}

#endif // !__cuda_blas_level_3_hpp__
