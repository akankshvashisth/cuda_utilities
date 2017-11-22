#ifndef __cuda_blas_level_1_hpp__
#define __cuda_blas_level_1_hpp__

#include "cuda_blas_manager.hpp"
#include "multi_dim_vector.hpp"
#include "cuda_pointer.hpp"
#include "cuda_blas_utilities.hpp"
#include <cassert>

namespace aks
{
    namespace cuda_blas 
    {
        int abs_max_index(cuda_blas_manager const& mgr, cuda_vector<double> const& xs)
        {
            cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
            int result = 0;
            status = cuda_blas_monad(status, cublasIdamax, mgr.m_handle, get_max_dim<0>(xs.view()), xs.view().data(), 1, &result);
            assert(status == cuda_blas_manager::success_value);
            return result - 1; //this is 1 based, back to 0 based
        }

        int abs_min_index(cuda_blas_manager const& mgr, cuda_vector<double> const& xs)
        {
            cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
            int result = 0;
            status = cuda_blas_monad(status, cublasIdamin, mgr.m_handle, get_max_dim<0>(xs.view()), xs.view().data(), 1, &result);
            assert(status == cuda_blas_manager::success_value);
            return result - 1; //this is 1 based, back to 0 based
        }

        double abs_sum(cuda_blas_manager const& mgr, cuda_vector<double> const& xs)
        {
            cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
            double result = 0;
            status = cuda_blas_monad(status, cublasDasum, mgr.m_handle, get_max_dim<0>(xs.view()), xs.view().data(), 1, &result);
            assert(status == cuda_blas_manager::success_value);
            return result;
        }

        double dot(cuda_blas_manager const& mgr, cuda_vector<double> const& xs, cuda_vector<double> const& ys)
        {
            //assume xs and ys are equal length
            cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
            double result = 0;
            status = cuda_blas_monad(status, cublasDdot, mgr.m_handle, get_max_dim<0>(xs.view()), xs.view().data(), 1, ys.view().data(), 1, &result);
            assert(status == cuda_blas_manager::success_value);
            return result;
        }

        double norm_sq(cuda_blas_manager const& mgr, cuda_vector<double> const& xs)
        {
            cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
            double result = 0;
            status = cuda_blas_monad(status, cublasDnrm2, mgr.m_handle, get_max_dim<0>(xs.view()), xs.view().data(), 1, &result);
            assert(status == cuda_blas_manager::success_value);
            return result;
        }

        cuda_vector<double>& scale_in_place(cuda_blas_manager const& mgr, cuda_vector<double>& xs, double const alpha)
        {
            cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
            status = cuda_blas_monad(status, cublasDscal, mgr.m_handle, get_max_dim<0>(xs.view()), &alpha, xs.view().data(), 1);
            assert(status == cuda_blas_manager::success_value);
            return xs;
        }
    }
}

#endif __cuda_blas_level_1_hpp__
