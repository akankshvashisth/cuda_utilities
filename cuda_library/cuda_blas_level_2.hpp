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
        cuda_multi_dim_vector<double, 1> matrix_multiply(cuda_blas_manager const& mgr, cuda_multi_dim_vector<double, 2> const& A, cuda_multi_dim_vector<double, 1> const& x)
        {            
            //assume mxn and nx1
            cuda_blas_manager::status_type status = cuda_blas_manager::success_value;
            int m = get_max_dim<0>(A.cview()), n = get_max_dim<1>(A.cview());
            double alpha = 1.0, beta = 1.0;
            cuda_multi_dim_vector<double, 1> y(m);
            
            auto const pA = A.view().data();
            auto const pX = x.view().data();
            auto pY = y.view().data();

            status = cuda_blas_monad(
                status
                , cublasDgemv
                , mgr.m_handle
                , CUBLAS_OP_T
                , n, m
                , &alpha
                , pA, n
                , pX, 1
                , &beta
                , pY, 1);
            assert(status == cuda_blas_manager::success_value);
            return y; //this is 1 based, back to 0 based
        }
    }
}

#endif // !__cuda_blas_level_2_hpp__

