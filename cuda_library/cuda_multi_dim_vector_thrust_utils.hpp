#ifndef __cuda_multi_dim_vector_thrust_utils_hpp__
#define __cuda_multi_dim_vector_thrust_utils_hpp__

#include "cuda_multi_dim_vector.hpp"
#include <thrust/device_ptr.h>

namespace aks
{
    namespace thrust_utils
    {
        template<typename T>
        ::thrust::device_ptr<T> begin(aks::multi_dim_vector<T, 1>& vec)
        {
            return ::thrust::device_ptr<T>(vec.begin());
        }

        //template<typename T>
        //thrust::device_ptr<T const> thrust_begin(aks::multi_dim_vector<T, 1> const& vec)
        //{
        //    return thrust::device_ptr<T const>(vec.begin());
        //}

        template<typename T>
        ::thrust::device_ptr<T const> begin(aks::multi_dim_vector<T const, 1>& vec)
        {
            return ::thrust::device_ptr<T const>(vec.begin());
        }

        template<typename T>
        ::thrust::device_ptr<T const> begin(aks::multi_dim_vector<T const, 1> const& vec)
        {
            return ::thrust::device_ptr<T const>(vec.begin());
        }

        template<typename T>
        ::thrust::device_ptr<T> end(aks::multi_dim_vector<T, 1>& vec)
        {
            return ::thrust::device_ptr<T>(vec.end());
        }

        //template<typename T>
        //thrust::device_ptr<T const> thrust_end(aks::multi_dim_vector<T, 1>& const vec)
        //{
        //    return thrust::device_ptr<T const>(vec.end());
        //}

        template<typename T>
        ::thrust::device_ptr<T const> end(aks::multi_dim_vector<T const, 1>& vec)
        {
            return ::thrust::device_ptr<T const>(vec.end());
        }

        template<typename T>
        ::thrust::device_ptr<T const> end(aks::multi_dim_vector<T const, 1> const& vec)
        {
            return ::thrust::device_ptr<T const>(vec.end());
        }

        //template<typename T>
        //thrust::device_vector<T> to_device_vector(aks::multi_dim_vector<T, 1>& vec)
        //{
        //    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(vec.data());
        //    return thrust::device_vector<T>(dev_ptr, dev_ptr + aks::get_max_dim<0>(vec));
        //}

        //template<typename T>
        //thrust::device_vector<T>  to_device_vector(aks::multi_dim_vector<T const, 1>& vec)
        //{
        //    thrust::device_ptr<T const> dev_ptr = thrust::device_pointer_cast(vec.data());
        //    return thrust::device_vector<T>(dev_ptr, dev_ptr + aks::get_max_dim<0>(vec));
        //}

        //template<typename T>
        //thrust::device_vector<T> to_device_vector(aks::multi_dim_vector<T, 1> const& vec)
        //{
        //    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(vec.data());
        //    return thrust::device_vector<T>(dev_ptr, dev_ptr + aks::get_max_dim<0>(vec));
        //}

        //template<typename T>
        //thrust::device_vector<T>  to_device_vector(aks::multi_dim_vector<T const, 1> const& vec)
        //{
        //    thrust::device_ptr<T const> dev_ptr = thrust::device_pointer_cast(vec.data());
        //    return thrust::device_vector<T>(dev_ptr, dev_ptr + aks::get_max_dim<0>(vec));
        //}
    }
}

#endif // !__cuda_multi_dim_vector_thrust_utils_hpp__
