#ifndef __cuda_pointer_thrust_utils_hpp__
#define __cuda_pointer_thrust_utils_hpp__

#include "cuda_pointer.hpp"
#include <thrust/device_ptr.h>

namespace aks
{
    namespace thrust_utils
    {
        template<typename T>
        ::thrust::device_ptr<T> to_device_ptr(cuda_pointer<T> const& ptr)
        {
            return ::thrust::device_ptr<T>(ptr.data());
        }

        template<typename T>
        ::thrust::device_ptr<T> begin(cuda_pointer<T>& ptr)
        {
            return ::thrust::device_ptr<T>(ptr.data());
        }

		template<typename T>
		::thrust::device_ptr<T> end(cuda_pointer<T>& ptr)
		{
			return ::thrust::device_ptr<T>(ptr.data() + ptr.size());
		}
    }
}

#endif // !__cuda_pointer_thrust_utils_hpp__
