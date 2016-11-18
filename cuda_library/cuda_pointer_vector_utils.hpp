#ifndef __cuda_pointer_vector_utils_hpp__
#define __cuda_pointer_vector_utils_hpp__

#include "cuda_pointer.hpp"
#include <vector>

namespace aks
{
    template<typename T>
    cuda_pointer<T> make_cuda_pointer(std::vector<T>& data)
    {
        return cuda_pointer<T>(data.size(), data.begin());
    }

    template<typename T>
    cuda_pointer<T const> make_cuda_pointer(std::vector<T> const& data)
    {
        return cuda_pointer<T const>(data.size(), &(*data.begin()));
    }

    template<typename T>
    std::vector<T> from_cuda_pointer(cuda_pointer<T> const& data)
    {
        std::vector<T> ret(data.size());
        data.load(&(*ret.begin()));
        return ret;
    }

    template<typename T>
    std::vector<T> from_cuda_pointer(cuda_pointer<T const> const& data)
    {
        std::vector<T> ret(data.size());
        data.load(ret.begin());
        return ret;
    }
}

#endif // __cuda_pointer_vector_utils_hpp__
