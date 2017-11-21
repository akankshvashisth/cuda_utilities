#ifndef __cuda_pointer_hpp__
#define __cuda_pointer_hpp__

#include <cuda_runtime.h>
#include <cassert>

namespace aks
{
    template<typename F, typename... Ts>
    cudaError_t cuda_monad(cudaError_t status, F func, Ts... ts)
    {
        if (status == cudaSuccess) {
            return func(ts...);
        } else {
            return status;
        }
    }

    template<typename _T>
    struct cuda_pointer
    {
        typedef _T value_type;
        typedef size_t size_type;
        typedef value_type* pointer_type;
        typedef value_type const * const_pointer_type;
        typedef cudaError_t status_type;

        cuda_pointer() : m_data(nullptr), m_size(0), m_status(cudaSuccess)
        {            
            assert(!has_error_occurred());
        }

        cuda_pointer(size_type const size) : m_data(nullptr), m_size(size), m_status(cudaSuccess)
        {                        
            allocate();
            assert(!has_error_occurred());
        }

        cuda_pointer(size_type const size, const_pointer_type data_ptr) : m_data(nullptr), m_size(size), m_status(cudaSuccess)
        {                        
            allocate();
            m_status = cuda_monad(m_status, cudaMemcpy, (void*)data(), (void const*)data_ptr, data_size(), cudaMemcpyHostToDevice);
            assert(!has_error_occurred());
        }

        cuda_pointer(cuda_pointer&& other) : m_data(other.data()), m_size(other.size()), m_status(other.status())
        {
            other.m_data = nullptr;
            other.reset();
            assert(!has_error_occurred());
        }

        cuda_pointer& operator=(cuda_pointer&& other)
        {
            if (this != &other)
            {
                reset();
                copy(other);
                other.reset();
            }
            assert(!has_error_occurred());
			return *this;
        }

        template<typename U>
        void deep_copy_from(cuda_pointer<U> const& other)
        {
            reset();
            init(other.size());
            allocate();
            auto src = (void const*)other.data();
            m_status = cuda_monad(m_status, cudaMemcpy, (void*)data(), src, data_size(), cudaMemcpyDeviceToDevice);
            assert(!has_error_occurred());
        }

        void load(pointer_type data_ptr) const
        {   
            assert(!has_error_occurred());
			auto dst = (void*)data_ptr;
			auto src = (void const*)data();
            m_status = cuda_monad(m_status, cudaMemcpy, dst, src, data_size(), cudaMemcpyDeviceToHost);
            assert(!has_error_occurred());
        }

        pointer_type data() { return m_data; }
        const_pointer_type data() const { return m_data; }
        size_type size() const { return m_size; }
        status_type status() const { return m_status; }
        bool has_error_occurred() const { return m_status != cudaSuccess; }

        ~cuda_pointer()
        {
            reset();
        }

    private:
		cuda_pointer(cuda_pointer&) = delete;
		cuda_pointer& operator=(cuda_pointer&) = delete;

        void reset()
        {
            cudaFree((void*)m_data);
            m_data = nullptr;
            m_size = 0;
            m_status = cudaSuccess;
        }

        void allocate()
        {
            m_status = cuda_monad(m_status, cudaMalloc<void>, (void**)&m_data, data_size());
        }

        void init(size_type size)
        {
            m_data = nullptr;
            m_status = cudaSuccess;
            m_size = size;
        }

        void copy(cuda_pointer& other)
        {
            m_data = other.data();
            m_size = other.size();
            m_status = other.status();
        }

        size_type data_size() const
        {
            return size() * sizeof(value_type);
        }

        pointer_type m_data;
        size_type m_size;
        mutable status_type m_status;
    };
}

#endif //__cuda_pointer_hpp__ !
