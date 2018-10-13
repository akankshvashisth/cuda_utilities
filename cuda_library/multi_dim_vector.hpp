#ifndef __multi_dim_vector_hpp__
#define __multi_dim_vector_hpp__

#include "cuda_multi_dim_vector.hpp"
#include "multi_dim_vector_with_memory.hpp"
#include "cuda_pointer.hpp"
#include "cuda_pointer_vector_utils.hpp"
#include <vector>

namespace aks
{
	template<typename T, size_t N>
	using host_multi_dim_vector = aks::detail::multi_dim_vector_with_memory<T, N, std::vector<T>, e_device_type::HOST>;

	template<typename T, size_t N>
	using cuda_multi_dim_vector = aks::detail::multi_dim_vector_with_memory<T, N, aks::cuda_pointer<T>, e_device_type::DEVICE>;

	template<typename T, size_t N>
	auto& operator<<(host_multi_dim_vector<T, N>& host, cuda_multi_dim_vector<T, N> const& device)
	{
		host.m_dimensions = device.m_dimensions;
		host.m_data = from_cuda_pointer(device.m_data);
		return host;
	}

	template<typename T, size_t N>
	auto to_host(cuda_multi_dim_vector<T, N> const& device)
	{
		host_multi_dim_vector<T, N> host(device.m_dimensions);
		host << device;
		return host;
	}

	template<typename T, size_t N>
	cuda_multi_dim_vector<T, N> to_device(host_multi_dim_vector<T, N> const& host)
	{
		return{ host.view().data(), host.m_dimensions };
	}

	template<typename T, size_t N>
	auto& operator>>(host_multi_dim_vector<T, N> const& host, cuda_multi_dim_vector<T, N>& device)
	{
		device = std::move(to_device(host));
		return device;
	}
}

#endif // !__multi_dim_vector_hpp__