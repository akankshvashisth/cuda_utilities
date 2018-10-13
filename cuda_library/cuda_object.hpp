#ifndef __cuda_object_hpp__
#define __cuda_object_hpp__

#include "cuda_pointer.hpp"
#include <vector>

namespace aks {
	template<typename T>
	struct object_view
	{
		typedef T value_type;
		AKS_FUNCTION_PREFIX_ATTR object_view(value_type* data) :m_data(data) {}
		AKS_FUNCTION_PREFIX_ATTR operator object_view<value_type const>() {
			return object_view<value_type const>(m_data);
		}
		AKS_FUNCTION_PREFIX_ATTR value_type const * operator->() const { return m_data; }
		AKS_FUNCTION_PREFIX_ATTR value_type * operator->() { return m_data; }
		AKS_FUNCTION_PREFIX_ATTR value_type const& operator*() const { return *m_data; }
		AKS_FUNCTION_PREFIX_ATTR value_type & operator*() { return *m_data; }
		value_type* m_data;
	};

	template<typename T>
	struct cuda_object
	{
		typedef T value_type;
		object_view<value_type> view() { return object_view<value_type>(m_ptr.data()); }
		object_view<value_type const> view() const { return object_view<value_type const>(m_ptr.data()); }
		object_view<value_type const> cview() const { return object_view<value_type const>(m_ptr.data()); }
		aks::cuda_pointer<value_type> m_ptr;
		cuda_object(value_type const& v) :m_ptr(1, &v) {}
	};

	template<typename T>
	struct host_object
	{
		typedef T value_type;
		object_view<value_type> view() { return object_view<value_type>(m_ptr.data()); }
		object_view<value_type const> view() const { return object_view<value_type const>(m_ptr.data()); }
		object_view<value_type const> cview() const { return object_view<value_type const>(m_ptr.data()); }
		std::vector<value_type> m_ptr;
		host_object(value_type const& v) :m_ptr(1, v) {}
	};
}

#endif // __cuda_object_hpp__