#ifndef __grid_stride_range_hpp__
#define __grid_stride_range_hpp__

#include "defines.hpp"

namespace aks
{
	template<typename T>
	struct stride_iterator
	{
		typedef T value_type;
		AKS_FUNCTION_PREFIX_ATTR stride_iterator(value_type current, value_type step) : m_current(current), m_step(step) {}
		AKS_FUNCTION_PREFIX_ATTR value_type operator*() { return m_current; }
		AKS_FUNCTION_PREFIX_ATTR value_type const * operator ->() { return &m_current; }
		AKS_FUNCTION_PREFIX_ATTR stride_iterator& operator++() { m_current += m_step; return *this; }
		AKS_FUNCTION_PREFIX_ATTR stride_iterator operator++(int) { auto temp = *this; ++(*this); return temp; }
		AKS_FUNCTION_PREFIX_ATTR stride_iterator& operator--() { m_current -= m_step; return *this; }
		AKS_FUNCTION_PREFIX_ATTR stride_iterator operator--(int) { auto temp = *this; --(*this); return temp; }
		AKS_FUNCTION_PREFIX_ATTR bool operator==(stride_iterator const& other) const { return m_step > 0 ? m_current >= other.m_current : m_current < other.m_current; }
		AKS_FUNCTION_PREFIX_ATTR bool operator!=(stride_iterator const& other) const { return !(*this == other); }
		value_type m_current;
		value_type m_step;
	};


	template<typename _iterator_type>
	struct range
	{
		typedef _iterator_type iterator_type;

		AKS_FUNCTION_PREFIX_ATTR range(iterator_type const b, iterator_type const e) : m_begin(b), m_end(e) {}
		AKS_FUNCTION_PREFIX_ATTR iterator_type begin() const { return m_begin; }
		AKS_FUNCTION_PREFIX_ATTR iterator_type end()   const { return m_end; }

		iterator_type m_begin;
		iterator_type m_end;
	};

	template<typename T>
	struct step_range
	{
		typedef T value_type;
		typedef stride_iterator<value_type> iterator_type;

		AKS_FUNCTION_PREFIX_ATTR step_range(T const b, T const e, T const step) : m_begin(b, step), m_end(e, step) {}
		AKS_FUNCTION_PREFIX_ATTR iterator_type begin() const { return m_begin; }
		AKS_FUNCTION_PREFIX_ATTR iterator_type end()   const { return m_end; }

		iterator_type m_begin;
		iterator_type m_end;
	};

	template<typename T>
	AKS_FUNCTION_PREFIX_ATTR step_range<T> make_stride_range(T b, T e, T step)
	{
		return step_range<T>{ b, e, step };
	}
	
	struct grid_stride_range
	{
		template<typename T>
		static AKS_FUNCTION_PREFIX_ATTR auto x(T b, T e)
		{
			return make_stride_range(T(b + blockDim.x * blockIdx.x + threadIdx.x), e, T(gridDim.x * blockDim.x));
		}

		template<typename T>
		static AKS_FUNCTION_PREFIX_ATTR auto y(T b, T e)
		{
			return make_stride_range(T(b + blockDim.y * blockIdx.y + threadIdx.y), e, T(gridDim.y * blockDim.y));
		}

		template<typename T>
		static AKS_FUNCTION_PREFIX_ATTR auto z(T b, T e)
		{
			return make_stride_range(T(b + blockDim.z * blockIdx.z + threadIdx.z), e, T(gridDim.z * blockDim.z));
		}
	};

}


#endif // !__grid_stride_range_hpp__

