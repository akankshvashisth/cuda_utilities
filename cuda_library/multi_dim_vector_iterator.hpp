#ifndef __multi_dim_vector_iterator_hpp__
#define __multi_dim_vector_iterator_hpp__

#include "defines.hpp"
#include "multi_dim_vector.hpp"
#include <utility>

namespace aks
{
	namespace iterator_detail
	{
		template<typename View, std::size_t... I>
		AKS_FUNCTION_PREFIX_ATTR auto& get_element_impl(View& view, std::size_t const * const dims, std::index_sequence<I...>)
		{
			return view(dims[I]...);
		}

		template<typename View>
		AKS_FUNCTION_PREFIX_ATTR auto& get_element(View& view, std::size_t const * const dims)
		{
			return get_element_impl(view, dims, std::make_index_sequence<View::dimensions>());
		}

		template<typename View, std::size_t... I>
		AKS_FUNCTION_PREFIX_ATTR auto& get_element_impl(View const& view, std::size_t const * const dims, std::index_sequence<I...>)
		{
			return view(dims[I]...);
		}

		template<typename View>
		AKS_FUNCTION_PREFIX_ATTR auto& get_element(View const& view, std::size_t const * const dims)
		{
			return get_element_impl(view, dims, std::make_index_sequence<View::dimensions>());
		}

		template <typename A, typename B, typename... C>
		struct index_of
		{
			enum { value = std::is_same<A, B>::value ? 0 : index_of<A, C...>::value + 1 };
		};

		template <typename A, typename B>
		struct index_of<A, B>
		{
			enum { value = std::is_same<A, B>::value ? 0 : 200000 /*some very large number*/ };
		};
	}

	struct token
	{
		AKS_FUNCTION_PREFIX_ATTR operator std::size_t() { return 0; }
	};

	template<typename _value_type>
	struct const_multi_dim_iterator
	{
		typedef _value_type value_type;
		typedef value_type const const_value_type;
		typedef const_value_type* const_pointer_type;
		typedef std::size_t size_type;

		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator() : m_data_pointer(), m_increment() {}

		template<typename _V, size_t _D>
		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator(multi_dim_vector<_V, _D> const& view, size_type const * const dims, size_type const moving_dimension)
			: m_data_pointer(&iterator_detail::get_element(view, dims))
			, m_increment(multi_dim_vector<_V, _D>::dimensions > moving_dimension+1 ? view.m_products[moving_dimension+1] : 1){}

		AKS_FUNCTION_PREFIX_ATTR const_value_type& operator*() { return *m_data_pointer; }
		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator& operator++() { m_data_pointer += m_increment; return *this; }
		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator operator++(int) { auto temp = *this; ++(*this); return temp; }
		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator& operator--() { m_data_pointer -= m_increment; return *this; }
		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator operator--(int) { auto temp = *this; --(*this); return temp; }
		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator& operator+=(size_type const increment) { m_data_pointer += m_increment * increment; return *this; }
		AKS_FUNCTION_PREFIX_ATTR const_multi_dim_iterator& operator-=(size_type const increment) { m_data_pointer -= m_increment * increment; return *this; }

		const_pointer_type m_data_pointer;
		size_type m_increment;
	};

	template<typename _value_type>
	struct multi_dim_iterator
	{
		typedef _value_type value_type;
		typedef value_type* pointer_type;
		typedef std::size_t size_type;

		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator() : m_data_pointer(), m_increment() {}

		template<typename _V, size_t _D>
		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator(multi_dim_vector<_V, _D>& view, size_type const * const dims, size_type const moving_dimension)
			: m_data_pointer(&iterator_detail::get_element(view, dims))
			, m_increment(multi_dim_vector<_V, _D>::dimensions > moving_dimension + 1 ? view.m_products[moving_dimension + 1] : 1) {}

		AKS_FUNCTION_PREFIX_ATTR value_type& operator*() { return *m_data_pointer; }
		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator& operator++() { m_data_pointer += m_increment; return *this; }
		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator operator++(int) { auto temp = *this; ++(*this); return temp; }
		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator& operator--() { m_data_pointer -= m_increment; return *this; }
		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator operator--(int) { auto temp = *this; --(*this); return temp; }
		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator& operator+=(size_type const increment) { m_data_pointer += m_increment * increment; return *this; }
		AKS_FUNCTION_PREFIX_ATTR multi_dim_iterator& operator-=(size_type const increment) { m_data_pointer -= m_increment * increment; return *this; }
		AKS_FUNCTION_PREFIX_ATTR operator const_multi_dim_iterator<value_type>() const
		{ 
			const_multi_dim_iterator<value_type> ret;
			ret.m_data_pointer = this->m_data_pointer;
			ret.m_increment = this->m_increment;
			return ret;
		}

		pointer_type m_data_pointer;
		size_type m_increment;
	};


	template<typename value_type, size_t dimensions, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto begin(multi_dim_vector<value_type, dimensions>& view, args... as)
	{
		static_assert(dimensions == sizeof...(as), "mismatch");
		std::size_t dimension[dimensions] = { std::size_t(as)... };
		return multi_dim_iterator<value_type>(view, dimension, iterator_detail::index_of<token, args...>::value);
	}

	template<typename value_type, size_t dimensions, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto end(multi_dim_vector<value_type, dimensions>& view, args... as)
	{
		static_assert(dimensions == sizeof...(as), "mismatch");
		return begin(view, as...) += get_max_dim<iterator_detail::index_of<token, args...>::value>(view);
	}

	template<typename value_type, size_t dimensions, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto begin(multi_dim_vector<value_type, dimensions> const& view, args... as)
	{
		static_assert(dimensions == sizeof...(as), "mismatch");
		std::size_t dimension[dimensions] = { std::size_t(as)... };
		return const_multi_dim_iterator<value_type>(view, dimension, iterator_detail::index_of<token, args...>::value);
	}

	template<typename value_type, size_t dimensions, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto end(multi_dim_vector<value_type, dimensions> const& view, args... as)
	{
		static_assert(dimensions == sizeof...(as), "mismatch");
		return begin(view, as...) += get_max_dim<iterator_detail::index_of<token, args...>::value>(view);
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator==(const_multi_dim_iterator<value_type> const& lhs, const_multi_dim_iterator<value_type> const& rhs)
	{
		if (lhs.m_data_pointer != rhs.m_data_pointer)
			return false;
		if (lhs.m_increment != rhs.m_increment)
			return false;
		return true;
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator==(multi_dim_iterator<value_type> const& lhs, const_multi_dim_iterator<value_type> const& rhs)
	{
		if (lhs.m_data_pointer != rhs.m_data_pointer)
			return false;
		if (lhs.m_increment != rhs.m_increment)
			return false;
		return true;
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator==(const_multi_dim_iterator<value_type> const& lhs, multi_dim_iterator<value_type> const& rhs)
	{
		if (lhs.m_data_pointer != rhs.m_data_pointer)
			return false;
		if (lhs.m_increment != rhs.m_increment)
			return false;
		return true;
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator==(multi_dim_iterator<value_type> const& lhs, multi_dim_iterator<value_type> const& rhs)
	{
		if (lhs.m_data_pointer != rhs.m_data_pointer)
			return false;
		if (lhs.m_increment != rhs.m_increment)
			return false;
		return true;
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator!=(const_multi_dim_iterator<value_type> const& lhs, const_multi_dim_iterator<value_type> const& rhs)
	{
		return !(lhs == rhs);
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator!=(multi_dim_iterator<value_type> const& lhs, const_multi_dim_iterator<value_type> const& rhs)
	{
		return !(lhs == rhs);
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator!=(const_multi_dim_iterator<value_type> const& lhs, multi_dim_iterator<value_type> const& rhs)
	{
		return !(lhs == rhs);
	}

	template<typename value_type>
	AKS_FUNCTION_PREFIX_ATTR bool operator!=(multi_dim_iterator<value_type> const& lhs, multi_dim_iterator<value_type> const& rhs)
	{
		return !(lhs == rhs);
	}
	
}

#endif // !__multi_dim_vector_iterator_hpp__

