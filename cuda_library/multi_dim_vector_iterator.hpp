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
			enum { value = std::is_same<A, B>::value ? 0 : (index_of<A, C...>::value >= 0) ? 1 + index_of<A, C...>::value : -1 };
		};

		template <typename A, typename B>
		struct index_of<A, B>
		{
			enum { value = std::is_same<A, B>::value - 1 };
		};
	}

	struct token
	{
		AKS_FUNCTION_PREFIX_ATTR operator std::size_t() { return 0; }
	};

	template<typename _value_type, std::size_t _dimensions, std::size_t _moving_dimension>
	struct iterator
	{
		typedef _value_type value_type;
		enum {
			dimensions = _dimensions
			, moving_dimension = _moving_dimension
		};
		typedef multi_dim_vector<value_type, dimensions> view_type;
		typedef typename view_type::size_type size_type;

		AKS_FUNCTION_PREFIX_ATTR iterator(view_type const view, size_type const * const dims)
			: m_dimensions()
			, m_view(view)
		{
			for (size_t i = 0; i < dimensions; ++i)
				m_dimensions[i] = dims[i];
		}

		AKS_FUNCTION_PREFIX_ATTR value_type& operator*() { return iterator_detail::get_element(m_view, m_dimensions); }

		AKS_FUNCTION_PREFIX_ATTR iterator& operator++()
		{
			++m_dimensions[moving_dimension];
			return *this;
		}

		AKS_FUNCTION_PREFIX_ATTR iterator operator++(int)
		{
			auto temp = *this;
			++(*this);
			return temp;
		}

		AKS_FUNCTION_PREFIX_ATTR iterator& operator--()
		{
			--m_dimensions[moving_dimension];
			return *this;
		}

		AKS_FUNCTION_PREFIX_ATTR iterator operator--(int)
		{
			auto temp = *this;
			--(*this);
			return temp;
		}

		AKS_FUNCTION_PREFIX_ATTR iterator& operator+=(size_type const increment)
		{
			m_dimensions[moving_dimension] += increment;
			return *this;
		}

		AKS_FUNCTION_PREFIX_ATTR iterator& operator-=(size_type const increment)
		{
			m_dimensions[moving_dimension] -= increment;
			return *this;
		}

		size_type m_dimensions[dimensions];
		view_type m_view;
	};

	template<typename value_type, size_t dimensions, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto begin(multi_dim_vector<value_type, dimensions>& view, args... as)
	{
		static_assert(dimensions == sizeof...(as), "mismatch");
		std::size_t dimension[dimensions] = { std::size_t(as)... };
		return iterator<value_type, dimensions, iterator_detail::index_of<token, args...>::value>(view, dimension);
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
		return iterator<value_type const, dimensions, iterator_detail::index_of<token, args...>::value>(view, dimension);
	}

	template<typename value_type, size_t dimensions, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto end(multi_dim_vector<value_type, dimensions> const& view, args... as)
	{
		static_assert(dimensions == sizeof...(as), "mismatch");
		return begin(view, as...) += get_max_dim<iterator_detail::index_of<token, args...>::value>(view);
	}

	template<typename value_type, std::size_t dimensions, std::size_t moving_dimension>
	AKS_FUNCTION_PREFIX_ATTR bool operator==(iterator<value_type, dimensions, moving_dimension> const& lhs, iterator<value_type, dimensions, moving_dimension> const& rhs)
	{
		if (lhs.m_view != rhs.m_view)
			return false;
		for (size_t i = 0; i < dimensions; ++i)
			if (lhs.m_dimensions[i] != rhs.m_dimensions[i])
				return false;
		return true;
	}

	template<typename value_type, std::size_t dimensions, std::size_t moving_dimension>
	AKS_FUNCTION_PREFIX_ATTR bool operator!=(iterator<value_type, dimensions, moving_dimension> const& lhs, iterator<value_type, dimensions, moving_dimension> const& rhs)
	{
		return !(lhs == rhs);
	}
	
}

#endif // !__multi_dim_vector_iterator_hpp__

