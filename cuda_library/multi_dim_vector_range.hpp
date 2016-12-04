#ifndef __multi_dim_vector_range__hpp__
#define __multi_dim_vector_range__hpp__

#include "defines.hpp"

namespace aks
{
	template<typename _iterator_type>
	struct multi_dim_vector_range
	{
		typedef _iterator_type iterator_type;

		AKS_FUNCTION_PREFIX_ATTR multi_dim_vector_range(iterator_type const b, iterator_type const e)
			: m_begin(b)
			, m_end(e) {}

		AKS_FUNCTION_PREFIX_ATTR iterator_type begin() const { return m_begin; }
		AKS_FUNCTION_PREFIX_ATTR iterator_type end()   const { return m_end; }

		iterator_type m_begin;
		iterator_type m_end;
	};

	template<typename iterator_type>
	AKS_FUNCTION_PREFIX_ATTR auto make_multi_dim_vector_range_from_iterators(iterator_type const begin, iterator_type const end)
	{
		return multi_dim_vector_range<iterator_type>(begin, end);
	}

	template<typename view_type, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto make_multi_dim_vector_range(view_type& view, args... as)
	{
		return make_multi_dim_vector_range_from_iterators(begin(view, as...), end(view, as...));
	}

	template<typename view_type, typename... args>
	AKS_FUNCTION_PREFIX_ATTR auto make_multi_dim_vector_range(view_type const& view, args... as)
	{
		return make_multi_dim_vector_range_from_iterators(begin(view, as...), end(view, as...));
	}
}

#endif // !__multi_dim_vector_range__hpp__

