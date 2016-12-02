#ifndef __multi_dim_vector_with_memory_hpp__
#define __multi_dim_vector_with_memory_hpp__

#include "cuda_multi_dim_vector.hpp"
#include "tuple_helper_utilities.hpp"
#include "variadic_arg_helpers.hpp"

namespace aks
{
	namespace detail
	{
		template<typename V, size_t... Is, typename... Ts>
		multi_dim_vector<V, sizeof...(Ts)> make_multi_dim_vector_impl(V* v, ::aks::tuple_utils::index_sequence<Is...>, std::tuple<Ts...> const& t)
		{
			return ::aks::make_multi_dim_vector(v, std::get<Is>(t)...);
		}

		template<typename V, typename... Ts>
		multi_dim_vector<V, sizeof...(Ts)> make_multi_dim_vector(V* v, std::tuple<Ts...> const& t)
		{
			return make_multi_dim_vector_impl(v, ::aks::tuple_utils::make_index_sequence<sizeof...(Ts)>(), t);
		}

		template<typename _T, size_t _N, typename _Storage>
		struct multi_dim_vector_with_memory
		{
			typedef _T value_type;
			typedef value_type const const_value_type;
			typedef value_type* value_type_pointer;
			typedef const_value_type * const_value_type_pointer;
			enum { dimensions = _N };

			typedef _Storage storage_type;
			typedef ::aks::tuple_utils::tuple_of_length<size_t, dimensions> dimensions_type;
			typedef multi_dim_vector<value_type, dimensions> view_type;
			typedef multi_dim_vector<const_value_type, dimensions> const_view_type;

			multi_dim_vector_with_memory(dimensions_type ts) : m_data(tuple_utils::reduce(ts, variadic_arg_helpers::product())), m_dimensions(ts)
			{
			}

			multi_dim_vector_with_memory(value_type_pointer data, dimensions_type ts) : m_data(tuple_utils::reduce(ts, variadic_arg_helpers::product()), data), m_dimensions(ts)
			{
			}

			multi_dim_vector_with_memory(const_value_type_pointer data, dimensions_type ts) : m_data(tuple_utils::reduce(ts, variadic_arg_helpers::product()), data), m_dimensions(ts)
			{
			}

			template<typename... Ts>
			multi_dim_vector_with_memory(Ts... ts) : m_data(variadic_arg_helpers::reduce<variadic_arg_helpers::product>::apply(ts...)), m_dimensions(ts...)
			{
				static_assert(sizeof...(ts) == dimensions, "Incorrect number of arguments");
			}

			template<typename... Ts>
			multi_dim_vector_with_memory(value_type_pointer data, Ts... ts) : m_data(variadic_arg_helpers::reduce<variadic_arg_helpers::product>::apply(ts...), data), m_dimensions(ts...)
			{
				static_assert(sizeof...(ts) == dimensions, "Incorrect number of arguments");
			}

			template<typename... Ts>
			multi_dim_vector_with_memory(const_value_type_pointer data, Ts... ts) : m_data(variadic_arg_helpers::reduce<variadic_arg_helpers::product>::apply(ts...), data), m_dimensions(ts...)
			{
				static_assert(sizeof...(ts) == dimensions, "Incorrect number of arguments");
			}

			view_type       view() { return detail::make_multi_dim_vector(m_data.data(), m_dimensions); }
			const_view_type view() const { return detail::make_multi_dim_vector(static_cast<const_value_type const*>(m_data.data()), m_dimensions); }

			const_view_type cview() const { return detail::make_multi_dim_vector(static_cast<const_value_type const*>(m_data.data()), m_dimensions); }

			storage_type m_data;
			dimensions_type m_dimensions;
		};
	}
}

#endif // !__multi_dim_vector_with_memory_hpp__

