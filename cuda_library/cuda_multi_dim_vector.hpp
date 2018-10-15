#ifndef __cuda_multi_dim_vector_hpp__
#define __cuda_multi_dim_vector_hpp__

#include <cstddef>
#include <cassert>
#include <type_traits>

#include "defines.hpp"
#include "variadic_arg_helpers.hpp"

namespace aks
{
	enum e_device_type
	{
		HOST,
		DEVICE,
		UNKNOWN
	};

	template<typename _value_type, std::size_t _dimensions>
	struct multi_dim_vector;

	namespace multi_dim_vector_detail
	{
		AKS_FUNCTION_PREFIX_ATTR inline size_t multiply(size_t const * begin, size_t const* end)
		{
			size_t ret = 1;
			for (auto it = begin; it != end; ++it)
			{
				ret *= *it;
			}
			return ret;
		}

		template<size_t N>
		AKS_FUNCTION_PREFIX_ATTR size_t index(size_t const max_dims[N])
		{
			return 0;
		}

		template<size_t N, typename T, typename... Ts>
		AKS_FUNCTION_PREFIX_ATTR size_t index(size_t const max_dims[N], T idx, Ts... ts)
		{
			static_assert(sizeof...(ts) + 1 <= N, "failed");
			auto const begin = N - (sizeof...(ts) + 1) + 1; //use the next one.
			auto const mult = multiply(&max_dims[begin], &max_dims[N]);
			auto const add = index<N>(max_dims, ts...);
			if (idx < max_dims[begin - 1]) {
				void* brkpnt = nullptr;
				//printf("idx(%s) >= range(%s)\n", idx, max_dims[begin - 1]);
			}
			auto const ret = idx * mult + add;
			//std::cout << idx << " * " << mult << " + " << add << " (" << begin << "," << sizeof...(ts) << ") = " << ret << std::endl;
			return ret;
		}

		template<size_t N, typename T>
		AKS_FUNCTION_PREFIX_ATTR size_t index_with_products(size_t const max_dims[N], T idx)
		{
			return idx;
		}

		template<size_t N, typename T, typename... Ts>
		AKS_FUNCTION_PREFIX_ATTR size_t index_with_products(size_t const max_dims[N], T idx, Ts... ts)
		{
			static_assert(sizeof...(ts) + 1 <= N, "failed");
			auto const add = index_with_products<N>(max_dims, ts...);
			auto const begin = N - (sizeof...(ts) + 1) + 1; //use the next one.
			auto const mult = max_dims[begin];
			if (idx < max_dims[begin - 1]) {
				void* brkpnt = nullptr;
				//printf("idx(%s) >= range(%s)\n", idx, max_dims[begin - 1]);
			}
			auto const ret = idx * mult + add;
			//std::cout << idx << " * " << mult << " + " << add << " (" << begin << "," << sizeof...(ts) << ") = " << ret << std::endl;
			return ret;
		}

		AKS_FUNCTION_PREFIX_ATTR inline void copy(size_t* /*data*/)
		{
		}

		template<typename T, typename... Ts>
		AKS_FUNCTION_PREFIX_ATTR void copy(size_t* data, T t, Ts... ts)
		{
			data[0] = t;
			copy(data + 1, ts...);
		}

		AKS_FUNCTION_PREFIX_ATTR inline void product(size_t* /*data*/)
		{
		}

		template<typename T, typename... Ts>
		AKS_FUNCTION_PREFIX_ATTR void product(size_t* data, T t, Ts... ts)
		{
			data[0] = aks::variadic_arg_helpers::reduce<aks::variadic_arg_helpers::product>::apply(t, ts...);
			product(data + 1, ts...);
		}

		template<size_t N>
		struct max_dimension
		{
			template<size_t M>
			AKS_FUNCTION_PREFIX_ATTR static size_t apply(size_t const* products)
			{
				return products[M] / products[M + 1];
			}
		};

		template<>
		struct max_dimension<1>
		{
			template<size_t M>
			AKS_FUNCTION_PREFIX_ATTR static size_t apply(size_t const* products)
			{
				return products[M];
			}
		};
	}

	template<typename _value_type, std::size_t _dimensions>
	struct multi_dim_vector
	{
		typedef _value_type value_type;
		typedef value_type const const_value_type;
		typedef value_type& reference;
		typedef value_type const& const_reference;
		typedef value_type* iterator;
		typedef value_type const* const_iterator;
		typedef value_type* pointer;
		typedef value_type const* const_pointer;
		typedef size_t size_type;
		typedef std::ptrdiff_t difference_type;
		enum { dimensions = _dimensions };

		template<typename... Ds>
		AKS_FUNCTION_PREFIX_ATTR multi_dim_vector(e_device_type device_type, pointer data, Ds... dims) : m_device_type(device_type), m_data(data), m_products()
		{
			static_assert(dimensions == sizeof...(dims), "incorrect number of arguments provided");
			multi_dim_vector_detail::product(m_products, dims...);
		}

		template<typename T>
		AKS_FUNCTION_PREFIX_ATTR multi_dim_vector(multi_dim_vector<T, dimensions> const& other) : m_device_type(other.m_device_type), m_data(other.m_data), m_products()
		{
			for (size_t i = 0; i < dimensions; ++i)
				m_products[i] = other.m_products[i];
		}

		template<typename... Ds>
		AKS_FUNCTION_PREFIX_ATTR reference operator()(Ds... idxs) { return data()[index(idxs...)]; }

		template<typename... Ds>
		AKS_FUNCTION_PREFIX_ATTR const_reference operator()(Ds... idxs) const { return data()[index(idxs...)]; }

		AKS_FUNCTION_PREFIX_ATTR pointer data() { return m_data; }
		AKS_FUNCTION_PREFIX_ATTR const_pointer data() const { return m_data; }

		AKS_FUNCTION_PREFIX_ATTR pointer begin() { return m_data; }
		AKS_FUNCTION_PREFIX_ATTR const_pointer begin() const { return m_data; }
		AKS_FUNCTION_PREFIX_ATTR const_pointer cbegin() const { return m_data; }

		AKS_FUNCTION_PREFIX_ATTR pointer end() { return data() + total_size(); }
		AKS_FUNCTION_PREFIX_ATTR const_pointer end() const { return data() + total_size(); }
		AKS_FUNCTION_PREFIX_ATTR const_pointer cend() const { return data() + total_size(); }

		AKS_FUNCTION_PREFIX_ATTR size_type total_size() const
		{
			return m_products[0];
		}

		template<typename other_type, size_t other_dimensions>
		AKS_FUNCTION_PREFIX_ATTR bool is_shape_same_as(multi_dim_vector<other_type, other_dimensions> const& other) const
		{
			if (other.dimensions != dimensions) {
				return false;
			}

			for (size_t i = 0; i < dimensions; ++i) {
				if (this->m_products[i] != other.m_products[i])
					return false;
			}

			return true;
		}

		template<size_t N>
		AKS_FUNCTION_PREFIX_ATTR size_type max_dimension() const
		{
			static_assert(N < dimensions, "Index is more than dimensions");
			return multi_dim_vector_detail::max_dimension<dimensions - N>::template apply<N>(m_products);
		}

		AKS_FUNCTION_PREFIX_ATTR e_device_type device_type() const
		{
			return this - m_device_type;
		}

		template<typename T>
		AKS_FUNCTION_PREFIX_ATTR bool operator==(multi_dim_vector<T, dimensions> const& other) const
		{
			if (this - device_type() != other.device_type())
				return false;

			if (this->data() != other.data())
				return false;

			for (size_t i = 0; i < dimensions; ++i)
				if (this->m_products[i] != other.m_products[i])
					return false;

			return true;
		}

		template<typename T>
		AKS_FUNCTION_PREFIX_ATTR bool operator!=(multi_dim_vector<T, dimensions> const& other) const
		{
			return !(*this == other);
		}

	private:
		template<typename... Ds>
		AKS_FUNCTION_PREFIX_ATTR size_type index(size_type idx, Ds... idxs) const
		{
			auto ret = multi_dim_vector_detail::index_with_products<dimensions>(m_products, idx, idxs...);
			//if (ret >= this->total_size()) {
			//	printf("aaahhhhggg....\n");
			//}
			return ret;
		}

		e_device_type m_device_type;
		pointer m_data;
		size_type m_products[dimensions];

		template<typename T, std::size_t D>
		friend struct multi_dim_vector;

		template<typename T>
		friend struct multi_dim_iterator;

		template<typename T>
		friend struct const_multi_dim_iterator;
	};

	template<size_t X, typename T, size_t N>
	AKS_FUNCTION_PREFIX_ATTR size_t get_max_dim(multi_dim_vector<T, N> const& v)
	{
		return v.template max_dimension<X>();
	}

	template<typename X, typename T, size_t N>
	AKS_FUNCTION_PREFIX_ATTR size_t get_max_dim(multi_dim_vector<T, N> const& v)
	{
		return v.template max_dimension<X::value>();
	}

	template<typename T, typename... Ns>
	AKS_FUNCTION_PREFIX_ATTR auto make_multi_dim_vector(e_device_type d, T* data, Ns... ns) -> multi_dim_vector < T, sizeof...(Ns)>
	{
		return multi_dim_vector<T, sizeof...(Ns)>(d, data, ns...);
	}

	template<typename T, typename U, size_t N, size_t M>
	AKS_FUNCTION_PREFIX_ATTR bool is_same_shape(aks::multi_dim_vector<T, N> const& a, aks::multi_dim_vector<U, M> const& b)
	{
		return a.is_shape_same_as(b);
	}
}

//http://coliru.stacked-crooked.com/a/5a7ef607b2b88bd6
//<?xml version="1.0" encoding="utf-8"?>
//<AutoVisualizer xmlns="http://schemas.microsoft.com/vstudio/debugger/natvis/2010">
//  <Type Name="multi_dim_vector&lt;*,1&gt;">
//    <DisplayString>{{ size={{{m_max_dim}}} }}</DisplayString>
//    <Expand>
//        <Item Name="[size 0]">m_max_dim</Item>
//        <!-- <ArrayItems>
//            <Size>m_max_dim</Size>
//            <ValuePointer>m_data</ValuePointer>
//        </ArrayItems> -->
//    </Expand>
//  </Type>
//
//  <Type Name="multi_dim_vector&lt;*,2&gt;">
//    <DisplayString>{{ size={{{m_max_dim},{m_data.m_max_dim}}} }</DisplayString>
//    <Expand>
//        <Item Name="[size 0]">m_max_dim</Item>
//        <Item Name="[size 1]">m_data.m_max_dim</Item>
//        <Item Name="[total]">m_max_dim * m_data.m_max_dim</Item>
//        <!-- <ArrayItems>
//            <Size>m_max_dim * m_data.m_max_dim</Size>
//            <ValuePointer>m_data.m_data</ValuePointer>
//        </ArrayItems> -->
//    </Expand>
//  </Type>
//
//  <Type Name="multi_dim_vector&lt;*,3&gt;">
//    <DisplayString>{{ size={{{m_max_dim},{m_data.m_max_dim},{m_data.m_data.m_max_dim}}} }}</DisplayString>
//    <Expand>
//        <Item Name="[size 0]">m_max_dim</Item>
//        <Item Name="[size 1]">m_data.m_max_dim</Item>
//        <Item Name="[size 2]">m_data.m_data.m_max_dim</Item>
//        <Item Name="[total]">m_max_dim * m_data.m_max_dim * m_data.m_data.m_max_dim</Item>
//        <!-- <ArrayItems>
//            <Size>m_max_dim * m_data.m_max_dim * m_data.m_data.m_max_dim</Size>
//            <ValuePointer>m_data.m_data.m_data</ValuePointer>
//        </ArrayItems> -->
//    </Expand>
//  </Type>
//
//  <Type Name="multi_dim_vector&lt;*,4&gt;">
//    <DisplayString>{{ size={{{m_max_dim},{m_data.m_max_dim},{m_data.m_data.m_max_dim},{m_data.m_data.m_data.m_max_dim}}} }}</DisplayString>
//    <Expand>
//        <Item Name="[size 0]">m_max_dim</Item>
//        <Item Name="[size 1]">m_data.m_max_dim</Item>
//        <Item Name="[size 2]">m_data.m_data.m_max_dim</Item>
//        <Item Name="[size 3]">m_data.m_data.m_data.m_max_dim</Item>
//        <Item Name="[total]">m_max_dim * m_data.m_max_dim * m_data.m_data.m_max_dim * m_data.m_data.m_data.m_max_dim</Item>
//        <!-- <ArrayItems>
//            <Size>m_max_dim * m_data.m_max_dim * m_data.m_data.m_max_dim * m_data.m_data.m_data.m_max_dim</Size>
//            <ValuePointer>m_data.m_data.m_data.m_data</ValuePointer>
//        </ArrayItems> -->
//    </Expand>
//  </Type>
//</AutoVisualizer>

#endif // !__cuda_multi_dim_vector_hpp__