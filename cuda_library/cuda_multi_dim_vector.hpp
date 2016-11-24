#ifndef __cuda_multi_dim_vector_hpp__
#define __cuda_multi_dim_vector_hpp__

#include <cstddef>
#include <cassert>

#include "defines.hpp"
#include "variadic_arg_helpers.hpp"

namespace aks
{
template<typename _value_type, std::size_t _dimensions>
struct multi_dim_vector;

namespace multi_dim_vector_detail
{
	AKS_FUNCTION_PREFIX_ATTR size_t multiply(size_t const * begin, size_t const* end)
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
		static_assert(sizeof...(ts)+1 <= N, "failed");
		auto const begin = N - (sizeof...(ts)+1) + 1; //use the next one.
		auto const mult = multiply(&max_dims[begin], &max_dims[N]);
		auto const add = index<N>(max_dims, ts...);
		//assert(idx < max_dims[begin - 1]);
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
		static_assert(sizeof...(ts)+1 <= N, "failed");
		auto const add = index_with_products<N>(max_dims, ts...);
		auto const begin = N - (sizeof...(ts)+1) + 1; //use the next one.
		auto const mult = max_dims[begin];		
		//assert(idx < max_dims[begin - 1]);
		auto const ret = idx * mult + add;
		//std::cout << idx << " * " << mult << " + " << add << " (" << begin << "," << sizeof...(ts) << ") = " << ret << std::endl;
		return ret;
	}

	AKS_FUNCTION_PREFIX_ATTR void copy(size_t* /*data*/)
	{

	}

	template<typename T, typename... Ts>
	AKS_FUNCTION_PREFIX_ATTR void copy(size_t* data, T t, Ts... ts)
	{
		data[0] = t;
		copy(data + 1, ts...);
	}

	AKS_FUNCTION_PREFIX_ATTR void product(size_t* /*data*/)
	{

	}

	template<typename T, typename... Ts>
	AKS_FUNCTION_PREFIX_ATTR void product(size_t* data, T t, Ts... ts)
	{
		data[0] = aks::variadic_arg_helpers::reduce<aks::variadic_arg_helpers::product>::apply(t,ts...);
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
	AKS_FUNCTION_PREFIX_ATTR multi_dim_vector(pointer data, Ds... dims) : m_data(data), m_products()
	{
		multi_dim_vector_detail::product(m_products, dims...);
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

	template<size_t N>
	AKS_FUNCTION_PREFIX_ATTR size_type max_dimension() const
	{
		static_assert(N < dimensions, "Index is more than dimensions");
		return multi_dim_vector_detail::max_dimension<dimensions - N>::template apply<N>(m_products);
	}
private:
	template<typename... Ds>
	AKS_FUNCTION_PREFIX_ATTR size_type index(size_type idx, Ds... idxs) const
	{
		return multi_dim_vector_detail::index_with_products<dimensions>(m_products, idx, idxs...);
	}

	pointer m_data;
	size_type m_products[dimensions];
};

template<size_t X, typename T, size_t N>
AKS_FUNCTION_PREFIX_ATTR size_t get_max_dim(multi_dim_vector<T, N> const& v) { return v.template max_dimension<X>(); }

template<typename X, typename T, size_t N>
AKS_FUNCTION_PREFIX_ATTR size_t get_max_dim(multi_dim_vector<T, N> const& v) { return v.template max_dimension<X::value>(); }

template<typename T, typename... Ns>
auto make_multi_dim_vector(T* data, Ns... ns)->multi_dim_vector <T, sizeof...(Ns)> { return multi_dim_vector <T, sizeof...(Ns)>(data, ns...); }

}
//#include <iostream>
//#include <vector>
//template<typename binary_op>
//struct reduce
//{
//    template<typename T, typename... Ts>
//    static T apply(T x, Ts... xs)
//    {
//        return binary_op::apply(x, reduce<binary_op>::apply(xs...));
//    }
//
//    template<typename T>
//    static T apply(T x)
//    {
//        return x;
//    }
//};
//
//struct product
//{
//    template<typename T>
//    static T apply(T x, T y)
//    {
//        return x * y;
//    }
//};
//
//struct add
//{
//    template<typename T>
//    static T apply(T x, T y)
//    {
//        return x + y;
//    }
//};
//
//template<typename T, typename... Ns>
//auto make_multi_dim_vector(std::vector<T>& data, Ns... ns)->multi_dim_vector <T, sizeof...(Ns)>
//{
//    assert(reduce<product>::apply(ns...) <= data.size());
//    return multi_dim_vector <T, sizeof...(Ns)>(data.data(), ns...);
//}
//
//template<typename T, typename... Ns>
//auto make_multi_dim_vector(std::vector<T> const& data, Ns... ns)->multi_dim_vector <T const, sizeof...(Ns)>
//{
//    assert(reduce<product>::apply(ns...) <= data.size());
//    return multi_dim_vector <T const, sizeof...(Ns)>(data.data(), ns...);
//}
//
//namespace tags
//{
//    struct X { enum{ value = 0 }; };
//    struct Y { enum{ value = 1 }; };
//    struct Z { enum{ value = 2 }; };
//    struct W { enum{ value = 3 }; };
//}
//
//namespace aks{
//    template<typename T>
//    struct MyVec
//    {
//        int m_size;
//        T* m_data;
//    };
//}
//
//#include<utility>
//#include<tuple>
//
//template<typename T, std::size_t N>
//using multi_dim_vector_with_memory = std::tuple<std::vector<T>, multi_dim_vector<T, N>>;
//
//template<typename T, typename... Ns>
//auto make_multi_dim_vector_with_memory(Ns... ns)->multi_dim_vector_with_memory <T, sizeof...(Ns)>
//{
//    std::vector<T> data(reduce<product>::apply(ns...));
//    auto x = make_multi_dim_vector(data, ns...);
//    return multi_dim_vector_with_memory<T, sizeof...(Ns)>(std::move(data), std::move(x));
//}
//
//
//int main()
//{
//    std::vector<int> a = { 2, 3, 4 };
//    aks::MyVec<int> p = { a.size(), &a[0] };
//    aks::MyVec<int> q = { 0, nullptr };
//
//    size_t const X = 5, Y = 3, Z = 2, W = 4;
//    std::vector<int long long> v, w;
//    v.resize(X*Y*Z*W);
//    w.resize(X*Y*Z*W);
//
//    auto const u = v;
//    auto xu = make_multi_dim_vector(u, X, Y, Z, W);
//
//    //auto x = make_multi_dim_vector(v, X, Y, Z, W);
//
//    auto xm = make_multi_dim_vector_with_memory<int>(X, Y, Z, W);
//    auto& x = std::get<1>(xm);
//
//    for (size_t i = 0; i < X; ++i)
//        for (size_t j = 0; j < Y; ++j)
//            for (size_t k = 0; k < Z; ++k)
//                for (size_t l = 0; l < W; ++l)
//                {
//                    auto const n = &x(i, j, k, l) - x.begin();
//                    w[n] += 1;
//                    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", i, j, k, l, n, w[n], xu(i, j, k, l));
//                    x(i, j, k, l) = n;
//                }
//
//
//    for (auto i = &x(2, 2, 1, 0), e = i + get_max_dim<tags::W>(x); i != e; ++i)
//        printf("%d ", *i);
//
//    for (auto i : x) { printf("%d\n", i); }
//    //x(0, 40, 1, 40);
//
//    printf("\n0:X:%d 1:Y:%d 2:Z:%d 3:W:%d\n", get_max_dim<0>(x), get_max_dim<tags::Y>(x), get_max_dim<tags::Z>(x), get_max_dim<tags::W>(x));
//    printf("%d\n", x.end() - x.begin());
//    printf("%d %d %d\n..", sizeof(multi_dim_vector<int, 4>), sizeof(multi_dim_vector<int, 4>::pointer), sizeof(multi_dim_vector<int, 4>::size_type));
//    return 0;
//}
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
