#ifndef __cuda_multi_dim_vector_hpp__
#define __cuda_multi_dim_vector_hpp__

#include <cstddef>
#include <cassert>


#define AKS_FUNCTION_PREFIX_ATTR __device__ __host__

namespace aks
{
template<typename _value_type, std::size_t _dimensions>
struct multi_dim_vector;

namespace multi_dim_vector_detail
{
    template<size_t _X>
    struct max_dim
    {
        enum{ dimension = _X };
        template<typename T, size_t N>
        AKS_FUNCTION_PREFIX_ATTR static size_t apply(multi_dim_vector<T, N> const& v) { return max_dim<dimension - 1>::apply(v.m_data); }
    };

    template<>
    struct max_dim<0>
    {
        template<typename T, size_t N>
        AKS_FUNCTION_PREFIX_ATTR static size_t apply(multi_dim_vector<T, N> const& v) { return v.m_max_dim; }
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
    enum{ dimensions = _dimensions };

    template<typename... Ds>
    AKS_FUNCTION_PREFIX_ATTR multi_dim_vector(pointer data, size_type maxDim, Ds... dims) : m_max_dim(maxDim), m_data(data, dims...) {}

    template<typename... Ds>
    AKS_FUNCTION_PREFIX_ATTR reference operator()(Ds... idxs) { return data()[index(idxs...)]; }

    template<typename... Ds>
    AKS_FUNCTION_PREFIX_ATTR const_reference operator()(Ds... idxs) const { return data()[index(idxs...)]; }

    AKS_FUNCTION_PREFIX_ATTR pointer data() { return m_data.data(); }
    AKS_FUNCTION_PREFIX_ATTR const_pointer data() const { return m_data.data(); }

    AKS_FUNCTION_PREFIX_ATTR pointer begin() { return m_data.data(); }
    AKS_FUNCTION_PREFIX_ATTR const_pointer begin() const { return m_data.data(); }
    AKS_FUNCTION_PREFIX_ATTR const_pointer cbegin() const { return m_data.data(); }

    AKS_FUNCTION_PREFIX_ATTR pointer end() { return data() + total_size(); }
    AKS_FUNCTION_PREFIX_ATTR const_pointer end() const { return data() + total_size(); }
    AKS_FUNCTION_PREFIX_ATTR const_pointer cend() const { return data() + total_size(); }

    AKS_FUNCTION_PREFIX_ATTR size_type total_size() const { return (m_max_dim * total_dimension_of_interior()); }
private:
    typedef multi_dim_vector<value_type, dimensions - 1> interior_type;

    template<typename... Ds>
    AKS_FUNCTION_PREFIX_ATTR size_type index(size_type idx, Ds... idxs) const
    {
        assert(idx < m_max_dim);
        return idx * total_dimension_of_interior() + m_data.index(idxs...);
    }

    AKS_FUNCTION_PREFIX_ATTR size_t total_dimension_of_interior() const { return m_data.m_max_dim * m_data.total_dimension_of_interior(); }

    size_type m_max_dim;
    interior_type m_data;

    friend struct multi_dim_vector<value_type, dimensions + 1>;

    template<size_type _M>
    friend struct multi_dim_vector_detail::max_dim;
};

template<typename _value_type>
struct multi_dim_vector<_value_type, 1>
{
    typedef _value_type value_type;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef value_type* iterator;
    typedef value_type const* const_iterator;
    typedef value_type* pointer;
    typedef value_type const* const_pointer;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    enum{ dimensions = 1 };

    AKS_FUNCTION_PREFIX_ATTR multi_dim_vector(pointer data, std::size_t maxDim) :m_max_dim(maxDim), m_data(data){}

    AKS_FUNCTION_PREFIX_ATTR reference operator()(std::size_t idx) { return m_data[index(idx)]; }
    AKS_FUNCTION_PREFIX_ATTR const_reference operator()(std::size_t idx) const { return m_data[index(idx)]; }

    AKS_FUNCTION_PREFIX_ATTR pointer data() { return m_data; }
    AKS_FUNCTION_PREFIX_ATTR const_pointer data() const { return m_data; }

    AKS_FUNCTION_PREFIX_ATTR pointer begin() { return data(); }
    AKS_FUNCTION_PREFIX_ATTR const_pointer begin() const { return data(); }
    AKS_FUNCTION_PREFIX_ATTR const_pointer cbegin() const { return data(); }

    AKS_FUNCTION_PREFIX_ATTR pointer end() { return data() + m_max_dim; }
    AKS_FUNCTION_PREFIX_ATTR const_pointer end() const { return data() + m_max_dim; }
    AKS_FUNCTION_PREFIX_ATTR const_pointer cend() const { return data() + m_max_dim; }
private:
    AKS_FUNCTION_PREFIX_ATTR size_type index(size_type idx) const
    {
        assert(idx < m_max_dim);
        return idx;
    }

    AKS_FUNCTION_PREFIX_ATTR size_type total_dimension_of_interior() const { return 1; }

    size_type m_max_dim;
    pointer m_data;

    friend struct multi_dim_vector<value_type, dimensions + 1>;

    template<size_t _M>
    friend struct multi_dim_vector_detail::max_dim;
};

template<size_t X, typename T, size_t N>
AKS_FUNCTION_PREFIX_ATTR size_t get_max_dim(multi_dim_vector<T, N> const& v) { return multi_dim_vector_detail::max_dim<X>::apply(v); }

template<typename X, typename T, size_t N>
AKS_FUNCTION_PREFIX_ATTR size_t get_max_dim(multi_dim_vector<T, N> const& v) { return multi_dim_vector_detail::max_dim<X::value>::apply(v); }

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
