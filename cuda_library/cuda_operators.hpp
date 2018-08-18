#ifndef __cuda_operators_hpp__
#define __cuda_operators_hpp__

#include "multi_dim_vector.hpp"
#include "grid_stride_range.hpp"
#include <tuple>
#include <thrust/gather.h>

namespace aks
{
	template<typename T, size_t N, typename Func>
	__global__ void unaryOpInplaceKernel(aks::multi_dim_vector<T, N> a, Func f)
	{
		auto a_ = a.data();
		for (auto i : aks::grid_stride_range::x(size_t(0), a.total_size()))
		{
			a_[i] = f(a_[i]);
		}
	}

	template<typename V, typename T, size_t N, typename Func>
	__global__ void unaryOpKernel(aks::multi_dim_vector<V, N> b, aks::multi_dim_vector<T const, N> a, Func f)
	{
		auto a_ = a.data();
		auto b_ = b.data();
		for (auto i : aks::grid_stride_range::x(size_t(0), a.total_size()))
		{
			b_[i] = f(a_[i]);
		}
	}

	template<typename T, typename U, size_t N, typename Func>
	__global__ void binaryOpInplaceKernel(aks::multi_dim_vector<T, N> a, aks::multi_dim_vector<U const, N> const b, Func f)
	{
		auto a_ = a.data();
		auto b_ = b.data();
		for (auto i : aks::grid_stride_range::x(size_t(0), a.total_size()))
		{
			a_[i] = f(a_[i], b_[i]);
		}
	}

	template<typename V, typename T, typename U, size_t N, typename Func>
	__global__ void binaryOpKernel(aks::multi_dim_vector<V, N> c, aks::multi_dim_vector<T const, N> a, aks::multi_dim_vector<U const, N> const b, Func f)
	{
		auto a_ = a.data();
		auto b_ = b.data();
		auto c_ = c.data();
		for (auto i : aks::grid_stride_range::x(size_t(0), a.total_size()))
		{
			c_[i] = f(a_[i], b_[i]);
		}
	}


	template<typename T, typename V, typename Func, size_t N>
	__global__ void reduceDim0Kernel(aks::multi_dim_vector<T, N - 1> c, aks::multi_dim_vector<V const, N> a, Func f);

	template<typename T, typename V, typename Func, size_t N>
	__global__ void reduceDim1Kernel(aks::multi_dim_vector<T, N - 1> c, aks::multi_dim_vector<V const, N> a, Func f);

	template<typename T, typename V, typename Func>
	__global__ void reduceDim0Kernel(aks::multi_dim_vector<T, 1> c, aks::multi_dim_vector<V const, 2> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<1>(a)))
		{
			auto it = aks::begin(a, token(), i), e = aks::end(a, token(), i);
			T reduced = *it;
			++it;
			for (; it != e; ++it) {
				reduced = f(reduced, *it);
			}
			c(i) = reduced;
		}
	}

	template<typename T, typename V, typename Func>
	__global__ void reduceDim1Kernel(aks::multi_dim_vector<T, 1> c, aks::multi_dim_vector<V const, 2> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
		{
			auto it = aks::begin(a, i, token()), e = aks::end(a, i, token());
			T reduced = *it;
			++it;
			for (; it != e; ++it) {
				reduced = f(reduced, *it);
			}
			c(i) = reduced;
		}
	}

	template<typename T, typename V, typename Func>
	__global__ void reduceDim0Kernel(aks::multi_dim_vector<T, 2> c, aks::multi_dim_vector<V const, 3> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<1>(a)))
			for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<2>(a)))
		{
			auto it = aks::begin(a, token(), i, j), e = aks::end(a, token(), i, j);
			T reduced = *it;
			++it;
			for (; it != e; ++it) {
				reduced = f(reduced, *it);
			}
			c(i, j) = reduced;
		}
	}

	template<typename T, typename V, typename Func>
	__global__ void reduceDim1Kernel(aks::multi_dim_vector<T, 2> c, aks::multi_dim_vector<V const, 3> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
			for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<2>(a)))
		{
			auto it = aks::begin(a, i, token(), j), e = aks::end(a, i, token(), j);
			T reduced = *it;
			++it;
			for (; it != e; ++it) {
				reduced = f(reduced, *it);
			}
			c(i, j) = reduced;
		}
	}

	template<typename T, typename V, typename Func>
	__global__ void reduceDim2Kernel(aks::multi_dim_vector<T, 2> c, aks::multi_dim_vector<V const, 3> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
			for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<1>(a)))
			{
				auto it = aks::begin(a, i, j, token()), e = aks::end(a, i, j, token());
				T reduced = *it;
				++it;
				for (; it != e; ++it) {
					reduced = f(reduced, *it);
				}
				c(i, j) = reduced;
			}
	}


	template<typename T, typename V, typename Func>
	__global__ void reduceDim0Kernel(aks::multi_dim_vector<T, 3> c, aks::multi_dim_vector<V const, 4> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<1>(a)))
			for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<2>(a)))
				for (auto k : aks::grid_stride_range::z(size_t(0), get_max_dim<3>(a)))
			{
				auto it = aks::begin(a, token(), i, j, k), e = aks::end(a, token(), i, j, k);
				T reduced = *it;
				++it;
				for (; it != e; ++it) {
					reduced = f(reduced, *it);
				}
				c(i, j, k) = reduced;
			}
	}

	template<typename T, typename V, typename Func>
	__global__ void reduceDim1Kernel(aks::multi_dim_vector<T, 3> c, aks::multi_dim_vector<V const, 4> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
			for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<2>(a)))
				for (auto k : aks::grid_stride_range::z(size_t(0), get_max_dim<3>(a)))
			{
				auto it = aks::begin(a, i, token(), j, k), e = aks::end(a, i, token(), j, k);
				T reduced = *it;
				++it;
				for (; it != e; ++it) {
					reduced = f(reduced, *it);
				}
				c(i, j, k) = reduced;
			}
	}

	template<typename T, typename V, typename Func>
	__global__ void reduceDim2Kernel(aks::multi_dim_vector<T, 3> c, aks::multi_dim_vector<V const, 4> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
			for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<1>(a)))
				for (auto k : aks::grid_stride_range::z(size_t(0), get_max_dim<3>(a)))
			{
				auto it = aks::begin(a, i, j, token(), k), e = aks::end(a, i, j, token(), k);
				T reduced = *it;
				++it;
				for (; it != e; ++it) {
					reduced = f(reduced, *it);
				}
				c(i, j, k) = reduced;
			}
	}

	template<typename T, typename V, typename Func>
	__global__ void reduceDim3Kernel(aks::multi_dim_vector<T, 3> c, aks::multi_dim_vector<V const, 4> a, Func f)
	{
		for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
			for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<1>(a)))
				for (auto k : aks::grid_stride_range::z(size_t(0), get_max_dim<2>(a)))
				{
					auto it = aks::begin(a, i, j, k, token()), e = aks::end(a, i, j, k, token());
					T reduced = *it;
					++it;
					for (; it != e; ++it) {
						reduced = f(reduced, *it);
					}
					c(i, j, k) = reduced;
				}
	}

	std::tuple<dim3, dim3> calculateDims(int x, int y, int z)
	{
		auto blockDimCalc = [](int a) { return a > 1024 ? a / 1024 : 1; };
		auto threadDimCalc = [](int a) { return a <= 1024 ? a : 1024; };
		dim3 blockDim(blockDimCalc(x), blockDimCalc(y), blockDimCalc(z));
		dim3 threadDim(threadDimCalc(x), threadDimCalc(y), threadDimCalc(z));
		return { blockDim, threadDim };
	}

	template<typename T, typename V, typename Func>
	void reduceDim(aks::multi_dim_vector<T, 1> c, aks::multi_dim_vector<V const, 2> a, Func f, int axis)
	{
		switch (axis)
		{
		case 0:
			if (aks::get_max_dim<1>(a) == aks::get_max_dim<0>(c))
			{				
				auto dims = calculateDims(aks::get_max_dim<0>(c), 1, 1);
				reduceDim0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (aks::get_max_dim<0>(a) == aks::get_max_dim<0>(c))
			{
				auto dims = calculateDims(aks::get_max_dim<0>(c), 1, 1);
				reduceDim1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename T, typename V, typename Func>
	void reduceDim(aks::multi_dim_vector<T, 2> c, aks::multi_dim_vector<V const, 3> a, Func f, int axis)
	{
		switch (axis)
		{
		case 0:
			if (aks::get_max_dim<1>(a) == aks::get_max_dim<0>(c) && 
				aks::get_max_dim<2>(a) == aks::get_max_dim<1>(c))
			{				
				auto dims = calculateDims(aks::get_max_dim<0>(c), aks::get_max_dim<1>(c), 1);				
				reduceDim0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (aks::get_max_dim<0>(a) == aks::get_max_dim<0>(c) &&
				aks::get_max_dim<2>(a) == aks::get_max_dim<1>(c))
			{
				auto dims = calculateDims(aks::get_max_dim<0>(c), aks::get_max_dim<1>(c), 1);
				reduceDim1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 2:
			if (aks::get_max_dim<0>(a) == aks::get_max_dim<0>(c) &&
				aks::get_max_dim<1>(a) == aks::get_max_dim<1>(c))
			{
				auto dims = calculateDims(aks::get_max_dim<0>(c), aks::get_max_dim<1>(c), 1);
				reduceDim2Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename T, typename V, typename Func>
	void reduceDim(aks::multi_dim_vector<T, 3> c, aks::multi_dim_vector<V const, 4> a, Func f, int axis)
	{
		switch (axis)
		{
		case 0:
			if (aks::get_max_dim<1>(a) == aks::get_max_dim<0>(c) &&
				aks::get_max_dim<2>(a) == aks::get_max_dim<1>(c) &&
				aks::get_max_dim<3>(a) == aks::get_max_dim<2>(c))
			{
				auto dims = calculateDims(aks::get_max_dim<0>(c), aks::get_max_dim<1>(c), aks::get_max_dim<2>(c));
				reduceDim0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (aks::get_max_dim<0>(a) == aks::get_max_dim<0>(c) &&
				aks::get_max_dim<2>(a) == aks::get_max_dim<1>(c) &&
				aks::get_max_dim<3>(a) == aks::get_max_dim<2>(c))
			{
				auto dims = calculateDims(aks::get_max_dim<0>(c), aks::get_max_dim<1>(c), aks::get_max_dim<2>(c));
				reduceDim1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 2:
			if (aks::get_max_dim<0>(a) == aks::get_max_dim<0>(c) &&
				aks::get_max_dim<1>(a) == aks::get_max_dim<1>(c) &&
				aks::get_max_dim<3>(a) == aks::get_max_dim<2>(c))
			{
				auto dims = calculateDims(aks::get_max_dim<0>(c), aks::get_max_dim<1>(c), aks::get_max_dim<2>(c));
				reduceDim2Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 3:
			if (aks::get_max_dim<0>(a) == aks::get_max_dim<0>(c) &&
				aks::get_max_dim<1>(a) == aks::get_max_dim<1>(c) &&
				aks::get_max_dim<2>(a) == aks::get_max_dim<2>(c))
			{
				auto dims = calculateDims(aks::get_max_dim<0>(c), aks::get_max_dim<1>(c), aks::get_max_dim<2>(c));
				reduceDim3Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename V, typename T, size_t N, typename Func>
	void unaryOp(aks::multi_dim_vector<V, N> mb, aks::multi_dim_vector<T const, N> ma, Func f)
	{
		if (aks::is_same_shape(ma, mb)) {			
			auto dims = calculateDims(ma.total_size(), 1, 1);
			unaryOpKernel << <std::get<0>(dims), std::get<1>(dims) >> > (mb, ma, f);
		}
	}

	template<typename T, size_t N, typename Func>
	void unaryOpInplace(aks::multi_dim_vector<T, N> ma, Func f)
	{
		auto dims = calculateDims(ma.total_size(), 1, 1);
		unaryOpInplaceKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, f);
	}

	template<typename V, typename T, typename U, size_t N, typename Func>
	void binaryOp(aks::multi_dim_vector<V, N> mc, aks::multi_dim_vector<T const, N> ma, aks::multi_dim_vector<U const, N> const mb, Func f)
	{
		if (aks::is_same_shape(ma, mb) && aks::is_same_shape(mb, mc)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			binaryOpKernel << <std::get<0>(dims), std::get<1>(dims) >> > (mc, ma, mb, f);
		}
	}

	template<typename T, typename U, size_t N, typename Func>
	void binaryOpInplace(aks::multi_dim_vector<T, N> ma, aks::multi_dim_vector<U const, N> const mb, Func f)
	{
		if (aks::is_same_shape(ma, mb)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			binaryOpInplaceKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, mb, f);
		}
	}
	
	namespace detail_sort {
		__global__ void fillAxis0(aks::multi_dim_vector<int, 2> a)
		{
			for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
				for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<1>(a)))
			{
					a(i, j) = j;
			}
		}

		__global__ void fillIndices(aks::multi_dim_vector<int, 2> a)
		{
			int const R = get_max_dim<0>(a);
			int const C = get_max_dim<1>(a);
			for (auto r : aks::grid_stride_range::x(0, R))
				for (auto c : aks::grid_stride_range::y(0, C))
				{
					auto i = r * C + c;
					a(r, c) = (i / R) + (i%R) * C;
				}
		}

		__global__ void fillAxis1(aks::multi_dim_vector<int, 2> a)
		{
			for (auto i : aks::grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
				for (auto j : aks::grid_stride_range::y(size_t(0), get_max_dim<1>(a)))
				{
					a(i, j) = i;
				}
		}
	}

	template<typename T>
	void sortAxis(aks::cuda_multi_dim_vector<T, 2>& a, int axis)
	{		
		using ::aks::thrust_utils::begin;
		using ::aks::thrust_utils::end;
		cuda_multi_dim_vector<int, 2> temp(get_max_dim<0>(a.view()), get_max_dim<1>(a.view()));		
		auto dims = calculateDims(get_max_dim<0>(a.view()), get_max_dim<1>(a.view()), 1);

		if (axis == 0)
		{	
			detail_sort::fillAxis0 << <std::get<0>(dims), std::get<1>(dims) >> > (temp.view());
		} else if (axis == 1){								
			detail_sort::fillAxis1 << <std::get<0>(dims), std::get<1>(dims) >> > (temp.view());
		};

		thrust::stable_sort_by_key(begin(a.m_data), end(a.m_data), begin(temp.m_data));
		thrust::stable_sort_by_key(begin(temp.m_data), end(temp.m_data), begin(a.m_data));
		
		if (axis == 0) {
			cuda_multi_dim_vector<int, 2> map(get_max_dim<0>(a.view()), get_max_dim<1>(a.view()));
			detail_sort::fillIndices << <std::get<0>(dims), std::get<1>(dims) >> > (map.view());
			thrust::scatter(begin(a.m_data), end(a.m_data), begin(map.m_data), begin(a.m_data));
		}
		//unaryOp(a.view(), temp.cview(), [] AKS_FUNCTION_PREFIX_ATTR(int x) { return x; });
	}

}

#endif // !__cuda_operators_hpp__

