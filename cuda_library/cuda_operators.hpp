#ifndef __cuda_operators_hpp__
#define __cuda_operators_hpp__

#include "multi_dim_vector.hpp"
#include "grid_stride_range.hpp"
#include <tuple>
#include <thrust/gather.h>

namespace aks
{
	template<typename T, size_t N, typename Func>
	__global__ void unaryOpInplaceKernel(multi_dim_vector<T, N> a, Func f)
	{
		auto a_ = a.data();
		for (auto i : grid_stride_range::x(size_t(0), a.total_size()))
		{
			a_[i] = f(a_[i]);
		}
	}

	template<typename V, typename T, size_t N, typename Func>
	__global__ void unaryOpKernel(multi_dim_vector<V, N> b, multi_dim_vector<T const, N> a, Func f)
	{
		auto a_ = a.data();
		auto b_ = b.data();
		for (auto i : grid_stride_range::x(size_t(0), a.total_size()))
		{
			b_[i] = f(a_[i]);
		}
	}

	template<typename T, typename U, size_t N, typename Func>
	__global__ void binaryOpInplaceKernel(multi_dim_vector<T, N> a, multi_dim_vector<U const, N> const b, Func f)
	{
		auto a_ = a.data();
		auto b_ = b.data();
		for (auto i : grid_stride_range::x(size_t(0), a.total_size()))
		{
			a_[i] = f(a_[i], b_[i]);
		}
	}

	template<typename V, typename T, typename U, size_t N, typename Func>
	__global__ void binaryOpKernel(multi_dim_vector<V, N> c, multi_dim_vector<T const, N> a, multi_dim_vector<U const, N> const b, Func f)
	{
		auto a_ = a.data();
		auto b_ = b.data();
		auto c_ = c.data();
		for (auto i : grid_stride_range::x(size_t(0), a.total_size()))
		{
			c_[i] = f(a_[i], b_[i]);
		}
	}



	template<typename T, typename V, typename Func, size_t N>
	__global__ void reduceDim0Kernel(multi_dim_vector<T, N - 1> c, multi_dim_vector<V const, N> a, Func f);

	template<typename T, typename V, typename Func, size_t N>
	__global__ void reduceDim1Kernel(multi_dim_vector<T, N - 1> c, multi_dim_vector<V const, N> a, Func f);

#define REDUCE_DIM_1_2_KERNEL_BEGIN(S, X)																					\
	template<typename T, typename V, typename Func>																			\
	__global__ void reduceDim ## S ## Kernel(multi_dim_vector<T, 1> c, multi_dim_vector<V const, 2> a, Func f)	\
	{																														\
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim<X>(a)))												\
		{																													\

#define REDUCE_DIM_2_3_KERNEL_BEGIN(S, X, Y)																				\
	template<typename T, typename V, typename Func>																			\
	__global__ void reduceDim ## S ## Kernel(multi_dim_vector<T, 2> c, multi_dim_vector<V const, 3> a, Func f)	\
	{																														\
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim<X>(a)))												\
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim<Y>(a)))											\
			{																												\

#define REDUCE_DIM_3_4_KERNEL_BEGIN(S, X, Y, Z)																		\
	template<typename T, typename V, typename Func>																	\
	__global__ void reduceDim ## S ## Kernel(multi_dim_vector<T, 3> c, multi_dim_vector<V const, 4> a, Func f)\
	{																												\
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< X >(a)))										\
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim< Y >(a)))									\
				for (auto k : grid_stride_range::z(size_t(0), get_max_dim< Z >(a)))								\
				{																										

#define REDUCE_DIM_KERNEL_END }}

	//BINARY REDUCERS
#define REDUCE_DIM_KERNEL_MID				\
	T reduced = *it;						\
		++it;								\
		for (; it != e; ++it) {				\
				reduced = f(reduced, *it);	\
		}

	REDUCE_DIM_1_2_KERNEL_BEGIN(0, 1)
		auto it = begin(a, token(), i), e = end(a, token(), i);
	REDUCE_DIM_KERNEL_MID
		c(i) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_1_2_KERNEL_BEGIN(1, 0)
		auto it = begin(a, i, token()), e = end(a, i, token());
	REDUCE_DIM_KERNEL_MID
		c(i) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_2_3_KERNEL_BEGIN(0, 1, 2)
		auto it = begin(a, token(), i, j), e = end(a, token(), i, j);
	REDUCE_DIM_KERNEL_MID
		c(i, j) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_2_3_KERNEL_BEGIN(1, 0, 2)
		auto it = begin(a, i, token(), j), e = end(a, i, token(), j);
	REDUCE_DIM_KERNEL_MID
		c(i, j) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_2_3_KERNEL_BEGIN(2, 0, 1)
		auto it = begin(a, i, j, token()), e = end(a, i, j, token());
	REDUCE_DIM_KERNEL_MID
		c(i, j) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(0, 1, 2, 3)
		auto it = begin(a, token(), i, j, k), e = end(a, token(), i, j, k);
	REDUCE_DIM_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(1, 0, 2, 3)
		auto it = begin(a, i, token(), j, k), e = end(a, i, token(), j, k);
	REDUCE_DIM_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(2, 0, 1, 3)
		auto it = begin(a, i, j, token(), k), e = end(a, i, j, token(), k);
	REDUCE_DIM_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(3, 0, 1, 2)
		auto it = begin(a, i, j, k, token()), e = end(a, i, j, k, token());
	REDUCE_DIM_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		//RANGE REDUCERS
#define REDUCE_DIM_RANGE_KERNEL_MID			\
	T reduced = f(it, e);					\

		REDUCE_DIM_1_2_KERNEL_BEGIN(Range0, 1)
		auto it = begin(a, token(), i), e = end(a, token(), i);
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_1_2_KERNEL_BEGIN(Range1, 0)
		auto it = begin(a, i, token()), e = end(a, i, token());
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_2_3_KERNEL_BEGIN(Range0, 1, 2)
		auto it = begin(a, token(), i, j), e = end(a, token(), i, j);
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i, j) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_2_3_KERNEL_BEGIN(Range1, 0, 2)
		auto it = begin(a, i, token(), j), e = end(a, i, token(), j);
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i, j) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_2_3_KERNEL_BEGIN(Range2, 0, 1)
		auto it = begin(a, i, j, token()), e = end(a, i, j, token());
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i, j) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(Range0, 1, 2, 3)
		auto it = begin(a, token(), i, j, k), e = end(a, token(), i, j, k);
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(Range1, 0, 2, 3)
		auto it = begin(a, i, token(), j, k), e = end(a, i, token(), j, k);
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(Range2, 0, 1, 3)
		auto it = begin(a, i, j, token(), k), e = end(a, i, j, token(), k);
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		REDUCE_DIM_3_4_KERNEL_BEGIN(Range3, 0, 1, 2)
		auto it = begin(a, i, j, k, token()), e = end(a, i, j, k, token());
	REDUCE_DIM_RANGE_KERNEL_MID
		c(i, j, k) = reduced;
	REDUCE_DIM_KERNEL_END

		std::tuple<dim3, dim3> calculateDims(int x, int y, int z)
	{
		auto blockDimCalc = [](int a) { return a > 1024 ? a / 1024 : 1; };
		auto threadDimCalc = [](int a) { return a <= 1024 ? a : 1024; };
		dim3 blockDim(blockDimCalc(x), blockDimCalc(y), blockDimCalc(z));
		dim3 threadDim(threadDimCalc(x), threadDimCalc(y), threadDimCalc(z));
		return { blockDim, threadDim };
	}

	template<typename T, typename V, typename Func>
	void reduceDim(multi_dim_vector<T, 1> c, multi_dim_vector<V const, 2> a, Func f, int axis)
	{
		auto dims = calculateDims(get_max_dim<0>(c), 1, 1);
		switch (axis)
		{
		case 0:
			if (get_max_dim<1>(a) == get_max_dim<0>(c)) {
				reduceDim0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (get_max_dim<0>(a) == get_max_dim<0>(c)) {
				reduceDim1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename T, typename V, typename Func>
	void reduceDim(multi_dim_vector<T, 2> c, multi_dim_vector<V const, 3> a, Func f, int axis)
	{
		auto dims = calculateDims(get_max_dim<0>(c), get_max_dim<1>(c), 1);
		switch (axis)
		{
		case 0:
			if (get_max_dim<1>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c)) {
				reduceDim0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c)) {
				reduceDim1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 2:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<1>(a) == get_max_dim<1>(c)) {
				reduceDim2Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename T, typename V, typename Func>
	void reduceDim(multi_dim_vector<T, 3> c, multi_dim_vector<V const, 4> a, Func f, int axis)
	{
		auto dims = calculateDims(get_max_dim<0>(c), get_max_dim<1>(c), get_max_dim<2>(c));

		switch (axis)
		{
		case 0:
			if (get_max_dim<1>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c) &&
				get_max_dim<3>(a) == get_max_dim<2>(c)) {
				reduceDim0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c) &&
				get_max_dim<3>(a) == get_max_dim<2>(c)) {
				reduceDim1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 2:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<1>(a) == get_max_dim<1>(c) &&
				get_max_dim<3>(a) == get_max_dim<2>(c)) {
				reduceDim2Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 3:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<1>(a) == get_max_dim<1>(c) &&
				get_max_dim<2>(a) == get_max_dim<2>(c)) {
				reduceDim3Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename T, typename V, typename Func>
	void reduceDimRange(multi_dim_vector<T, 1> c, multi_dim_vector<V const, 2> a, Func f, int axis)
	{
		auto dims = calculateDims(get_max_dim<0>(c), 1, 1);
		switch (axis)
		{
		case 0:
			if (get_max_dim<1>(a) == get_max_dim<0>(c)) {
				reduceDimRange0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (get_max_dim<0>(a) == get_max_dim<0>(c)) {
				reduceDimRange1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename T, typename V, typename Func>
	void reduceDimRange(multi_dim_vector<T, 2> c, multi_dim_vector<V const, 3> a, Func f, int axis)
	{
		auto dims = calculateDims(get_max_dim<0>(c), get_max_dim<1>(c), 1);
		switch (axis)
		{
		case 0:
			if (get_max_dim<1>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c)) {
				reduceDimRange0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c)) {
				reduceDimRange1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 2:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<1>(a) == get_max_dim<1>(c)) {
				reduceDimRange2Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename T, typename V, typename Func>
	void reduceDimRange(multi_dim_vector<T, 3> c, multi_dim_vector<V const, 4> a, Func f, int axis)
	{
		auto dims = calculateDims(get_max_dim<0>(c), get_max_dim<1>(c), get_max_dim<2>(c));

		switch (axis)
		{
		case 0:
			if (get_max_dim<1>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c) &&
				get_max_dim<3>(a) == get_max_dim<2>(c)) {
				reduceDimRange0Kernel << <std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 1:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<2>(a) == get_max_dim<1>(c) &&
				get_max_dim<3>(a) == get_max_dim<2>(c)) {
				reduceDimRange1Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 2:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<1>(a) == get_max_dim<1>(c) &&
				get_max_dim<3>(a) == get_max_dim<2>(c)) {
				reduceDimRange2Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		case 3:
			if (get_max_dim<0>(a) == get_max_dim<0>(c) &&
				get_max_dim<1>(a) == get_max_dim<1>(c) &&
				get_max_dim<2>(a) == get_max_dim<2>(c)) {
				reduceDimRange3Kernel << < std::get<0>(dims), std::get<1>(dims) >> > (c, a, f);
			}
			break;
		}
	}

	template<typename V, typename T, size_t N, typename Func>
	void unaryOp(multi_dim_vector<V, N> mb, multi_dim_vector<T const, N> ma, Func f)
	{
		if (is_same_shape(ma, mb)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			unaryOpKernel << <std::get<0>(dims), std::get<1>(dims) >> > (mb, ma, f);
		}
	}

	template<typename T, size_t N, typename Func>
	void unaryOpInplace(multi_dim_vector<T, N> ma, Func f)
	{
		auto dims = calculateDims(ma.total_size(), 1, 1);
		unaryOpInplaceKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, f);
	}

	template<typename V, typename T, typename U, size_t N, typename Func>
	void binaryOp(multi_dim_vector<V, N> mc, multi_dim_vector<T const, N> ma, multi_dim_vector<U const, N> const mb, Func f)
	{
		if (is_same_shape(ma, mb) && is_same_shape(mb, mc)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			binaryOpKernel << <std::get<0>(dims), std::get<1>(dims) >> > (mc, ma, mb, f);
		}
	}

	template<typename T, typename U, size_t N, typename Func>
	void binaryOpInplace(multi_dim_vector<T, N> ma, multi_dim_vector<U const, N> const mb, Func f)
	{
		if (is_same_shape(ma, mb)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			binaryOpInplaceKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, mb, f);
		}
	}

	namespace detail_sort {
		__global__ void fillAxis0(multi_dim_vector<int, 2> a)
		{
			for (auto i : grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
				for (auto j : grid_stride_range::y(size_t(0), get_max_dim<1>(a)))
				{
					a(i, j) = j;
				}
		}

		__global__ void fillIndices(multi_dim_vector<int, 2> a)
		{
			int const R = get_max_dim<0>(a);
			int const C = get_max_dim<1>(a);
			for (auto r : grid_stride_range::x(0, R))
				for (auto c : grid_stride_range::y(0, C))
				{
					auto i = r * C + c;
					a(r, c) = (i / R) + (i%R) * C;
				}
		}

		__global__ void fillAxis1(multi_dim_vector<int, 2> a)
		{
			for (auto i : grid_stride_range::x(size_t(0), get_max_dim<0>(a)))
				for (auto j : grid_stride_range::y(size_t(0), get_max_dim<1>(a)))
				{
					a(i, j) = i;
				}
		}
	}

	template<typename T>
	void sortAxis(cuda_multi_dim_vector<T, 2>& a, int axis)
	{
		using thrust_utils::begin;
		using thrust_utils::end;
		cuda_multi_dim_vector<int, 2> temp(get_max_dim<0>(a.view()), get_max_dim<1>(a.view()));
		auto dims = calculateDims(get_max_dim<0>(a.view()), get_max_dim<1>(a.view()), 1);

		if (axis == 0) {
			detail_sort::fillAxis0 << <std::get<0>(dims), std::get<1>(dims) >> > (temp.view());
		}
		else if (axis == 1) {
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

