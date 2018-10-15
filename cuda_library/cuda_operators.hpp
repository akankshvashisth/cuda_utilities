#ifndef __cuda_operators_hpp__
#define __cuda_operators_hpp__

#include "multi_dim_vector.hpp"
#include "grid_stride_range.hpp"
#include "multi_dim_vector_iterator.hpp"
#include "cuda_pointer_thrust_utils.hpp"
#include "defines.hpp"
#include "cuda_context.hpp"
#include <tuple>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

namespace aks
{
	template<typename T, size_t N>
	struct point;

	template<typename T>
	struct point<T, 0>
	{};

	template<typename T> struct point<T, 1> : point<T, 0> { T x; };
	template<typename T> struct point<T, 2> : point<T, 1> { T y; };
	template<typename T> struct point<T, 3> : point<T, 2> { T z; };
	template<typename T> struct point<T, 4> : point<T, 3> { T w; };

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

	template<typename Func, typename V, size_t N, typename... Us>
	__global__ void naryOpKernel(multi_dim_vector<V, N> c, Func f, multi_dim_vector<Us const, N>... a)
	{
		for (auto i : grid_stride_range::x(size_t(0), c.total_size()))
		{
			c.data()[i] = f(a.data()[i]...);
		}
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndexTiled(multi_dim_vector<V, 2> c, point<size_t, 2> tile, point<size_t, 2> start, Func f, multi_dim_vector<Us const, 2>... a)
	{
		for (auto ix : grid_stride_range::x(size_t(0), tile.x))
			for (auto jy : grid_stride_range::y(size_t(0), tile.y)) {
				auto i = start.x + ix;
				auto j = start.y + jy;
				if (i < get_max_dim< 0 >(c) && j < get_max_dim< 1 >(c))
					c(i, j) = f(i, j, a(i, j)...);
			}
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndexTiled(multi_dim_vector<V, 3> c, point<size_t, 3> tile, point<size_t, 3> start, Func f, multi_dim_vector<Us const, 3>... a)
	{
		for (auto ix : grid_stride_range::x(size_t(0), tile.x))
			for (auto jy : grid_stride_range::y(size_t(0), tile.y)) {
				for (auto kz : grid_stride_range::z(size_t(0), tile.z)) {
					auto i = start.x + ix;
					auto j = start.y + jy;
					auto k = start.z + kz;
					auto safe_i = i < get_max_dim< 0 >(c);
					auto safe_j = j < get_max_dim< 1 >(c);
					auto safe_k = k < get_max_dim< 2 >(c);
					if (safe_i && safe_j && safe_k)
					{
						c(i, j, k) = f(i, j, k); // , a(i, j, k)...);
					}
				}
			}
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndex(multi_dim_vector<V, 1> c, Func f, multi_dim_vector<Us const, 1>... a)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< 0 >(c)))
			c(i) = f(i, a(i)...);
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndex(multi_dim_vector<V, 2> c, Func f, multi_dim_vector<Us const, 2>... a)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< 0 >(c)))
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim< 1 >(c)))
				c(i, j) = f(i, j, a(i, j)...);
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndex(multi_dim_vector<V, 3> c, Func f, multi_dim_vector<Us const, 3>... a)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< 0 >(c)))
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim< 1 >(c)))
				for (auto k : grid_stride_range::z(size_t(0), get_max_dim< 2 >(c)))
					c(i, j, k) = f(i, j, k, a(i, j, k)...);
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndex(multi_dim_vector<V, 4> c, Func f, multi_dim_vector<Us const, 4>... a)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< 0 >(c)))
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim< 1 >(c)))
				for (auto k : grid_stride_range::z(size_t(0), get_max_dim< 2 >(c)))
					for (size_t x = 0; x < get_max_dim< 3 >(c); ++x)
						c(i, j, k, x) = f(i, j, k, x, a(i, j, k, x)...);
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndex(multi_dim_vector<V, 5> c, Func f, multi_dim_vector<Us const, 5>... a)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< 0 >(c)))
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim< 1 >(c)))
				for (auto k : grid_stride_range::z(size_t(0), get_max_dim< 2 >(c)))
					for (size_t x = 0; x < get_max_dim< 3 >(c); ++x)
						for (size_t y = 0; y < get_max_dim< 4 >(c); ++y)
							c(i, j, k, x, y) = f(i, j, k, x, y, a(i, j, k, x, y)...);
	}

	template<typename Func, typename V, typename... Us>
	__global__ void naryOpKernelWithIndex(multi_dim_vector<V, 6> c, Func f, multi_dim_vector<Us const, 6>... a)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< 0 >(c)))
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim< 1 >(c)))
				for (auto k : grid_stride_range::z(size_t(0), get_max_dim< 2 >(c)))
					for (size_t x = 0; x < get_max_dim< 3 >(c); ++x)
						for (size_t y = 0; y < get_max_dim< 4 >(c); ++y)
							for (size_t z = 0; z < get_max_dim< 5 >(c); ++z)
								c(i, j, k, x, y, z) = f(i, j, k, x, y, z, a(i, j, k, x, y, z)...);
	}

	template<typename T, typename V, typename Func, size_t N>
	__global__ void reduceDim0Kernel(multi_dim_vector<T, N - 1> c, multi_dim_vector<V const, N> a, Func f);

	template<typename T, typename V, typename Func, size_t N>
	__global__ void reduceDim1Kernel(multi_dim_vector<T, N - 1> c, multi_dim_vector<V const, N> a, Func f);

#define REDUCE_DIM_1_2_KERNEL_BEGIN(S, X)																					\
	template<typename T, typename V, typename Func>																			\
	__global__ void reduceDim ## S ## Kernel(multi_dim_vector<T, 1> c, multi_dim_vector<V const, 2> a, Func f)				\
	{																														\
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim<X>(a)))												    \
		{																													\

#define REDUCE_DIM_2_3_KERNEL_BEGIN(S, X, Y)																				\
	template<typename T, typename V, typename Func>																			\
	__global__ void reduceDim ## S ## Kernel(multi_dim_vector<T, 2> c, multi_dim_vector<V const, 3> a, Func f)				\
	{																														\
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim<X>(a)))													\
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim<Y>(a)))												\
			{																												\

#define REDUCE_DIM_3_4_KERNEL_BEGIN(S, X, Y, Z)																		\
	template<typename T, typename V, typename Func>																	\
	__global__ void reduceDim ## S ## Kernel(multi_dim_vector<T, 3> c, multi_dim_vector<V const, 4> a, Func f)		\
	{																												\
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim< X >(a)))											\
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim< Y >(a)))										\
				for (auto k : grid_stride_range::z(size_t(0), get_max_dim< Z >(a))){								\

#define REDUCE_DIM_KERNEL_END }}

	//BINARY REDUCERS
#define REDUCE_DIM_KERNEL_MID T reduced = *it; ++it; for(; it != e; ++it){reduced = f(reduced, *it);}

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

		//

		template<typename T, typename V, typename Func>
	__global__ void subArrayKernel(multi_dim_vector<T, 1> c, multi_dim_vector<V const, 1> a, point<int, 1> from, Func f)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim<0>(c)))
		{
			c(i) = f(a(i + from.x));
		}
	}

	template<typename T, typename V, typename Func>
	__global__ void subArrayKernel(multi_dim_vector<T, 2> c, multi_dim_vector<V const, 2> a, point<int, 2> from, Func f)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim<0>(c)))
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim<1>(c)))
			{
				c(i, j) = f(a(i + from.x, j + from.y));
			}
	}

	template<typename T, typename V, typename Func>
	__global__ void subArrayKernel(multi_dim_vector<T, 3> c, multi_dim_vector<V const, 3> a, point<int, 3> from, Func f)
	{
		for (auto i : grid_stride_range::x(size_t(0), get_max_dim<0>(c)))
			for (auto j : grid_stride_range::y(size_t(0), get_max_dim<1>(c)))
				for (auto k : grid_stride_range::z(size_t(0), get_max_dim<2>(c)))
				{
					c(i, j, k) = f(a(i + from.x, j + from.y, k + from.z));
				}
	}

	//

	std::tuple<dim3, dim3> calculateDims(int x, int y, int z)
	{
		auto threadDimCalc = [&](int maxThreadCount) {
			dim3 ret;
			int maxThreadCountLeft = maxThreadCount;
			ret.x = x > maxThreadCountLeft ? maxThreadCountLeft : x;
			maxThreadCountLeft /= ret.x;
			ret.y = y > maxThreadCountLeft ? maxThreadCountLeft : y;
			maxThreadCountLeft /= ret.y;
			ret.z = z > maxThreadCountLeft ? maxThreadCountLeft : z;
			return ret;
		};
		auto blockDimCalc = [&](dim3 threads) {
			dim3 ret;
			ret.x = x / threads.x + int(x%threads.x > 0);
			ret.y = y / threads.y + int(y%threads.y > 0);
			ret.z = z / threads.z + int(z%threads.z > 0);
			return ret;
		};

		dim3 threadDim = threadDimCalc(1024);
		dim3 blockDim = blockDimCalc(threadDim);

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
		gpu_error_check(last_status());
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
		gpu_error_check(last_status());
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
		gpu_error_check(last_status());
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
		gpu_error_check(last_status());
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
		gpu_error_check(last_status());
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
		gpu_error_check(last_status());
	}

	template<typename V, typename T, size_t N, typename Func>
	void unaryOp(multi_dim_vector<V, N> mb, multi_dim_vector<T const, N> ma, Func f)
	{
		if (is_same_shape(ma, mb)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			unaryOpKernel << <std::get<0>(dims), std::get<1>(dims) >> > (mb, ma, f);
		}
		gpu_error_check(last_status());
	}

	template<typename T, size_t N, typename Func>
	void unaryOpInplace(multi_dim_vector<T, N> ma, Func f)
	{
		auto dims = calculateDims(ma.total_size(), 1, 1);
		unaryOpInplaceKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, f);
		gpu_error_check(last_status());
	}

	template<typename V, typename T, typename U, size_t N, typename Func>
	void binaryOp(multi_dim_vector<V, N> mc, multi_dim_vector<T const, N> ma, multi_dim_vector<U const, N> const mb, Func f)
	{
		if (is_same_shape(ma, mb) && is_same_shape(mb, mc)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			binaryOpKernel << <std::get<0>(dims), std::get<1>(dims) >> > (mc, ma, mb, f);
		}
		gpu_error_check(last_status());
	}

	template<typename T, typename U, size_t N, typename Func>
	void binaryOpInplace(multi_dim_vector<T, N> ma, multi_dim_vector<U const, N> const mb, Func f)
	{
		if (is_same_shape(ma, mb)) {
			auto dims = calculateDims(ma.total_size(), 1, 1);
			binaryOpInplaceKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, mb, f);
		}
		gpu_error_check(last_status());
	}

	namespace operator_detail
	{
		template<typename Func, typename T>
		bool test_equal(Func f, T a)
		{
			return true;
		}

		template<typename Func, typename T, typename U, typename... Ts>
		bool test_equal(Func f, T a, U b, Ts... ts)
		{
			return f(a, b) && test_equal(f, a, ts...);
		}

		struct is_same_shape_wrapper
		{
			template<typename T, typename U>
			AKS_FUNCTION_PREFIX_ATTR bool operator()(T const& a, U const& b) const
			{
				return is_same_shape(a, b);
			}
		};
	};

	template<typename Func, typename V, size_t N, typename... Us>
	void naryOp(multi_dim_vector<V, N> mc, Func f, multi_dim_vector<Us const, N>... ma)
	{
		if (operator_detail::test_equal(operator_detail::is_same_shape_wrapper(), mc, ma...)) {
			auto dims = calculateDims(mc.total_size(), 1, 1);
			naryOpKernel << <std::get<0>(dims), std::get<1>(dims) >> > (mc, f, ma...);
		}
		gpu_error_check(last_status());
	}

	template<typename T>
	std::tuple<dim3, dim3> calcDims(multi_dim_vector<T, 1> const& mc)
	{
		return calculateDims(get_max_dim<0>(mc), 1, 1);
	}

	template<typename T>
	std::tuple<dim3, dim3> calcDims(multi_dim_vector<T, 2> const& mc)
	{
		return calculateDims(get_max_dim<0>(mc), 1, 1);
	}

	template<typename T, size_t N>
	std::tuple<dim3, dim3> calcDims(multi_dim_vector<T, N> const& mc)
	{
		return calculateDims(get_max_dim<0>(mc), 1, 1);
	}

	template<typename Func, typename V, size_t N, typename... Us>
	void naryOpWithIndex(multi_dim_vector<V, N> mc, Func f, multi_dim_vector<Us const, N>... ma)
	{
		if (operator_detail::test_equal(operator_detail::is_same_shape_wrapper(), mc, ma...)) {
			std::tuple<dim3, dim3> dims = calcDims(mc);
			naryOpKernelWithIndex << <std::get<0>(dims), std::get<1>(dims) >> > (mc, f, ma...);
			//naryOpKernelWithIndex << <1, 1 >> > (mc, f, ma...);
		}
		gpu_error_check(last_status());
	}

	std::tuple<dim3, dim3> calcDims(point<size_t, 1> tile)
	{
		return calculateDims(tile.x, 1, 1);
	}

	std::tuple<dim3, dim3> calcDims(point<size_t, 2> tile)
	{
		return calculateDims(tile.x, tile.y, 1);
	}

	template<size_t N>
	std::tuple<dim3, dim3> calcDims(point<size_t, N> tile)
	{
		return calculateDims(tile.x, tile.y, tile.z);
	}

	template<typename Func, typename V, size_t N, typename... Us>
	void naryOpWithIndexTiled(multi_dim_vector<V, N> mc, point<size_t, N> tile, point<size_t, N> start, Func f, multi_dim_vector<Us const, N>... ma)
	{
		if (operator_detail::test_equal(operator_detail::is_same_shape_wrapper(), mc, ma...)) {
			std::tuple<dim3, dim3> dims = calcDims(tile);
			naryOpKernelWithIndexTiled << <std::get<0>(dims), std::get<1>(dims) >> > (mc, tile, start, f, ma...);
			//naryOpKernelWithIndexTiled << <1, 1 >> > (mc, tile, start, f, ma...);
			//naryOpKernelWithIndex << <1, 1 >> > (mc, f, ma...);
		}
		gpu_error_check(last_status());
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

	template<typename T, typename Comp>
	void sortAxis(cuda_multi_dim_vector<T, 2> & a, int axis, Comp comp)
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
		gpu_error_check(last_status());

		thrust::stable_sort_by_key(begin(a.m_data), end(a.m_data), begin(temp.m_data), comp);
		thrust::stable_sort_by_key(begin(temp.m_data), end(temp.m_data), begin(a.m_data));

		gpu_error_check(last_status());

		if (axis == 0) {
			//cuda_multi_dim_vector<int, 2> map(get_max_dim<0>(a.view()), get_max_dim<1>(a.view()));
			detail_sort::fillIndices << <std::get<0>(dims), std::get<1>(dims) >> > (temp.view());
			thrust::scatter(begin(a.m_data), end(a.m_data), begin(temp.m_data), begin(a.m_data));
		}
		//unaryOp(a.view(), temp.cview(), [] AKS_FUNCTION_PREFIX_ATTR(int x) { return x; });
		gpu_error_check(last_status());
	}

	template<typename T, typename U, typename Func>
	void subArray(multi_dim_vector<T, 1> ma, multi_dim_vector<U const, 1> const mb, point<int, 1> from, Func f)
	{
		if (get_max_dim<0>(ma) + from.x <= get_max_dim<0>(mb))
		{
			auto dims = calculateDims(get_max_dim<0>(ma), 1, 1);
			subArrayKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, mb, from, f);
		}
	}

	template<typename T, typename U, typename Func>
	void subArray(multi_dim_vector<T, 2> ma, multi_dim_vector<U const, 2> const mb, point<int, 2> from, Func f)
	{
		if (get_max_dim<0>(ma) + from.x <= get_max_dim<0>(mb) &&
			get_max_dim<1>(ma) + from.y <= get_max_dim<1>(mb))
		{
			auto dims = calculateDims(get_max_dim<0>(ma), get_max_dim<1>(ma), 1);
			subArrayKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, mb, from, f);
		}
		gpu_error_check(last_status());
	}

	template<typename T, typename U, typename Func>
	void subArray(multi_dim_vector<T, 3> ma, multi_dim_vector<U const, 3> const mb, point<int, 3> from, Func f)
	{
		if (get_max_dim<0>(ma) + from.x <= get_max_dim<0>(mb) &&
			get_max_dim<1>(ma) + from.y <= get_max_dim<1>(mb) &&
			get_max_dim<2>(ma) + from.z <= get_max_dim<2>(mb))
		{
			auto dims = calculateDims(get_max_dim<0>(ma), get_max_dim<1>(ma), get_max_dim<2>(ma));
			subArrayKernel << <std::get<0>(dims), std::get<1>(dims) >> > (ma, mb, from, f);
		}
		gpu_error_check(last_status());
	}
			}

#endif // !__cuda_operators_hpp__