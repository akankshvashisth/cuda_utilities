#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_context.hpp"
#include "cuda_pointer.hpp"
#include "cuda_pointer_vector_utils.hpp"
#include "cuda_multi_dim_vector.hpp"
#include "cuda_pointer_thrust_utils.hpp"
#include "cuda_multi_dim_vector_thrust_utils.hpp"
#include "multi_dim_vector_with_memory.hpp"

#include "compile_time_differentiation_tests.hpp"
#include "experiments.hpp"

#include <thrust/functional.h>
#include <thrust/sort.h>
#include <assert.h>

#include <memory>
#include "multi_dim_vector.hpp"
#include "multi_dim_vector_iterator.hpp"
#include "multi_dim_vector_range.hpp"
#include "cuda_blas_manager.hpp"
#include "cuda_blas_level_1.hpp"
#include "cuda_blas_level_2.hpp"
#include "cuda_blas_level_3.hpp"
#include "cuda_operators.hpp"
#include "cuda_object.hpp"

struct vec
{
	float x, y, z, w;
};

void object_check()
{
	//gpu_error_check(cudaError::cudaErrorAssert);
	using namespace aks;
	vec a = { 1,2,3,4 };
	vec b = { 5,6,7,8 };
	host_object<vec> ha(a);
	host_object<vec> hb(b);
	object_view<vec> va = ha.view();
	object_view<vec> vb = hb.view();
	object_view<vec const> vav = ha.view();
	object_view<vec const> vbv = hb.view();
	object_view<vec const> vacv = ha.cview();
	object_view<vec const> vbcv = hb.cview();
	host_object<vec> const& cha = ha;
	host_object<vec> const& chb = hb;
	object_view<vec const> cvav = cha.view();
	object_view<vec const> cvbv = chb.view();
	object_view<vec const> cvacv = cha.cview();
	object_view<vec const> cvbcv = chb.cview();
	auto& a0 = va->w;
	auto& a1 = (*vb).x;
	auto& a2 = vav->x;
	auto& a3 = vacv->x;
	std::cout << a0 << a1 << a2 << a3 << std::endl;
}

cudaError_t addWithCuda(std::vector<int>& c, std::vector<int> const& a, std::vector<int> const& b);
cudaError_t addWithCuda2(std::vector<int>& c, std::vector<int> const& a, std::vector<int> const& b);

__global__ void addKernel(aks::multi_dim_vector<int, 1> c, aks::multi_dim_vector<int const, 1> a, aks::multi_dim_vector<int const, 1> b)
{
	int i = threadIdx.x;
	int sum = 0;
	for (auto it = aks::begin(a, aks::token(5)), end = aks::end(a, aks::token(5)); it != end; ++it)
		sum += *it;
	for (auto const& x : aks::make_multi_dim_vector_range(b, aks::token()))
		sum += x;
	c(i) = a(i) + b(i) - sum;
}

__global__ void addKernel(aks::multi_dim_vector<int, 3> c, aks::multi_dim_vector<int const, 3> const a, aks::multi_dim_vector<int const, 3> const b)
{
	int const i = threadIdx.x;
	int const j = threadIdx.y;
	int const k = threadIdx.z;

	int sum = 0;
	for (auto it = aks::begin(a, aks::token(), j, k), end = aks::end(a, aks::token(), j, k); it != end; ++it)
		sum += *it;
	for (auto it = aks::begin(a, i, aks::token(), k), end = aks::end(a, i, aks::token(), k); it != end; ++it)
		sum += *it;
	for (auto const& x : aks::make_multi_dim_vector_range(b, i, j, aks::token()))
		sum += x;

	c(i, j, k) = sum;
}

void check2()
{
	compile_time_differentiation_tests();
	{
		aks::host_multi_dim_vector<int, 3> vec(3, 4, 5);
		auto view = vec.view();
		auto const& const_vec = vec;
		auto const_view = const_vec.view();
		printf("");
	}
	if (true)
	{
		aks::cuda_context ctxt(aks::cuda_device(0));
		aks::host_multi_dim_vector<int, 3> host_vec(3, 4, 5);
		auto host_view = host_vec.view();
		auto m0 = host_view.max_dimension<0>();
		auto m1 = host_view.max_dimension<1>();
		auto m2 = host_view.max_dimension<2>();
		auto m00 = aks::get_max_dim<0>(host_view);
		auto m01 = aks::get_max_dim<1>(host_view);
		auto m02 = aks::get_max_dim<2>(host_view);
		for (size_t x = 0; x < 3; ++x)
			for (size_t y = 0; y < 4; ++y)
				for (size_t z = 0; z < 5; ++z)
				{
					host_view(x, y, z) = x * 4 * 5 + y * 5 + z;
				}

		aks::cuda_multi_dim_vector<int, 3> vec = aks::to_device(host_vec);// (host_vec.view().data(), 3, 4, 5);
		aks::cuda_multi_dim_vector<int, 3> res(3, 4, 5);

		dim3 threadsPerBlock(3, 4, 5);
		{
			aks::cuda_sync_context sync_ctxt;
			addKernel << < 1, threadsPerBlock >> > (res.view(), vec.view(), vec.cview());
		}

		//auto view = vec.view();
		//auto const& const_vec = vec;
		//auto const_view = const_vec.view();

		//auto tmp = aks::from_cuda_pointer(vec.m_data);

		//std::vector<int> ret(view.total_size());
		//vec.m_data.load(ret.data());

		aks::host_multi_dim_vector<int, 3> ret_vec(3, 4, 5);
		ret_vec << res;

		auto ret_vec2 = aks::to_host(res);

		printf("");
	}
}

template<typename T>
void print2D(T const& y)
{
	using namespace aks;
	for (auto i = 0; i < get_max_dim<0>(y); ++i) {
		for (auto j = 0; j < get_max_dim<1>(y); ++j) {
			std::cout << y(i, j) << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

template<typename T>
void print1D(T const& y)
{
	using namespace aks;
	for (auto j = 0; j < aks::get_max_dim<0>(y); ++j) {
		std::cout << y(j) << "\t";
	}
	std::cout << std::endl;
	std::cout << std::endl;
}

void blas_checks()
{
	using namespace aks;
	using namespace aks::cuda_blas;
	cuda_context ctxt(cuda_device(0));
	cuda_blas_manager blas_mgr;
	{
		std::vector<double> const a = { 1., 2., 3., 4., 50., 8., -9., 23.0 };
		aks::cuda_multi_dim_vector<double, 1> cuda_vec(a.data(), std::tuple<size_t>(a.size()));
		int value = abs_max_index(blas_mgr, cuda_vec);
		std::cout << a[value] << std::endl;
		value = abs_min_index(blas_mgr, cuda_vec);
		std::cout << a[value] << std::endl;
		double sum = abs_sum(blas_mgr, cuda_vec);
		std::cout << sum << std::endl;
		std::vector<double> const b = { 1., -2.,  3. };
		std::vector<double> const c = { 2.,  2.,  2. };
		aks::cuda_multi_dim_vector<double, 1> cuda_vecb(b.data(), std::tuple<size_t>(b.size()));
		aks::cuda_multi_dim_vector<double, 1> cuda_vecc(c.data(), std::tuple<size_t>(c.size()));
		double dt = dot(blas_mgr, cuda_vecb, cuda_vecc);
		std::cout << dt << std::endl;
		std::cout << norm_sq(blas_mgr, cuda_vecb) << std::endl;
		scale_in_place(blas_mgr, cuda_vecb, 1.5);
		auto const dev_vecb = to_host(cuda_vecb);
		for (auto it = dev_vecb.view().cbegin(), end = dev_vecb.view().cend(); it != end; ++it) {
			std::cout << *it << ",";
		}
		std::cout << std::endl;
	}
	{
		host_multi_dim_vector<double, 2> Avec(2, 5);
		host_multi_dim_vector<double, 1> xvec(5);
		auto A = Avec.view();
		auto x = xvec.view();
		A(0, 0) = 1.0;
		A(0, 1) = 2.0;
		A(0, 2) = 3.0;
		A(0, 3) = 4.0;
		A(0, 4) = 5.0;
		A(1, 0) = 6.0;
		A(1, 1) = 7.0;
		A(1, 2) = 8.0;
		A(1, 3) = 9.0;
		A(1, 4) = 10.0;
		x(0) = 1.5;
		x(1) = 2.5;
		x(2) = 3.5;
		x(3) = 4.5;
		x(4) = 5.5;
		auto devA = to_device(Avec);
		auto devx = to_device(xvec);
		auto devy = matrix_multiply(blas_mgr, devA, devx);
		auto yvec = to_host(devy);
		for (auto it = yvec.view().cbegin(), end = yvec.view().cend(); it != end; ++it) {
			std::cout << *it << ",";
		}
		std::cout << std::endl;
	}
	{
		host_multi_dim_vector<double, 2> Avec(2, 5);
		host_multi_dim_vector<double, 2> Bvec(5, 3);
		auto A = Avec.view();
		auto B = Bvec.view();
		A(0, 0) = 1.0;
		A(0, 1) = 2.0;
		A(0, 2) = 3.0;
		A(0, 3) = 4.0;
		A(0, 4) = 5.0;
		A(1, 0) = 6.0;
		A(1, 1) = 7.0;
		A(1, 2) = 8.0;
		A(1, 3) = 9.0;
		A(1, 4) = 10.0;

		B(0, 0) = 1.0;
		B(0, 1) = 2.0;
		B(0, 2) = 3.0;
		B(1, 0) = 4.0;
		B(1, 1) = 5.0;
		B(1, 2) = 6.0;
		B(2, 0) = 7.0;
		B(2, 1) = 8.0;
		B(2, 2) = 9.0;
		B(3, 0) = 10.0;
		B(3, 1) = 11.0;
		B(3, 2) = 12.0;
		B(4, 0) = 13.0;
		B(4, 1) = 14.0;
		B(4, 2) = 15.0;

		auto devA = to_device(Avec);
		auto devB = to_device(Bvec);
		auto devy = matrix_multiply(blas_mgr, devA, devB);
		auto yvec = to_host(devy);
		auto y = yvec.view();
		print2D(y);
		auto devyt = transpose(blas_mgr, devy);
		auto ytvec = to_host(devyt);
		auto yt = ytvec.view();
		print2D(yt);

		auto devaabb = alpha_A_plus_beta_B(blas_mgr, -1.0, devyt, -1.0, devyt);
		auto yaabbvec = to_host(devaabb);
		auto yaabb = yaabbvec.view();
		print2D(yaabb);
	}
}

#define TEST_PREAMBLE() aks::cuda_context ctxt(aks::cuda_device(0)); \
 aks::cuda_sync_context sync; \
 std::cout << "---" << testNum++ << "---\n";

int operatorCheck()
{
	aks::host_multi_dim_vector<double, 2> host_in(3000, 5000);
	aks::host_multi_dim_vector<double, 2> host_out(3000, 5000);

	auto in_ = host_in.view();
	auto out_ = host_out.view();
	auto m0 = aks::get_max_dim<0>(in_);
	auto m1 = aks::get_max_dim<1>(in_);
	for (auto x = 0; x < m0; ++x)
		for (auto y = 0; y < m1; ++y) {
			in_(x, y) = 8.3;
			out_(x, y) = 12.1;
		}

	aks::cuda_multi_dim_vector<double, 2> vec_in = aks::to_device(host_in);
	aks::cuda_multi_dim_vector<double, 2> vec_out = aks::to_device(host_out);

	int testNum = 0;
	std::cout << "---" << testNum++ << "---\n";
	aks::binaryOpInplace(vec_out.view(), vec_in.cview(), thrust::plus<double>());
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "*** ---" << testNum++ << "--- ***\n";
	aks::naryOp(vec_out.view(), thrust::multiplies<double>(), vec_out.cview(), vec_in.cview());
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "*** ---" << testNum++ << "--- ***\n";
	aks::naryOp(vec_out.view(), [] AKS_FUNCTION_PREFIX_ATTR(double x) { return x / 8.3; }, vec_out.cview());
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "*** ---" << testNum++ << "--- ***\n";
	aks::naryOp(vec_out.view(), [] AKS_FUNCTION_PREFIX_ATTR() { return 9.1; });
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "*** ---" << testNum++ << "--- ***\n";
	aks::naryOp(vec_out.view(), [] AKS_FUNCTION_PREFIX_ATTR(double x, double y, double z, double w) { return (x / y) + (z / w); }, vec_out.cview(), vec_out.cview(), vec_out.cview(), vec_out.cview());
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "---" << testNum++ << "---\n";
	aks::binaryOp(vec_out.view(), vec_out.cview(), vec_in.cview(), thrust::plus<double>());
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "---" << testNum++ << "---\n";
	aks::unaryOpInplace(vec_out.view(), [] AKS_FUNCTION_PREFIX_ATTR(double x) { return x * 2; });
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "---" << testNum++ << "---\n";
	aks::unaryOp(vec_out.view(), vec_out.cview(), [] AKS_FUNCTION_PREFIX_ATTR(double x) { return x * -1; });
	host_out << vec_out;
	print2D(host_out.cview());
	std::cout << "---" << testNum++ << "---\n";
	aks::naryOp(vec_out.view(), [] AKS_FUNCTION_PREFIX_ATTR() { return 1.0; });
	host_out << vec_out;
	print2D(host_out.cview());
	auto const me = vec_out.cview();
	int const maxX = aks::get_max_dim<0>(me);
	int const maxY = aks::get_max_dim<1>(me);
	int const kernelX = 3;
	int const kernelY = 3;
	aks::cuda_multi_dim_vector<double, 2> vec_out2(vec_out.m_dimensions);
	aks::naryOpWithIndex(vec_out2.view(), [me, maxX, maxY, kernelX, kernelY] AKS_FUNCTION_PREFIX_ATTR(int i, int j, double x) {
		x = 0;
		for (int m = i - kernelX; m <= i + kernelX; ++m)
			for (int n = j - kernelY; n <= j + kernelY; ++n)
			{
				if (!(0 > m || m >= maxX) && !(0 > n || n >= maxY))
				{
					x += me(m, n);
				}
			}
		return x;
	}, vec_out.cview());
	host_out << vec_out2;
	print2D(host_out.cview());

	return 0;
}

struct plusRange
{
	template<typename Iter>
	AKS_FUNCTION_PREFIX_ATTR auto operator()(Iter b, Iter e) const
	{
		int sum = 0;
		for (; b != e; ++b)
			sum += *b;
		return sum;
	}
};

int reduceDimCheck()
{
	int testNum = 0;
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 2> host_in(10, 5);
		aks::host_multi_dim_vector<int, 1> host_out_0(5);
		aks::host_multi_dim_vector<int, 1> host_out_1(10);
		auto in_ = host_in.view();
		auto m0 = aks::get_max_dim<0>(in_);
		auto m1 = aks::get_max_dim<1>(in_);
		for (auto x = 0; x < m0; ++x)
			for (auto y = 0; y < m1; ++y)
				in_(x, y) = 8;

		aks::cuda_multi_dim_vector<int, 2> vec_in = aks::to_device(host_in);
		aks::cuda_multi_dim_vector<int, 1> vec_out_0 = aks::to_device(host_out_0);
		aks::cuda_multi_dim_vector<int, 1> vec_out_1 = aks::to_device(host_out_1);

		aks::reduceDim(vec_out_0.view(), vec_in.cview(), thrust::plus<int>(), 0);
		host_out_0 << vec_out_0;
		print1D(host_out_0.cview());

		aks::reduceDim(vec_out_1.view(), vec_in.cview(), thrust::plus<int>(), 1);
		host_out_1 << vec_out_1;
		print1D(host_out_1.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 3> host_x(7, 10, 5);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 3;
		}
		aks::cuda_multi_dim_vector<int, 2> out(10, 5);
		aks::cuda_multi_dim_vector<int, 3> in = aks::to_device(host_x);
		aks::reduceDim(out.view(), in.cview(), thrust::plus<int>(), 0);
		auto res = aks::to_host(out);
		print2D(res.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 3> host_x(10, 8, 5);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 3;
		}
		aks::cuda_multi_dim_vector<int, 2> out(10, 5);
		aks::cuda_multi_dim_vector<int, 3> in = aks::to_device(host_x);
		aks::reduceDim(out.view(), in.cview(), thrust::plus<int>(), 1);
		auto res = aks::to_host(out);
		print2D(res.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 3> host_x(10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 3;
		}
		aks::cuda_multi_dim_vector<int, 2> out(10, 5);
		aks::cuda_multi_dim_vector<int, 3> in = aks::to_device(host_x);
		aks::reduceDim(out.view(), in.cview(), thrust::plus<int>(), 2);
		auto res = aks::to_host(out);
		print2D(res.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 3;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(10, 5, 9);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		aks::reduceDim(out1.view(), in.cview(), thrust::plus<int>(), 0);

		aks::cuda_multi_dim_vector<int, 2> out2(5, 9);
		aks::reduceDim(out2.view(), out1.cview(), thrust::plus<int>(), 0);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(9);
		aks::reduceDim(out3.view(), out2.cview(), thrust::plus<int>(), 0);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 3;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(10, 5, 9);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		aks::reduceDimRange(out1.view(), in.cview(), plusRange(), 0);

		aks::cuda_multi_dim_vector<int, 2> out2(5, 9);
		aks::reduceDimRange(out2.view(), out1.cview(), plusRange(), 0);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(9);
		aks::reduceDimRange(out3.view(), out2.cview(), plusRange(), 0);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 1;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(3, 5, 9);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		aks::reduceDim(out1.view(), in.cview(), thrust::plus<int>(), 1);

		aks::cuda_multi_dim_vector<int, 2> out2(3, 9);
		aks::reduceDim(out2.view(), out1.cview(), thrust::plus<int>(), 1);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(3);
		aks::reduceDim(out3.view(), out2.cview(), thrust::plus<int>(), 1);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 1;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(3, 5, 9);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		aks::reduceDimRange(out1.view(), in.cview(), plusRange(), 1);

		aks::cuda_multi_dim_vector<int, 2> out2(3, 9);
		aks::reduceDimRange(out2.view(), out1.cview(), plusRange(), 1);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(3);
		aks::reduceDimRange(out3.view(), out2.cview(), plusRange(), 1);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 1;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(3, 10, 9);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		host_x >> in;
		aks::reduceDim(out1.view(), in.cview(), thrust::plus<int>(), 2);

		aks::cuda_multi_dim_vector<int, 2> out2(3, 10);
		aks::reduceDim(out2.view(), out1.cview(), thrust::plus<int>(), 2);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(3);
		aks::reduceDim(out3.view(), out2.cview(), thrust::plus<int>(), 1);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 1;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(3, 10, 9);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		aks::reduceDimRange(out1.view(), in.cview(), plusRange(), 2);

		aks::cuda_multi_dim_vector<int, 2> out2(3, 10);
		aks::reduceDimRange(out2.view(), out1.cview(), plusRange(), 2);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(3);
		aks::reduceDimRange(out3.view(), out2.cview(), plusRange(), 1);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 1;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(3, 10, 5);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		aks::reduceDim(out1.view(), in.cview(), thrust::plus<int>(), 3);

		aks::cuda_multi_dim_vector<int, 2> out2(3, 10);
		aks::reduceDim(out2.view(), out1.cview(), thrust::plus<int>(), 2);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(3);
		aks::reduceDim(out3.view(), out2.cview(), thrust::plus<int>(), 1);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		std::cout << "---" << testNum++ << "---\n";
		aks::host_multi_dim_vector<int, 4> host_x(3, 10, 5, 9);
		for (auto i = host_x.view().begin(), e = host_x.view().end(); i != e; ++i) {
			*i = 1;
		}
		aks::cuda_multi_dim_vector<int, 3> out1(3, 10, 5);
		aks::cuda_multi_dim_vector<int, 4> in = aks::to_device(host_x);
		aks::reduceDimRange(out1.view(), in.cview(), plusRange(), 3);

		aks::cuda_multi_dim_vector<int, 2> out2(3, 10);
		aks::reduceDimRange(out2.view(), out1.cview(), plusRange(), 2);

		auto res = aks::to_host(out2);
		print2D(res.cview());

		aks::cuda_multi_dim_vector<int, 1> out3(3);
		aks::reduceDimRange(out3.view(), out2.cview(), plusRange(), 1);

		auto res2 = aks::to_host(out3);
		print1D(res2.cview());
	}
	{
		{
			std::cout << "---" << testNum++ << "---\n";
			auto testCase = []() -> aks::host_multi_dim_vector<double, 2> {
				aks::host_multi_dim_vector<double, 2> Bvec(5, 3);
				auto B = Bvec.view();
				B(0, 0) = 1.0;
				B(0, 1) = -2.0;
				B(0, 2) = 0.0;
				B(1, 0) = -4.0;
				B(1, 1) = 5.0;
				B(1, 2) = -6.0;
				B(2, 0) = 2.0;
				B(2, 1) = -8.0;
				B(2, 2) = 9.0;
				B(3, 0) = -10.0;
				B(3, 1) = 11.0;
				B(3, 2) = -12.0;
				B(4, 0) = 1.0;
				B(4, 1) = -14.0;
				B(4, 2) = 15.0;
				return Bvec;
			};
			auto testCase1D = []() -> aks::host_multi_dim_vector<double, 1> {
				aks::host_multi_dim_vector<double, 1> Bvec(10);
				auto B = Bvec.view();
				B(0) = 1.0;
				B(1) = -2.0;
				B(2) = 0.0;
				B(3) = -4.0;
				B(4) = 5.0;
				B(5) = -6.0;
				B(6) = 2.0;
				B(7) = -8.0;
				B(8) = 9.0;
				B(9) = -10.0;
				return Bvec;
			};
			auto Bvec(testCase());
			print2D(Bvec.cview());
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase1D());
				auto Bcuda = aks::to_device(Bvec);
				aks::cuda_multi_dim_vector<double, 1> C(3);
				aks::point<int, 1> from; from.x = 3;
				aks::subArray(C.view(), Bcuda.cview(), from, [] AKS_FUNCTION_PREFIX_ATTR(double x) { return x; });
				auto res = aks::to_host(C);
				print1D(Bvec.view());
				print1D(res.view());
			}
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase());
				auto Bcuda = aks::to_device(Bvec);
				aks::cuda_multi_dim_vector<double, 2> C(3, 2);
				aks::point<int, 2> from; from.x = 1, from.y = 1;
				aks::subArray(C.view(), Bcuda.cview(), from, [] AKS_FUNCTION_PREFIX_ATTR(double x) { return x; });
				auto res = aks::to_host(C);
				print2D(Bvec.view());
				print2D(res.view());
			}
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase());
				auto Bcuda = aks::to_device(Bvec);
				aks::cuda_multi_dim_vector<double, 2> C(5, 1);
				aks::point<int, 2> from; from.x = 0, from.y = 1;
				aks::subArray(C.view(), Bcuda.cview(), from, [] AKS_FUNCTION_PREFIX_ATTR(double x) { return x; });
				auto res = aks::to_host(C);
				print2D(Bvec.view());
				print2D(res.view());
				aks::cuda_multi_dim_vector<double, 1> C2(5);
				aks::reduceDim(C2.view(), C.cview(), thrust::plus<double>(), 1);
				auto res2 = aks::to_host(C2);
				print1D(res2.view());
			}
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase());
				auto Bcuda = aks::to_device(Bvec);
				aks::sortAxis(Bcuda, 0, thrust::less<double>());
				Bvec << Bcuda;
				print2D(Bvec.view());
			}
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase());
				auto Bcuda = aks::to_device(Bvec);
				aks::sortAxis(Bcuda, 1, thrust::less<double>());
				Bvec << Bcuda;
				print2D(Bvec.view());
			}
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase());
				auto Bcuda = aks::to_device(Bvec);
				aks::sortAxis(Bcuda, 0, thrust::greater<double>());
				Bvec << Bcuda;
				print2D(Bvec.view());
			}
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase());
				auto Bcuda = aks::to_device(Bvec);
				aks::sortAxis(Bcuda, 1, thrust::greater<double>());
				Bvec << Bcuda;
				print2D(Bvec.view());
			}
		}
		{
			std::cout << "---" << testNum++ << "---\n";
			auto testCase2 = []() -> aks::host_multi_dim_vector<double, 2> {
				aks::host_multi_dim_vector<double, 2> Bvec(3, 5);
				auto B = Bvec.view();
				B(0, 0) = 1.0;
				B(0, 1) = -2.0;
				B(0, 2) = 0.0;
				B(0, 3) = -4.0;
				B(0, 4) = 5.0;
				B(1, 0) = -6.0;
				B(1, 1) = 2.0;
				B(1, 2) = -8.0;
				B(1, 3) = 9.0;
				B(1, 4) = -10.0;
				B(2, 0) = 11.0;
				B(2, 1) = -12.0;
				B(2, 2) = 1.0;
				B(2, 3) = -14.0;
				B(2, 4) = 15.0;
				return Bvec;
			};
			auto Bvec(testCase2());
			print2D(Bvec.cview());
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase2());
				auto Bcuda = aks::to_device(Bvec);
				aks::sortAxis(Bcuda, 0, thrust::less<double>());
				Bvec << Bcuda;
				print2D(Bvec.view());
			}
			{
				std::cout << "---" << testNum++ << "---\n";
				auto Bvec(testCase2());
				auto Bcuda = aks::to_device(Bvec);
				aks::sortAxis(Bcuda, 1, thrust::less<double>());
				Bvec << Bcuda;
				print2D(Bvec.view());
			}
		}
	}

	return 0;
}

int main()
{
	object_check();
	operatorCheck();
	reduceDimCheck();
	blas_checks();
	//   //return 0;
	   //main2();
	compile_time_differentiation_tests();
	run_experiments();
	//check2();

	//aks::cuda_context ctxt;

	std::vector<int> const a = { 1, 2, 3, 4, 5 };
	std::vector<int> const b = { 10, 20, 30, 40, 50 };
	std::vector<int> c;

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda2(c, a, b);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda2(std::vector<int>& c, std::vector<int> const& a, std::vector<int> const& b)
{
	using namespace aks;

	cuda_context ctxt(cuda_device(0));
	auto da = make_cuda_pointer(a);
	auto db = make_cuda_pointer(b);
	//cuda_pointer<int const> db(size, b);
	cuda_pointer<int> dc(a.size());
	//dc.deep_copy_from(db);
	auto const ma = make_multi_dim_vector(aks::e_device_type::DEVICE, da.data(), da.size());
	auto const mb = make_multi_dim_vector(aks::e_device_type::DEVICE, db.data(), da.size());
	auto mc = make_multi_dim_vector(aks::e_device_type::DEVICE, dc.data(), da.size());

	{
		cuda_sync_context sync_ctxt;
		addKernel << <1, a.size() >> > (mc, ma, mb);
	}

	thrust::transform(thrust_utils::begin(ma), thrust_utils::end(ma), thrust_utils::begin(mb), thrust_utils::begin(mc), thrust::plus<int>());

	printf("%d\n", thrust::reduce(thrust_utils::begin(ma), thrust_utils::end(ma), (int)0));
	printf("%d\n", thrust::reduce(thrust_utils::begin(mb), thrust_utils::end(mb), (int)0));
	printf("%d\n", thrust::reduce(thrust_utils::begin(mc), thrust_utils::end(mc), (int)0));

	thrust::transform(thrust_utils::begin(mc), thrust_utils::end(mc), thrust_utils::begin(mc), thrust::negate<int>());
	thrust::sort(thrust_utils::begin(mc), thrust_utils::end(mc));

	c = from_cuda_pointer(dc);

	assert(!da.has_error_occurred() && !db.has_error_occurred() && !dc.has_error_occurred());

	//check();

	return last_status();
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(std::vector<int>& c, std::vector<int> const& a, std::vector<int> const& b)
{
	aks::cuda_context ctxt(aks::cuda_device(0));
	aks::cuda_pointer<int const> da = aks::make_cuda_pointer(a);
	aks::cuda_pointer<int const> db = aks::make_cuda_pointer(b);
	//aks::cuda_pointer<int const> db(size, b);
	aks::cuda_pointer<int> dc(a.size());
	//dc.deep_copy_from(db);
	aks::multi_dim_vector<int const, 1> const ma = aks::make_multi_dim_vector(aks::e_device_type::DEVICE, da.data(), da.size());
	aks::multi_dim_vector<int const, 1> const mb = aks::make_multi_dim_vector(aks::e_device_type::DEVICE, db.data(), da.size());
	aks::multi_dim_vector<int, 1> mc = aks::make_multi_dim_vector(aks::e_device_type::DEVICE, dc.data(), da.size());

	{
		aks::cuda_sync_context sync_ctxt;
		addKernel << <1, a.size() >> > (mc, ma, mb);
	}

	c = aks::from_cuda_pointer(dc);

	//thrust::device_vector<int> const tva = aks::to_thrust_device_vector(ma);
	//thrust::device_vector<int> const tvb = aks::to_thrust_device_vector(mb);
	//thrust::device_vector<int> tvc = aks::to_thrust_device_vector(mc);

	thrust::transform(aks::thrust_utils::begin(ma), aks::thrust_utils::end(ma), aks::thrust_utils::begin(mb), aks::thrust_utils::begin(mc), thrust::plus<int>());

	printf("%d\n", thrust::reduce(aks::thrust_utils::begin(ma), aks::thrust_utils::end(ma), (int)0));
	printf("%d\n", thrust::reduce(aks::thrust_utils::begin(mb), aks::thrust_utils::end(mb), (int)0));
	printf("%d\n", thrust::reduce(aks::thrust_utils::begin(mc), aks::thrust_utils::end(mc), (int)0));

	thrust::transform(aks::thrust_utils::begin(mc), aks::thrust_utils::end(mc), aks::thrust_utils::begin(mc), thrust::negate<int>());
	thrust::sort(aks::thrust_utils::begin(mc), aks::thrust_utils::end(mc));

	c = aks::from_cuda_pointer(dc);

	assert(!da.has_error_occurred() && !db.has_error_occurred() && !dc.has_error_occurred());

	//check();

	return aks::last_status();
}