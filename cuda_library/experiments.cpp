#include "experiments.hpp"
#include "multi_dim_vector.hpp"
#include "multi_dim_vector_iterator.hpp"
#include <tuple>

template<size_t N, typename F, typename... Ts>
struct curry
{
	enum{remaining = N-sizeof...(Ts)};
	typedef std::tuple<Ts...> tuple_type;

	curry(F f, Ts... ts) :args(ts...) {}
	curry(tuple_type ts) :args(ts) {}

	F func;
	tuple_type args;
};

template<size_t N, typename F, typename... Ts, typename... As>
auto cat(curry<N, F, Ts...> c0, curry<N, F, As...> c1)
{
	return curry<N, F, Ts..., As...>(c0.func, std::tuple_cat(c0.args, c1.args));
}

template<size_t N, typename F, typename... Ts>
auto make_curry(F f, Ts... ts)
{
	return curry<N, F, Ts...>(f, ts...);
}

#include <iostream>

void show(int x, int y, int z, int w)
{
	std::cout << x << y << z << w << std::endl;
}

void addKernel3(aks::multi_dim_vector<int, 3> c, aks::multi_dim_vector<int const, 3> const a, aks::multi_dim_vector<int const, 3> const b)
{
	for (size_t i = 0; i<3; ++i)
		for (size_t j = 0; j<4; ++j)
			for (size_t k = 0; k < 5; ++k) {
				int sum = 0;
				for (auto it = aks::begin(a, aks::token(), j, k), end = aks::end(a, aks::token(), j, k); it != end; ++it)
					sum += *it;
				for (auto it = aks::begin(a, i, aks::token(), k), end = aks::end(a, i, aks::token(), k); it != end; ++it)
					sum += *it;
				for (auto it = aks::begin(b, i, j, aks::token()), end = aks::end(b, i, j, aks::token()); it != end; ++it)
					sum += *it;

				c(i, j, k) = sum;
			}
}

template<typename... As>
size_t index_find(As...)
{
	return aks::iterator_detail::index_of<aks::token, As...>::value;
}

int run_experiments()
{
	{
		aks::host_multi_dim_vector<int, 1> host_vec(10);
		auto view = host_vec.view();
		
		printf("%zd\n", index_find(aks::token()));
		printf("%zd\n", index_find(2,2,aks::token()));
		printf("%zd\n", index_find(2, 2, aks::token(),3,4));
		for (auto it = aks::begin(view, aks::token()), end = aks::end(view, aks::token()); it != end; ++it)
			*it = 23;
		auto const cview = view;
		for (auto it = aks::begin(cview, aks::token()), end = aks::end(cview, aks::token()); it != end; ++it)
			printf("%d, ", *it);
		printf("\n");
	}

	{
		aks::host_multi_dim_vector<int, 3> host_vec(3, 4, 5);
		auto res_vec = host_vec;
		auto host_view = host_vec.view();
		for (size_t x = 0; x < 3; ++x)
			for (size_t y = 0; y < 4; ++y)
				for (size_t z = 0; z < 5; ++z)
				{
					host_view(x, y, z) = int(x * 4 * 5 + y * 5 + z);
				}
		addKernel3(res_vec.view(), host_view, host_view);
		printf("");
	}

	{
		aks::host_multi_dim_vector<int, 4> v(2, 3, 4, 5);
		//aks::func(1, 2, 3, aks::token);

		{
			auto view = v.view();
			size_t dims[] = { 1,2,3,4 };
			aks::iterator_detail::get_element(view, dims);// = 23;
			//printf("%d", view(1, 2, 3, 4));

			auto iter = aks::begin(view, aks::token(), 2, 3, 4);
			++iter;
			auto end = aks::end(view, aks::token(), 2, 3, 4);
			//*(iter) = 23;

			auto iter2 = aks::begin(view, 1, aks::token(), 3, 4);
			++iter2;
			auto end2 = aks::end(view, 1, aks::token(), 3, 4);
			*(iter2) = 23;

			for (auto it = aks::begin(view, 0, 0, 0, aks::token()), end = aks::end(view, 0, 0, 0, aks::token()); it != end; ++it)
				(*it) = 53;

			auto const cview = view;
			for (auto it = aks::begin(cview, 0, 0, 0, aks::token()), end = aks::end(cview, 0, 0, 0, aks::token()); it != end; ++it)
				printf("%d\n", *it);

			printf("");
		}
		{
			auto view = v.view();
			auto const cview = view;
			auto it = aks::begin(view, aks::token(), 2, 3, 4);
			auto cit = aks::begin(cview, aks::token(), 2, 3, 4);
			auto end = aks::end(view, aks::token(), 2, 3, 4);
			auto cend = aks::end(cview, aks::token(), 2, 3, 4);
			auto b0 = it != it;
			auto b1 = cit != cit;
			auto b2 = it != cit;
			auto b3 = cit != it;
			*it = 23;
			//*cit = 23;
			cit = it;
		}
	}

	return 0;
}