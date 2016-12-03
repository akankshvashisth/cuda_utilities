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





int run_experiments()
{
	aks::host_multi_dim_vector<int, 4> v(2, 3, 4, 5);
	//aks::func(1, 2, 3, aks::token);

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

	for (auto it = aks::begin(view, 0, aks::token(), 2, 3), end = aks::end(view, 0, aks::token(), 2, 3); it != end; ++it)
		(*it) = 53;

	return 0;
}