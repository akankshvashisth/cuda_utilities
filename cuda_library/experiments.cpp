#include "experiments.hpp"

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
	return 0;
}