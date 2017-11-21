#include "experiments.hpp"
#include "multi_dim_vector.hpp"
#include "multi_dim_vector_iterator.hpp"
#include "multi_dim_vector_range.hpp"
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
				for (auto const& x : aks::make_multi_dim_vector_range(b, i, j, aks::token()))
					sum += x;

				c(i, j, k) = sum;
			}
}

template<size_t _dimensions>
struct multi_dim_counter
{
	enum{dimensions = _dimensions};
	multi_dim_counter(
		  bool * const varying_dimensions
		, size_t * const current
		, size_t * const maximums
	) {
		auto copy_over = [](auto const * from, auto* to) {
			for (size_t i = 0; i < dimensions; ++i)
				to[i] = from[i];
		};
		copy_over(varying_dimensions, m_varying_dimensions);
		copy_over(current, m_current);
		copy_over(maximums, m_maximums);
	}

	size_t const * current() const {
		return m_current;
	}

	void increment() {
		for (int i = dimensions-1; i >= int(0); --i) {
			if (m_varying_dimensions[i]) {
				++m_current[i];
				if (m_current[i] == m_maximums[i]) {
					m_current[i] = 0;
				} else {
					break;
				}
			}
		}
	}

	bool m_varying_dimensions[dimensions];
	size_t m_current[dimensions];
	size_t m_maximums[dimensions];
};

template<size_t dimensions>
std::ostream& operator<<(std::ostream& o, multi_dim_counter<dimensions> const& c)
{
	auto print = [&](auto const* x){
		o << '{';
		o << x[0];
		for (size_t i = 1; i < dimensions; ++i) {
			o << ", " << x[i];
		}
		o << '}';
	};

	print(c.m_current);
	return o;
}

template<typename... As>
size_t index_find(As...)
{
	return aks::iterator_detail::index_of<aks::token, As...>::value;
}

template<typename... args>
auto make_counter(args... as)
{

}

#include <type_traits>

struct token {};

template<typename T, typename... Ts>
struct count_tokens
{
	enum { value = (std::is_same<T, token>::value ? 1 : 0) + count_tokens<Ts...>::value };
};

template<typename T>
struct count_tokens<T>
{
	enum { value = std::is_same<T, token>::value ? 1 : 0 };
};

template<size_t C, size_t... Is>
struct dummy
{
	static void apply()
	{
		printf("");
	}
};

template<typename... Ts>
void something(Ts... ts)
{
	dummy<count_tokens<Ts...>::value, std::is_same<Ts, token>::value...>::apply();
}

template<size_t C, size_t I, size_t... Is>
struct token_idx
{
	static_assert(sizeof...(Is) > 0, "Miscounted?");
	enum { value = 1 + token_idx<C - I, Is...>::value };
};

template<size_t... Is>
struct token_idx<0, 1, Is...>
{
	enum { value = 0 };
};

template<size_t N, size_t... Is>
struct MovingIndices
{
	static void apply()
	{
		printf("");
	}
};

template<size_t N, size_t... Ts, size_t... Is>
auto moving_impl(std::index_sequence<Is...>)
{
	return MovingIndices<N, token_idx<Is, Ts...>::value...>();
}

template<typename... Ts>
auto moving(Ts... ts)
{
	return moving_impl<sizeof...(Ts), std::is_same<Ts, token>::value...>(std::make_index_sequence<count_tokens<Ts...>::value>());
}

int exp_01()
{
	auto ct = count_tokens<int, token, token, int, token>::value;
	auto ct2 = token_idx<0, 0, 0, 0, 0, 1>::value;
	auto ct22 = token_idx<0, 1>::value;
	auto ct3 = token_idx<1, 0, 1, 1, 0, 1>::value;
	auto ct4 = token_idx<2, 0, 1, 1, 0, 1>::value;
	auto ct5 = token_idx<0, 0, 1, 1, 0, 1, 0, 1, 1, 0>::value;
	auto ct6 = token_idx<1, 0, 1, 1, 0, 1, 0, 1, 1, 0>::value;
	auto ct7 = token_idx<2, 0, 1, 1, 0, 1, 0, 1, 1, 0>::value;
	auto ct8 = token_idx<3, 0, 1, 1, 0, 1, 0, 1, 1, 0>::value;
	auto ct9 = token_idx<4, 0, 1, 1, 0, 1, 0, 1, 1, 0>::value;
	//auto ct10 = token_idx<5, 0, 1, 1, 0, 1, 0, 1, 1, 0>::value;


	something(2, token(), 2, token(), token(), 3.0);
	auto m = moving(2, token(), token(), token(), 1.0, token(), 3.0);

	return 0;
}


int run_experiments()
{
	exp_01();
	{
		size_t current[4] = { 0,3,2,0 };
		size_t max[4] = { 4,4,4,4 };
		bool vars[4] = { 0,1,1,0 };
		multi_dim_counter<4> counter(vars, current, max);
		for (size_t i = 0; i < 80; ++i)
		{
			std::cout << counter << std::endl;
			counter.increment();
		}
	}

	{
		aks::host_multi_dim_vector<int, 1> host_vec(10);
		auto view = host_vec.view();

		for (auto const& x : view)
		{
			printf("%d\n", x);
		}
		
		printf("%zd\n", index_find(aks::token()));
		printf("%zd\n", index_find(2,2,aks::token()));
		printf("%zd\n", index_find(2, 2, aks::token(),3,4));
		for (auto it = aks::begin(view, aks::token(5)), end = aks::end(view, aks::token(5)); it != end; ++it)
			*it = 23;
		auto const cview = view;
		for (auto it = aks::begin(cview, aks::token()), end = aks::end(cview, aks::token()); it != end; ++it)
			printf("%d, ", *it);
		for (auto& x : aks::make_multi_dim_vector_range(view, aks::token(10)) )
		{
			printf("%d, ", x);
		}
		for (auto& x : aks::make_multi_dim_vector_range(cview, aks::token()))
		{
			printf("%d, ", x);
		}
		printf("\n");
	}

	{
		//aks::host_multi_dim_vector<int, 3> host_vec(3, 4, 5);
		//auto res_vec = host_vec;
		//auto host_view = host_vec.view();
		//for (size_t x = 0; x < 3; ++x)
		//	for (size_t y = 0; y < 4; ++y)
		//		for (size_t z = 0; z < 5; ++z)
		//		{
		//			host_view(x, y, z) = int(x * 4 * 5 + y * 5 + z);
		//		}
		//addKernel3(res_vec.view(), host_view, host_view);
		//printf("");
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
			auto rng = aks::make_multi_dim_vector_range_from_iterators(it, end);
			auto rng2 = aks::make_multi_dim_vector_range_from_iterators(cit, cend);
			for (auto& x : rng) { x = 23; }
			//for (auto& x : rng2) { x = 23; }
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