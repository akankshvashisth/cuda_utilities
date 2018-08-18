#ifndef __cuda_blas_utilities_hpp__
#define __cuda_blas_utilities_hpp__

#include "multi_dim_vector.hpp"

namespace aks
{
	namespace cuda_blas
	{
		template<typename T>
		using cuda_matrix = cuda_multi_dim_vector<T, 2>;

		template<typename T>
		using cuda_vector = cuda_multi_dim_vector<T, 1>;

		template<typename T>
		using matrix = multi_dim_vector<T, 2>;

		template<typename T>
		size_t rows(cuda_matrix<T> const& t) { return get_max_dim<0>(t); }

		template<typename T>
		size_t cols(cuda_matrix<T> const& t) { return get_max_dim<1>(t); }

		template<typename T>
		size_t rows(cuda_vector<T> const& t) { return get_max_dim<0>(t); }

		template<typename T>
		size_t cols(cuda_vector<T> const& t) { return 1; }

		template<typename T>
		size_t rows(matrix<T> const& t) { return get_max_dim<0>(t); }

		template<typename T>
		size_t cols(matrix<T> const& t) { return get_max_dim<1>(t); }
	}
}

#endif // !__cuda_blas_utilities_hpp__