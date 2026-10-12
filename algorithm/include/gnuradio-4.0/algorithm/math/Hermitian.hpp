#ifndef GNURADIO_HERMITIAN_HPP
#define GNURADIO_HERMITIAN_HPP

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/SVD.hpp>
#include <gnuradio-4.0/algorithm/math/TensorMath.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <expected>
#include <format>
#include <limits>
#include <numeric>
#include <source_location>
#include <vector>

/**
 * Eigendecomposition of a Hermitian (real: symmetric) matrix and its split into a signal and a noise
 * subspace, for real and complex floating-point element types. Each function returns its result in a
 * std::expected and reports a bad input through its gr::Error, which names the function that refused
 * the input and the fault: wrong tensor rank, mismatched extents, a non-finite entry or norm, an
 * out-of-range split, or off-diagonal entries that stay above the tolerance after maxIterations sweeps.
 */
namespace gr::math {

namespace eig {
/// Eigenvalues in ascending order and the unitary matrix with one eigenvector per column.
template<typename T, typename RealT = gr::meta::fundamental_base_value_type_t<T>>
struct Decomposition {
    Tensor<RealT> values;
    Tensor<T>     vectors;
};

/// The signal and the noise subspace, one basis vector per column, each in ascending order of eigenvalue.
template<typename T>
struct Subspaces {
    Tensor<T> signal;
    Tensor<T> noise;
};

template<typename T>
struct Config {
    std::size_t maxIterations = 0;     /// sweeps over every off-diagonal pair; 0 means 30 * n
    T           tolerance     = T{-1}; /// a pair rotates while its entry exceeds tolerance * ||A||_F; negative means eps
};

/// Splits off the `count` largest eigenvalues as the signal subspace.
struct Rank {
    std::size_t count;
};

/// Splits off the eigenvalues above `value` as the signal subspace.
template<std::floating_point T>
struct Threshold {
    T value;
};
} // namespace eig

namespace detail {

/// ||M||_F from the squares of M * 2^-e, where 2^e is the power of two at or below the largest magnitude, so no square under- or overflows.
template<typename T, typename RealT = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] RealT scaledFrobeniusNorm(const Tensor<T>& M) noexcept {
    RealT largest{0};
    for (const T& m : M) {
        largest = std::max(largest, static_cast<RealT>(std::abs(m)));
    }
    const int   exponent = std::clamp(std::ilogb(largest), std::numeric_limits<RealT>::min_exponent - 1, std::numeric_limits<RealT>::max_exponent - 1);
    const RealT down     = std::ldexp(RealT{1}, -exponent);
    RealT       sumSq{0};
    for (const T& m : M) {
        sumSq += squaredMagnitude(m * down);
    }
    return std::ldexp(std::sqrt(sumSq), exponent);
}

/// Multiplies rows i and j of the first `cols` columns of M by J^H from the left, for J = [[c, s], [-conj(s), c]].
template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void rotateRows(Tensor<T>& M, std::size_t i, std::size_t j, BaseValueType c, T s, std::size_t cols) noexcept {
    for (std::size_t k = 0UZ; k < cols; ++k) {
        const T mi = M[i, k];
        const T mj = M[j, k];
        M[i, k]    = c * mi - s * mj;
        M[j, k]    = conj(s) * mi + c * mj;
    }
}

} // namespace detail

/**
 * @brief Eigendecomposition A = V diag(values) V^H of a Hermitian matrix by cyclic Jacobi rotations.
 *
 * Reads the upper triangle of A and the real part of its diagonal. `values` is real and ascending;
 * column k of the n x n unitary `vectors` is the eigenvector of values[k]. Each sweep visits every
 * pair p < q and applies the two-sided rotation that zeroes entry (p, q). The decomposition
 * converges when a sweep finds no entry above config.tolerance * ||A||_F. A decomposition that
 * has not converged after config.maxIterations sweeps is an error, and so is an A whose ||A||_F
 * exceeds the largest finite value of the element type.
 */
template<TensorLike TensorA, typename T = typename std::remove_cvref_t<TensorA>::value_type, typename RealT = gr::meta::fundamental_base_value_type_t<T>>
requires detail::FloatingElement<T>
[[nodiscard]] std::expected<eig::Decomposition<T, RealT>, gr::Error> eigh(const TensorA& A, const eig::Config<RealT>& config = {}, std::source_location loc = std::source_location::current()) {
    if (A.rank() != 2UZ || A.extent(0) != A.extent(1) || !isFinite(A)) {
        return std::unexpected(gr::Error("eigh: A is not a finite square matrix", loc));
    }
    const std::size_t n = A.extent(0);

    Tensor<T> W({n, n});
    for (std::size_t i = 0UZ; i < n; ++i) {
        W[i, i] = T{static_cast<RealT>(std::real(A[i, i]))};
        for (std::size_t j = i + 1UZ; j < n; ++j) {
            W[i, j] = A[i, j];
            W[j, i] = detail::conj(A[i, j]);
        }
    }
    Tensor<T> V({n, n});
    V.fill(T{0});
    for (std::size_t i = 0UZ; i < n; ++i) {
        V[i, i] = T{1};
    }

    const RealT norm = detail::scaledFrobeniusNorm(W);
    if (!std::isfinite(norm)) {
        return std::unexpected(gr::Error("eigh: the Frobenius norm of A is not finite", loc));
    }
    const RealT       tolerance = config.tolerance < RealT{0} ? std::numeric_limits<RealT>::epsilon() : config.tolerance;
    const RealT       threshold = tolerance * norm;
    const std::size_t maxSweeps = config.maxIterations == 0UZ ? 30UZ * std::max(n, 1UZ) : config.maxIterations;

    bool converged = false;
    for (std::size_t sweep = 0UZ; sweep < maxSweeps && !converged; ++sweep) {
        converged = true;
        for (std::size_t p = 0UZ; p + 1UZ < n; ++p) {
            for (std::size_t q = p + 1UZ; q < n; ++q) {
                const T apq = W[p, q];
                if (!(std::abs(apq) > threshold)) {
                    continue;
                }
                converged          = false;
                const RealT app    = std::real(W[p, p]);
                const RealT aqq    = std::real(W[q, q]);
                const auto  rotate = detail::jacobiRotation(app, aqq, apq);
                detail::rotateColumns(W, p, q, rotate.c, rotate.s, n);
                detail::rotateRows(W, p, q, rotate.c, rotate.s, n);
                // the rotation zeroes entry (p, q); the 2 x 2 block is set from its closed form, which leaves no rounding residue there
                W[p, p] = T{app - rotate.shift};
                W[q, q] = T{aqq + rotate.shift};
                W[p, q] = T{0};
                W[q, p] = T{0};
                detail::rotateColumns(V, p, q, rotate.c, rotate.s, n);
            }
        }
    }

    if (!converged) {
        return std::unexpected(gr::Error(std::format("eigh: no convergence within {} sweep{}", maxSweeps, maxSweeps == 1UZ ? "" : "s"), loc));
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0UZ);
    std::ranges::stable_sort(order, [&W](std::size_t a, std::size_t b) { return std::real(W[a, a]) < std::real(W[b, b]); });

    eig::Decomposition<T, RealT> result{.values = Tensor<RealT>({n}), .vectors = Tensor<T>({n, n})};
    for (std::size_t k = 0UZ; k < n; ++k) {
        result.values[k] = std::real(W[order[k], order[k]]);
        for (std::size_t i = 0UZ; i < n; ++i) {
            result.vectors[i, k] = V[i, order[k]];
        }
    }
    return result;
}

namespace detail {

template<typename T, typename RealT>
[[nodiscard]] eig::Subspaces<T> splitSubspace(const Tensor<RealT>& values, const Tensor<T>& vectors, std::size_t signalCount) {
    const std::size_t n          = values.size();
    const std::size_t noiseCount = n - signalCount;
    eig::Subspaces<T> result{.signal = Tensor<T>({n, signalCount}), .noise = Tensor<T>({n, noiseCount})};
    for (std::size_t i = 0UZ; i < n; ++i) {
        for (std::size_t k = 0UZ; k < noiseCount; ++k) {
            result.noise[i, k] = vectors[i, k];
        }
        for (std::size_t k = 0UZ; k < signalCount; ++k) {
            result.signal[i, k] = vectors[i, noiseCount + k];
        }
    }
    return result;
}

template<typename T, typename RealT>
[[nodiscard]] bool isEigenPair(const Tensor<RealT>& values, const Tensor<T>& vectors) {
    const std::size_t n = values.size();
    return values.rank() == 1UZ && vectors.rank() == 2UZ && vectors.extent(0) == n && vectors.extent(1) == n && std::ranges::is_sorted(values) && isFinite(values) && isFinite(vectors);
}

} // namespace detail

/**
 * @brief Splits the eigenvectors of eigh into the signal subspace of the `rank.count` largest eigenvalues and the noise subspace of the rest.
 *
 * `values` is ascending and `vectors` holds one eigenvector per column, as eigh returns them.
 * `noise` takes the first n - count columns and `signal` the last count columns, both in ascending
 * order of eigenvalue. Returns an error when count exceeds n, when the inputs are not such a pair,
 * or when an entry of either is not finite.
 */
template<typename T, typename RealT>
[[nodiscard]] std::expected<eig::Subspaces<T>, gr::Error> subspace(const Tensor<RealT>& values, const Tensor<T>& vectors, eig::Rank rank, std::source_location loc = std::source_location::current()) {
    if (!detail::isEigenPair(values, vectors)) {
        return std::unexpected(gr::Error("subspace: values and vectors are not a finite ascending eigenvalue and eigenvector pair", loc));
    }
    if (rank.count > values.size()) {
        return std::unexpected(gr::Error(std::format("subspace: a rank of {} exceeds the {} eigenvalues", rank.count, values.size()), loc));
    }
    return detail::splitSubspace(values, vectors, rank.count);
}

/**
 * @brief Splits the eigenvectors of eigh into the signal subspace of the eigenvalues above a threshold and the noise subspace of the rest.
 *
 * The column order and the requirements on the inputs are those of the Rank form.
 */
template<typename T, typename RealT, typename U>
[[nodiscard]] std::expected<eig::Subspaces<T>, gr::Error> subspace(const Tensor<RealT>& values, const Tensor<T>& vectors, eig::Threshold<U> threshold, std::source_location loc = std::source_location::current()) {
    if (!detail::isEigenPair(values, vectors)) {
        return std::unexpected(gr::Error("subspace: values and vectors are not a finite ascending eigenvalue and eigenvector pair", loc));
    }
    if (!std::isfinite(threshold.value)) {
        return std::unexpected(gr::Error("subspace: the threshold is not finite", loc));
    }
    const auto        above       = std::ranges::count_if(values, [&threshold](RealT v) { return v > static_cast<RealT>(threshold.value); });
    const std::size_t signalCount = static_cast<std::size_t>(above);
    return detail::splitSubspace(values, vectors, signalCount);
}

} // namespace gr::math

#endif // GNURADIO_HERMITIAN_HPP
