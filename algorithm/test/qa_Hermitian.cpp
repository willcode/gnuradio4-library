#include <boost/ut.hpp>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/Hermitian.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

namespace {

using gr::Tensor;
using namespace gr::math;

using ElementTypes = std::tuple<float, double, std::complex<float>, std::complex<double>>;

template<typename T>
using Real = gr::meta::fundamental_base_value_type_t<T>;

template<typename T>
std::string typeName() {
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return "complex<float>";
    } else {
        return "complex<double>";
    }
}

/// An element from a real and an imaginary part; a real element type drops the imaginary part.
template<typename T>
T element(double re, double im = 0.0) {
    if constexpr (gr::meta::complex_like<T>) {
        return T{static_cast<Real<T>>(re), static_cast<Real<T>>(im)};
    } else {
        return static_cast<T>(re);
    }
}

template<typename T>
T conjugate(const T& x) {
    if constexpr (gr::meta::complex_like<T>) {
        return std::conj(x);
    } else {
        return x;
    }
}

/// The error bound of a decomposition of an n x n matrix relative to its largest entry: a multiple of n * eps.
template<typename T>
Real<T> tolerance(std::size_t n) {
    return Real<T>{64} * static_cast<Real<T>>(n) * std::numeric_limits<Real<T>>::epsilon();
}

template<typename T>
Tensor<T> matrix(std::size_t rows, std::size_t cols, std::initializer_list<std::pair<double, double>> values) {
    Tensor<T> M({rows, cols});
    auto      it = M.begin();
    for (const auto& [re, im] : values) {
        *it++ = element<T>(re, im);
    }
    return M;
}

template<typename T>
Tensor<T> multiply(const Tensor<T>& A, const Tensor<T>& B) {
    Tensor<T> C({A.extent(0), B.extent(1)});
    C.fill(T{0});
    for (std::size_t i = 0UZ; i < A.extent(0); ++i) {
        for (std::size_t j = 0UZ; j < B.extent(1); ++j) {
            for (std::size_t k = 0UZ; k < A.extent(1); ++k) {
                C[i, j] += A[i, k] * B[k, j];
            }
        }
    }
    return C;
}

template<typename T>
Tensor<T> adjoint(const Tensor<T>& A) {
    Tensor<T> H({A.extent(1), A.extent(0)});
    for (std::size_t i = 0UZ; i < A.extent(0); ++i) {
        for (std::size_t j = 0UZ; j < A.extent(1); ++j) {
            H[j, i] = conjugate(A[i, j]);
        }
    }
    return H;
}

template<typename T>
Tensor<T> identity(std::size_t n) {
    Tensor<T> I({n, n});
    I.fill(T{0});
    for (std::size_t i = 0UZ; i < n; ++i) {
        I[i, i] = T{1};
    }
    return I;
}

/// The largest element-wise distance between two tensors of equal size, relative to the largest magnitude in `reference`.
template<typename T>
Real<T> relativeError(const Tensor<T>& actual, const Tensor<T>& reference) {
    Real<T> scale{0};
    Real<T> error{0};
    for (std::size_t k = 0UZ; k < reference.size(); ++k) {
        scale = std::max(scale, static_cast<Real<T>>(std::abs(reference.data()[k])));
        error = std::max(error, static_cast<Real<T>>(std::abs(actual.data()[k] - reference.data()[k])));
    }
    return error / std::max(scale, std::numeric_limits<Real<T>>::min());
}

/// A Hermitian 5 x 5 matrix written by its upper triangle.
template<typename T>
Tensor<T> chosenHermitian() {
    Tensor<T> A = matrix<T>(5, 5,
        {{4, 0}, {1, 2}, {0, -1}, {2, 0}, {-1, 1},    //
            {0, 0}, {-3, 0}, {1, 1}, {0, 2}, {3, 0},  //
            {0, 0}, {0, 0}, {2, 0}, {-2, 1}, {1, -1}, //
            {0, 0}, {0, 0}, {0, 0}, {5, 0}, {0, 1},   //
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-1, 0}});
    for (std::size_t i = 0UZ; i < 5UZ; ++i) {
        for (std::size_t j = i + 1UZ; j < 5UZ; ++j) {
            A[j, i] = conjugate(A[i, j]);
        }
    }
    return A;
}

template<typename T>
Tensor<T> diagonal(const Tensor<Real<T>>& values) {
    Tensor<T> D({values.size(), values.size()});
    D.fill(T{0});
    for (std::size_t i = 0UZ; i < values.size(); ++i) {
        D[i, i] = T{values[i]};
    }
    return D;
}

/// The largest entry of A V - V diag(values) relative to `scale`, with each factor divided by `scale`, a power of two, before the product. A NaN entry gives NaN.
template<typename T>
Real<T> scaledResidual(const Tensor<T>& A, const Tensor<T>& V, const Tensor<Real<T>>& values, Real<T> scale) {
    const std::size_t n = values.size();
    Real<T>           residual{0};
    for (std::size_t i = 0UZ; i < n; ++i) {
        for (std::size_t k = 0UZ; k < n; ++k) {
            T av{0};
            for (std::size_t j = 0UZ; j < n; ++j) {
                av += (A[i, j] / scale) * V[j, k];
            }
            const Real<T> entry = static_cast<Real<T>>(std::abs(av - V[i, k] * (values[k] / scale)));
            if (std::isnan(entry)) {
                return entry;
            }
            residual = std::max(residual, entry);
        }
    }
    return residual;
}

/// ||M^H a|| / ||a||: the share of a vector that the columns of M capture.
template<typename T>
Real<T> projectionNorm(const Tensor<T>& M, const Tensor<T>& a) {
    const Tensor<T> projection = multiply(adjoint(M), a);
    Real<T>         numerator{0};
    Real<T>         denominator{0};
    for (const T& p : projection) {
        numerator += static_cast<Real<T>>(std::norm(p));
    }
    for (const T& x : a) {
        denominator += static_cast<Real<T>>(std::norm(x));
    }
    return std::sqrt(numerator / denominator);
}

} // namespace

const boost::ut::suite<"eigh"> _eigh = [] {
    using namespace boost::ut;

    "reconstruction A = V diag(values) V^H and orthonormal V"_test = []<typename T> {
        const Tensor<T> A      = chosenHermitian<T>();
        const auto      result = eigh(A);
        expect(fatal(result.has_value())) << typeName<T>();
        const auto& [values, V] = *result;
        expect(eq(values.size(), 5UZ) && eq(V.extent(0), 5UZ) && eq(V.extent(1), 5UZ)) << typeName<T>();
        expect(le(relativeError(multiply(multiply(V, diagonal<T>(values)), adjoint(V)), A), tolerance<T>(5))) << typeName<T>() << "A = V diag V^H";
        expect(le(relativeError(multiply(adjoint(V), V), identity<T>(5)), tolerance<T>(5))) << typeName<T>() << "V^H V = I";
        expect(std::ranges::is_sorted(values)) << typeName<T>() << "ascending";
    } | ElementTypes{};

    "a matrix near either end of the range decomposes as at unit scale"_test = []<typename T> {
        // Above sqrt(max) the square of an entry overflows; below sqrt(denorm_min) it underflows to zero.
        const Real<T> large = std::sqrt(std::numeric_limits<Real<T>>::max()) * Real<T>{4};
        const Real<T> small = std::sqrt(std::numeric_limits<Real<T>>::denorm_min()) / Real<T>{16};
        for (const Real<T> scale : {large, small}) {
            Tensor<T> A = chosenHermitian<T>();
            for (auto& a : A) {
                a *= scale;
            }
            const auto result = eigh(A);
            expect(fatal(result.has_value())) << typeName<T>() << scale;
            const auto& [values, V] = *result;
            expect(le(relativeError(multiply(multiply(V, diagonal<T>(values)), adjoint(V)), A), tolerance<T>(5))) << typeName<T>() << scale << "A = V diag V^H";
            expect(le(relativeError(multiply(adjoint(V), V), identity<T>(5)), tolerance<T>(5))) << typeName<T>() << scale << "V^H V = I";
        }

        Tensor<T> beyond = chosenHermitian<T>(); // finite entries whose Frobenius norm exceeds max
        for (auto& a : beyond) {
            a *= std::numeric_limits<Real<T>>::max() / Real<T>{8};
        }
        const auto refused = eigh(beyond);
        expect(fatal(!refused.has_value())) << typeName<T>();
        expect(refused.error().message.contains("Frobenius norm of A is not finite")) << typeName<T>() << refused.error().message;
    } | ElementTypes{};

    "a pair whose diagonal difference overflows decomposes as it does at unit scale"_test = []<typename T> {
        // [[-s, g], [conj(g), s]] with |g| = s / 8 has eigenvalues -s sqrt(65) / 8 and s sqrt(65) / 8.
        // For s = 2^(max_exponent - 1) the eigenvalues and the Frobenius norm are finite, and 2 s overflows.
        const Real<T>                   top    = std::ldexp(Real<T>{1}, std::numeric_limits<Real<T>>::max_exponent - 1);
        const std::pair<double, double> g      = gr::meta::complex_like<T> ? std::pair{0.6, 0.8} : std::pair{1.0, 0.0};
        const Real<T>                   lambda = std::sqrt(Real<T>{65}) / Real<T>{8};
        for (const Real<T> scale : {Real<T>{1}, top}) {
            Tensor<T> A = matrix<T>(2, 2, {{-1, 0}, {g.first / 8, g.second / 8}, {g.first / 8, -g.second / 8}, {1, 0}});
            for (auto& a : A) {
                a *= scale;
            }
            const auto result = eigh(A);
            expect(fatal(result.has_value())) << typeName<T>() << scale;
            const auto& [values, V] = *result;
            expect(le(scaledResidual(A, V, values, scale), tolerance<T>(2))) << typeName<T>() << scale << "A V = V diag(values)";
            expect(le(std::abs(values[0] / scale + lambda), tolerance<T>(2))) << typeName<T>() << scale << values[0];
            expect(le(std::abs(values[1] / scale - lambda), tolerance<T>(2))) << typeName<T>() << scale << values[1];
        }
    } | ElementTypes{};

    "a pair whose doubled off-diagonal entry overflows decomposes as it does at unit scale"_test = []<typename T> {
        // [[0, g], [conj(g), s]] with |g| = s has eigenvalues s (1 - sqrt(5)) / 2 and s (1 + sqrt(5)) / 2.
        // For s = 2^(max_exponent - 1) the eigenvalues and the Frobenius norm are finite, and 2 |g| overflows.
        const Real<T>                   top   = std::ldexp(Real<T>{1}, std::numeric_limits<Real<T>>::max_exponent - 1);
        const std::pair<double, double> g     = gr::meta::complex_like<T> ? std::pair{0.6, 0.8} : std::pair{1.0, 0.0};
        const Real<T>                   root5 = std::sqrt(Real<T>{5});
        for (const Real<T> scale : {Real<T>{1}, top}) {
            Tensor<T> A = matrix<T>(2, 2, {{0, 0}, g, {g.first, -g.second}, {1, 0}});
            for (auto& a : A) {
                a *= scale;
            }
            const auto result = eigh(A);
            expect(fatal(result.has_value())) << typeName<T>() << scale;
            const auto& [values, V] = *result;
            expect(std::isfinite(values[0]) && std::isfinite(values[1])) << typeName<T>() << scale << values[0] << values[1];
            expect(le(scaledResidual(A, V, values, scale), tolerance<T>(2))) << typeName<T>() << scale << "A V = V diag(values)";
            expect(le(std::abs(values[0] / scale - (Real<T>{1} - root5) / Real<T>{2}), tolerance<T>(2))) << typeName<T>() << scale << values[0];
            expect(le(std::abs(values[1] / scale - (Real<T>{1} + root5) / Real<T>{2}), tolerance<T>(2))) << typeName<T>() << scale << values[1];
        }
    } | ElementTypes{};

    "a pair with nearly equal diagonal entries decomposes with a small residual"_test = []<typename T> {
        // [[1, g], [conj(g), 1 + gap]] with |g| = 1e-5. For float the gap 2e-4 is about 1700 eps. The rotation for that
        // pair is about 2.9 degrees, far from the 45 degrees of an equal pair. The gaps 0 and 1e-3 are controls.
        const std::pair<double, double> g = gr::meta::complex_like<T> ? std::pair{0.6e-5, 0.8e-5} : std::pair{1e-5, 0.0};
        for (const double gap : {0.0, 2e-4, 1e-3}) {
            const Tensor<T> A      = matrix<T>(2, 2, {{1, 0}, g, {g.first, -g.second}, {1.0 + gap, 0}});
            const auto      result = eigh(A);
            expect(fatal(result.has_value())) << typeName<T>() << gap;
            const auto& [values, V] = *result;
            expect(le(scaledResidual(A, V, values, Real<T>{1}), tolerance<T>(2))) << typeName<T>() << gap << "A V = V diag(values)";
            expect(le(relativeError(multiply(adjoint(V), V), identity<T>(2)), tolerance<T>(2))) << typeName<T>() << gap << "V^H V = I";
        }
    } | ElementTypes{};

    "eigenvalues worked by hand"_test = []<typename T> {
        // [[2, g], [conj(g), 2]] with |g| = 1 has eigenvalues 2 - |g| and 2 + |g|
        const std::pair<double, double> g    = gr::meta::complex_like<T> ? std::pair{0.6, 0.8} : std::pair{1.0, 0.0};
        const Tensor<T>                 A    = matrix<T>(2, 2, {{2, 0}, g, {g.first, -g.second}, {2, 0}});
        const auto                      pair = eigh(A);
        expect(fatal(pair.has_value())) << typeName<T>();
        const auto& values = pair->values;
        expect(le(std::abs(values[0] - Real<T>{1}), tolerance<T>(2) * Real<T>{3})) << typeName<T>() << values[0];
        expect(le(std::abs(values[1] - Real<T>{3}), tolerance<T>(2) * Real<T>{3})) << typeName<T>() << values[1];

        // a diagonal matrix is sorted, and its eigenvectors are the permuted unit vectors
        const Tensor<T> D            = matrix<T>(3, 3, {{3, 0}, {0, 0}, {0, 0}, {0, 0}, {-1, 0}, {0, 0}, {0, 0}, {0, 0}, {2, 0}});
        const auto      diagonalPair = eigh(D);
        expect(fatal(diagonalPair.has_value())) << typeName<T>();
        const auto& [sorted, V] = *diagonalPair;
        expect(sorted[0] == Real<T>{-1} && sorted[1] == Real<T>{2} && sorted[2] == Real<T>{3}) << typeName<T>();
        expect(V[1, 0] == T{1} && V[2, 1] == T{1} && V[0, 2] == T{1}) << typeName<T>();
    } | ElementTypes{};

    "only the upper triangle is read"_test = []<typename T> {
        const Tensor<T> A     = chosenHermitian<T>();
        Tensor<T>       upper = A;
        for (std::size_t i = 0UZ; i < 5UZ; ++i) {
            for (std::size_t j = 0UZ; j < i; ++j) {
                upper[i, j] = element<T>(100, -50);
            }
        }
        const auto full      = eigh(A);
        const auto fromUpper = eigh(upper);
        expect(fatal(full.has_value() && fromUpper.has_value())) << typeName<T>();
        expect(std::ranges::equal(full->values, fromUpper->values)) << typeName<T>();
    } | ElementTypes{};

    "too few sweeps are an error; invalid input is refused"_test = []<typename T> {
        const Tensor<T> A          = chosenHermitian<T>();
        const auto      unfinished = eigh(A, eig::Config<Real<T>>{.maxIterations = 1UZ});
        expect(fatal(!unfinished.has_value())) << typeName<T>();
        expect(unfinished.error().message.contains("no convergence within 1 sweeps")) << typeName<T>() << unfinished.error().message;

        const Tensor<T> rectangular({2UZ, 3UZ});
        const auto      notSquare = eigh(rectangular);
        expect(fatal(!notSquare.has_value())) << typeName<T>();
        expect(notSquare.error().message.contains("not a finite square matrix")) << typeName<T>() << notSquare.error().message;
        Tensor<T> withInf = A;
        withInf[0, 1]     = element<T>(std::numeric_limits<double>::infinity());
        expect(!eigh(withInf).has_value()) << typeName<T>();
    } | ElementTypes{};
};

const boost::ut::suite<"subspace"> _subspace = [] {
    using namespace boost::ut;

    "two plane waves: the noise subspace is orthogonal to both steering vectors"_test = []<typename T> {
        // six elements; a1 advances the phase by 90 degrees per element and a2 by 180 degrees.
        // A real element type keeps the real parts, (1, 0, -1, 0, 1, 0) and (1, -1, 1, -1, 1, -1).
        const Tensor<T> a1 = matrix<T>(6, 1, {{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 0}, {0, 1}});
        const Tensor<T> a2 = matrix<T>(6, 1, {{1, 0}, {-1, 0}, {1, 0}, {-1, 0}, {1, 0}, {-1, 0}});

        // R = a1 a1^H + 2 a2 a2^H + noisePower I
        const Real<T> noisePower = static_cast<Real<T>>(0.01);
        Tensor<T>     R          = multiply(a1, adjoint(a1));
        const auto    R2         = multiply(a2, adjoint(a2));
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            for (std::size_t j = 0UZ; j < 6UZ; ++j) {
                R[i, j] += T{2} * R2[i, j];
            }
            R[i, i] += T{noisePower};
        }

        const auto pair = eigh(R);
        expect(fatal(pair.has_value())) << typeName<T>();
        const auto& [values, V] = *pair;
        for (std::size_t k = 0UZ; k < 4UZ; ++k) {
            expect(le(std::abs(values[k] - noisePower), tolerance<T>(6) * Real<T>{12})) << typeName<T>() << "noise eigenvalue" << k;
        }
        expect(gt(values[4], Real<T>{1})) << typeName<T>();

        const auto byRank = subspace(values, V, eig::Rank{2UZ});
        expect(fatal(byRank.has_value())) << typeName<T>();
        const auto& [signal, noise] = *byRank;
        expect(eq(signal.extent(1), 2UZ) && eq(noise.extent(1), 4UZ) && eq(noise.extent(0), 6UZ)) << typeName<T>();
        const Real<T> bound = tolerance<T>(6) * Real<T>{12} / (values[4] - values[3]); // the error of R over the eigenvalue gap bounds the subspace error
        expect(le(projectionNorm(noise, a1), bound)) << typeName<T>() << "En^H a1 = 0";
        expect(le(projectionNorm(noise, a2), bound)) << typeName<T>() << "En^H a2 = 0";
        expect(le(std::abs(projectionNorm(signal, a1) - Real<T>{1}), bound)) << typeName<T>() << "a1 lies in the signal subspace";
        expect(le(std::abs(projectionNorm(signal, a2) - Real<T>{1}), bound)) << typeName<T>() << "a2 lies in the signal subspace";

        const auto byThreshold = subspace(values, V, eig::Threshold{Real<T>{10} * noisePower});
        expect(fatal(byThreshold.has_value())) << typeName<T>();
        expect(std::ranges::equal(byThreshold->signal, signal) && std::ranges::equal(byThreshold->noise, noise)) << typeName<T>() << "a threshold above the noise floor splits as rank 2";
    } | ElementTypes{};

    "an out-of-range rank is refused"_test = []<typename T> {
        const auto pair = eigh(identity<T>(3));
        expect(fatal(pair.has_value())) << typeName<T>();
        const auto& [values, V] = *pair;
        const auto tooLarge     = subspace(values, V, eig::Rank{4UZ});
        expect(fatal(!tooLarge.has_value())) << typeName<T>();
        expect(tooLarge.error().message.contains("a rank of 4 exceeds the 3 eigenvalues")) << typeName<T>() << tooLarge.error().message;
        const auto notFinite = subspace(values, V, eig::Threshold{std::numeric_limits<Real<T>>::quiet_NaN()});
        expect(fatal(!notFinite.has_value())) << typeName<T>();
        expect(notFinite.error().message.contains("threshold is not finite")) << typeName<T>() << notFinite.error().message;
        const auto allSignal = subspace(values, V, eig::Rank{3UZ});
        expect(allSignal.has_value() && eq(allSignal->noise.extent(1), 0UZ)) << typeName<T>();
        const auto allNoise = subspace(values, V, eig::Threshold{Real<T>{2}});
        expect(allNoise.has_value() && eq(allNoise->signal.extent(1), 0UZ)) << typeName<T>();
    } | ElementTypes{};

    "an eigenpair with a non-finite entry is refused"_test = []<typename T> {
        const auto result = eigh(chosenHermitian<T>());
        expect(fatal(result.has_value())) << typeName<T>();
        const auto& [values, V] = *result;
        expect(subspace(values, V, eig::Rank{2UZ}).has_value()) << typeName<T>() << "a finite eigenpair";

        const eig::Rank               rank{3UZ};
        const eig::Threshold<Real<T>> threshold{std::numeric_limits<Real<T>>::lowest()};
        for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            Tensor<T> badVectors      = V;
            badVectors[1, 2]          = element<T>(bad);
            Tensor<Real<T>> badValues = values;
            badValues[4]              = static_cast<Real<T>>(bad); // the order test alone admits NaN and +inf as the last value
            const auto refused        = [bad](const auto& split, const char* what) {
                expect(fatal(!split.has_value())) << typeName<T>() << bad << what;
                expect(split.error().message.contains("not a finite ascending eigenvalue and eigenvector pair")) << typeName<T>() << bad << what << split.error().message;
            };
            refused(subspace(values, badVectors, rank), "in vectors, by rank");
            refused(subspace(values, badVectors, threshold), "in vectors, by threshold");
            refused(subspace(badValues, V, rank), "in values, by rank");
            refused(subspace(badValues, V, threshold), "in values, by threshold");
        }
    } | ElementTypes{};
};

int main() { /* tests are automatically registered and executed */ return 0; }
