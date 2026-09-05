#ifndef GNURADIO_ALGORITHM_KURTOSIS_HPP
#define GNURADIO_ALGORITHM_KURTOSIS_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

/**
 * @brief The normalized fourth moment, over a stream and over a run of spectra, with its bias and its spread in closed form.
 *
 * Kurtosis is the statistic that separates noise from structure, and spectral kurtosis is the same statistic per
 * frequency bin across a run of spectra, which is how a pulsed or bursty interferer is found without knowing
 * what it is. Both are here because they are one piece of arithmetic and a second convention for it would be a
 * silent disagreement.
 *
 * **The normalization, stated once.** For a stream `x` with `m2 = mean|x|^2` and `m4 = mean|x|^4`,
 *
 * ```
 * K = m4 / m2^2
 * ```
 *
 * `K` is dimensionless and scale free: replacing `x` by `a*x` multiplies both terms by `|a|^4`. This is the
 * quantity the tree's `m2m4` signal-to-noise estimator already calls `ka` and `kn`, and nothing here changes it.
 * The **excess** is `K` minus the value a Gaussian of the same domain takes, so that a Gaussian reads zero: 3 for
 * real input and 2 for complex, and both are derived rather than quoted.
 *
 * - Real. For `x ~ N(0, s^2)`, `E[x^2] = s^2` and `E[x^4] = 3 s^4`, so `K = 3`.
 * - Complex, circular. For `z = u + jv` with `u, v ~ N(0, s^2/2)` independent — the tree's noise convention,
 *   `E|z|^2 = s^2` — the quantity `|z|^2` is **exponential** with mean `s^2`. An exponential's second moment is
 *   twice its mean squared, so `E|z|^4 = 2 s^4` and `K = 2`. The factor of two, not three, is the whole of the
 *   difference, and it comes from `|z|^2` being a sum of two squared normals rather than one.
 * - Constant modulus. `|z| = A` gives `K = 1` and an excess of `-1`.
 * - Two equal complex tones. `|z|^2 = 2 + 2*cos(D)` with `D` uniform over a long window, so `E|z|^2 = 2` and
 *   `E|z|^4 = 6`: `K = 3/2` and the excess is exactly `-0.5`.
 * - A real sinusoid. `m4 = 3A^4/8` and `m2 = A^2/2`, so `K = 3/2` and the excess is exactly `-1.5`.
 *
 * Both structured values are negative and the impulsive ones are positive: bounded structure drives the
 * normalized fourth moment *down*, impulsive structure drives it *up*, and that sign is the discriminant the
 * statistic exists for.
 *
 * **The estimator's bias and spread, in closed form.** The plug-in estimator `m4_hat/m2_hat^2` is biased low.
 * By the delta method on `u = |z|^2` exponential of mean `mu`, with `E[u^2]=2mu^2`, `E[u^3]=6mu^3` and
 * `E[u^4]=24mu^4`, writing `A = mean(u^2)` and `B = mean(u)`:
 *
 * ```
 * Var(A) = 20 mu^4 / N     Var(B) = mu^2 / N     Cov(A,B) = 4 mu^3 / N
 * E[A/B^2] ~ 2 - 8/N + 6/N = 2 - 2/N
 * Var(A/B^2) ~ 20/N + 16/N - 32/N = 4/N
 * ```
 *
 * so the complex excess is biased by `-2/N` and its standard deviation is `2/sqrt(N)`. The same construction on
 * the real Gaussian gives `-6/N` and the classical `24/N`, a standard deviation of `4.899/sqrt(N)`. These two
 * numbers are the envelope every kurtosis criterion is written against, and they are why a QA case asserts a
 * band rather than a value.
 *
 * There is no `unbiased` mode. The bias is `O(1/N)` and known in closed form for both domains, so a caller who
 * needs it corrects from `kurtosisBias`; a setting that silently switched between two estimators would make two
 * estimators with one name, and there is no standard unbiased form for the complex case to switch to.
 */
namespace gr::measurement {

/// The four sample types the moments are defined over: the two real domains and the two complex ones.
template<typename T>
concept MomentSample = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>;

namespace detail {

template<typename T>
struct MomentDomain {
    static constexpr bool isComplex = false;
};

template<std::floating_point F>
struct MomentDomain<std::complex<F>> {
    static constexpr bool isComplex = true;
};

} // namespace detail

/// @brief What a Gaussian of the given domain reads on `m4/m2^2`: 3 real, 2 complex, both derived in the header note.
[[nodiscard]] constexpr double gaussianFourthMoment(bool complexDomain) noexcept { return complexDomain ? 2. : 3.; }

/// @brief The plug-in estimator's bias on the excess over `n` samples: `-2/n` complex, `-6/n` real.
[[nodiscard]] constexpr double kurtosisBias(std::size_t n, bool complexDomain) noexcept { return n == 0UZ ? 0. : -(complexDomain ? 2. : 6.) / static_cast<double>(n); }

/// @brief The estimator's variance over `n` samples: `4/n` complex, `24/n` real.
[[nodiscard]] constexpr double kurtosisVariance(std::size_t n, bool complexDomain) noexcept { return n == 0UZ ? 0. : (complexDomain ? 4. : 24.) / static_cast<double>(n); }

/// @brief The standard deviation of one window's reading: `2/sqrt(n)` complex, `4.899/sqrt(n)` real.
[[nodiscard]] inline double kurtosisSpread(std::size_t n, bool complexDomain) noexcept { return std::sqrt(kurtosisVariance(n, complexDomain)); }

/**
 * @brief `n`, `sum|x|^2` and `sum|x|^4` in `double`, and the three figures they make.
 *
 * A value type and nothing more: the arithmetic has exactly one home and a block over it is thin.
 *
 * A plain sum in `double`, with the bound derived rather than assumed. The relative error of a naive sum of `N`
 * non-negative terms is at most `N * eps` with `eps = 2^-52 = 2.22e-16`, so at a window of `2^24` samples it is
 * `3.7e-9` — nine orders below the `2/sqrt(N) = 2.4e-4` spread of the estimate itself at that window. Kahan
 * summation would be measuring the adder rather than the signal.
 *
 * A window whose `sum|x|^2` is exactly zero has no defined ratio, and this reports `degenerate()` rather than a
 * NaN or a guess. The test is exact equality with zero and not an epsilon, because there is no defensible
 * epsilon and a chosen one would be a number without a derivation.
 */
template<MomentSample T>
class KurtosisAccumulator {
public:
    static constexpr bool kComplexDomain = detail::MomentDomain<T>::isComplex;

    constexpr void reset() noexcept {
        _n  = 0UZ;
        _m2 = 0.;
        _m4 = 0.;
    }

    constexpr void add(T x) noexcept {
        const double p = power(x);
        ++_n;
        _m2 += p;
        _m4 += p * p;
    }

    constexpr void add(std::span<const T> in) noexcept {
        double sum2 = _m2;
        double sum4 = _m4;
        for (const T x : in) {
            const double p = power(x);
            sum2 += p;
            sum4 += p * p;
        }
        _n += in.size();
        _m2 = sum2;
        _m4 = sum4;
    }

    [[nodiscard]] constexpr std::size_t count() const noexcept { return _n; }
    [[nodiscard]] constexpr double      sumPower() const noexcept { return _m2; }
    [[nodiscard]] constexpr double      sumPowerSquared() const noexcept { return _m4; }
    [[nodiscard]] constexpr bool        degenerate() const noexcept { return _n == 0UZ || _m2 == 0.; }
    [[nodiscard]] constexpr double      meanPower() const noexcept { return _n == 0UZ ? 0. : _m2 / static_cast<double>(_n); }

    /// @brief `m4/m2^2`, and `0` on a degenerate window rather than a NaN a consumer would have to test for.
    [[nodiscard]] constexpr double normalized() const noexcept { return degenerate() ? 0. : static_cast<double>(_n) * _m4 / (_m2 * _m2); }

    /// @brief `normalized()` less the Gaussian reference for this domain, so that a Gaussian reads zero.
    [[nodiscard]] constexpr double excess() const noexcept { return degenerate() ? 0. : normalized() - gaussianFourthMoment(kComplexDomain); }

    /// @brief This window's own bias and spread, from the closed forms, so a consumer never re-derives them.
    [[nodiscard]] constexpr double bias() const noexcept { return kurtosisBias(_n, kComplexDomain); }
    [[nodiscard]] double           spread() const noexcept { return kurtosisSpread(_n, kComplexDomain); }

private:
    [[nodiscard]] static constexpr double power(T x) noexcept {
        if constexpr (kComplexDomain) {
            const double re = static_cast<double>(x.real());
            const double im = static_cast<double>(x.imag());
            return re * re + im * im;
        } else {
            const double v = static_cast<double>(x);
            return v * v;
        }
    }

    std::size_t _n{0UZ};
    double      _m2{0.};
    double      _m4{0.};
};

/**
 * @brief Spectral kurtosis over `M` independent estimates of one bin, and the two figures that make it assertable.
 *
 * With `u_k` the bin's power in estimate `k`, `S1 = sum u_k` and `S2 = sum u_k^2`,
 *
 * ```
 * SK = ((M*d + 1) / (M - 1)) * ( M * S2 / S1^2 - 1 )
 * ```
 *
 * with `d` the shape parameter: `d = 1` for a single periodogram per estimate, `d = n` when each estimate is
 * already the mean of `n` periodograms. The normalization is derived, not taken. For `u_k` i.i.d. Gamma of shape
 * `d`, the normalized vector `u_k/S1` is Dirichlet with all parameters `d` and **independent of `S1`**, so with
 * `Q = S2/S1^2` the Dirichlet moment `E[w^2] = d(d+1)/(Md(Md+1))` gives
 *
 * ```
 * E[Q] = (d+1)/(Md+1)          E[M*Q - 1] = (M - 1)/(Md + 1)
 * ```
 *
 * and the leading factor is exactly the reciprocal of that: **`E[SK] = 1` exactly**, at every `M` and every `d`,
 * not asymptotically. That exactness is what makes the criterion assertable rather than approximate.
 */
[[nodiscard]] inline double spectralKurtosis(double s1, double s2, std::size_t m, double shape) noexcept {
    if (!(s1 > 0.) || m < 2UZ) {
        return 0.;
    }
    const double count = static_cast<double>(m);
    return ((count * shape + 1.) / (count - 1.)) * (count * s2 / (s1 * s1) - 1.);
}

/// @brief What spectral kurtosis reads on noise, exactly and at every `M` and `d`.
[[nodiscard]] constexpr double spectralKurtosisExpectation() noexcept { return 1.; }

/**
 * @brief `Var(SK) = 2 M^2 d (d+1) / ((M-1)(Md+2)(Md+3))`, from the same Dirichlet moments.
 *
 * It tends to `2(d+1)/(dM)`, and at `d = 1` it is `4M^2/((M-1)(M+2)(M+3))`, approaching `4/M`. Carrying it
 * beside a spectral-kurtosis record is what lets a downstream threshold be set in standard deviations without
 * the consumer re-deriving it.
 */
[[nodiscard]] inline double spectralKurtosisVariance(std::size_t m, double shape) noexcept {
    if (m < 2UZ) {
        return 0.;
    }
    const double count = static_cast<double>(m);
    const double md    = count * shape;
    return 2. * count * count * shape * (shape + 1.) / ((count - 1.) * (md + 2.) * (md + 3.));
}

/// @brief The standard deviation of one spectral-kurtosis reading.
[[nodiscard]] inline double spectralKurtosisSpread(std::size_t m, double shape) noexcept { return std::sqrt(spectralKurtosisVariance(m, shape)); }

/**
 * @brief What spectral kurtosis reads on a carrier in noise at bin signal-to-noise ratio @p rho: `(2*rho + 1)/(rho + 1)^2`.
 *
 * `u = |A + n|^2` has `E[u] = S + N0` and `E[u^2] = S^2 + 4 S N0 + 2 N0^2`, so the large-`M` limit of `SK` is
 * `E[u^2]/E[u]^2 - 1`, which reduces to the expression above. It runs from 1 at `rho = 0` — noise — down through
 * 0.75, 0.556, 0.174 and 0.0197 at `rho = 1, 2, 10, 100` and to 0 for a noiseless tone, which is the other side
 * of the discriminant from the values above 1 an intermittent interferer produces.
 */
[[nodiscard]] constexpr double spectralKurtosisCw(double rho) noexcept { return (2. * rho + 1.) / ((rho + 1.) * (rho + 1.)); }

/// What one spectral estimate did to the accumulation.
enum class SpectrumResponse : std::uint8_t {
    accepted,      ///< folded in
    wrongBinCount, ///< a different bin count from the accumulation in progress
    negativeBin    ///< a power that is not a power; nothing was folded in
};

/**
 * @brief `S1` and `S2` per bin over a run of spectra, and the per-bin statistic they make.
 *
 * Two `double` vectors and a count. Per input record one add and one multiply-add per bin, and at the end one
 * evaluation per bin — the whole cost of finding an intermittent interferer without knowing what it is.
 *
 * **The scale cancels.** `SK` is homogeneous of degree zero in `u`: replacing `u_k` by `a*u_k` leaves
 * `M*S2/S1^2` unchanged. So a linear power-density spectrum can be consumed directly and no calibration
 * argument is needed at all — the window's equivalent noise bandwidth, the full-scale reference and the
 * `1/(fs * sum(w^2))` factor are one positive scalar per bin and every one of them divides out. Against a
 * spectrum already converted to decibels the cancellation would not exist and the conversion would have to be
 * inverted first.
 *
 * **Independence is a precondition and this cannot check it.** The derivation requires the `M` spectra to be
 * independent, which overlapping Welch segments are not: at 50 % overlap consecutive periodograms share half
 * their samples, `S1`'s variance falls and `E[SK]` is biased. A producer's overlap is metadata a block reads,
 * not something the arithmetic can see, so it is stated here and refused there.
 */
class SpectralKurtosisAccumulator {
public:
    SpectralKurtosisAccumulator() = default;
    explicit SpectralKurtosisAccumulator(std::size_t nBins) { resize(nBins); }

    /// @brief Size the accumulation for @p nBins bins and clear it.
    void resize(std::size_t nBins) {
        _s1.assign(nBins, 0.);
        _s2.assign(nBins, 0.);
        _m = 0UZ;
    }

    void reset() noexcept {
        std::ranges::fill(_s1, 0.);
        std::ranges::fill(_s2, 0.);
        _m = 0UZ;
    }

    /// @brief Fold one spectral estimate in, leaving the accumulation untouched where the record cannot be used.
    SpectrumResponse accumulate(std::span<const float> spectrum) noexcept {
        if (spectrum.size() != _s1.size()) {
            ++_refused;
            return SpectrumResponse::wrongBinCount;
        }
        for (const float value : spectrum) {
            if (!(value >= 0.f)) {
                ++_refused;
                return SpectrumResponse::negativeBin;
            }
        }
        for (std::size_t bin = 0UZ; bin < spectrum.size(); ++bin) {
            const double u = static_cast<double>(spectrum[bin]);
            _s1[bin] += u;
            _s2[bin] += u * u;
        }
        ++_m;
        return SpectrumResponse::accepted;
    }

    /**
     * @brief The per-bin statistic at shape @p shape, and how many bins had no signal at all to divide by.
     *
     * A bin whose `S1` is exactly zero has no defined `SK` and is written as zero and counted. One dead bin is a
     * property of the signal rather than of the record, so the count is what a consumer reads and not a flag on
     * the whole spectrum.
     */
    [[nodiscard]] std::size_t evaluate(double shape, std::span<double> out) const {
        if (out.size() != _s1.size()) {
            throw std::invalid_argument("gr::measurement::SpectralKurtosisAccumulator::evaluate: the output span must have one entry per bin");
        }
        std::size_t degenerate = 0UZ;
        for (std::size_t bin = 0UZ; bin < out.size(); ++bin) {
            if (!(_s1[bin] > 0.)) {
                out[bin] = 0.;
                ++degenerate;
                continue;
            }
            out[bin] = spectralKurtosis(_s1[bin], _s2[bin], _m, shape);
        }
        return degenerate;
    }

    [[nodiscard]] std::size_t             bins() const noexcept { return _s1.size(); }
    [[nodiscard]] std::size_t             count() const noexcept { return _m; }
    [[nodiscard]] std::uint64_t           nRefused() const noexcept { return _refused; }
    [[nodiscard]] std::span<const double> sums() const noexcept { return _s1; }
    [[nodiscard]] std::span<const double> sumSquares() const noexcept { return _s2; }

private:
    std::vector<double> _s1;
    std::vector<double> _s2;
    std::size_t         _m{0UZ};
    std::uint64_t       _refused{0ULL};
};

} // namespace gr::measurement

#endif // GNURADIO_ALGORITHM_KURTOSIS_HPP
