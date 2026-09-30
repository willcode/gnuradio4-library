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
 * @brief The normalized fourth moment of a stream and of a run of spectra, with its bias and spread in closed form.
 *
 * Kurtosis separates noise from structure. Spectral kurtosis is the same statistic per frequency bin across a run
 * of spectra. It finds a pulsed or bursty interferer without knowing what the interferer is. Both statistics are
 * in this file because they share one piece of arithmetic. A second convention for it could disagree without
 * any error.
 *
 * **The normalization.** For a stream `x` with `m2 = mean|x|^2` and `m4 = mean|x|^4`,
 *
 * ```
 * K = m4 / m2^2
 * ```
 *
 * `K` is dimensionless and independent of scale. Replacing `x` by `a*x` multiplies both terms by `|a|^4`. The
 * `m2m4` signal-to-noise estimator uses the same quantity under the names `ka` and `kn`. The **excess** is `K`
 * minus the value for a Gaussian of the same domain. A Gaussian then reads zero. That value is 3 for real input
 * and 2 for complex input. Both values are derived below.
 *
 * - Real. For `x ~ N(0, s^2)`, `E[x^2] = s^2` and `E[x^4] = 3 s^4`. Then `K = 3`.
 * - Complex, circular. Take `z = u + jv` with `u, v ~ N(0, s^2/2)` independent, so `E|z|^2 = s^2`. This is
 *   the noise convention used here. The quantity `|z|^2` is then **exponential** with mean `s^2`. The second
 *   moment of an exponential is twice its mean squared. Then `E|z|^4 = 2 s^4` and `K = 2`. The factor of two in
 *   place of three is the entire difference. It arises because `|z|^2` is a sum of two squared normals and not one.
 * - Constant modulus. `|z| = A` gives `K = 1` and an excess of `-1`.
 * - Two equal complex tones. `|z|^2 = 2 + 2*cos(D)` with `D` uniform over a long window. Then `E|z|^2 = 2` and
 *   `E|z|^4 = 6`. `K = 3/2`, and the excess is exactly `-0.5`.
 * - A real sinusoid. `m4 = 3A^4/8` and `m2 = A^2/2`. Then `K = 3/2`, and the excess is exactly `-1.5`.
 *
 * The structured values are negative, and the impulsive values are positive. Bounded structure drives the
 * normalized fourth moment *down*. Impulsive structure drives it *up*. The sign of the excess is the
 * discriminant the statistic provides.
 *
 * **The estimator's bias and spread, in closed form.** The plug-in estimator `m4_hat/m2_hat^2` is biased low.
 * Apply the delta method to `u = |z|^2`, exponential with mean `mu`. Then `E[u^2]=2mu^2`, `E[u^3]=6mu^3` and
 * `E[u^4]=24mu^4`. Write `A = mean(u^2)` and `B = mean(u)`.
 *
 * ```
 * Var(A) = 20 mu^4 / N     Var(B) = mu^2 / N     Cov(A,B) = 4 mu^3 / N
 * E[A/B^2] ~ 2 - 8/N + 6/N = 2 - 2/N
 * Var(A/B^2) ~ 20/N + 16/N - 32/N = 4/N
 * ```
 *
 * The complex excess is therefore biased by `-2/N`, and its standard deviation is `2/sqrt(N)`. The same
 * construction on the real Gaussian gives a bias of `-6/N` and the classical variance `24/N`. That is a standard
 * deviation of `4.899/sqrt(N)`. A measured excess falls in a band set by these two numbers, not at one value.
 *
 * There is no `unbiased` mode. The bias is `O(1/N)` and known in closed form for both domains. A caller that
 * needs an unbiased value corrects it with `kurtosisBias`. A setting that switched between two estimators would
 * give two estimators one name. The complex case also has no standard unbiased form.
 */
namespace gr::measurement {

/// The four sample types the moments are defined over, two real and two complex.
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

/// @brief The value of `m4/m2^2` for a Gaussian of the given domain, 3 real and 2 complex, derived in the file note.
[[nodiscard]] constexpr double gaussianFourthMoment(bool complexDomain) noexcept { return complexDomain ? 2. : 3.; }

/// @brief The plug-in estimator's bias on the excess over `n` samples, `-2/n` complex and `-6/n` real.
[[nodiscard]] constexpr double kurtosisBias(std::size_t n, bool complexDomain) noexcept { return n == 0UZ ? 0. : -(complexDomain ? 2. : 6.) / static_cast<double>(n); }

/// @brief The estimator's variance over `n` samples, `4/n` complex and `24/n` real.
[[nodiscard]] constexpr double kurtosisVariance(std::size_t n, bool complexDomain) noexcept { return n == 0UZ ? 0. : (complexDomain ? 4. : 24.) / static_cast<double>(n); }

/// @brief The standard deviation of one window's reading, `2/sqrt(n)` complex and `4.899/sqrt(n)` real.
[[nodiscard]] inline double kurtosisSpread(std::size_t n, bool complexDomain) noexcept { return std::sqrt(kurtosisVariance(n, complexDomain)); }

/**
 * @brief `n`, `sum|x|^2` and `sum|x|^4` in `double`, and the three figures they make.
 *
 * A plain value type. It holds the arithmetic and no stream handling.
 *
 * The sums are plain sums in `double`, and their error bound is derived. The relative error of a naive sum of
 * `N` non-negative terms is at most `N * eps`, with `eps = 2^-52 = 2.22e-16`. At a window of `2^24` samples that
 * is `3.7e-9`. The spread of the estimate at that window is `2/sqrt(N) = 2.4e-4`, nine orders of magnitude
 * larger. Kahan summation would improve the adder and not the measurement.
 *
 * A window whose `sum|x|^2` is exactly zero has no defined ratio. The type reports `degenerate()` for it, and
 * no NaN or estimate. The test is exact equality with zero and not an epsilon. No epsilon could be derived, and
 * a chosen one would be an arbitrary number.
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

    /// @brief `m4/m2^2`, or `0` on a degenerate window in place of a NaN.
    [[nodiscard]] constexpr double normalized() const noexcept { return degenerate() ? 0. : static_cast<double>(_n) * _m4 / (_m2 * _m2); }

    /// @brief `normalized()` less the Gaussian reference for this domain. A Gaussian reads zero.
    [[nodiscard]] constexpr double excess() const noexcept { return degenerate() ? 0. : normalized() - gaussianFourthMoment(kComplexDomain); }

    /// @brief This window's own bias and spread, from the closed forms.
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
 * @brief Spectral kurtosis over `M` independent estimates of one bin.
 *
 * With `u_k` the bin's power in estimate `k`, `S1 = sum u_k` and `S2 = sum u_k^2`,
 *
 * ```
 * SK = ((M*d + 1) / (M - 1)) * ( M * S2 / S1^2 - 1 )
 * ```
 *
 * with `d` the shape parameter. `d = 1` for a single periodogram per estimate. `d = n` when each estimate is
 * already the mean of `n` periodograms. The normalization is derived here. For `u_k` i.i.d. Gamma of shape `d`,
 * the normalized vector `u_k/S1` is Dirichlet with all parameters `d`. It is **independent of `S1`**. With
 * `Q = S2/S1^2`, the Dirichlet moment `E[w^2] = d(d+1)/(Md(Md+1))` gives
 *
 * ```
 * E[Q] = (d+1)/(Md+1)          E[M*Q - 1] = (M - 1)/(Md + 1)
 * ```
 *
 * The leading factor is exactly the reciprocal of that. **`E[SK] = 1` exactly**, at every `M` and every `d`, and
 * not only asymptotically. The expected value on noise is therefore exact and not approximate.
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
 * It tends to `2(d+1)/(dM)`. At `d = 1` it is `4M^2/((M-1)(M+2)(M+3))`, which approaches `4/M`. With the variance
 * beside a spectral-kurtosis record, a threshold can be set in standard deviations without a new derivation.
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
 * @brief Spectral kurtosis of a carrier in noise at bin signal-to-noise ratio @p rho, `(2*rho + 1)/(rho + 1)^2`.
 *
 * `u = |A + n|^2` has `E[u] = S + N0` and `E[u^2] = S^2 + 4 S N0 + 2 N0^2`. The large-`M` limit of `SK` is
 * `E[u^2]/E[u]^2 - 1`, which reduces to the expression above. It is 1 at `rho = 0`, which is noise alone. It
 * falls through 0.75, 0.556, 0.174 and 0.0197 at `rho = 1, 2, 10, 100`. It reaches 0 for a noiseless tone. An
 * intermittent interferer gives values above 1, on the other side of the discriminant.
 */
[[nodiscard]] constexpr double spectralKurtosisCw(double rho) noexcept { return (2. * rho + 1.) / ((rho + 1.) * (rho + 1.)); }

/// What one spectral estimate did to the accumulation.
enum class SpectrumResponse : std::uint8_t {
    accepted,      ///< folded in
    wrongBinCount, ///< a different bin count from the accumulation in progress
    negativeBin    ///< a negative power, and nothing was folded in
};

/**
 * @brief `S1` and `S2` per bin over a run of spectra, and the per-bin statistic they make.
 *
 * The state is two `double` vectors and a count. Each input record costs one add and one multiply-add per bin.
 * The final evaluation costs one evaluation per bin. That is the entire cost of finding an intermittent
 * interferer without knowing what it is.
 *
 * **The scale cancels.** `SK` is homogeneous of degree zero in `u`. Replacing `u_k` by `a*u_k` leaves
 * `M*S2/S1^2` unchanged. A linear power-density spectrum can therefore be used directly, with no calibration.
 * The window's equivalent noise bandwidth, the full-scale reference and the `1/(fs * sum(w^2))` factor form one
 * positive scalar per bin. Each of them divides out. A spectrum in decibels does not cancel this way. The
 * conversion to decibels would have to be inverted first.
 *
 * **Independence is a precondition, and this class cannot check it.** The derivation requires the `M` spectra
 * to be independent. Overlapping Welch segments are not independent. At 50 % overlap consecutive periodograms
 * share half their samples. The variance of `S1` then falls, and `E[SK]` is biased. The arithmetic cannot see a
 * producer's overlap. The caller must refuse overlapped spectra.
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

    /// @brief Fold one spectral estimate in. A record that cannot be used leaves the accumulation unchanged.
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
     * @brief The per-bin statistic at shape @p shape, and the count of bins with no signal to divide by.
     *
     * A bin whose `S1` is exactly zero has no defined `SK`. It is written as zero and counted. One dead bin is a
     * property of the signal and not of the record. The function therefore returns a count and sets no flag on
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
