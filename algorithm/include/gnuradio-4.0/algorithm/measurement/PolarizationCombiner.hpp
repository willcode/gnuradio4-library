#ifndef GNURADIO_ALGORITHM_POLARIZATION_COMBINER_HPP
#define GNURADIO_ALGORITHM_POLARIZATION_COMBINER_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

#include <gnuradio-4.0/algorithm/math/Decibel.hpp>

/**
 * @brief Combines two orthogonal antenna branches at the maximal-ratio optimum, from the 2x2 covariance in closed form.
 *
 * Two branches carry one signal through two channels, `r_i(k) = h_i s(k) + n_i(k)`. The branch noises are
 * independent. A linear combiner forms `y = conj(w_0) r_0 + conj(w_1) r_1`, and
 *
 * ```
 * SNR_out = P * |sum_i conj(w_i) h_i|^2 / (sum_i |w_i|^2 N_i)
 * ```
 *
 * By Cauchy-Schwarz this is maximized at `w_i` proportional to `h_i / N_i`, where
 *
 * ```
 * SNR_out = P * sum_i |h_i|^2 / N_i = SNR_0 + SNR_1
 * ```
 *
 * **Output signal-to-noise ratios add.** For two branches of equal quality the gain is
 * `10*log10(2) = 3.0103 dB` over either branch and over selection of the stronger one. Against selection with
 * unequal branches the gain is `10*log10(1 + SNR_min/SNR_max)`. That is `3.0103 / 1.7609 / 0.9691 / 0.4139 dB`
 * at ratios `1 / 0.5 / 0.25 / 0.1`.
 *
 * **The estimator.** The combiner does not have `h`. It measures the branch covariance.
 *
 * ```
 * R = E[ r r^H ] = P h h^H + diag(N_0, N_1) = [[a, c], [conj(c), b]]
 * ```
 *
 * Under equal branch noise this is `P h h^H + N I`. Its eigenvectors are `h` and its orthogonal complement. The
 * principal eigenvector *is* therefore the maximal-ratio weight vector. The 2x2 case has a closed form with no
 * iteration.
 *
 * ```
 * lambda_+ = (a+b)/2 + sqrt( ((a-b)/2)^2 + |c|^2 )
 * lambda_- = (a+b)/2 - sqrt( ((a-b)/2)^2 + |c|^2 )
 * v        = [ c , lambda_+ - a ]
 * ```
 *
 * `v` is exact and not approximate. `(lambda_+ - a)(lambda_+ - b) = |c|^2` is the characteristic equation
 * itself. The second row of `(R - lambda_+ I) v = 0` requires exactly that. Every reported value follows from
 * `v`. Three of the consequences are derived here.
 *
 * - **The phase and the amplitude ratio are free of noise bias.** The branch noises are independent, and
 *   `c = E[r_0 conj(r_1)] = P h_0 conj(h_1)` exactly. `arg(c) = arg(h_0) - arg(h_1)` then carries no noise term.
 *   The naive amplitude ratio `sqrt(a/b) = sqrt((P|h_0|^2+N)/(P|h_1|^2+N))` is biased toward 1 by the noise.
 *   `|c|/(lambda_+ - a)` is not biased, because `lambda_+ - a = P|h_1|^2` and `|c| = P|h_0||h_1|`.
 * - **The reported ratios are self-consistent.** `a - lambda_- = P|h_0|^2` and `b - lambda_- = P|h_1|^2`. The
 *   two branch figures therefore sum to `(a + b - 2 lambda_-)/lambda_- = (lambda_+ - lambda_-)/lambda_-`, the
 *   combined figure. `SNR_out = SNR_0 + SNR_1` holds as an identity on the estimates, for any data, and not
 *   only on the true values. A sign error or an eigenvalue swap breaks the identity at once.
 * - **The orthogonal output nulls the signal.** For a unit `w`, `u = [-conj(w_1), conj(w_0)]` satisfies
 *   `w^H u = 0` identically. `u` carries no component of `h`. It is the channel of interference and noise alone.
 *
 * **Unequal branch noise reduces to the equal case** and needs no second estimator. Whitening by
 * `r_i/sqrt(N_i)` gives both noises unit power and turns `h_i` into `h_i/sqrt(N_i)`. The solution above then
 * returns the whitened weights. Unwhitened, they are `h_i/N_i`, the optimum of the first expression. The code
 * needs one line for this and no second solver.
 *
 * **The gauge is fixed.** An eigenvector is defined up to a complex scalar of unit modulus. A solver that
 * returned one with an arbitrary phase would put an arbitrary phase step on the combined output at *every*
 * window boundary. Those random discontinuities would break a carrier loop fed by the output. The purpose of
 * the combiner is to improve that signal. The gauge is therefore pinned. `v` is normalized and rotated to
 * `arg(v_0) = 0` exactly. This aligns the phase of the combined output with branch 0. With a slowly varying
 * channel the weights then move slowly. The output phase is then continuous up to the channel's own rate of
 * change.
 */
namespace gr::measurement {

/// The rule that turns the two branches into one.
enum class PolarizationMode : std::uint8_t {
    mrc,      ///< the maximal-ratio combination, where the output ratios add
    selection ///< the stronger branch alone, chosen from the same covariance
};

/// The scaling of the reported weight vector.
enum class PolarizationNormalization : std::uint8_t {
    unit_noise, ///< the output noise power equals one branch's, and the reported ratios compare directly to a branch's
    unit_signal ///< the output signal power is 1
};

[[nodiscard]] inline constexpr std::string_view polarizationModeName(PolarizationMode mode) noexcept { return mode == PolarizationMode::selection ? "selection" : "mrc"; }
[[nodiscard]] inline constexpr std::string_view polarizationNormalizationName(PolarizationNormalization norm) noexcept { return norm == PolarizationNormalization::unit_signal ? "unit_signal" : "unit_noise"; }

[[nodiscard]] inline constexpr std::optional<PolarizationMode> polarizationModeFrom(std::string_view name) noexcept {
    if (name == "mrc") {
        return PolarizationMode::mrc;
    }
    if (name == "selection") {
        return PolarizationMode::selection;
    }
    return std::nullopt;
}

[[nodiscard]] inline constexpr std::optional<PolarizationNormalization> polarizationNormalizationFrom(std::string_view name) noexcept {
    if (name == "unit_noise") {
        return PolarizationNormalization::unit_noise;
    }
    if (name == "unit_signal") {
        return PolarizationNormalization::unit_signal;
    }
    return std::nullopt;
}

/// The figure reported in place of an unbounded one, as a linear power ratio of 60 dB. Every use of it is flagged.
/// A finite number here is then not mistaken for a measurement.
inline constexpr double kPolarizationSaturation = 1e6;

/// @brief The result of the closed-form eigensolution, and the weights for a combiner to apply.
struct PolarizationEstimate {
    double lambdaPlus{0.};  ///< the principal eigenvalue of the whitened covariance
    double lambdaMinus{0.}; ///< the other eigenvalue, the noise power in whitened units

    std::complex<double> weight0{1., 0.}; ///< applied as `conj(weight0) * r_0 + conj(weight1) * r_1`
    std::complex<double> weight1{0., 0.};
    std::complex<double> ortho0{0., 0.}; ///< the orthogonal complement, with the signal nulled
    std::complex<double> ortho1{1., 0.};

    double relativePhase{0.};     ///< `arg(h_0) - arg(h_1)`, in radians, free of noise bias
    double amplitudeRatio{1.};    ///< `|h_0| / |h_1|`, free of noise bias
    double branchSnr0{0.};        ///< linear power ratio
    double branchSnr1{0.};        ///< linear power ratio
    double combinedSnr{0.};       ///< linear power ratio, identically the sum of the two above
    double scale{1.};             ///< the normalization scalar the weights already carry
    int    selectedBranch{-1};    ///< the branch `selection` chose, or -1 under `mrc`
    bool   saturatedSnr{false};   ///< `lambda_- == 0`, no noise in either branch and unbounded ratios
    bool   saturatedRatio{false}; ///< `lambda_+ - a == 0`, nothing correlated in branch 1

    [[nodiscard]] double branchSnr(std::size_t branch) const noexcept { return branch == 0UZ ? branchSnr0 : branchSnr1; }
    [[nodiscard]] double branchSnrDb(std::size_t branch) const noexcept { return toDb(branchSnr(branch)); }
    [[nodiscard]] double combinedSnrDb() const noexcept { return toDb(combinedSnr); }
    /// @brief The gain of the combination over the better of the two branches, in dB.
    [[nodiscard]] double combiningGainDb() const noexcept { return combinedSnrDb() - std::max(branchSnrDb(0UZ), branchSnrDb(1UZ)); }

    [[nodiscard]] static double toDb(double linear) noexcept { return gr::math::toDb(linear); }
};

/// @brief The estimate for use before any measurement. Branch @p branch passes through unchanged.
/// The other branch goes to the orthogonal port. The output is a valid signal. With the default branch 0,
/// later estimates keep the output phase continuous, because their gauge is branch 0. No phase is made up.
[[nodiscard]] inline PolarizationEstimate polarizationPassthrough(std::size_t branch = 0UZ) {
    PolarizationEstimate out;
    out.selectedBranch = static_cast<int>(branch);
    if (branch == 0UZ) {
        out.weight0 = {1., 0.};
        out.weight1 = {0., 0.};
        out.ortho0  = {0., 0.};
        out.ortho1  = {1., 0.};
    } else {
        out.weight0 = {0., 0.};
        out.weight1 = {1., 0.};
        out.ortho0  = {1., 0.};
        out.ortho1  = {0., 0.};
    }
    return out;
}

/**
 * @brief The three covariance entries over a window, and the closed-form solution on them.
 *
 * `a`, `b` and `c` are held in `double` and accumulated in stream order. A caller that passes one span and a
 * caller that passes a hundred get the same sums to the bit. Each sample costs four real multiply-adds and no
 * branch.
 */
class BranchCovariance {
public:
    void reset() noexcept {
        _n   = 0UZ;
        _a   = 0.;
        _b   = 0.;
        _cRe = 0.;
        _cIm = 0.;
    }

    void add(std::complex<float> r0, std::complex<float> r1) noexcept {
        const double x0 = static_cast<double>(r0.real());
        const double y0 = static_cast<double>(r0.imag());
        const double x1 = static_cast<double>(r1.real());
        const double y1 = static_cast<double>(r1.imag());
        _a += x0 * x0 + y0 * y0;
        _b += x1 * x1 + y1 * y1;
        _cRe += x0 * x1 + y0 * y1; // r_0 * conj(r_1)
        _cIm += y0 * x1 - x0 * y1;
        ++_n;
    }

    void add(std::span<const std::complex<float>> in0, std::span<const std::complex<float>> in1) {
        if (in0.size() != in1.size()) {
            throw std::invalid_argument(std::format("gr::measurement::BranchCovariance::add: {} samples on branch 0 against {} on branch 1", in0.size(), in1.size()));
        }
        for (std::size_t k = 0UZ; k < in0.size(); ++k) {
            add(in0[k], in1[k]);
        }
    }

    [[nodiscard]] std::size_t          count() const noexcept { return _n; }
    [[nodiscard]] double               a() const noexcept { return _n == 0UZ ? 0. : _a / static_cast<double>(_n); }
    [[nodiscard]] double               b() const noexcept { return _n == 0UZ ? 0. : _b / static_cast<double>(_n); }
    [[nodiscard]] std::complex<double> c() const noexcept { return _n == 0UZ ? std::complex<double>{} : std::complex<double>{_cRe / static_cast<double>(_n), _cIm / static_cast<double>(_n)}; }

    /**
     * @brief Solve for the weights, the channel estimates and the three signal-to-noise ratios.
     *
     * @param mode        maximal ratio, or the stronger branch alone
     * @param norm        what the weights are scaled to
     * @param noisePowers empty for equal branch noise, or exactly two positive entries to whiten by
     */
    [[nodiscard]] PolarizationEstimate solve(PolarizationMode mode = PolarizationMode::mrc, PolarizationNormalization norm = PolarizationNormalization::unit_noise, std::span<const double> noisePowers = {}) const {
        double n0 = 1.;
        double n1 = 1.;
        if (!noisePowers.empty()) {
            if (noisePowers.size() != 2UZ) {
                throw std::invalid_argument(std::format("gr::measurement::BranchCovariance::solve: {} noise powers — two branches take two", noisePowers.size()));
            }
            for (const double power : noisePowers) {
                if (!std::isfinite(power) || !(power > 0.)) {
                    throw std::invalid_argument(std::format("gr::measurement::BranchCovariance::solve: a noise power of {} is not positive and finite", power));
                }
            }
            n0 = noisePowers[0];
            n1 = noisePowers[1];
        }

        // Whitening scales the covariance and not the data. It costs three divisions per window instead of two per
        // sample. The estimator after it is unchanged.
        const double               whitenedA = a() / n0;
        const double               whitenedB = b() / n1;
        const std::complex<double> whitenedC = c() / std::sqrt(n0 * n1);

        PolarizationEstimate out;
        const double         half = 0.5 * (whitenedA - whitenedB);
        const double         disc = std::sqrt(half * half + std::norm(whitenedC));
        out.lambdaPlus            = 0.5 * (whitenedA + whitenedB) + disc;
        out.lambdaMinus           = 0.5 * (whitenedA + whitenedB) - disc;

        const double magnitude = std::abs(whitenedC);
        const double along     = out.lambdaPlus - whitenedA; // `P|h_1|^2` under the model

        std::complex<double> v0{};
        std::complex<double> v1{};
        if (magnitude > 0.) {
            // Rotate by `exp(-j arg(c))` to make `arg(v_0) = 0` exactly. `along` is real. The rotation therefore
            // falls wholly on the second component.
            v0 = {magnitude, 0.};
            v1 = along * std::conj(whitenedC) / magnitude;
        } else if (whitenedA >= whitenedB) {
            // Nothing is correlated between the branches. The weights reduce to branch 0 alone. That is the
            // correct result and not an error. The amplitude ratio is unbounded.
            v0 = {1., 0.};
            v1 = {0., 0.};
        } else {
            v0 = {0., 0.};
            v1 = {1., 0.};
        }

        if (mode == PolarizationMode::selection) {
            // `a - lambda_-` and `b - lambda_-` differ by the same constant. The stronger branch is the larger
            // diagonal entry after whitening. The same covariance decides both modes.
            out.selectedBranch = (whitenedA >= whitenedB) ? 0 : 1;
            v0                 = (out.selectedBranch == 0) ? std::complex<double>{1., 0.} : std::complex<double>{};
            v1                 = (out.selectedBranch == 0) ? std::complex<double>{} : std::complex<double>{1., 0.};
        }

        const double norm2 = std::sqrt(std::norm(v0) + std::norm(v1));
        if (norm2 > 0.) {
            v0 /= norm2;
            v1 /= norm2;
        }

        out.relativePhase  = (magnitude > 0.) ? std::arg(whitenedC) : 0.;
        out.saturatedRatio = !(along > 0.);
        out.amplitudeRatio = out.saturatedRatio ? kPolarizationSaturation : magnitude / along;

        out.saturatedSnr = !(out.lambdaMinus > 0.);
        if (out.saturatedSnr) {
            // All three figures are the cap. The additive identity below does not hold here.
            // `saturatedSnr` reports that. Read the flag before the numbers.
            out.branchSnr0  = kPolarizationSaturation;
            out.branchSnr1  = kPolarizationSaturation;
            out.combinedSnr = kPolarizationSaturation;
        } else {
            out.branchSnr0  = (whitenedA - out.lambdaMinus) / out.lambdaMinus;
            out.branchSnr1  = (whitenedB - out.lambdaMinus) / out.lambdaMinus;
            out.combinedSnr = (out.lambdaPlus - out.lambdaMinus) / out.lambdaMinus;
        }

        double scale = 1.;
        if (norm == PolarizationNormalization::unit_signal) {
            const double signalPower = out.lambdaPlus - out.lambdaMinus;
            scale                    = (signalPower > 0.) ? 1. / std::sqrt(signalPower) : 1.;
        }
        out.scale = scale;

        // Unwhiten last. The weight applied to `r_i` is the whitened one over `sqrt(N_i)`. That is `h_i/N_i` up
        // to the scale. The output noise power `sum |w_i|^2 N_i` is then exactly the whitened norm.
        out.weight0 = scale * v0 / std::sqrt(n0);
        out.weight1 = scale * v1 / std::sqrt(n1);

        const std::complex<double> u0 = -std::conj(v1);
        const std::complex<double> u1 = std::conj(v0);
        out.ortho0                    = u0 / std::sqrt(n0);
        out.ortho1                    = u1 / std::sqrt(n1);
        return out;
    }

private:
    std::size_t _n{0UZ};
    double      _a{0.};
    double      _b{0.};
    double      _cRe{0.};
    double      _cIm{0.};
};

/**
 * @brief Apply an estimate's weights, `out = conj(w_0) r_0 + conj(w_1) r_1`, with the orthogonal channel beside it.
 *
 * Each output sample costs two complex multiply-accumulates. The orthogonal output costs two more. There is no
 * transcendental function and no branch. Pass an empty @p ortho to skip the second output and its arithmetic.
 */
inline void polarizationCombine(std::span<const std::complex<float>> in0, std::span<const std::complex<float>> in1, const PolarizationEstimate& estimate, std::span<std::complex<float>> out, std::span<std::complex<float>> ortho = {}) {
    if (in0.size() != in1.size()) {
        throw std::invalid_argument(std::format("gr::measurement::polarizationCombine: {} samples on branch 0 against {} on branch 1", in0.size(), in1.size()));
    }
    if (out.size() < in0.size()) {
        throw std::invalid_argument(std::format("gr::measurement::polarizationCombine: {} outputs for {} samples", out.size(), in0.size()));
    }
    if (!ortho.empty() && ortho.size() < in0.size()) {
        throw std::invalid_argument(std::format("gr::measurement::polarizationCombine: {} orthogonal outputs for {} samples", ortho.size(), in0.size()));
    }

    const std::complex<float> w0 = static_cast<std::complex<float>>(std::conj(estimate.weight0));
    const std::complex<float> w1 = static_cast<std::complex<float>>(std::conj(estimate.weight1));
    const std::complex<float> u0 = static_cast<std::complex<float>>(std::conj(estimate.ortho0));
    const std::complex<float> u1 = static_cast<std::complex<float>>(std::conj(estimate.ortho1));

    for (std::size_t k = 0UZ; k < in0.size(); ++k) {
        out[k] = w0 * in0[k] + w1 * in1[k];
    }
    if (!ortho.empty()) {
        for (std::size_t k = 0UZ; k < in0.size(); ++k) {
            ortho[k] = u0 * in0[k] + u1 * in1[k];
        }
    }
}

} // namespace gr::measurement

#endif // GNURADIO_ALGORITHM_POLARIZATION_COMBINER_HPP
