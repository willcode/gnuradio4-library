#ifndef GNURADIO_ALGORITHM_SPECTRAL_CALIBRATION_HPP
#define GNURADIO_ALGORITHM_SPECTRAL_CALIBRATION_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <span>

namespace gr::algorithm::fft {

/**
 * @brief The scalars that turn an unnormalized transform into a calibrated power spectral density.
 *
 * The stored quantity is density, in power per hertz referred to a full-scale sine. For an unnormalized
 * forward transform `X[k] = sum_n x[n] w[n] exp(-j 2 pi k n / N)`,
 *
 *     psd[k] = |X[k]|^2 / (fs * sum(w^2))
 *
 * and the band integral is `sum_k psd[k] * fs / N`. Parseval makes that come out at the signal's own mean
 * power: `sum_k |X[k]|^2 = N * sum_n |x[n] w[n]|^2`, so the integral reduces to
 * `sum_n |x[n]|^2 w[n]^2 / sum(w^2)`, which for a constant-envelope input of amplitude A is exactly `A^2`,
 * whatever window was used. That is what makes the density calibration window-independent, and it is why
 * the density is the quantity kept.
 *
 * A tone read off its own peak bin is the other question and needs the other correction. The window spreads
 * a tone over several bins, so the peak density under-states it by the window's equivalent noise bandwidth;
 * multiplying the peak density by `enbwHz` recovers the tone's power. The two normalizations cannot both
 * hold per bin, so `enbwBins` travels in the record's metadata and a consumer applies it when it is reading
 * a tone rather than a noise floor.
 */
template<std::floating_point F>
struct SpectralScale {
    F density{};  ///< multiplies |X[k]|^2 to give power per hertz
    F enbwBins{}; ///< the window's equivalent noise bandwidth, in bins
    F binWidthHz{};

    /// @brief The equivalent noise bandwidth in hertz: what a peak density is multiplied by to read a tone's power.
    [[nodiscard]] constexpr F enbwHz() const noexcept { return enbwBins * binWidthHz; }
};

/// @brief Sum of a window's samples — the coherent gain, before division by the length.
template<std::floating_point F>
[[nodiscard]] constexpr F windowSum(std::span<const F> window) noexcept {
    F total{0};
    for (const F value : window) {
        total += value;
    }
    return total;
}

/// @brief Sum of a window's squared samples — the incoherent gain, and the density denominator.
template<std::floating_point F>
[[nodiscard]] constexpr F windowPower(std::span<const F> window) noexcept {
    F total{0};
    for (const F value : window) {
        total += value * value;
    }
    return total;
}

/// @brief `N * sum(w^2) / sum(w)^2`, the window's noise bandwidth expressed in bins. One for a rectangle.
template<std::floating_point F>
[[nodiscard]] constexpr F enbwBins(std::span<const F> window) noexcept {
    const F coherent = windowSum(window);
    if (coherent == F{0}) {
        return std::numeric_limits<F>::quiet_NaN();
    }
    return static_cast<F>(window.size()) * windowPower(window) / (coherent * coherent);
}

/// @brief The scalars for one transform length, window and sample rate. `sampleRate` and a non-empty window
/// with non-zero power are the caller's responsibility; a degenerate window yields a NaN density rather than
/// a division by zero, which a caller's own validation is expected to have refused first.
template<std::floating_point F>
[[nodiscard]] constexpr SpectralScale<F> spectralScale(std::span<const F> window, F sampleRate) noexcept {
    const F power = windowPower(window);
    return SpectralScale<F>{
        .density    = (sampleRate > F{0} && power > F{0}) ? F{1} / (sampleRate * power) : std::numeric_limits<F>::quiet_NaN(),
        .enbwBins   = enbwBins(window),
        .binWidthHz = window.empty() ? std::numeric_limits<F>::quiet_NaN() : sampleRate / static_cast<F>(window.size()),
    };
}

/**
 * @brief Accumulates `|X[k]|^2 * scale` into `out`, one-sided when asked.
 *
 * A one-sided spectrum keeps bins `0 .. N/2` and doubles every bin that has a mirrored twin — which is all
 * of them but DC and Nyquist, the two that are their own reflection. That is the fold `fft_common.hpp`
 * applies to the magnitude spectrum, restated here in power rather than amplitude, so a one-sided density
 * integrates to the same total as the two-sided one it came from.
 *
 * `out` is added to rather than overwritten, so a caller averaging segments accumulates in place.
 */
template<std::floating_point F>
constexpr void accumulatePowerSpectrum(std::span<const std::complex<F>> spectrum, F scale, bool oneSided, std::span<F> out) noexcept {
    const std::size_t n    = spectrum.size();
    const std::size_t bins = std::min(out.size(), oneSided ? n / 2UZ + 1UZ : n);
    const std::size_t nyq  = n / 2UZ;
    for (std::size_t k = 0UZ; k < bins; ++k) {
        const std::complex<F>& value  = spectrum[k];
        const F                power  = value.real() * value.real() + value.imag() * value.imag();
        const bool             folded = oneSided && k != 0UZ && k != nyq;
        out[k] += (folded ? F{2} : F{1}) * power * scale;
    }
}

/// @brief Power ratio to decibels, with a floor so an empty bin reports a number rather than negative infinity.
template<std::floating_point F>
[[nodiscard]] constexpr F powerToDb(F power, F floorDb = F{-300}) noexcept {
    if (!(power > F{0})) {
        return floorDb;
    }
    const F db = F{10} * std::log10(power);
    return db < floorDb ? floorDb : db;
}

/// @brief The band integral of a density: `sum(psd) * binWidth`. What criterion 1 reads to check a calibration.
template<std::floating_point F>
[[nodiscard]] constexpr F integratePowerDensity(std::span<const F> density, F binWidthHz) noexcept {
    F total{0};
    for (const F value : density) {
        total += value;
    }
    return total * binWidthHz;
}

} // namespace gr::algorithm::fft

#endif // GNURADIO_ALGORITHM_SPECTRAL_CALIBRATION_HPP
