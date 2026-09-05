#ifndef GNURADIO_ALGORITHM_LTE_SYNC_SIGNALS_HPP
#define GNURADIO_ALGORITHM_LTE_SYNC_SIGNALS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#include <vir/simd.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace gr::lte {

/**
 * @brief The two E-UTRA synchronization signals, their frame geometry, and the two kernels that read them.
 *
 * Everything here follows 3GPP TS 36.211: the primary synchronization signal is section 6.11.1, the secondary
 * is 6.11.2, the symbol positions are 6.11.1.3, 6.11.2.3 and section 4, and the cyclic-prefix lengths are
 * section 6.12 with Table 6.12-1. The sample rate is fixed at 1.92 MS/s, the smallest the numerology allows
 * (128 * 15 kHz), at which the central six resource blocks that carry both signals are the whole band and one
 * OFDM symbol is 128 useful samples.
 *
 * Two signals carry the whole answer. The primary one says N_ID^(2) and where the symbol sits; the secondary
 * one says N_ID^(1), which of the two half-frames the detection came from, the duplex mode and the cyclic-prefix
 * type — the last two because the spacing from the secondary symbol to the primary one differs for each
 * combination, so the hypothesis that decodes is the one that names them.
 */

/// Subcarriers each synchronization signal occupies: a length-63 Zadoff-Chu sequence with its middle element removed.
inline constexpr std::size_t kSignalLength = 62UZ;
/// Useful part of one OFDM symbol at 1.92 MS/s.
inline constexpr std::size_t kSymbolSamples = 128UZ;
/// Length of each of the three m-sequences the secondary signal is built from.
inline constexpr std::size_t kMSequenceLength = 31UZ;
/// The only sample rate this geometry is written for.
inline constexpr float kSampleRate = 1'920'000.f;

inline constexpr std::size_t kSlotSamples      = 960UZ;
inline constexpr std::size_t kSubframeSamples  = 1920UZ;
inline constexpr std::size_t kHalfFrameSamples = 9600UZ;
inline constexpr std::size_t kFrameSamples     = 19200UZ;

/// Physical cell identities, `3 * N_ID^(1) + N_ID^(2)`.
inline constexpr std::uint32_t kCellIdentities = 504U;
/// Cell-identity groups, the range of `N_ID^(1)`.
inline constexpr std::uint32_t kCellGroups = 168U;

/// Zadoff-Chu roots for `N_ID^(2) = 0, 1, 2`. Roots 29 and 34 sum to 63, which is what makes their sequences conjugates.
inline constexpr std::array<std::uint32_t, 3UZ> kPssRoots{25U, 29U, 34U};

/// Frame structure type 1 (paired spectrum) or type 2 (unpaired), which the two signals' spacing distinguishes.
enum class DuplexMode : std::uint8_t { Fdd = 0, Tdd = 1 };

/// Which of Table 6.12-1's two cyclic-prefix columns the symbol grid uses.
enum class CyclicPrefix : std::uint8_t { Normal = 0, Extended = 1 };

/**
 * @brief Where the symbols sit, in samples at 1.92 MS/s, for one duplex mode and cyclic-prefix type.
 *
 * Every offset is stated against `p`, the index of the primary symbol's first **useful** sample — the first
 * sample after its own cyclic prefix — because that is the position a correlation against the useful symbol
 * finds. A slot is 960 samples and a subframe 1920 whichever prefix is in use: the normal prefix spends
 * `10 + 6*9 = 64` samples on prefixes across seven symbols and the extended one `6*32 = 192` across six, and
 * `960 - 7*128 = 64` and `960 - 6*128 = 192` respectively, so the two grids close on the same boundaries.
 */
struct FrameGeometry {
    DuplexMode   duplex{DuplexMode::Fdd};
    CyclicPrefix cyclicPrefix{CyclicPrefix::Normal};

    [[nodiscard]] constexpr bool extended() const noexcept { return cyclicPrefix == CyclicPrefix::Extended; }

    /// Symbols in one 0.5 ms slot.
    [[nodiscard]] constexpr std::size_t symbolsPerSlot() const noexcept { return extended() ? 6UZ : 7UZ; }
    /// Cyclic prefix of a slot's first symbol, 160 or 512 units of T_s.
    [[nodiscard]] constexpr std::size_t firstPrefix() const noexcept { return extended() ? 32UZ : 10UZ; }
    /// Cyclic prefix of every other symbol of a slot, 144 or 512 units of T_s.
    [[nodiscard]] constexpr std::size_t otherPrefix() const noexcept { return extended() ? 32UZ : 9UZ; }

    /**
     * @brief Start of the secondary symbol's useful part, relative to `p`.
     *
     * Paired spectrum puts the secondary signal in the symbol immediately before the primary one, so the gap is
     * that symbol's 128 useful samples plus the primary symbol's own prefix. Unpaired spectrum puts the primary
     * signal in symbol 2 of the subframe after the one whose last symbol carries the secondary signal, so the
     * gap is everything from the subframe boundary back over one more symbol.
     */
    [[nodiscard]] constexpr std::ptrdiff_t secondaryOffset() const noexcept {
        const std::ptrdiff_t symbol = static_cast<std::ptrdiff_t>(kSymbolSamples);
        if (duplex == DuplexMode::Fdd) {
            return -(symbol + static_cast<std::ptrdiff_t>(otherPrefix()));
        }
        return -(primaryIntoSubframe() + symbol);
    }

    /// How far the primary symbol's useful part sits into its own subframe. Unpaired spectrum only; paired spectrum
    /// puts the signal at the end of a slot instead, which `slotOffset()` states.
    [[nodiscard]] constexpr std::ptrdiff_t primaryIntoSubframe() const noexcept {
        const std::ptrdiff_t prefixes = static_cast<std::ptrdiff_t>(firstPrefix() + 2UZ * otherPrefix());
        return prefixes + 2 * static_cast<std::ptrdiff_t>(kSymbolSamples);
    }

    /// Start of the slot carrying the primary symbol, relative to `p`. Paired spectrum only, and the same 832 for
    /// both prefix types: `10 + 128 + 5*(9 + 128) + 9` and `5*(32 + 128) + 32`.
    [[nodiscard]] constexpr std::ptrdiff_t slotOffset() const noexcept {
        const std::ptrdiff_t symbol   = static_cast<std::ptrdiff_t>(kSymbolSamples);
        const std::size_t    symbols  = symbolsPerSlot();
        std::ptrdiff_t       into     = static_cast<std::ptrdiff_t>(firstPrefix()) + symbol;
        const std::size_t    interior = symbols - 2UZ;
        into += static_cast<std::ptrdiff_t>(interior) * (static_cast<std::ptrdiff_t>(otherPrefix()) + symbol);
        into += static_cast<std::ptrdiff_t>(otherPrefix());
        return -into;
    }

    /**
     * @brief Start of the 10 ms radio frame, relative to `p`, for a detection in half-frame `halfFrame`.
     *
     * Paired spectrum carries the signals in slots 0 and 10, so the frame begins at the slot the primary symbol
     * ends, less a half-frame when the secondary signal decoded in its subframe-5 form. Unpaired spectrum carries
     * them in subframes 1 and 6, so the frame begins one subframe before the primary symbol's own.
     */
    [[nodiscard]] constexpr std::ptrdiff_t frameStartOffset(std::uint32_t halfFrame) const noexcept {
        const std::ptrdiff_t half = static_cast<std::ptrdiff_t>(halfFrame) * static_cast<std::ptrdiff_t>(kHalfFrameSamples);
        if (duplex == DuplexMode::Fdd) {
            return slotOffset() - half;
        }
        return -primaryIntoSubframe() - static_cast<std::ptrdiff_t>(kSubframeSamples) - half;
    }
};

/// The four structures a detection is tested against, in the order the decoder reports them.
inline constexpr std::array<FrameGeometry, 4UZ> kStructures{{
    {DuplexMode::Fdd, CyclicPrefix::Normal},
    {DuplexMode::Fdd, CyclicPrefix::Extended},
    {DuplexMode::Tdd, CyclicPrefix::Normal},
    {DuplexMode::Tdd, CyclicPrefix::Extended},
}};

/// The furthest back any structure places the secondary symbol, and so the context a window must hold before it.
inline constexpr std::size_t kMaxSecondaryLookBehind = 480UZ;

/// @brief `d(n)`'s bin in a 128-point transform: subcarriers -31..-1 for `n = 0..30` and +1..+31 for `n = 31..61`,
/// with DC left empty (TS 36.211 sections 6.11.1.2 and 6.11.2.2).
[[nodiscard]] constexpr std::size_t subcarrierBin(std::size_t n) noexcept { return n <= 30UZ ? n + 97UZ : n - 30UZ; }

/// @brief `N_ID^cell = 3 * N_ID^(1) + N_ID^(2)`.
[[nodiscard]] constexpr std::uint32_t cellIdentity(std::uint32_t nId1, std::uint32_t nId2) noexcept { return 3U * nId1 + nId2; }

/// The two cyclic shifts of the s-sequence that a cell-identity group selects.
struct ShiftPair {
    std::uint32_t m0{0U};
    std::uint32_t m1{0U};

    [[nodiscard]] constexpr bool operator==(const ShiftPair&) const noexcept = default;
};

/// @brief TS 36.211 section 6.11.2.1's closed form for the `(m0, m1)` table. All 168 pairs are distinct.
[[nodiscard]] constexpr ShiftPair m0m1(std::uint32_t nId1) {
    if (nId1 >= kCellGroups) {
        throw std::out_of_range(std::format("lte: N_ID^(1) must be below {}, got {}", kCellGroups, nId1));
    }
    const std::uint32_t qPrime = nId1 / 30U;
    const std::uint32_t q      = (nId1 + qPrime * (qPrime + 1U) / 2U) / 30U;
    const std::uint32_t mPrime = nId1 + q * (q + 1U) / 2U;
    const std::uint32_t m0     = mPrime % 31U;
    return ShiftPair{m0, (m0 + mPrime / 31U + 1U) % 31U};
}

/// @brief The cell-identity group a shift pair names, or nothing when the pair is not in the table.
[[nodiscard]] constexpr std::optional<std::uint32_t> nId1FromPair(std::uint32_t m0, std::uint32_t m1) noexcept {
    for (std::uint32_t nId1 = 0U; nId1 < kCellGroups; ++nId1) {
        const std::uint32_t qPrime = nId1 / 30U;
        const std::uint32_t q      = (nId1 + qPrime * (qPrime + 1U) / 2U) / 30U;
        const std::uint32_t mPrime = nId1 + q * (q + 1U) / 2U;
        if (mPrime % 31U == m0 && (mPrime % 31U + mPrime / 31U + 1U) % 31U == m1) {
            return nId1;
        }
    }
    return std::nullopt;
}

namespace detail {

/// @brief One of TS 36.211 section 6.11.2.1's five-stage m-sequences, seeded `x(0..4) = 0 0 0 0 1` and mapped `1 - 2x`.
/// @param taps stage indices summed modulo two to form `x(i+5)`
template<std::size_t nTaps>
[[nodiscard]] constexpr std::array<float, kMSequenceLength> mSequence(const std::array<std::size_t, nTaps>& taps) {
    std::array<std::uint8_t, kMSequenceLength + 5UZ> x{};
    x[4UZ] = 1U;
    for (std::size_t i = 0UZ; i + 5UZ < x.size(); ++i) {
        std::uint8_t next = 0U;
        for (const std::size_t tap : taps) {
            next = static_cast<std::uint8_t>(next ^ x[i + tap]);
        }
        x[i + 5UZ] = next;
    }
    std::array<float, kMSequenceLength> out{};
    for (std::size_t i = 0UZ; i < kMSequenceLength; ++i) {
        out[i] = 1.f - 2.f * static_cast<float>(x[i]);
    }
    return out;
}

} // namespace detail

/// @brief The s-sequence, `x(i+5) = (x(i+2) + x(i)) mod 2`, whose two cyclic shifts carry the cell-identity group.
[[nodiscard]] constexpr std::array<float, kMSequenceLength> sSequence() { return detail::mSequence(std::array<std::size_t, 2UZ>{2UZ, 0UZ}); }
/// @brief The c-sequence, `x(i+5) = (x(i+3) + x(i)) mod 2`, whose two shifts scramble by `N_ID^(2)`.
[[nodiscard]] constexpr std::array<float, kMSequenceLength> cSequence() { return detail::mSequence(std::array<std::size_t, 2UZ>{3UZ, 0UZ}); }
/// @brief The z-sequence, `x(i+5) = (x(i+4) + x(i+2) + x(i+1) + x(i)) mod 2`, the second scrambling of the odd subcarriers.
[[nodiscard]] constexpr std::array<float, kMSequenceLength> zSequence() { return detail::mSequence(std::array<std::size_t, 4UZ>{4UZ, 2UZ, 1UZ, 0UZ}); }

/**
 * @brief The primary synchronization signal's 62 frequency-domain values (TS 36.211 section 6.11.1.1).
 *
 * The length-63 Zadoff-Chu sequence `exp(-j*pi*u*n*(n+1)/63)` with its middle element punctured, which is why the
 * two halves take different arguments. The exponent is periodic in `u*k` with period 126, so it is reduced before
 * it becomes an angle: that keeps the argument small for every root instead of rounding a value near 26000, and it
 * makes the central symmetry `d(n) = d(61-n)` hold exactly rather than to within a rounding of a large angle.
 */
[[nodiscard]] inline std::array<std::complex<double>, kSignalLength> pssSequence(std::uint32_t nId2) {
    if (nId2 >= kPssRoots.size()) {
        throw std::out_of_range(std::format("lte: N_ID^(2) must be below 3, got {}", nId2));
    }
    const std::uint64_t                             u = kPssRoots[nId2];
    std::array<std::complex<double>, kSignalLength> d{};
    for (std::size_t n = 0UZ; n < kSignalLength; ++n) {
        const std::uint64_t k     = n <= 30UZ ? std::uint64_t{n} * (n + 1UZ) : std::uint64_t{n + 1UZ} * (n + 2UZ);
        const std::uint64_t phase = (u * k) % 126ULL;
        d[n]                      = std::polar(1.0, -std::numbers::pi * static_cast<double>(phase) / 63.0);
    }
    return d;
}

/**
 * @brief The primary signal's useful symbol in time, 128 samples of unit average power.
 *
 * The 62 values on subcarriers -31..-1 and +1..+31, DC empty, through a 128-point inverse transform. The
 * sequence's central symmetry makes the result symmetric in time as well, `x(m) = x(128-m)` for `m > 0`, which is
 * the property a receiver's half-symbol estimators rely on.
 */
[[nodiscard]] inline std::array<std::complex<float>, kSymbolSamples> pssTimeDomain(std::uint32_t nId2) {
    const std::array<std::complex<double>, kSignalLength> d = pssSequence(nId2);

    std::vector<std::complex<float>> bins(kSymbolSamples, std::complex<float>(0.f, 0.f));
    for (std::size_t n = 0UZ; n < kSignalLength; ++n) {
        bins[subcarrierBin(n)] = std::complex<float>(static_cast<float>(d[n].real()), static_cast<float>(d[n].imag()));
    }

    gr::algorithm::FFT<std::complex<float>, std::complex<float>, gr::algorithm::Direction::Backward> inverse;
    const auto                                                                                       samples = inverse.compute(bins);

    double power = 0.;
    for (const std::complex<float>& sample : samples) {
        power += static_cast<double>(std::norm(sample));
    }
    const float scale = power > 0. ? static_cast<float>(std::sqrt(static_cast<double>(kSymbolSamples) / power)) : 1.f;

    std::array<std::complex<float>, kSymbolSamples> out{};
    for (std::size_t m = 0UZ; m < kSymbolSamples; ++m) {
        out[m] = samples[m] * scale;
    }
    return out;
}

/**
 * @brief The secondary synchronization signal's 62 values, +/-1 (TS 36.211 section 6.11.2.1).
 *
 * Two interleaved cyclic shifts of the s-sequence, scrambled by two shifts of the c-sequence selected by
 * `N_ID^(2)` and, on the odd subcarriers, by a shift of the z-sequence selected by the other half's shift. The
 * two half-frames exchange the roles of the two s-shifts and of the two z-shifts, which is exactly what lets a
 * receiver tell a 5 ms position from a 10 ms one.
 *
 * @param subframe 0 for the first half-frame's form, 5 for the second's
 */
[[nodiscard]] inline std::array<float, kSignalLength> sssSequence(std::uint32_t nId1, std::uint32_t nId2, std::uint32_t subframe) {
    if (nId2 >= kPssRoots.size()) {
        throw std::out_of_range(std::format("lte: N_ID^(2) must be below 3, got {}", nId2));
    }
    if (subframe != 0U && subframe != 5U) {
        throw std::out_of_range(std::format("lte: the secondary signal has a subframe-0 and a subframe-5 form only, got {}", subframe));
    }
    const ShiftPair                           pair = m0m1(nId1);
    const std::array<float, kMSequenceLength> s    = sSequence();
    const std::array<float, kMSequenceLength> c    = cSequence();
    const std::array<float, kMSequenceLength> z    = zSequence();

    std::array<float, kSignalLength> d{};
    for (std::uint32_t n = 0U; n < kMSequenceLength; ++n) {
        const float s0 = s[(n + pair.m0) % kMSequenceLength];
        const float s1 = s[(n + pair.m1) % kMSequenceLength];
        const float c0 = c[(n + nId2) % kMSequenceLength];
        const float c1 = c[(n + nId2 + 3U) % kMSequenceLength];
        const float z0 = z[(n + pair.m0 % 8U) % kMSequenceLength];
        const float z1 = z[(n + pair.m1 % 8U) % kMSequenceLength];
        if (subframe == 0U) {
            d[2UZ * n]       = s0 * c0;
            d[2UZ * n + 1UZ] = s1 * c1 * z0;
        } else {
            d[2UZ * n]       = s1 * c0;
            d[2UZ * n + 1UZ] = s0 * c1 * z1;
        }
    }
    return d;
}

/// What one half of a secondary-signal reading resolved to.
struct ShiftMatch {
    std::uint32_t shift{0U};
    float         quality{0.f}; ///< peak correlation over the 31 shifts, normalized so a clean match is 1
};

/**
 * @brief The cyclic shift of `basis` that best matches `values`, and how well.
 *
 * The quality is the peak inner product divided by `sqrt(31) * ||values||`, which is the Cauchy-Schwarz bound for
 * a unit-modulus basis: 1 for a clean match whatever the values are scaled by, and near `2/sqrt(31) = 0.36` on
 * noise, where the 31 shifts each draw an inner product of standard deviation `sqrt(31) * ||values|| / sqrt(31)`
 * and the largest of 31 such draws sits about two standard deviations out. The inner product is signed rather
 * than absolute because coherent equalization against the primary symbol fixes the sign.
 */
[[nodiscard]] inline ShiftMatch bestShift(std::span<const float, kMSequenceLength> values, const std::array<float, kMSequenceLength>& basis) noexcept {
    double energy = 0.;
    for (const float value : values) {
        energy += static_cast<double>(value) * static_cast<double>(value);
    }
    if (!(energy > 0.)) {
        return ShiftMatch{};
    }
    const double norm = std::sqrt(energy * static_cast<double>(kMSequenceLength));
    ShiftMatch   best{0U, -std::numeric_limits<float>::infinity()};
    for (std::uint32_t shift = 0U; shift < kMSequenceLength; ++shift) {
        double sum = 0.;
        for (std::uint32_t n = 0U; n < kMSequenceLength; ++n) {
            sum += static_cast<double>(values[n]) * static_cast<double>(basis[(n + shift) % kMSequenceLength]);
        }
        const float quality = static_cast<float>(sum / norm);
        if (quality > best.quality) {
            best = ShiftMatch{shift, quality};
        }
    }
    return best;
}

/// One reading of the 62 soft secondary-signal values under one assumed half-frame form.
struct SssReading {
    bool          found{false};
    std::uint32_t nId1{0U};
    std::uint32_t subframe{0U}; ///< 0 or 5, the form that decoded
    float         metric{0.f};  ///< mean of the two halves' shift qualities
};

/**
 * @brief Read a cell-identity group out of 62 soft secondary-signal values (TS 36.211 section 6.11.2.1).
 *
 * The even subcarriers carry one shift of the s-sequence under the `c0` scrambling and the odd ones the other
 * shift under `c1` and a z-shift the first half names, so the two halves are read in that order. Both half-frame
 * forms are tried and the better one wins; a shift pair outside the table is not a cell and is refused for that
 * form, which is what makes the reading a check as well as a decode.
 */
[[nodiscard]] inline SssReading decodeSssValues(std::span<const float, kSignalLength> soft, std::uint32_t nId2) {
    if (nId2 >= kPssRoots.size()) {
        throw std::out_of_range(std::format("lte: N_ID^(2) must be below 3, got {}", nId2));
    }
    const std::array<float, kMSequenceLength> s = sSequence();
    const std::array<float, kMSequenceLength> c = cSequence();
    const std::array<float, kMSequenceLength> z = zSequence();

    std::array<float, kMSequenceLength> even{};
    std::array<float, kMSequenceLength> odd{};
    for (std::uint32_t n = 0U; n < kMSequenceLength; ++n) {
        even[n] = soft[2UZ * n] * c[(n + nId2) % kMSequenceLength];
        odd[n]  = soft[2UZ * n + 1UZ] * c[(n + nId2 + 3U) % kMSequenceLength];
    }

    const ShiftMatch first = bestShift(std::span<const float, kMSequenceLength>(even), s);

    std::array<float, kMSequenceLength> descrambled{};
    for (std::uint32_t n = 0U; n < kMSequenceLength; ++n) {
        descrambled[n] = odd[n] * z[(n + first.shift % 8U) % kMSequenceLength];
    }
    const ShiftMatch second = bestShift(std::span<const float, kMSequenceLength>(descrambled), s);

    const float metric = 0.5f * (first.quality + second.quality);

    // The first half is m0 in the subframe-0 form and m1 in the subframe-5 form; only one of the two orderings
    // can name a cell, so testing both costs one table lookup and settles the half-frame at the same time.
    if (const std::optional<std::uint32_t> group = nId1FromPair(first.shift, second.shift); group.has_value()) {
        return SssReading{true, *group, 0U, metric};
    }
    if (const std::optional<std::uint32_t> group = nId1FromPair(second.shift, first.shift); group.has_value()) {
        return SssReading{true, *group, 5U, metric};
    }
    return SssReading{false, 0U, 0U, metric};
}

/// One primary-signal detection: where the symbol is, which root it carries, and how strongly.
struct PssDetection {
    bool          found{false};
    std::size_t   position{0UZ}; ///< window index of the symbol's first useful sample
    std::uint32_t nId2{0U};
    std::size_t   hypothesis{0UZ};   ///< which integer frequency hypothesis won
    float         hypothesisHz{0.f}; ///< that hypothesis's own frequency
    float         frequencyHz{0.f};  ///< the hypothesis plus the fractional estimate
    float         metric{0.f};       ///< peak power over the mean correlation power across the window
};

/**
 * @brief Finds the primary synchronization signal in a window of samples.
 *
 * The statistic is the complex correlation of the window against each of the three 128-sample references at every
 * position, and the metric of the best position is its power divided by the mean correlation power over the whole
 * window. That ratio is scale-free by construction, so no gain control is a precondition: at the symbol the peak
 * is `(128*A)^2` for a per-sample amplitude `A`, while the window mean is `128*(P_signal + sigma^2)` because every
 * other symbol's content is as uncorrelated with the reference as the noise is. The metric is therefore
 * `128 * SNR / (1 + SNR)` when the symbol carries the signal's average power — 64 at 0 dB, 116 at 10 dB, and
 * saturating at 128 — and on noise alone the correlation power is exponentially distributed, so the largest of
 * `N` positions sits near `ln(N) + 0.577` times the mean.
 *
 * A carrier offset costs `|sinc(offset * 66.7 us)|` across the 128-sample correlation, a null at one subcarrier
 * spacing. Hypotheses are therefore spaced half a subcarrier, 7.5 kHz, so the worst case before the fractional
 * estimate is 0.9 dB, and the fractional estimator covers exactly the gap between two hypotheses: with `c1` the
 * correlation over the first 64 samples and `c2` over the last 64, `arg(c2 * conj(c1))` is unambiguous over
 * +/-15 kHz and its noise standard deviation is `1/(8*sqrt(SNR))` radians.
 *
 * The correlation is evaluated directly rather than through a transform. The direct form costs `3*K*128` complex
 * multiply-accumulates per input sample against a transform form's `3*K` inverse transforms per window, so it is
 * the dearer of the two for a wide search; it is exact at every position with no transform-length quantization of
 * the hypothesis grid, and a wide search is what `search_interval` on the consuming block is for.
 *
 * The references are held as split real and imaginary arrays, and the window is split the same way once per
 * search, because a `std::complex<float>` product carries infinity and NaN rules that stop a compiler
 * vectorizing the inner loop at all; the loop then accumulates one vector register of lanes at a time, which is
 * what a floating-point sum needs to be told explicitly because reassociating it is not a transformation a
 * compiler may make on its own.
 */
class PssCorrelator {
public:
    /// Half a subcarrier: the spacing at which the worst-case correlation loss before the fractional estimate is 0.9 dB.
    static constexpr float kHypothesisSpacingHz = 7'500.f;
    /// @param searchHalfWidthHz half-width of the integer frequency search; 0 is the single zero hypothesis
    explicit PssCorrelator(float searchHalfWidthHz = 0.f, float sampleRate = kSampleRate) : _sampleRate(sampleRate) {
        if (!(sampleRate > 0.f)) {
            throw std::invalid_argument(std::format("lte: the sample rate must be positive, got {}", sampleRate));
        }
        const std::size_t half = static_cast<std::size_t>(std::ceil(static_cast<double>(std::max(0.f, searchHalfWidthHz)) / static_cast<double>(kHypothesisSpacingHz)));
        _hypotheses            = 2UZ * half + 1UZ;
        _firstHypothesis       = -static_cast<float>(half) * kHypothesisSpacingHz;

        _referenceReal.resize(kPssRoots.size() * _hypotheses * kSymbolSamples);
        _referenceImag.resize(_referenceReal.size());
        for (std::uint32_t nId2 = 0U; nId2 < kPssRoots.size(); ++nId2) {
            const std::array<std::complex<float>, kSymbolSamples> base = pssTimeDomain(nId2);
            for (std::size_t k = 0UZ; k < _hypotheses; ++k) {
                // The reference carries the hypothesis rather than removing it: the correlation conjugates the
                // reference, so a reference at `+f_k` is the one that cancels a signal sitting at `+f_k` and the
                // winning hypothesis is then the offset itself rather than its negative.
                const double      turn = 2. * std::numbers::pi * static_cast<double>(hypothesisFrequency(k)) / static_cast<double>(_sampleRate);
                const std::size_t at   = referenceAt(nId2, k);
                for (std::size_t n = 0UZ; n < kSymbolSamples; ++n) {
                    const std::complex<double> rotated = std::complex<double>(base[n]) * std::polar(1.0, turn * static_cast<double>(n));
                    _referenceReal[at + n]             = static_cast<float>(rotated.real());
                    _referenceImag[at + n]             = static_cast<float>(rotated.imag());
                }
            }
        }
    }

    [[nodiscard]] std::size_t hypotheses() const noexcept { return _hypotheses; }

    /// @brief The integer frequency hypothesis at index `k`, in Hz.
    [[nodiscard]] float hypothesisFrequency(std::size_t k) const noexcept { return _firstHypothesis + static_cast<float>(k) * kHypothesisSpacingHz; }

    /**
     * @brief The best position and hypothesis for each of the three roots.
     *
     * `nPositions` positions are evaluated, the first at the window's own start, and the window must therefore
     * hold `nPositions + 127` samples. Every position is a candidate on its own merits: the peak is the maximum
     * over the window rather than the first crossing of any level, and the caller arranges its windows so that a
     * symbol beginning at the last position is still evaluated whole.
     *
     * @return one detection per `N_ID^(2)`, always populated; the caller applies its own threshold to `metric`
     */
    [[nodiscard]] std::array<PssDetection, 3UZ> search(std::span<const std::complex<float>> window, std::size_t nPositions) {
        if (nPositions == 0UZ || window.size() < nPositions + kSymbolSamples - 1UZ) {
            throw std::invalid_argument(std::format("lte: a window of {} samples cannot carry {} correlation positions", window.size(), nPositions));
        }
        const std::size_t span = nPositions + kSymbolSamples - 1UZ;
        _windowReal.resize(span);
        _windowImag.resize(span);
        for (std::size_t i = 0UZ; i < span; ++i) {
            _windowReal[i] = window[i].real();
            _windowImag[i] = window[i].imag();
        }

        std::array<PssDetection, 3UZ> best{};
        for (std::uint32_t nId2 = 0U; nId2 < kPssRoots.size(); ++nId2) {
            best[nId2].nId2 = nId2;
            for (std::size_t k = 0UZ; k < _hypotheses; ++k) {
                const std::size_t at    = referenceAt(nId2, k);
                double            total = 0.;
                double            peak  = -1.;
                std::size_t       where = 0UZ;
                for (std::size_t i = 0UZ; i < nPositions; ++i) {
                    const std::complex<double> c     = correlate(i, at, 0UZ, kSymbolSamples);
                    const double               power = c.real() * c.real() + c.imag() * c.imag();
                    total += power;
                    if (power > peak) {
                        peak  = power;
                        where = i;
                    }
                }
                const double mean   = total / static_cast<double>(nPositions);
                const float  metric = mean > 0. ? static_cast<float>(peak / mean) : 0.f;
                if (metric > best[nId2].metric) {
                    best[nId2] = PssDetection{true, where, nId2, k, hypothesisFrequency(k), hypothesisFrequency(k) + fractionalOffset(where, at), metric};
                }
            }
        }
        return best;
    }

private:
    [[nodiscard]] std::size_t referenceAt(std::uint32_t nId2, std::size_t k) const noexcept { return (static_cast<std::size_t>(nId2) * _hypotheses + k) * kSymbolSamples; }

    /// @brief `sum(x[i+n] * conj(r[at+n]))` over `[from, to)`, one vector register of lanes at a time.
    [[nodiscard]] std::complex<double> correlate(std::size_t i, std::size_t at, std::size_t from, std::size_t to) const noexcept {
        using Lanes = vir::stdx::native_simd<float>;

        const float* const xr = _windowReal.data() + i;
        const float* const xi = _windowImag.data() + i;
        const float* const rr = _referenceReal.data() + at;
        const float* const ri = _referenceImag.data() + at;

        Lanes       accumulatedReal(0.f);
        Lanes       accumulatedImag(0.f);
        std::size_t n = from;
        for (; n + Lanes::size() <= to; n += Lanes::size()) {
            const Lanes windowReal(xr + n, vir::stdx::element_aligned);
            const Lanes windowImag(xi + n, vir::stdx::element_aligned);
            const Lanes referenceReal(rr + n, vir::stdx::element_aligned);
            const Lanes referenceImag(ri + n, vir::stdx::element_aligned);
            accumulatedReal += windowReal * referenceReal + windowImag * referenceImag;
            accumulatedImag += windowImag * referenceReal - windowReal * referenceImag;
        }
        double sumReal = static_cast<double>(vir::stdx::reduce(accumulatedReal));
        double sumImag = static_cast<double>(vir::stdx::reduce(accumulatedImag));
        for (; n < to; ++n) { // a register width that does not divide the symbol leaves a tail
            sumReal += static_cast<double>(xr[n] * rr[n] + xi[n] * ri[n]);
            sumImag += static_cast<double>(xi[n] * rr[n] - xr[n] * ri[n]);
        }
        return {sumReal, sumImag};
    }

    /// @brief The residual offset the two half-symbol correlations imply, in Hz, from their phase difference.
    [[nodiscard]] float fractionalOffset(std::size_t i, std::size_t at) const noexcept {
        const std::size_t          half = kSymbolSamples / 2UZ;
        const std::complex<double> c1   = correlate(i, at, 0UZ, half);
        const std::complex<double> c2   = correlate(i, at, half, kSymbolSamples);
        const double               turn = std::arg(c2 * std::conj(c1));
        return static_cast<float>(turn * static_cast<double>(_sampleRate) / (2. * std::numbers::pi * static_cast<double>(half)));
    }

    float              _sampleRate{kSampleRate};
    std::size_t        _hypotheses{1UZ};
    float              _firstHypothesis{0.f};
    std::vector<float> _referenceReal{};
    std::vector<float> _referenceImag{};
    std::vector<float> _windowReal{};
    std::vector<float> _windowImag{};
};

/// What the secondary signal added to a primary detection.
struct SssDecision {
    bool                             confirmed{false};
    std::uint32_t                    nId1{0U};
    std::uint32_t                    halfFrame{0U}; ///< 0 for the subframe-0 form, 1 for the subframe-5 form
    FrameGeometry                    geometry{};
    float                            metric{0.f};
    std::array<float, kSignalLength> soft{}; ///< the winning hypothesis's soft values, in subcarrier order
};

/**
 * @brief Reads the secondary synchronization signal beside a located primary one.
 *
 * The primary symbol is the channel estimate: its transmitted values are known and unit modulus, so
 * `H(k) = Y_pss(k) * conj(d(k))` is the channel on each of the 62 subcarriers with nothing to divide by. Equalizing
 * the secondary symbol against it, `Re(Y_sss(k) * conj(H(k)))`, is coherent and real-valued, because the secondary
 * signal is real in frequency and the estimate carries the same residual phase the symbol does.
 *
 * The four structures place the secondary symbol at four different distances behind the primary one, so all four
 * are transformed and the one that decodes with the best metric names the duplex mode and the cyclic-prefix type
 * together with the cell-identity group. A structure whose symbol would start before the window is skipped rather
 * than clamped: a clamped transform would read the wrong 128 samples and could win on them.
 */
class SssDecoder {
public:
    explicit SssDecoder(float sampleRate = kSampleRate) : _sampleRate(sampleRate) {
        if (!(sampleRate > 0.f)) {
            throw std::invalid_argument(std::format("lte: the sample rate must be positive, got {}", sampleRate));
        }
    }

    /**
     * @brief Decode the secondary signal for a primary detection at window index `p`.
     *
     * @param window samples holding at least `p + 128` entries, with as much context before `p` as the structures need
     * @param p index of the primary symbol's first useful sample
     * @param nId2 the root the primary detection carried
     * @param frequencyOffsetHz the offset the primary detection reported, removed before either transform
     */
    [[nodiscard]] SssDecision decode(std::span<const std::complex<float>> window, std::size_t p, std::uint32_t nId2, float frequencyOffsetHz) {
        if (window.size() < p + kSymbolSamples) {
            throw std::invalid_argument(std::format("lte: a window of {} samples cannot hold a symbol at {}", window.size(), p));
        }
        const std::array<std::complex<double>, kSignalLength> reference = pssSequence(nId2);

        derotate(window, p, frequencyOffsetHz);
        _fft.compute(_symbol, _spectrum);

        std::array<std::complex<float>, kSignalLength> channel{};
        for (std::size_t n = 0UZ; n < kSignalLength; ++n) {
            channel[n] = _spectrum[subcarrierBin(n)] * std::conj(std::complex<float>(static_cast<float>(reference[n].real()), static_cast<float>(reference[n].imag())));
        }

        SssDecision best{};
        for (const FrameGeometry& geometry : kStructures) {
            const std::ptrdiff_t start = static_cast<std::ptrdiff_t>(p) + geometry.secondaryOffset();
            if (start < 0) {
                continue;
            }
            derotate(window, static_cast<std::size_t>(start), frequencyOffsetHz);
            _fft.compute(_symbol, _spectrum);

            std::array<float, kSignalLength> soft{};
            for (std::size_t n = 0UZ; n < kSignalLength; ++n) {
                soft[n] = (_spectrum[subcarrierBin(n)] * std::conj(channel[n])).real();
            }

            const SssReading reading = decodeSssValues(std::span<const float, kSignalLength>(soft), nId2);
            if (reading.found && reading.metric > best.metric) {
                best = SssDecision{true, reading.nId1, reading.subframe == 0U ? 0U : 1U, geometry, reading.metric, soft};
            }
        }
        return best;
    }

private:
    /// Copy 128 samples out of the window with the carrier offset removed, keeping the window's own index as the
    /// phase origin so that the primary and secondary symbols share a time base and their common phase cancels.
    void derotate(std::span<const std::complex<float>> window, std::size_t start, float frequencyOffsetHz) {
        const double turn = -2. * std::numbers::pi * static_cast<double>(frequencyOffsetHz) / static_cast<double>(_sampleRate);
        _symbol.resize(kSymbolSamples);
        for (std::size_t n = 0UZ; n < kSymbolSamples; ++n) {
            const std::complex<double> rotated = std::complex<double>(window[start + n]) * std::polar(1.0, turn * static_cast<double>(start + n));
            _symbol[n]                         = std::complex<float>(static_cast<float>(rotated.real()), static_cast<float>(rotated.imag()));
        }
    }

    float                                   _sampleRate{kSampleRate};
    gr::algorithm::FFT<std::complex<float>> _fft{};
    std::vector<std::complex<float>>        _symbol{};
    std::vector<std::complex<float>>        _spectrum{};
};

/// One identification: where the primary symbol was, and what the secondary signal said about it.
struct CellDetection {
    PssDetection primary{};
    SssDecision  secondary{};
};

/// What examining a window did beyond the identifications it confirmed.
struct ExamineCounts {
    std::uint64_t primaryFound{0ULL};      ///< primary detections that cleared their threshold
    std::uint64_t secondaryRejected{0ULL}; ///< of those, the ones no secondary reading confirmed
};

/**
 * @brief The two kernels run together over one window: the whole identification, less the reporting.
 *
 * A window is a run of positions with context on both sides. The first `kMaxSecondaryLookBehind`
 * samples are the context the furthest secondary hypothesis reaches back into and are never
 * themselves candidate positions; the last `kSymbolSamples - 1` are what a primary symbol beginning
 * at the final position occupies. Between them sit exactly `nPositions` candidates, each evaluated
 * once, which is what lets a caller cut a stream into windows that neither miss a symbol nor
 * evaluate one twice.
 *
 * The thresholds are the caller's per call rather than the detector's, because a consumer that
 * sweeps a band spends them differently from one watching a single carrier.
 */
class CellDetector {
public:
    explicit CellDetector(float searchHalfWidthHz = 0.f, float sampleRate = kSampleRate) : _correlator(searchHalfWidthHz, sampleRate), _decoder(sampleRate) {}

    [[nodiscard]] std::size_t hypotheses() const noexcept { return _correlator.hypotheses(); }

    /// @brief Samples a window of `nPositions` candidate positions holds, context on both sides included.
    [[nodiscard]] static constexpr std::size_t windowFor(std::size_t nPositions) noexcept { return kMaxSecondaryLookBehind + nPositions + kSymbolSamples - 1UZ; }

    /**
     * @brief Every identification the window confirms, at most one per root.
     *
     * @return a view of the detector's own storage, valid until the next call
     */
    [[nodiscard]] std::span<const CellDetection> examine(std::span<const std::complex<float>> window, std::size_t nPositions, float pssThreshold, float sssThreshold, ExamineCounts& counts) {
        _found.clear();
        if (window.size() < windowFor(nPositions)) {
            throw std::invalid_argument(std::format("lte: a window of {} samples cannot carry {} positions with their context", window.size(), nPositions));
        }
        const std::array<PssDetection, 3UZ> located = _correlator.search(window.subspan(kMaxSecondaryLookBehind), nPositions);
        for (const PssDetection& detection : located) {
            if (!(detection.metric > pssThreshold)) {
                continue;
            }
            ++counts.primaryFound;
            const SssDecision decision = _decoder.decode(window, kMaxSecondaryLookBehind + detection.position, detection.nId2, detection.frequencyHz);
            if (!decision.confirmed || !(decision.metric > sssThreshold)) {
                ++counts.secondaryRejected;
                continue;
            }
            _found.push_back(CellDetection{detection, decision});
        }
        return _found;
    }

private:
    PssCorrelator              _correlator;
    SssDecoder                 _decoder;
    std::vector<CellDetection> _found{};
};

} // namespace gr::lte

#endif // GNURADIO_ALGORITHM_LTE_SYNC_SIGNALS_HPP
