#ifndef GNURADIO_FRACTIONAL_DELAY_HPP
#define GNURADIO_FRACTIONAL_DELAY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/ArbitraryResampler.hpp>

/**
 * @brief A delay of a signal by a time that changes while the signal runs, over the arbitrary resampler's own bank.
 *
 * A trajectory does two things to a signal and this is the second: the carrier is shifted, and the envelope is
 * *moved*. Over a ten-minute low-orbit pass at 7 km/s the one-way delay changes by
 * `7000 * 600 / 299792458 = 14.0097 ms` — 672.5 samples at 48 kS/s and 14010 at 1 MS/s. A symbol-timing loop
 * that is not told about that has to track it, and a graph that wants to test whether it can must be able to
 * generate it.
 *
 * Everything a fractional delay needs is already in `ArbitraryResampler`, driven by the wrong thing: its read
 * position advances by a fixed `step`, and a scheduled delay's advances by `1 - dtau/dt`. So the **prototype
 * design and the arm blend are shared** — `detail::polyphaseArmBank` cuts the prototype the same way for both
 * and `detail::polyphaseArmBlend` reads it the same way — and only the phase driver differs. Reaching into the
 * resampler to move its phase per sample instead would break exactly the guarantee that kernel spends a section
 * establishing, that its realized ratio is a stated rational which never drifts.
 *
 * **The delay is fixed point, `Q32`**, matching `kArbitraryFractionBits`: the whole part selects how far back in
 * history to read and the fraction selects the arm and `mu`. A sample index is never a `double` and the delay is
 * never a `double` after the one conversion that produces it. The bound that conversion carries is derived: the
 * product must fit `std::int64_t`, so the largest delay is `2^63 / 2^32 = 2^31 = 2147483648` samples — 44739 s
 * at 48 kS/s and 2147.5 s at 1 MS/s, four to twelve hours, three orders beyond the 8.339 ms a 2500 km slant
 * range costs. It is a guard, not a limit. The `double` seconds that feed it are far finer than the fixed point
 * they land in: a `double`'s spacing at 1 s is 2.22e-16 s, which even at `fs = 10^9` is 2.2e-7 samples, seven
 * orders below the `2^-32` the representation keeps.
 *
 * **The interpolation-error bound.** For a signal band-limited to `f_max = (1 - rolloff)/2` cycles per sample
 * and arms `1/L` apart, the deviation from an ideal fractional delay is `h/2`, `h^2/8` and `h^4 * 9/384` at
 * orders 0, 1 and 3, with `h = 2*pi*f_max/L` — the Lagrange remainder, and the closed form
 * `arbitraryInterpolationError` already computes. At `rolloff = 0.2`:
 *
 * ```
 *    L    q = 0                q = 1                 q = 3
 *    8    1.571e-1 (-16.1 dB)  1.234e-2 (-38.2 dB)   2.283e-4  (-72.8 dB)
 *   16    7.854e-2 (-22.1 dB)  3.084e-3 (-50.2 dB)   1.427e-5  (-96.9 dB)
 *   32    3.927e-2 (-28.1 dB)  7.711e-4 (-62.3 dB)   8.918e-7 (-121.0 dB)
 *   64    1.964e-2 (-34.1 dB)  1.928e-4 (-74.3 dB)   5.574e-8 (-145.1 dB)
 *  128    9.818e-3 (-40.2 dB)  4.819e-5 (-86.3 dB)   3.484e-9 (-169.2 dB)
 * ```
 *
 * These bound the *ideal* interpolator; the realized floor is the prototype's own delivered passband ripple,
 * so a bank sized past that point buys nothing without also raising the attenuation. The default here is
 * `q = 1` at a 60 dB design, hence `L = 32`, which `arbitraryBankSize` reads off.
 *
 * **Why the bound holds under a moving delay.** It is derived for a fixed fractional offset. A moving delay
 * compresses the band by `1 - dtau/dt`, so it applies with `f_max` replaced by `f_max / (1 - dtau/dt)`. Since
 * `dtau/dt = v/c`, at `|v| <= 30 km/s` — an order above any orbital range rate — `|dtau/dt| <= 1.0e-4`, the band
 * changes by at most 0.01 % and the `q = 1` bound by 0.02 %. It is therefore quoted unchanged in that regime.
 *
 * This is the hot path: per output one shift, one multiply, one `llround`-free index and `q+1` dot products of
 * `B+1` taps, or one blended dot product for complex `T`. No allocation, no transcendental, no branch beyond
 * the one that chooses between the history and the caller's own span.
 */
namespace gr::filter {

/// @brief The largest delay `Q32` samples can carry: `2^63 / 2^32`. See the class note for what it is in seconds.
inline constexpr std::uint64_t kMaxFractionalDelaySamples = 1ULL << 31;

/**
 * @brief The prototype for a pure delay over a bank of @p bankSize arms.
 *
 * `designArbitraryResampler` with `minRate = 1`, which is what a pure delay is: the stopband edge is `0.5/L`
 * and the bank neither decimates nor interpolates, it only shifts. Named here so that a caller building a delay
 * line does not have to know that the rate it would otherwise pass is the identity.
 */
[[nodiscard]] inline ResamplerDesign designFractionalDelay(std::size_t bankSize, double rolloff = 0.2, double attenuationDb = 60.0, double maxRippleDb = 0.1) { return designArbitraryResampler(bankSize, 1.0, rolloff, attenuationDb, maxRippleDb); }

/// @brief A delay in whole and fractional samples as the `Q32` the line consumes.
[[nodiscard]] inline std::uint64_t fractionalDelayQ32(double delaySamples) {
    if (!std::isfinite(delaySamples) || delaySamples < 0.) {
        throw std::invalid_argument(std::format("gr::filter::fractionalDelayQ32: a delay of {} samples is not a finite non-negative number", delaySamples));
    }
    if (delaySamples >= static_cast<double>(kMaxFractionalDelaySamples)) {
        throw std::invalid_argument(std::format("gr::filter::fractionalDelayQ32: {} samples is at or past the {} the fixed point carries", delaySamples, kMaxFractionalDelaySamples));
    }
    return static_cast<std::uint64_t>(std::llround(delaySamples * static_cast<double>(kArbitraryOne)));
}

/**
 * @brief A span of delays in seconds — as `gr::timing::DelaySchedule::valuesFor` writes them — as `Q32` samples.
 *
 * The one place a delay stops being a `double`. It is a free function rather than a step inside `process` so
 * that the schedule walk and the delay line stay independent of one another: the line takes fixed point from
 * wherever it comes, and a caller with no schedule at all is not made to acquire one.
 */
inline void fractionalDelayQ32(std::span<const double> delaySeconds, double sampleRateHz, std::span<std::uint64_t> out) {
    if (out.size() != delaySeconds.size()) {
        throw std::invalid_argument(std::format("gr::filter::fractionalDelayQ32: {} delays against {} outputs", delaySeconds.size(), out.size()));
    }
    if (!std::isfinite(sampleRateHz) || !(sampleRateHz > 0.)) {
        throw std::invalid_argument(std::format("gr::filter::fractionalDelayQ32: a sample rate of {} Hz is not positive and finite", sampleRateHz));
    }
    for (std::size_t i = 0UZ; i < delaySeconds.size(); ++i) {
        out[i] = fractionalDelayQ32(delaySeconds[i] * sampleRateHz);
    }
}

/**
 * @brief The variable fractional-delay line: one output per (input sample, delay) pair, phase continuous and chunk independent.
 *
 * The line carries the history a delay needs, so the stream it produces is a pure function of the absolute
 * sample index and the delay commanded at it. A caller that hands over one span and a caller that hands over a
 * hundred get the same samples to the bit, which is the property a delay whose value moves has to have and the
 * one a naive ring buffer loses at every call boundary.
 *
 * **The lag, and where the extra sample comes from.** At a commanded delay of zero the output is the input
 * `latencySamples()` behind, which is `1 + groupDelaySamples()`. The group delay is the prototype's own,
 * `(N-1)/(2L)` input samples, stated and not compensated — the Lagrange weights add none of their own, being
 * exact at their nodes. The extra whole sample is the bank's: the wrap arm's first tap reads one input sample
 * past the window's anchor, so the anchor is placed one sample behind the output to keep every read inside the
 * span the caller handed over. Paying it is what makes the stream chunk independent; not paying it would make
 * the last output of every chunk depend on the first sample of the next.
 */
template<typename T>
requires PolyphaseSample<T>
class FractionalDelayLine {
    using Traits      = detail::PolyphaseTraits<T, float>;
    using SampleValue = typename Traits::SampleValue;

    static constexpr std::size_t kSampleLanes = Traits::kSampleLanes;

public:
    using sample_type = T;

    /**
     * @brief A bank of @p bankSize arms at interpolation @p order over @p taps, holding history for @p maxWholeDelay samples.
     *
     * @p order is 0 (nearest arm), 1 (linear) or 3 (cubic Lagrange), the same three the resampler offers and for
     * the same reason: the even orders put the interpolation nodes off-center for no gain, and the weights above
     * order 3 cost more than the extra arms they save.
     */
    FractionalDelayLine(std::size_t bankSize, int order, std::span<const float> taps, std::uint64_t maxWholeDelay = 0ULL) : _bankSize(bankSize), _order(order) {
        if (bankSize == 0UZ) {
            throw std::invalid_argument("FractionalDelayLine: the bank must have at least one arm");
        }
        if (order != 0 && order != 1 && order != 3) {
            throw std::invalid_argument("FractionalDelayLine: the interpolation order is 0, 1 or 3");
        }
        if (taps.empty()) {
            throw std::invalid_argument("FractionalDelayLine: no taps");
        }
        build(taps);
        setMaxDelay(maxWholeDelay);
    }

    /// @brief Forget the history: the next `process` starts a new stream on silence.
    void reset() { std::ranges::fill(_window, T{}); }

    /**
     * @brief Install a new prototype and clear the history.
     *
     * The window length follows the prototype, so the samples the line was holding no longer line up with the
     * taps that would read them; carrying them across would put a transient in the stream that no caller could
     * account for. The reset is the honest cost of a bank rebuild and is stated rather than hidden.
     */
    void setTaps(std::span<const float> taps) {
        if (taps.empty()) {
            throw std::invalid_argument("FractionalDelayLine: no taps");
        }
        build(taps);
        setMaxDelay(_maxWholeDelay);
    }

    /// @brief Size the history for delays up to @p maxWholeDelay whole samples, and clear it.
    void setMaxDelay(std::uint64_t maxWholeDelay) {
        if (maxWholeDelay >= kMaxFractionalDelaySamples) {
            throw std::invalid_argument(std::format("FractionalDelayLine: a history for {} samples is at or past the {} the fixed point carries", maxWholeDelay, kMaxFractionalDelaySamples));
        }
        _maxWholeDelay = maxWholeDelay;
        _history       = historyNeeded(maxWholeDelay);
        _window.assign(_history + _windowLength, T{});
    }

    /**
     * @brief Reach back for delays up to @p maxWholeDelay whole samples, keeping the history already held.
     *
     * A table that commands a deeper delay than the one before it needs a longer history, and the samples the
     * line is holding are still the samples that precede the next one: they line up with the same taps, because
     * the prototype did not change. So the window grows and the held samples move to its end, the older part
     * reading as the silence that preceded the stream. A request the line already covers is left alone, so a
     * schedule that alternates between two depths does not thrash its buffer.
     */
    void growMaxDelay(std::uint64_t maxWholeDelay) {
        if (maxWholeDelay <= _maxWholeDelay) {
            return;
        }
        if (maxWholeDelay >= kMaxFractionalDelaySamples) {
            throw std::invalid_argument(std::format("FractionalDelayLine: a history for {} samples is at or past the {} the fixed point carries", maxWholeDelay, kMaxFractionalDelaySamples));
        }
        const std::size_t grown = historyNeeded(maxWholeDelay);
        std::vector<T>    next(grown + _windowLength, T{});
        const std::size_t kept = std::min(_history, grown);
        std::copy_n(_window.begin() + static_cast<std::ptrdiff_t>(_history - kept), kept, next.begin() + static_cast<std::ptrdiff_t>(grown - kept));
        _window        = std::move(next);
        _maxWholeDelay = maxWholeDelay;
        _history       = grown;
    }

    /**
     * @brief The history a delay of @p maxDelayWhole whole samples needs, in samples.
     *
     * `maxDelayWhole + tapsPerArm + 1`. Bounded, because a schedule clamps at its ends, so the largest delay
     * over all time is the largest delay in the table. As a number: a 2500 km slant range is 8.339 ms, so at
     * 1 MS/s the history is 8339 samples plus the taps — about 67 KiB for `std::complex<float>` — and a caller
     * who wants only the *variation* subtracts a constant from the table and pays for the variation alone.
     */
    [[nodiscard]] std::size_t historyNeeded(std::uint64_t maxDelayWhole) const noexcept { return static_cast<std::size_t>(maxDelayWhole) + _windowLength + 1UZ; }

    [[nodiscard]] std::size_t bankSize() const noexcept { return _bankSize; }
    [[nodiscard]] int         order() const noexcept { return _order; }
    /// @brief `B = ceil(N/L)`; an arm stores `B+1`, the extra one being the wrap term that is zero below arm `L`.
    [[nodiscard]] std::size_t tapsPerArm() const noexcept { return _windowLength; }
    /// @brief `N`, the prototype as supplied, before padding to a multiple of `L`.
    [[nodiscard]] std::size_t prototypeLength() const noexcept { return _prototypeLength; }
    /// @brief `(N-1)/(2L)` input samples, the prototype's own, stated and not compensated.
    [[nodiscard]] double groupDelaySamples() const noexcept { return 0.5 * static_cast<double>(_prototypeLength - 1UZ) / static_cast<double>(_bankSize); }
    /// @brief The output's lag behind the input at a commanded delay of zero: `1 + groupDelaySamples()`.
    [[nodiscard]] double        latencySamples() const noexcept { return 1. + groupDelaySamples(); }
    [[nodiscard]] std::uint64_t maxDelaySamples() const noexcept { return _maxWholeDelay; }
    [[nodiscard]] std::size_t   historySamples() const noexcept { return _history; }

    /**
     * @brief One output per (input sample, delay) pair: `out[k]` is the input at `k - latency - delayQ32[k]/2^32`.
     *
     * @param in        the input samples
     * @param delayQ32  one delay per input sample, in whole samples in the high 32 bits
     * @param out       at least `in.size()` outputs
     *
     * Refuses a delay whose whole part exceeds the history the line was sized for, rather than reading what is
     * not there: the check is one pass of compares over the delay span, which is off the dot products the
     * per-sample cost is actually made of.
     */
    void process(std::span<const T> in, std::span<const std::uint64_t> delayQ32, std::span<T> out) {
        if (in.size() != delayQ32.size()) {
            throw std::invalid_argument(std::format("FractionalDelayLine::process: {} samples against {} delays — the two must be paired", in.size(), delayQ32.size()));
        }
        if (out.size() < in.size()) {
            throw std::invalid_argument(std::format("FractionalDelayLine::process: {} outputs for {} samples", out.size(), in.size()));
        }
        if (in.empty()) {
            return;
        }

        std::uint64_t worst = 0ULL;
        for (const std::uint64_t delay : delayQ32) {
            worst = std::max(worst, delay);
        }
        if ((worst >> kArbitraryFractionBits) > _maxWholeDelay) {
            throw std::invalid_argument(std::format("FractionalDelayLine::process: a delay of {} whole samples past the {} this line holds history for", worst >> kArbitraryFractionBits, _maxWholeDelay));
        }

        const std::size_t n       = in.size();
        const std::size_t armStep = _windowLength + 1UZ;

        // The head of `in` copied into the tail of the window, so a read that straddles the boundary between the
        // history and the caller's span is still one contiguous run of B+1 samples.
        const std::size_t head = std::min(n, _windowLength);
        std::copy_n(in.begin(), head, _window.begin() + static_cast<std::ptrdiff_t>(_history));

        const float* const       bank    = _bank.data();
        const SampleValue* const windowF = reinterpret_cast<const SampleValue*>(_window.data());
        const SampleValue* const inF     = reinterpret_cast<const SampleValue*>(in.data());
        const std::int64_t       history = static_cast<std::int64_t>(_history);
        const std::int64_t       reach   = static_cast<std::int64_t>(_windowLength) - 1;

        for (std::size_t k = 0UZ; k < n; ++k) {
            // The commanded delay, plus the one whole sample the wrap arm's reach costs.
            const std::int64_t  position = (static_cast<std::int64_t>(k) << kArbitraryFractionBits) - static_cast<std::int64_t>(delayQ32[k]) - static_cast<std::int64_t>(kArbitraryOne);
            const std::int64_t  anchor   = position >> kArbitraryFractionBits; // arithmetic, so it floors on both sides of zero
            const std::uint64_t frac     = static_cast<std::uint64_t>(position) & kArbitraryMask;

            const std::uint64_t armPos = frac * _bankSize;
            const std::uint64_t arm    = armPos >> kArbitraryFractionBits;
            const double        mu     = static_cast<double>(armPos & kArbitraryMask) / static_cast<double>(kArbitraryOne);

            const std::int64_t       base = anchor - reach + history;
            const SampleValue* const x    = (base < history) ? windowF + kSampleLanes * static_cast<std::size_t>(base) : inF + kSampleLanes * static_cast<std::size_t>(base - history);

            out[k] = detail::polyphaseArmBlend<T>(bank + (arm + 1UZ) * armStep, x, armStep, mu, _order);
        }

        // The history is the last `_history` input samples, whether they came from this call alone or from this
        // call and what was already held.
        if (n >= _history) {
            std::copy_n(in.end() - static_cast<std::ptrdiff_t>(_history), _history, _window.begin());
        } else {
            std::copy_n(_window.begin() + static_cast<std::ptrdiff_t>(n), _history - n, _window.begin());
            std::copy_n(in.begin(), n, _window.begin() + static_cast<std::ptrdiff_t>(_history - n));
        }
    }

private:
    void build(std::span<const float> taps) {
        _prototypeLength      = taps.size();
        detail::ArmBank built = detail::polyphaseArmBank(taps, _bankSize, _order);
        _windowLength         = built.windowLength;
        _bank                 = std::move(built.taps);
    }

    std::size_t        _bankSize;
    int                _order;
    std::size_t        _prototypeLength = 0UZ;
    std::size_t        _windowLength    = 0UZ; ///< `B`
    std::vector<float> _bank;                  ///< arm-major, each arm reversed, adjacent arms adjacent
    std::uint64_t      _maxWholeDelay = 0ULL;
    std::size_t        _history       = 0UZ;
    std::vector<T>     _window; ///< `_history` past samples, then room for the head of the call
};

} // namespace gr::filter

#endif // GNURADIO_FRACTIONAL_DELAY_HPP
