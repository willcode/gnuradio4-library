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
 * A trajectory shifts the carrier of a signal and *moves* its envelope. This class models the envelope
 * movement. Over a ten-minute low-orbit pass at 7 km/s the one-way delay changes by
 * `7000 * 600 / 299792458 = 14.0097 ms`. That is 672.5 samples at 48 kS/s and 14010 samples at 1 MS/s. A
 * symbol-timing loop that is not given this delay has to track it. Testing that tracking needs a signal that
 * carries the delay.
 *
 * `ArbitraryResampler` already holds the parts a fractional delay needs, with a different phase driver. Its read
 * position advances by a fixed `step`. The read position of a scheduled delay advances by `1 - dtau/dt`. The
 * **prototype design and the arm blend are shared**. `detail::polyphaseArmBank` cuts the prototype the same way
 * for both, and `detail::polyphaseArmBlend` reads it the same way. Only the phase driver differs. Moving the
 * resampler's phase per sample would break a guarantee the resampler states. Its realized ratio is a stated
 * rational that does not drift.
 *
 * **The delay is fixed point, `Q32`**, the same as `kArbitraryFractionBits`. The whole part selects how far back
 * in history to read. The fraction selects the arm and `mu`. A sample index is not a `double`. The delay is not a
 * `double` after the one conversion that produces it. That conversion has a derived bound. The product must fit
 * `std::int64_t`, so the largest delay is `2^63 / 2^32 = 2^31 = 2147483648` samples. That is 44739 s at 48 kS/s
 * and 2147.5 s at 1 MS/s, or four to twelve hours. It is three orders of magnitude beyond the 8.339 ms of a
 * 2500 km slant range. The bound guards against overflow and limits no real use. The `double` seconds at the
 * input are far finer than the fixed point. The spacing of a `double` at 1 s is 2.22e-16 s. Even at
 * `fs = 10^9` that is 2.2e-7 samples, seven orders of magnitude below the `2^-32` the representation keeps.
 *
 * **The interpolation-error bound.** Take a signal band-limited to `f_max = (1 - rolloff)/2` cycles per sample
 * and arms `1/L` apart. The deviation from an ideal fractional delay is `h/2`, `h^2/8` and `h^4 * 9/384` at
 * orders 0, 1 and 3, with `h = 2*pi*f_max/L`. This is the Lagrange remainder. `arbitraryInterpolationError`
 * computes the same closed form. At `rolloff = 0.2` the bounds are these.
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
 * These values bound the *ideal* interpolator. The realized floor is the passband ripple the prototype
 * delivers. A larger bank improves nothing past that point unless the attenuation also rises. The default is
 * `q = 1` at a 60 dB design. `arbitraryBankSize` gives `L = 32` for it.
 *
 * **The bound also holds under a moving delay.** It is derived for a fixed fractional offset. A moving delay
 * compresses the band by `1 - dtau/dt`. The bound then applies with `f_max` replaced by `f_max / (1 - dtau/dt)`.
 * Here `dtau/dt = v/c`. At `|v| <= 30 km/s`, an order of magnitude above any orbital range rate,
 * `|dtau/dt| <= 1.0e-4`. The band then changes by at most 0.01 %, and the `q = 1` bound by 0.02 %. The bound
 * is quoted unchanged in this regime.
 *
 * The per-output cost is one shift, one multiply, one index without `llround`, and `q+1` dot products of `B+1`
 * taps. Complex `T` takes one blended dot product instead. The path has no allocation and no transcendental
 * function. Its one branch chooses between the history and the caller's span.
 */
namespace gr::filter {

/// @brief The largest delay `Q32` samples can carry, `2^63 / 2^32`. The file note gives it in seconds.
inline constexpr std::uint64_t kMaxFractionalDelaySamples = 1ULL << 31;

/**
 * @brief The prototype for a pure delay over a bank of @p bankSize arms.
 *
 * This is `designArbitraryResampler` with `minRate = 1`, the design for a pure delay. The stopband edge is
 * `0.5/L`. The bank neither decimates nor interpolates. It only shifts. The function has its own name. A caller
 * that builds a delay line need not know that the rate to pass is the identity.
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
 * @brief A span of delays in seconds converted to `Q32` samples.
 *
 * The delays are in the form `gr::timing::DelaySchedule::valuesFor` writes. This function converts a delay from
 * `double` to fixed point. It is a free function and not a step inside `process`. The schedule walk and the delay line
 * stay independent. The line takes fixed point from any source. A caller with no schedule does not need one.
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
 * @brief A variable fractional-delay line with one output per (input sample, delay) pair.
 *
 * The output is phase continuous and does not depend on how the input is split into calls. The line holds the
 * history a delay needs. Its output is a pure function of the absolute sample index and the delay commanded at
 * that index. A caller that passes one span and a caller that passes a hundred get the same samples to the bit.
 * A delay whose value moves needs this property. A naive ring buffer loses it at every call boundary.
 *
 * **The lag, and the source of the extra sample.** At a commanded delay of zero the output lags the input by
 * `latencySamples()`, which is `1 + groupDelaySamples()`. The group delay is the prototype's own, `(N-1)/(2L)`
 * input samples. It is stated and not compensated. The Lagrange weights add no delay, because they are exact at
 * their nodes. The extra whole sample comes from the bank. The first tap of the wrap arm reads one input sample
 * past the window's anchor. The anchor is therefore placed one sample behind the output. Every read then stays
 * inside the caller's span. That sample makes the output independent of the call split. Without it the last
 * output of each call would depend on the first sample of the next call.
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
     * @p order is 0 (nearest arm), 1 (linear) or 3 (cubic Lagrange). The resampler offers the same three, for
     * the same reason. The even orders put the interpolation nodes off center for no gain. The weights above
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

    /// @brief Clear the history. The next `process` starts a new stream on silence.
    void reset() { std::ranges::fill(_window, T{}); }

    /**
     * @brief Install a new prototype and clear the history.
     *
     * The window length follows the prototype. The held samples then do not line up with the taps that read
     * them. Keeping them would put an unexplained transient in the stream. A bank rebuild therefore clears the
     * history, and this function states that.
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
     * A table that commands a deeper delay than the previous one needs a longer history. The held samples still
     * precede the next sample. They line up with the same taps, because the prototype is unchanged. The window
     * grows, and the held samples move to its end. The older part reads as silence before the stream. A request
     * the line already covers changes nothing. A schedule that alternates between two depths therefore does not
     * reallocate its buffer each time.
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
     * The value is `maxDelayWhole + tapsPerArm + 1`. It is bounded, because a schedule clamps at its ends. The
     * largest delay over all time is the largest delay in the table. For example, a 2500 km slant range is
     * 8.339 ms. At 1 MS/s the history is 8339 samples plus the taps, about 67 KiB for `std::complex<float>`. A
     * caller that needs only the *variation* subtracts a constant from the table. The history then covers the
     * variation alone.
     */
    [[nodiscard]] std::size_t historyNeeded(std::uint64_t maxDelayWhole) const noexcept { return static_cast<std::size_t>(maxDelayWhole) + _windowLength + 1UZ; }

    [[nodiscard]] std::size_t bankSize() const noexcept { return _bankSize; }
    [[nodiscard]] int         order() const noexcept { return _order; }
    /// @brief `B = ceil(N/L)`. An arm stores `B+1` taps. The extra tap is the wrap term, zero below arm `L`.
    [[nodiscard]] std::size_t tapsPerArm() const noexcept { return _windowLength; }
    /// @brief `N`, the prototype as supplied, before padding to a multiple of `L`.
    [[nodiscard]] std::size_t prototypeLength() const noexcept { return _prototypeLength; }
    /// @brief `(N-1)/(2L)` input samples, the prototype's own, stated and not compensated.
    [[nodiscard]] double groupDelaySamples() const noexcept { return 0.5 * static_cast<double>(_prototypeLength - 1UZ) / static_cast<double>(_bankSize); }
    /// @brief The output's lag behind the input at a commanded delay of zero, `1 + groupDelaySamples()`.
    [[nodiscard]] double        latencySamples() const noexcept { return 1. + groupDelaySamples(); }
    [[nodiscard]] std::uint64_t maxDelaySamples() const noexcept { return _maxWholeDelay; }
    [[nodiscard]] std::size_t   historySamples() const noexcept { return _history; }

    /**
     * @brief One output per (input sample, delay) pair, `out[k]` the input at `k - latency - delayQ32[k]/2^32`.
     *
     * @param in        the input samples
     * @param delayQ32  one delay per input sample, in whole samples in the high 32 bits
     * @param out       at least `in.size()` outputs
     *
     * Throws for a delay whose whole part exceeds the history the line was sized for. It does not read past
     * the history. The check is one pass of compares over the delay span. The per-sample cost is in the dot
     * products, and the check adds little to it.
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

        // The head of `in` is copied into the tail of the window. A read across the boundary between the history
        // and the caller's span is then still one contiguous run of B+1 samples.
        const std::size_t head = std::min(n, _windowLength);
        std::copy_n(in.begin(), head, _window.begin() + static_cast<std::ptrdiff_t>(_history));

        const float* const       bank    = _bank.data();
        const SampleValue* const windowF = reinterpret_cast<const SampleValue*>(_window.data());
        const SampleValue* const inF     = reinterpret_cast<const SampleValue*>(in.data());
        const std::int64_t       history = static_cast<std::int64_t>(_history);
        const std::int64_t       reach   = static_cast<std::int64_t>(_windowLength) - 1;

        for (std::size_t k = 0UZ; k < n; ++k) {
            // The commanded delay, plus the one whole sample for the reach of the wrap arm.
            const std::int64_t  position = (static_cast<std::int64_t>(k) << kArbitraryFractionBits) - static_cast<std::int64_t>(delayQ32[k]) - static_cast<std::int64_t>(kArbitraryOne);
            const std::int64_t  anchor   = position >> kArbitraryFractionBits; // an arithmetic shift, which floors on both sides of zero
            const std::uint64_t frac     = static_cast<std::uint64_t>(position) & kArbitraryMask;

            const std::uint64_t armPos = frac * _bankSize;
            const std::uint64_t arm    = armPos >> kArbitraryFractionBits;
            const double        mu     = static_cast<double>(armPos & kArbitraryMask) / static_cast<double>(kArbitraryOne);

            const std::int64_t       base = anchor - reach + history;
            const SampleValue* const x    = (base < history) ? windowF + kSampleLanes * static_cast<std::size_t>(base) : inF + kSampleLanes * static_cast<std::size_t>(base - history);

            out[k] = detail::polyphaseArmBlend<T>(bank + (arm + 1UZ) * armStep, x, armStep, mu, _order);
        }

        // The history is the last `_history` input samples. They come from this call alone, or from this call
        // and the samples already held.
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
