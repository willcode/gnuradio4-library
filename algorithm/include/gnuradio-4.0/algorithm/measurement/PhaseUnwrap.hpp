#ifndef GNURADIO_ALGORITHM_PHASE_UNWRAP_HPP
#define GNURADIO_ALGORITHM_PHASE_UNWRAP_HPP

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
#include <string_view>

/**
 * @brief Instantaneous phase carried as an exact integer cycle count beside a fractional phase, never as a float sum.
 *
 * The instantaneous phase of a complex stream is `atan2(im, re)`, wrapped to `[-pi, pi)`. Unwrapping adds back
 * the whole turns, and the only question is what carries the turns. The answer is arithmetic:
 *
 * - A **`float`** accumulator holding total radians has a 24-bit significand. At `2*pi*10^9` radians its spacing
 *   is **512 radians — 81.5 whole cycles per representable step**. It stops resolving a milliradian past about
 *   **1335 cycles**, and past roughly `2^23` radians it cannot represent adjacent cycles at all. It does not
 *   degrade; it stops working, early.
 * - A **`double`** accumulator survives further — spacing `9.54e-7 rad` at `2*pi*10^9` radians, and a nanoradian
 *   is lost past about `7.2e5` cycles — but it still degrades monotonically, and every reading's resolution
 *   depends on how long the block has been running.
 * - An **`std::int64_t` cycle count beside a fractional phase** has neither property. The count is exact for as
 *   many turns as it can hold, and the fraction keeps a `float`'s spacing at `pi` (`2.38e-7 rad`) forever,
 *   because it never grows.
 *
 * The count's range is not a limit in any physical sense. The fastest a stream can turn and still be unwrappable
 * is `fs/2` cycles per second, so `2^63 - 1` cycles takes **584.55 years at `fs = 10^9` S/s** and 584 554 years
 * at 1 MS/s. The saturating check is carried anyway and counted, on the principle that a claim of
 * unreachability should be cheap to enforce.
 *
 * **The two parts are never summed here**, and that is the decision rather than an omission. Forming
 * `2*pi*cycles + phase` requires a floating type, and choosing one would reintroduce exactly the loss this
 * exists to avoid, at the point where nothing downstream can recover it. A consumer that wants one number
 * computes it in whatever type it can afford, having been handed the parts exactly; a consumer that wants
 * frequency differences the count in integers and the fraction in floats and never touches a large number at
 * all. `unwrappedRadians()` exists as a convenience for the current sample only, with its own spacing stated at
 * the accessor.
 *
 * **The precondition, derived.** The unwrap is correct exactly when the true phase advances by less than `pi`
 * per sample, that is when `|f| < fs/2`. Below that bound the wrapped difference determines the turn uniquely;
 * at or above it two different turn counts produce the same wrapped difference and no algorithm operating on the
 * samples alone can choose. That is Nyquist restated for phase, and it is not something this can check — it can
 * only observe that a step came close, which is what `maxStepFraction` and `nSuspectSteps()` are: an
 * observability hook, not a detector.
 */
namespace gr::measurement {

/// Where the count and the reported phase start.
enum class UnwrapOrigin : std::uint8_t {
    first_sample, ///< `cycles = 0` at the first sample, whose phase is reported as it is
    zero          ///< additionally subtract the first sample's phase, so the output starts at exactly zero
};

[[nodiscard]] inline constexpr std::string_view unwrapOriginName(UnwrapOrigin origin) noexcept { return origin == UnwrapOrigin::zero ? "zero" : "first_sample"; }

/// @brief The two spellings a setting may carry, and nothing else.
[[nodiscard]] inline constexpr std::optional<UnwrapOrigin> unwrapOriginFrom(std::string_view name) noexcept {
    if (name == "first_sample") {
        return UnwrapOrigin::first_sample;
    }
    if (name == "zero") {
        return UnwrapOrigin::zero;
    }
    return std::nullopt;
}

class CycleUnwrapper {
public:
    static constexpr float  kPi    = std::numbers::pi_v<float>;
    static constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

    CycleUnwrapper() = default;

    explicit CycleUnwrapper(UnwrapOrigin origin, double maxStepFraction = 0.9) : _origin(origin), _maxStepFraction(maxStepFraction) {
        if (!(maxStepFraction > 0.) || !(maxStepFraction <= 1.)) {
            throw std::invalid_argument(std::format("gr::measurement::CycleUnwrapper: max step fraction {} is not in (0, 1]", maxStepFraction));
        }
    }

    /// @brief Back to the state `start()` leaves: no origin, no count, no counters.
    void reset() noexcept {
        _started  = false;
        _cycles   = 0LL;
        _phase    = 0.f;
        _base     = 0.f;
        _index    = 0ULL;
        _suspect  = 0ULL;
        _resets   = 0ULL;
        _saturate = 0ULL;
        _lastSuspect.reset();
    }

    /**
     * @brief The reset a gap in the stream calls for: the count returns to zero and the origin is re-taken.
     *
     * After a gap the cycle count is no longer the number of turns the underlying signal made, because the turns
     * during the gap were not seen. Continuing it would be a number that looks exact and is wrong. Resetting
     * makes the discontinuity visible as a step to zero, and it is counted. A caller who would rather keep a
     * running count through gaps and account for them elsewhere simply does not call this, and the count's
     * meaning is then the caller's.
     *
     * Unlike `reset()` the counters are kept, because they are the record of what the stream did.
     */
    void markDiscontinuity() noexcept {
        _started = false;
        _cycles  = 0LL;
        _phase   = 0.f;
        _base    = 0.f;
        ++_resets;
    }

    /**
     * @brief One `(cycles, phase)` pair per input sample.
     *
     * @param in     the complex stream
     * @param cycles whole turns since the origin, exact
     * @param phase  the residual, in radians on `[-pi, pi)`
     */
    void process(std::span<const std::complex<float>> in, std::span<std::int64_t> cycles, std::span<float> phase) {
        if (cycles.size() < in.size() || phase.size() < in.size()) {
            throw std::invalid_argument(std::format("gr::measurement::CycleUnwrapper::process: {} samples against {} cycles and {} phases", in.size(), cycles.size(), phase.size()));
        }

        const float suspect = static_cast<float>(_maxStepFraction) * kPi;
        for (std::size_t k = 0UZ; k < in.size(); ++k) {
            const float raw = std::atan2(in[k].imag(), in[k].real());
            if (!_started) {
                _started = true;
                _base    = (_origin == UnwrapOrigin::zero) ? raw : 0.f;
                _phase   = wrap(raw - _base);
            } else {
                const float current = wrap(raw - _base);
                const float delta   = current - _phase;
                const float step    = (delta > kPi) ? (delta - 2.f * kPi) : ((delta < -kPi) ? (delta + 2.f * kPi) : delta);
                if (delta > kPi) {
                    turn(-1LL);
                } else if (delta < -kPi) {
                    turn(+1LL);
                }
                if (std::abs(step) > suspect) {
                    ++_suspect;
                    _lastSuspect = _index;
                }
                _phase = current;
            }
            cycles[k] = _cycles;
            phase[k]  = _phase;
            ++_index;
        }
    }

    [[nodiscard]] std::int64_t cycles() const noexcept { return _cycles; }
    [[nodiscard]] float        phase() const noexcept { return _phase; }

    /// @brief `2*pi*cycles + phase` for the current sample only, as a convenience and not as the contract: its
    /// spacing is `9.54e-7 rad` at `10^9` cycles, which is precisely why the two parts travel separately.
    [[nodiscard]] double unwrappedRadians() const noexcept { return kTwoPi * static_cast<double>(_cycles) + static_cast<double>(_phase); }

    [[nodiscard]] bool                         started() const noexcept { return _started; }
    [[nodiscard]] UnwrapOrigin                 origin() const noexcept { return _origin; }
    [[nodiscard]] double                       maxStepFraction() const noexcept { return _maxStepFraction; }
    [[nodiscard]] std::uint64_t                processed() const noexcept { return _index; }
    [[nodiscard]] std::uint64_t                nSuspectSteps() const noexcept { return _suspect; }
    [[nodiscard]] std::uint64_t                nResets() const noexcept { return _resets; }
    [[nodiscard]] std::uint64_t                nSaturations() const noexcept { return _saturate; }
    [[nodiscard]] std::optional<std::uint64_t> lastSuspectIndex() const noexcept { return _lastSuspect; }

private:
    /// `atan2` already returns `[-pi, pi]`, but `raw - _base` under the `zero` origin does not.
    [[nodiscard]] static float wrap(float radians) noexcept {
        float out = radians;
        while (out >= kPi) {
            out -= 2.f * kPi;
        }
        while (out < -kPi) {
            out += 2.f * kPi;
        }
        return out;
    }

    void turn(std::int64_t by) noexcept {
        if ((by > 0 && _cycles == std::numeric_limits<std::int64_t>::max()) || (by < 0 && _cycles == std::numeric_limits<std::int64_t>::min())) {
            ++_saturate;
            return;
        }
        _cycles += by;
    }

    UnwrapOrigin                 _origin{UnwrapOrigin::first_sample};
    double                       _maxStepFraction{0.9};
    bool                         _started{false};
    std::int64_t                 _cycles{0LL};
    float                        _phase{0.f};
    float                        _base{0.f};
    std::uint64_t                _index{0ULL};
    std::uint64_t                _suspect{0ULL};
    std::uint64_t                _resets{0ULL};
    std::uint64_t                _saturate{0ULL};
    std::optional<std::uint64_t> _lastSuspect{};
};

} // namespace gr::measurement

#endif // GNURADIO_ALGORITHM_PHASE_UNWRAP_HPP
