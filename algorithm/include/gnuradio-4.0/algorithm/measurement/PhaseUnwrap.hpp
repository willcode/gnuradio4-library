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
 * @brief Instantaneous phase as an exact integer cycle count beside a fractional phase, not as a float sum.
 *
 * The instantaneous phase of a complex stream is `atan2(im, re)`, wrapped to `[-pi, pi)`. Unwrapping adds back
 * the whole turns. The accumulator that holds those turns sets the accuracy.
 *
 * - A **`float`** accumulator of total radians has a 24-bit significand. At `2*pi*10^9` radians its spacing is
 *   **512 radians, or 81.5 whole cycles per representable step**. It stops resolving a milliradian past about
 *   **1335 cycles**. Past roughly `2^23` radians it cannot represent adjacent cycles at all. It fails early and
 *   does not degrade gradually.
 * - A **`double`** accumulator lasts longer. Its spacing at `2*pi*10^9` radians is `9.54e-7 rad`, and it loses a
 *   nanoradian past about `7.2e5` cycles. It still degrades monotonically. The resolution of every reading then
 *   depends on how long the accumulator has run.
 * - An **`std::int64_t` cycle count beside a fractional phase** has neither problem. The count is exact for as many
 *   turns as it can hold. The fraction keeps a `float`'s spacing at `pi` (`2.38e-7 rad`) for all time, because it
 *   does not grow.
 *
 * The count's range is no physical limit. The fastest a stream can turn and still be unwrapped is `fs/2` cycles per
 * second. `2^63 - 1` cycles then take **584.55 years at `fs = 10^9` S/s** and 584 554 years at 1 MS/s. The saturating
 * check is still made and counted. The check costs little and enforces the claim that the limit is unreachable.
 *
 * **The two parts are not summed here**, by design. Forming `2*pi*cycles + phase` requires a floating type.
 * Any choice of type would bring back the loss this class avoids, at a point where the loss cannot be recovered.
 * The class returns both parts exactly. A caller that needs one number forms it in a type of its choice. A caller
 * that needs frequency differences the count in integers and the fraction in floats, with no large number at
 * all. `unwrappedRadians()` is a convenience for the current sample only. Its spacing is stated at the accessor.
 *
 * **The precondition, derived.** The unwrap is correct exactly when the true phase advances by less than `pi`
 * per sample, that is when `|f| < fs/2`. Below that bound the wrapped difference determines the turn uniquely.
 * At or above it, two different turn counts produce the same wrapped difference. No algorithm that sees only the
 * samples can choose between them. This is the Nyquist limit restated for phase. The class cannot check it. It
 * can only observe that a step came close to the limit. `maxStepFraction` and `nSuspectSteps()` report that
 * observation. They are an observability hook and not a detector.
 */
namespace gr::measurement {

/// Where the count and the reported phase start.
enum class UnwrapOrigin : std::uint8_t {
    first_sample, ///< `cycles = 0` at the first sample, whose phase is reported as it is
    zero          ///< also subtract the first sample's phase, and the output starts at exactly zero
};

[[nodiscard]] inline constexpr std::string_view unwrapOriginName(UnwrapOrigin origin) noexcept { return origin == UnwrapOrigin::zero ? "zero" : "first_sample"; }

/// @brief The origin named by a setting's text, `first_sample` or `zero`, or no value for any other text.
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

    /// @brief Return to the state `start()` leaves, with no origin, no count and no counters.
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
     * @brief The reset for a gap in the stream. The count returns to zero and the origin is taken again.
     *
     * After a gap the cycle count is not the number of turns the signal made. The turns during the gap
     * were not seen. A continued count would look exact and be wrong. The reset shows the discontinuity as a
     * step to zero, and the step is counted. A caller that keeps a running count through gaps does not call
     * this function. The meaning of the count is then the caller's responsibility.
     *
     * Unlike `reset()`, this keeps the counters. They record what the stream did.
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

    /// @brief `2*pi*cycles + phase` for the current sample only, as a convenience outside the contract. Its
    /// spacing is `9.54e-7 rad` at `10^9` cycles. That loss is the reason the two parts are kept separate.
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
