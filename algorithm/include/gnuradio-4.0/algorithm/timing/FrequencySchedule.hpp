#ifndef GNURADIO_ALGORITHM_FREQUENCY_SCHEDULE_HPP
#define GNURADIO_ALGORITHM_FREQUENCY_SCHEDULE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/timing/SampleClock.hpp>

namespace gr::timing {

/// The speed of light in vacuum, exact, because the meter is defined from it.
inline constexpr double kSpeedOfLight = 299'792'458.;

/**
 * @brief The Doppler offset a radial velocity puts on a carrier, `-f_carrier * v_radial / c`.
 *
 * `v_radial` is the rate of change of range. A closing pass has a negative `v_radial` and a positive offset. An
 * approaching transmitter reads high. That is the whole sign convention. A sign error in Doppler inverts the
 * offset without any other symptom.
 *
 * This function converts a trajectory's velocity into a schedule's frequency. Orbit propagation, TLEs and station
 * geometry belong to the code that produced the velocity. The schedule takes frequencies.
 */
[[nodiscard]] constexpr double offsetFor(double vRadialMetersPerSecond, double fCarrierHz) noexcept { return -fCarrierHz * vRadialMetersPerSecond / kSpeedOfLight; }

/**
 * @brief The one-way propagation delay of a slant range, `range / c`, in seconds.
 *
 * This is the companion of `offsetFor`. A trajectory shifts the carrier by the offset and moves the envelope by
 * the delay. Both functions are in this file and share one definition of `c`.
 */
[[nodiscard]] constexpr double delayFor(double rangeMeters) noexcept { return rangeMeters / kSpeedOfLight; }

/// @brief How a schedule names itself and its values when it refuses a table.
struct ScheduleNaming {
    std::string_view type;     ///< the fully qualified class, as it opens every message
    std::string_view plural;   ///< the values, plural, for the pairing refusal
    std::string_view singular; ///< the value, singular, for the per-knot refusal
    std::string_view finite;   ///< what a finite value of this kind is called
};

/**
 * @brief An immutable table of knots on a `SampleClock`'s time axis, read piecewise linearly.
 *
 * A curve arrives as a few points and must become one number per sample. The table holds `(t_ns, value)`
 * knots, strictly increasing in time. The value between two knots lies on the straight line between them.
 * Outside the table the end values **hold**. The schedule does not extrapolate past its last knot. It keeps
 * the last value it was given.
 *
 * The segment walk uses a cursor and not a search. Sample times only increase. The cursor therefore advances at
 * most once per knot over a whole span. The per-sample cost does not depend on the number of knots. A benchmark
 * measures this at 100, 1000 and 10000 knots. A per-sample binary search would make a dense table cost more per
 * sample than a sparse one for the same curve.
 *
 * The table is a value. Nothing here is mutable, and nothing is carried between calls. The caller decides what
 * to do with a schedule, such as its sign, the refusal of an out-of-range value or the swap of one table for
 * another. The two derived schedules differ only in the meaning of their values and in the one output form each
 * adds. They share the knot table, the validation and the cursor. The two cannot diverge in those parts.
 */
class PiecewiseLinearSchedule {
public:
    /// Two knots make a line. With fewer there is nothing to interpolate. The ceiling on the knot count leaves
    /// room for a caller that wants a smoother curve than piecewise-linear to supply denser knots.
    static constexpr std::size_t kMinKnots = 2UZ;
    static constexpr std::size_t kMaxKnots = std::size_t{1} << 20;

    [[nodiscard]] std::size_t                   size() const noexcept { return _times.size(); }
    [[nodiscard]] std::span<const std::int64_t> times() const noexcept { return _times; }
    [[nodiscard]] std::span<const double>       values() const noexcept { return _values; }
    [[nodiscard]] std::int64_t                  firstTime() const noexcept { return _times.front(); }
    [[nodiscard]] std::int64_t                  lastTime() const noexcept { return _times.back(); }

    /// The value at a time, linear between knots and held at the end values outside the table.
    [[nodiscard]] double valueAt(std::int64_t t_ns) const noexcept { return evaluate(locate(t_ns), t_ns); }

    /**
     * @brief The slope, in value units per second, of the segment @p t_ns falls in.
     *
     * The derivative does not exist at a knot or outside the table. This value is therefore not a derivative.
     * The function returns the slope of the segment that holds the time. At both ends it clamps to a real segment,
     * as `valueAt` clamps its value. A caller that compares a table against a rate needs the segment slope. One
     * example is `worstRangeRateMismatch`. Without this function the caller would have to derive the slope from the
     * knots again.
     */
    [[nodiscard]] double segmentSlopePerSecond(std::int64_t t_ns) const noexcept { return _slopes[locate(t_ns)] * 1e9; }

    /**
     * @brief The value at each of `out.size()` samples starting at `firstIndex` on `clock`.
     *
     * This is the point-evaluation counterpart of `FrequencySchedule::phaseIncrementsFor`. It walks the same
     * `NsWalk` cursor and evaluates the line at each sample's own time. Each sample costs one compare, one
     * subtract and one multiply-add. Every value is a pure function of its absolute index. A caller that asks
     * for a span in one call and a caller that asks in a hundred calls get bit-identical values.
     */
    void valuesFor(const SampleClock& clock, std::uint64_t firstIndex, std::span<double> out) const noexcept {
        if (out.empty()) {
            return;
        }

        SampleClock::NsWalk walk    = clock.walkFrom(firstIndex);
        std::size_t         segment = locate(walk.t_ns);
        for (double& value : out) {
            while (segment + 2UZ < _times.size() && _times[segment + 1UZ] <= walk.t_ns) {
                ++segment;
            }
            value = evaluate(segment, walk.t_ns);
            walk.advance();
        }
    }

protected:
    PiecewiseLinearSchedule(std::span<const std::int64_t> timesNs, std::span<const double> values, const ScheduleNaming& naming) : _times(timesNs.begin(), timesNs.end()), _values(values.begin(), values.end()) {
        if (timesNs.size() != values.size()) {
            throw std::invalid_argument(std::format("{}: {} times against {} {} — the two must be paired", naming.type, timesNs.size(), values.size(), naming.plural));
        }
        if (_times.size() < kMinKnots || _times.size() > kMaxKnots) {
            throw std::invalid_argument(std::format("{}: {} knots — the table holds between {} and {}", naming.type, _times.size(), kMinKnots, kMaxKnots));
        }
        for (std::size_t i = 0UZ; i < _values.size(); ++i) {
            if (!std::isfinite(_values[i])) {
                throw std::invalid_argument(std::format("{}: knot {} has {} {}, which is not {}", naming.type, i, naming.singular, _values[i], naming.finite));
            }
        }
        for (std::size_t i = 1UZ; i < _times.size(); ++i) {
            if (_times[i] <= _times[i - 1UZ]) {
                throw std::invalid_argument(std::format("{}: knot {} is at {} ns, at or before knot {} at {} ns — times must strictly increase", naming.type, i, _times[i], i - 1UZ, _times[i - 1UZ]));
            }
        }

        // One slope per segment. Evaluating a point is then a multiply and not a divide. The divide, which could
        // differ between two evaluations of the same point, happens once here.
        _slopes.resize(_times.size() - 1UZ);
        for (std::size_t i = 0UZ; i + 1UZ < _times.size(); ++i) {
            _slopes[i] = (_values[i + 1UZ] - _values[i]) / static_cast<double>(_times[i + 1UZ] - _times[i]);
        }
    }

    /// The segment that holds a time, the last knot at or before it, clamped to a real segment at both ends.
    [[nodiscard]] std::size_t locate(std::int64_t t_ns) const noexcept {
        const std::size_t upper = static_cast<std::size_t>(std::upper_bound(_times.begin(), _times.end(), t_ns) - _times.begin());
        if (upper == 0UZ) {
            return 0UZ;
        }
        return std::min(upper - 1UZ, _times.size() - 2UZ);
    }

    /// The value at a time already known to lie in `segment`. At the two knot times it returns the knot values
    /// themselves and not a point on the line. A point evaluated as the end of one segment and as the start of
    /// the next then gives the same bits. The cursor walk can therefore replace a fresh search.
    [[nodiscard]] double evaluate(std::size_t segment, std::int64_t t_ns) const noexcept {
        if (t_ns <= _times.front()) {
            return _values.front();
        }
        if (t_ns >= _times.back()) {
            return _values.back();
        }
        if (t_ns == _times[segment]) {
            return _values[segment];
        }
        if (t_ns == _times[segment + 1UZ]) {
            return _values[segment + 1UZ];
        }
        return _values[segment] + _slopes[segment] * static_cast<double>(t_ns - _times[segment]);
    }

    std::vector<std::int64_t> _times;
    std::vector<double>       _values;
    std::vector<double>       _slopes; ///< value units per nanosecond across each segment
};

/**
 * @brief A table of frequency-offset knots, and the radian increments a phasor consumes.
 *
 * A pass's Doppler curve arrives as a few points and must become one radian increment per sample.
 * `phaseIncrementsFor` produces that form. It gives `Phasor::mixModulated` its input, one `double` radian
 * increment per sample. Each increment is `2*pi` times the **integral** of the offset across that sample's own
 * interval. It is not `2*pi*offset(t_k)/fs` sampled at the interval's start. The offset is piecewise linear.
 * Its integral is a trapezoid per segment in closed form and costs no more than the sampled form. The
 * accumulated phase is then the schedule's true integral. A coherent demodulator tracks the phase and not the
 * frequency. A sampled increment would give it an error that grows over the run.
 *
 * The sample interval boundaries come from `SampleClock::NsWalk`. They are the exact `timeOf` of the absolute
 * sample index and cost no division per sample. Every increment is therefore a pure function of its absolute
 * index. A caller that asks for a span in one call and a caller that asks in a hundred calls get bit-identical
 * values.
 */
class FrequencySchedule : public PiecewiseLinearSchedule {
public:
    static constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

    FrequencySchedule(std::span<const std::int64_t> timesNs, std::span<const double> offsetsHz) : PiecewiseLinearSchedule(timesNs, offsetsHz, ScheduleNaming{"gr::timing::FrequencySchedule", "offsets", "offset", "a finite frequency"}) {}

    /// The knot values, in hertz. `values()` returns the same values under a name without the unit.
    [[nodiscard]] std::span<const double> offsets() const noexcept { return _values; }

    /// The offset at a time, in hertz. This is `valueAt` under a name that carries the unit.
    [[nodiscard]] double offsetAt(std::int64_t t_ns) const noexcept { return valueAt(t_ns); }

    /**
     * @brief The radian increments for `out.size()` samples starting at `firstIndex` on `clock`.
     *
     * `out[k]` is `2*pi` times the integral of the offset, in cycles, across the interval from `timeOf` of
     * sample `firstIndex + k` to `timeOf` of the next one. `Phasor::mixModulated` consumes these increments, one
     * per sample, in the same `double` the phasor accumulates in.
     */
    void phaseIncrementsFor(const SampleClock& clock, std::uint64_t firstIndex, std::span<double> out) const noexcept {
        if (out.empty()) {
            return;
        }

        SampleClock::NsWalk walk    = clock.walkFrom(firstIndex);
        std::int64_t        start   = walk.t_ns;
        std::size_t         segment = locate(start);
        for (double& increment : out) {
            walk.advance();
            const std::int64_t end = walk.t_ns;
            increment              = kTwoPi * integrateCycles(start, end, segment);
            start                  = end;
        }
    }

private:
    /**
     * @brief The integral of the offset from `a` to `b`, in cycles, advancing the cursor as it goes.
     *
     * The offset is linear on each segment and constant beyond the ends. The trapezoid over a stretch that
     * crosses no knot is therefore the exact integral. A sample interval that crosses knots is split at each of
     * them. The loop exists for that case. At any usable sample rate it runs once. The cursor enters at or
     * before the segment that holds `a` and leaves at the segment that holds `b`. A whole span moves it at most
     * once per knot.
     */
    [[nodiscard]] double integrateCycles(std::int64_t a, std::int64_t b, std::size_t& segment) const noexcept {
        constexpr double kSecondsPerNs = 1e-9;

        double       cycles = 0.;
        std::int64_t x      = a;
        while (x < b) {
            while (segment + 2UZ < _times.size() && _times[segment + 1UZ] <= x) {
                ++segment;
            }
            // The next knot strictly after `x`. Before the table that is the first knot. There the held value
            // changes to the first line. That point is a breakpoint like any other. A walk that looked only at
            // the far end of the segment would miss it.
            const std::int64_t nextKnot = (x < _times.front()) ? _times.front() : _times[segment + 1UZ];
            const std::int64_t end      = (nextKnot > x && nextKnot < b) ? nextKnot : b;
            cycles += 0.5 * (evaluate(segment, x) + evaluate(segment, end)) * static_cast<double>(end - x) * kSecondsPerNs;
            x = end;
        }
        return cycles;
    }
};

/**
 * @brief A table of propagation-delay knots, in seconds, on the same axis and with the same reading rule.
 *
 * `FrequencySchedule` holds the effect of a pass on the carrier. This class holds its effect on the envelope.
 * `valuesFor` gives one delay per sample. The two extremes are exposed. A delay line sizes its history from the
 * largest delay in the table. The table clamps at its ends. The maximum over the table is therefore the maximum
 * over all time.
 */
class DelaySchedule : public PiecewiseLinearSchedule {
public:
    DelaySchedule(std::span<const std::int64_t> timesNs, std::span<const double> delaysSeconds) : PiecewiseLinearSchedule(timesNs, delaysSeconds, ScheduleNaming{"gr::timing::DelaySchedule", "delays", "delay", "a finite delay"}) {}

    /// The knot values, in seconds.
    [[nodiscard]] std::span<const double> delays() const noexcept { return _values; }

    /// The delay at a time, in seconds.
    [[nodiscard]] double delayAt(std::int64_t t_ns) const noexcept { return valueAt(t_ns); }

    /// The largest delay in the table, which is also the largest delay at any time.
    [[nodiscard]] double maxValue() const noexcept { return *std::ranges::max_element(_values); }
    [[nodiscard]] double minValue() const noexcept { return *std::ranges::min_element(_values); }
};

} // namespace gr::timing

#endif // GNURADIO_ALGORITHM_FREQUENCY_SCHEDULE_HPP
