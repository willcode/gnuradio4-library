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

/// The speed of light in vacuum, exact: the meter is defined from it.
inline constexpr double kSpeedOfLight = 299'792'458.;

/**
 * @brief The Doppler offset a radial velocity puts on a carrier: `-f_carrier * v_radial / c`.
 *
 * `v_radial` is the rate of change of range, so a closing pass has it negative and the offset comes out
 * positive: an approaching transmitter reads high. That is the whole sign convention, and it is the one thing
 * about Doppler that silently inverts, so the QA anchors it on a worked case rather than on this sentence.
 *
 * This is the only step between a trajectory and a schedule. Orbit propagation, TLEs and station geometry stay
 * with whatever produced the velocity; the schedule takes frequencies.
 */
[[nodiscard]] constexpr double offsetFor(double vRadialMetersPerSecond, double fCarrierHz) noexcept { return -fCarrierHz * vRadialMetersPerSecond / kSpeedOfLight; }

/**
 * @brief The one-way propagation delay a slant range costs: `range / c`, in seconds.
 *
 * The companion of `offsetFor` and the other half of the same physics: a trajectory shifts the carrier by the
 * first and moves the envelope by the second. Both are written here so that `c` appears once in this tree.
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
 * A curve arrives as a handful of points and has to become one number per sample. The table is `(t_ns, value)`
 * knots, strictly increasing in time, and the value between two knots is the straight line between them.
 * Outside the table the end values **hold**: a schedule that has run out does not extrapolate into a value
 * nobody asked for, it keeps the last one it was given.
 *
 * The segment walk is a cursor, not a search. Sample times only increase, so the cursor advances at most once
 * per knot over a whole span and the per-sample cost does not depend on how many knots the table holds — the
 * bench measures that against 100, 1000 and 10000 knots. A per-sample binary search would make a dense table
 * cost more per sample than a sparse one for the same curve, which is exactly backwards.
 *
 * The table is a value: nothing here is mutable, and nothing is carried between calls. What a block does with
 * a schedule — the sign of it, refusing a value out of range, swapping one table for another — belongs to the
 * block. The two schedules that derive from this differ only in what their values mean and in the one
 * consumer-facing form each adds; the knot table, the validation and the cursor are shared so that a
 * divergence between them cannot exist to be found.
 */
class PiecewiseLinearSchedule {
public:
    /// Two knots make a line; below that there is nothing to interpolate. The ceiling is the spec's, and it is
    /// there so a caller wanting a smoother curve than piecewise-linear can simply supply denser knots.
    static constexpr std::size_t kMinKnots = 2UZ;
    static constexpr std::size_t kMaxKnots = std::size_t{1} << 20;

    [[nodiscard]] std::size_t                   size() const noexcept { return _times.size(); }
    [[nodiscard]] std::span<const std::int64_t> times() const noexcept { return _times; }
    [[nodiscard]] std::span<const double>       values() const noexcept { return _values; }
    [[nodiscard]] std::int64_t                  firstTime() const noexcept { return _times.front(); }
    [[nodiscard]] std::int64_t                  lastTime() const noexcept { return _times.back(); }

    /// The value at a time: linear between knots, and held at the end values outside the table.
    [[nodiscard]] double valueAt(std::int64_t t_ns) const noexcept { return evaluate(locate(t_ns), t_ns); }

    /**
     * @brief The slope, in value units per second, of the segment @p t_ns falls in.
     *
     * The derivative does not exist at a knot and does not exist outside the table, so this is not a
     * derivative: it is the slope of the one segment the time belongs to, clamped to a real segment at both
     * ends the way `valueAt` clamps its value. A caller comparing a table against a rate — which is what
     * `worstRangeRateMismatch` does — needs exactly that, and would otherwise have to reach into the knots and
     * re-derive it.
     */
    [[nodiscard]] double segmentSlopePerSecond(std::int64_t t_ns) const noexcept { return _slopes[locate(t_ns)] * 1e9; }

    /**
     * @brief The value at each of `out.size()` samples starting at `firstIndex` on `clock`.
     *
     * The point-evaluation twin of `FrequencySchedule::phaseIncrementsFor`, walking the same `NsWalk` cursor
     * and evaluating the line at each sample's own time. One compare, one subtract and one multiply-add per
     * sample, and every value is a pure function of its absolute index, so a caller that asks for a span in
     * one call and a caller that asks for it in a hundred get bit-identical values.
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

        // One slope per segment, so evaluating a point is a multiply rather than a divide, and the divide that
        // could differ between two evaluations of the same point happens once here instead.
        _slopes.resize(_times.size() - 1UZ);
        for (std::size_t i = 0UZ; i + 1UZ < _times.size(); ++i) {
            _slopes[i] = (_values[i + 1UZ] - _values[i]) / static_cast<double>(_times[i + 1UZ] - _times[i]);
        }
    }

    /// The segment a time falls in: the last knot at or before it, clamped to a real segment at both ends.
    [[nodiscard]] std::size_t locate(std::int64_t t_ns) const noexcept {
        const std::size_t upper = static_cast<std::size_t>(std::upper_bound(_times.begin(), _times.end(), t_ns) - _times.begin());
        if (upper == 0UZ) {
            return 0UZ;
        }
        return std::min(upper - 1UZ, _times.size() - 2UZ);
    }

    /// The value at a time already known to belong to `segment`. The two knot times are answered with the knot
    /// values themselves rather than through the line, so a point evaluated as the end of one segment and as the
    /// start of the next gives the same bits — which is what lets the cursor walk stand in for a fresh search.
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
 * A pass's Doppler curve arrives as a handful of points and has to become one radian increment per sample.
 * `phaseIncrementsFor` is the consumer-facing form and gives `Phasor::mixModulated` exactly what it takes: one
 * `double` radian increment per sample. Each increment is `2*pi` times the **integral** of the offset across
 * that sample's own interval, not `2*pi*offset(t_k)/fs` sampled at its start. The offset is piecewise linear, so
 * that integral is a trapezoid per segment in closed form and costs no more than the sampled version; what it
 * buys is that the accumulated phase is the schedule's true integral. A coherent demodulator downstream rides
 * the phase, not the frequency, and a sampled increment leaves it with an error that grows with the run.
 *
 * The sample interval boundaries come from `SampleClock::NsWalk`, so they are the exact `timeOf` of the absolute
 * sample index and cost no division per sample. Every increment is therefore a pure function of its absolute
 * index, and a caller that asks for a span in one call and a caller that asks for it in a hundred get
 * bit-identical values.
 */
class FrequencySchedule : public PiecewiseLinearSchedule {
public:
    static constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

    FrequencySchedule(std::span<const std::int64_t> timesNs, std::span<const double> offsetsHz) : PiecewiseLinearSchedule(timesNs, offsetsHz, ScheduleNaming{"gr::timing::FrequencySchedule", "offsets", "offset", "a finite frequency"}) {}

    /// The knot values, in hertz. `values()` names them without their unit; this names the unit.
    [[nodiscard]] std::span<const double> offsets() const noexcept { return _values; }

    /// The offset at a time, in hertz: `valueAt` under the name that carries the unit.
    [[nodiscard]] double offsetAt(std::int64_t t_ns) const noexcept { return valueAt(t_ns); }

    /**
     * @brief The radian increments for `out.size()` samples starting at `firstIndex` on `clock`.
     *
     * `out[k]` is `2*pi` times the integral of the offset, in cycles, across the interval from `timeOf` of
     * sample `firstIndex + k` to `timeOf` of the next one. It is what `Phasor::mixModulated` consumes, one
     * increment per sample, in the same `double` the phasor accumulates in.
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
     * The offset is linear on each segment and constant beyond the ends, so the trapezoid over a stretch that
     * crosses no knot is not an approximation but the integral itself. A sample interval that does cross knots
     * is split at each of them, which is the only thing the loop is for: at any usable sample rate it runs once.
     * The cursor enters at or behind the segment holding `a` and leaves at the one holding `b`, so a whole span
     * moves it at most once per knot.
     */
    [[nodiscard]] double integrateCycles(std::int64_t a, std::int64_t b, std::size_t& segment) const noexcept {
        constexpr double kSecondsPerNs = 1e-9;

        double       cycles = 0.;
        std::int64_t x      = a;
        while (x < b) {
            while (segment + 2UZ < _times.size() && _times[segment + 1UZ] <= x) {
                ++segment;
            }
            // The next knot strictly after `x`. Before the table that is its first knot, where the held value
            // gives way to the first line — a breakpoint like any other, and one a walk that only ever looked at
            // the segment's far end would step straight over.
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
 * The other half of what a trajectory does: `FrequencySchedule` carries what the pass does to the carrier and
 * this carries what it does to the envelope. `valuesFor` is the consumer-facing form — one delay per sample —
 * and the two extremes are exposed because a delay line has to size its history from the largest delay the
 * table ever asks for, and the table clamps at its ends, so the maximum over the table is the maximum over all
 * time.
 */
class DelaySchedule : public PiecewiseLinearSchedule {
public:
    DelaySchedule(std::span<const std::int64_t> timesNs, std::span<const double> delaysSeconds) : PiecewiseLinearSchedule(timesNs, delaysSeconds, ScheduleNaming{"gr::timing::DelaySchedule", "delays", "delay", "a finite delay"}) {}

    /// The knot values, in seconds.
    [[nodiscard]] std::span<const double> delays() const noexcept { return _values; }

    /// The delay at a time, in seconds.
    [[nodiscard]] double delayAt(std::int64_t t_ns) const noexcept { return valueAt(t_ns); }

    /// The largest delay the table reaches, and therefore the largest it reaches at any time at all.
    [[nodiscard]] double maxValue() const noexcept { return *std::ranges::max_element(_values); }
    [[nodiscard]] double minValue() const noexcept { return *std::ranges::min_element(_values); }
};

} // namespace gr::timing

#endif // GNURADIO_ALGORITHM_FREQUENCY_SCHEDULE_HPP
