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
 * @brief An immutable table of frequency-offset knots on a `SampleClock`'s time axis, read piecewise linearly.
 *
 * A pass's Doppler curve arrives as a handful of points and has to become one radian increment per sample. The
 * table is `(t_ns, offset_hz)` knots, strictly increasing in time, and the offset between two knots is the
 * straight line between them. Outside the table the end values **hold**: a schedule that has run out does not
 * extrapolate into a frequency nobody asked for, it keeps the last one it was given.
 *
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
 *
 * The segment walk is a cursor, not a search. Sample times only increase, so the cursor advances at most once
 * per knot over a whole span and the per-sample cost does not depend on how many knots the table holds — the
 * bench measures that against 100, 1000 and 10000 knots. A per-sample binary search would make a dense table
 * cost more per sample than a sparse one for the same curve, which is exactly backwards.
 *
 * The table is a value: nothing here is mutable, and nothing is carried between calls. What a block does with
 * the schedule — the sign of it, refusing an offset past `fs/2`, swapping one table for another — belongs to
 * the block.
 */
class FrequencySchedule {
public:
    /// Two knots make a line; below that there is nothing to interpolate. The ceiling is the spec's, and it is
    /// there so a caller wanting a smoother curve than piecewise-linear can simply supply denser knots.
    static constexpr std::size_t kMinKnots = 2UZ;
    static constexpr std::size_t kMaxKnots = std::size_t{1} << 20;

    static constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

    FrequencySchedule(std::span<const std::int64_t> timesNs, std::span<const double> offsetsHz) : _times(timesNs.begin(), timesNs.end()), _offsets(offsetsHz.begin(), offsetsHz.end()) {
        if (timesNs.size() != offsetsHz.size()) {
            throw std::invalid_argument(std::format("gr::timing::FrequencySchedule: {} times against {} offsets — the two must be paired", timesNs.size(), offsetsHz.size()));
        }
        if (_times.size() < kMinKnots || _times.size() > kMaxKnots) {
            throw std::invalid_argument(std::format("gr::timing::FrequencySchedule: {} knots — the table holds between {} and {}", _times.size(), kMinKnots, kMaxKnots));
        }
        for (std::size_t i = 0UZ; i < _offsets.size(); ++i) {
            if (!std::isfinite(_offsets[i])) {
                throw std::invalid_argument(std::format("gr::timing::FrequencySchedule: knot {} has offset {}, which is not a finite frequency", i, _offsets[i]));
            }
        }
        for (std::size_t i = 1UZ; i < _times.size(); ++i) {
            if (_times[i] <= _times[i - 1UZ]) {
                throw std::invalid_argument(std::format("gr::timing::FrequencySchedule: knot {} is at {} ns, at or before knot {} at {} ns — times must strictly increase", i, _times[i], i - 1UZ, _times[i - 1UZ]));
            }
        }

        // One slope per segment, so evaluating a point is a multiply rather than a divide, and the divide that
        // could differ between two evaluations of the same point happens once here instead.
        _slopes.resize(_times.size() - 1UZ);
        for (std::size_t i = 0UZ; i + 1UZ < _times.size(); ++i) {
            _slopes[i] = (_offsets[i + 1UZ] - _offsets[i]) / static_cast<double>(_times[i + 1UZ] - _times[i]);
        }
    }

    [[nodiscard]] std::size_t                   size() const noexcept { return _times.size(); }
    [[nodiscard]] std::span<const std::int64_t> times() const noexcept { return _times; }
    [[nodiscard]] std::span<const double>       offsets() const noexcept { return _offsets; }
    [[nodiscard]] std::int64_t                  firstTime() const noexcept { return _times.front(); }
    [[nodiscard]] std::int64_t                  lastTime() const noexcept { return _times.back(); }

    /// The offset at a time: linear between knots, and held at the end values outside the table.
    [[nodiscard]] double offsetAt(std::int64_t t_ns) const noexcept { return evaluate(locate(t_ns), t_ns); }

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
    std::vector<std::int64_t> _times;
    std::vector<double>       _offsets;
    std::vector<double>       _slopes; ///< Hz per nanosecond across each segment

    /// The segment a time falls in: the last knot at or before it, clamped to a real segment at both ends.
    [[nodiscard]] std::size_t locate(std::int64_t t_ns) const noexcept {
        const std::size_t upper = static_cast<std::size_t>(std::upper_bound(_times.begin(), _times.end(), t_ns) - _times.begin());
        if (upper == 0UZ) {
            return 0UZ;
        }
        return std::min(upper - 1UZ, _times.size() - 2UZ);
    }

    /// The offset at a time already known to belong to `segment`. The two knot times are answered with the knot
    /// values themselves rather than through the line, so a point evaluated as the end of one segment and as the
    /// start of the next gives the same bits — which is what lets the cursor walk stand in for a fresh search.
    [[nodiscard]] double evaluate(std::size_t segment, std::int64_t t_ns) const noexcept {
        if (t_ns <= _times.front()) {
            return _offsets.front();
        }
        if (t_ns >= _times.back()) {
            return _offsets.back();
        }
        if (t_ns == _times[segment]) {
            return _offsets[segment];
        }
        if (t_ns == _times[segment + 1UZ]) {
            return _offsets[segment + 1UZ];
        }
        return _offsets[segment] + _slopes[segment] * static_cast<double>(t_ns - _times[segment]);
    }

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

} // namespace gr::timing

#endif // GNURADIO_ALGORITHM_FREQUENCY_SCHEDULE_HPP
