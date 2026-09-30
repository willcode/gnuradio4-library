#ifndef GNURADIO_ALGORITHM_SCHEDULE_ANCHOR_HPP
#define GNURADIO_ALGORITHM_SCHEDULE_ANCHOR_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

#include <gnuradio-4.0/algorithm/timing/SampleClock.hpp>

namespace gr::timing {

/// Where a schedule's time origin comes from.
enum class AnchorSource : std::uint8_t {
    setting,       ///< the configured index and time, with every trigger tag ignored and counted
    first_trigger, ///< set by the first trigger tag, with every later one ignored and counted
    every_trigger  ///< every trigger tag re-establishes it, with the discontinuity declared
};

[[nodiscard]] inline constexpr std::string_view anchorSourceName(AnchorSource source) noexcept {
    switch (source) {
    case AnchorSource::setting: return "setting";
    case AnchorSource::first_trigger: return "first_trigger";
    case AnchorSource::every_trigger: return "every_trigger";
    }
    return "";
}

/// @brief The source a setting's text names, or `nullopt` for any other text, which the caller refuses.
[[nodiscard]] inline constexpr std::optional<AnchorSource> anchorSourceFrom(std::string_view name) noexcept {
    if (name == "setting") {
        return AnchorSource::setting;
    }
    if (name == "first_trigger") {
        return AnchorSource::first_trigger;
    }
    if (name == "every_trigger") {
        return AnchorSource::every_trigger;
    }
    return std::nullopt;
}

/**
 * @brief The rule that ties sample zero of a stream to a schedule's own time axis.
 *
 * `anchor_index` and `anchor_ns` tie a stream to a schedule. A recording or a receiver may already state its
 * start time in the reserved `trigger_time` tag. A setting that restates it is a second source of truth, and
 * the two can disagree. The origin can therefore be taken from the stream instead. This type decides between
 * the two sources and defines what a *second* tag means. It is one shared type and not one copy per user. Two
 * users driven by one trajectory could otherwise disagree about its origin. A test of either user alone would
 * not see that defect.
 *
 * **`setting`.** The configured index and time are the anchor, from construction. Every `trigger_time` tag is
 * ignored and **counted**. A tagged source combined with a hand-set anchor then shows up in the count. The two
 * origins do not disagree unseen.
 *
 * **`first_trigger`.** The first tag establishes the anchor. The index becomes the tag's absolute stream index,
 * and the time becomes the tag's value. Every later tag is ignored and counted, not honored. The reason is
 * arithmetic. A schedule's accumulated phase is `2*pi*integral(offset)` from the anchor. Moving the anchor
 * mid-stream shifts the schedule against the signal and steps the phase by an unbounded amount. A coherent
 * demodulator tracks exactly that phase. On a continuous stream, a second tag either restates the same origin
 * or contradicts it. A restatement is harmless and is ignored. A contradiction is a fault. A stream with one
 * output sample per input sample can only count it. Before the first tag the anchor is not armed. The caller
 * reads that state and neither guesses an origin nor stalls.
 *
 * **`every_trigger`.** The burst form. Each tag re-anchors. The re-anchoring is returned as a declared
 * discontinuity, and the caller resets its phase state. This mode does not claim phase continuity across a
 * tag, and it states that. It is correct exactly when the stream is a sequence of independently demodulated
 * bursts. The reserved `trigger_*` keys were designed for that case.
 *
 * Two rules hold in all three modes.
 *
 * `trigger_time` is `std::uint64_t` nanoseconds and an anchor is `std::int64_t`. A value above `2^63 - 1` is
 * **refused and counted** and not truncated. `2^63 - 1` ns after the Unix epoch is
 * 2262-04-11T23:47:16.854775807Z. A correct producer cannot reach the refusal. A producer that put something
 * other than a nanosecond count in the field can.
 *
 * `trigger_offset` is a `float` in seconds, "sample delay w.r.t. the trigger". It is honored when
 * `honor_trigger_offset` is set. The anchored sample's time is then `trigger_time + llround(offset * 1e9)`.
 * The **sign is plus**. By convention a source sets `trigger_time = now - relOffset` and
 * `trigger_offset = relOffset`. The tagged sample's time is therefore the sum of the two. This is the one
 * float-to-nanosecond step here, and its resolution is stated. A `float` has a 24-bit significand. Its spacing
 * is 0.116 ns at 1 ms, 119.2 ns at 1 s and 3.815 us at 60 s. An offset of a few milliseconds is exact to well
 * under a nanosecond. That is the analog group delay the tag was defined for. An offset of a minute is not a
 * group delay, and its resolution shows it. A non-finite offset, or one whose nanoseconds leave the axis, is
 * refused and counted.
 */
class ScheduleAnchor {
public:
    /// The result of a fed tag. `reanchored` is the declared discontinuity, and the caller resets its phase state on it.
    enum class Response : std::uint8_t { ignored, armed, reanchored, refused };

    /// The largest `trigger_time` that is a nanosecond count on the signed axis, 2262-04-11T23:47:16.854775807Z.
    static constexpr std::uint64_t kMaxTriggerTimeNs = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

    constexpr ScheduleAnchor() noexcept = default;

    constexpr ScheduleAnchor(AnchorSource source, std::uint64_t anchorIndex, std::int64_t anchorNs, bool honorTriggerOffset = true) noexcept //
        : _source(source), _index(anchorIndex), _ns(anchorNs), _honorOffset(honorTriggerOffset), _armed(source == AnchorSource::setting), _settingIndex(anchorIndex), _settingNs(anchorNs) {}

    /// @brief Return to the state `start()` leaves, with the configured anchor and no history of tags.
    constexpr void reset() noexcept {
        _index    = _settingIndex;
        _ns       = _settingNs;
        _armed    = _source == AnchorSource::setting;
        _ignored  = 0ULL;
        _reanchor = 0ULL;
        _refused  = 0ULL;
    }

    /**
     * @brief Feed one `trigger_time` tag at its absolute stream index and return the rule's result.
     *
     * @param streamIndex the tag's absolute index in the stream
     * @param triggerTimeNs the reserved `trigger_time` value, nanoseconds since the Unix epoch
     * @param triggerOffsetSeconds the reserved `trigger_offset` value, ignored unless `honorTriggerOffset()`
     */
    Response onTrigger(std::uint64_t streamIndex, std::uint64_t triggerTimeNs, float triggerOffsetSeconds = 0.f) noexcept {
        const std::optional<std::int64_t> when = anchorNsFor(triggerTimeNs, _honorOffset ? triggerOffsetSeconds : 0.f);
        if (!when.has_value()) {
            ++_refused;
            return Response::refused;
        }
        if (_source == AnchorSource::setting || (_source == AnchorSource::first_trigger && _armed)) {
            ++_ignored;
            return Response::ignored;
        }

        const bool wasArmed = _armed;
        _index              = streamIndex;
        _ns                 = *when;
        _armed              = true;
        if (wasArmed) {
            ++_reanchor;
            return Response::reanchored;
        }
        return Response::armed;
    }

    /**
     * @brief Count a `trigger_time` that carries something other than the reserved key's own type.
     *
     * A tag whose `trigger_time` is not a `std::uint64_t` states a time this rule cannot read. That is the same
     * event as a value off the nanosecond axis, and it goes in the same counter. Reading the field as whatever
     * type it holds would anchor a pass at the Unix epoch with no report.
     */
    Response onUnreadableTrigger() noexcept {
        ++_refused;
        return Response::refused;
    }

    /**
     * @brief The anchor time a tag would produce, or `nullopt` where the tag is not a nanosecond count.
     *
     * It is public because it is the one expression in this group whose sign matters. A caller can check a tag
     * before feeding it without writing the expression again.
     */
    [[nodiscard]] static std::optional<std::int64_t> anchorNsFor(std::uint64_t triggerTimeNs, float triggerOffsetSeconds) noexcept {
        if (triggerTimeNs > kMaxTriggerTimeNs) {
            return std::nullopt;
        }
        if (!std::isfinite(triggerOffsetSeconds)) {
            return std::nullopt;
        }

        // `llround` is undefined past the integer range. The seconds are therefore bounded before the call.
        // 2^63 ns is 9.223e18, and a `double` holds that bound exactly.
        constexpr double kMaxNs  = 9.2233720368547758e18;
        const double     shiftNs = static_cast<double>(triggerOffsetSeconds) * 1e9;
        if (!(std::abs(shiftNs) < kMaxNs)) {
            return std::nullopt;
        }

        const std::int64_t base  = static_cast<std::int64_t>(triggerTimeNs);
        const std::int64_t shift = std::llround(shiftNs);
        if (shift > 0 && base > std::numeric_limits<std::int64_t>::max() - shift) {
            return std::nullopt;
        }
        if (shift < 0 && base < std::numeric_limits<std::int64_t>::min() - shift) {
            return std::nullopt;
        }
        return base + shift;
    }

    [[nodiscard]] constexpr AnchorSource  source() const noexcept { return _source; }
    [[nodiscard]] constexpr bool          honorTriggerOffset() const noexcept { return _honorOffset; }
    [[nodiscard]] constexpr bool          armed() const noexcept { return _armed; }
    [[nodiscard]] constexpr std::uint64_t anchorIndex() const noexcept { return _index; }
    [[nodiscard]] constexpr std::int64_t  anchorNs() const noexcept { return _ns; }
    [[nodiscard]] constexpr std::uint64_t nIgnoredAnchors() const noexcept { return _ignored; }
    [[nodiscard]] constexpr std::uint64_t nReanchors() const noexcept { return _reanchor; }
    [[nodiscard]] constexpr std::uint64_t nRefusedAnchors() const noexcept { return _refused; }

    /**
     * @brief The clock this anchor and a rate make.
     *
     * The anchor is half of a `SampleClock`, and the rate is the other half. The two are combined here. A
     * tag-driven anchor and a setting-driven anchor at the same index and time produce clocks that compare
     * equal. Their streams are then equal to the bit.
     */
    [[nodiscard]] SampleClock clock(std::uint64_t rateNum, std::uint64_t rateDen) const { return SampleClock(_index, _ns, rateNum, rateDen); }

private:
    AnchorSource  _source{AnchorSource::setting};
    std::uint64_t _index{0ULL};
    std::int64_t  _ns{0LL};
    bool          _honorOffset{true};
    bool          _armed{true};
    std::uint64_t _ignored{0ULL};
    std::uint64_t _reanchor{0ULL};
    std::uint64_t _refused{0ULL};
    std::uint64_t _settingIndex{0ULL};
    std::int64_t  _settingNs{0LL};
};

} // namespace gr::timing

#endif // GNURADIO_ALGORITHM_SCHEDULE_ANCHOR_HPP
