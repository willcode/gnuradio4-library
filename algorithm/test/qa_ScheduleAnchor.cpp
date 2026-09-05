#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/timing/FrequencySchedule.hpp>
#include <gnuradio-4.0/algorithm/timing/SampleClock.hpp>
#include <gnuradio-4.0/algorithm/timing/ScheduleAnchor.hpp>

using namespace boost::ut;
using gr::timing::AnchorSource;
using gr::timing::anchorSourceFrom;
using gr::timing::anchorSourceName;
using gr::timing::FrequencySchedule;
using gr::timing::SampleClock;
using gr::timing::ScheduleAnchor;

namespace {

constexpr std::uint64_t kTagIndex  = 12'345ULL;
constexpr std::uint64_t kTagTimeNs = 1'788'000'000'123'456'789ULL;

/// A pass table on the same axis the anchor places, so that a wrong anchor shows up as different samples
/// rather than as a different number in a getter.
[[nodiscard]] FrequencySchedule passSchedule() {
    std::vector<std::int64_t> times(31UZ);
    std::vector<double>       offsets(31UZ);
    for (std::size_t i = 0UZ; i < times.size(); ++i) {
        const double u = static_cast<double>(i) / 30.;
        times[i]       = static_cast<std::int64_t>(kTagTimeNs) + static_cast<std::int64_t>(600e9 * u);
        offsets[i]     = -10'000. * std::tanh(6. * (u - 0.5));
    }
    return FrequencySchedule(times, offsets);
}

[[nodiscard]] std::vector<double> incrementsOn(const SampleClock& clock, const FrequencySchedule& schedule, std::size_t n) {
    std::vector<double> out(n);
    schedule.phaseIncrementsFor(clock, 0ULL, out);
    return out;
}

} // namespace

const boost::ut::suite<"ScheduleAnchor"> _schedule_anchor = [] {
    "the three spellings, and nothing else"_test = [] {
        expect(anchorSourceFrom("setting").value() == AnchorSource::setting);
        expect(anchorSourceFrom("first_trigger").value() == AnchorSource::first_trigger);
        expect(anchorSourceFrom("every_trigger").value() == AnchorSource::every_trigger);
        expect(!anchorSourceFrom("first").has_value()) << "a near miss is not accepted";
        expect(!anchorSourceFrom("").has_value());
        expect(anchorSourceName(AnchorSource::every_trigger) == "every_trigger");
    };

    "the tag-driven anchor is the setting-driven one, bit for bit"_test = [] {
        constexpr std::size_t kSamples = 8'192UZ;

        ScheduleAnchor tagged(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(!tagged.armed()) << "before its first tag the anchor is not armed, and says so";
        expect(tagged.onTrigger(kTagIndex, kTagTimeNs) == ScheduleAnchor::Response::armed);
        expect(tagged.armed());
        expect(tagged.anchorIndex() == kTagIndex);
        expect(tagged.anchorNs() == static_cast<std::int64_t>(kTagTimeNs));

        const ScheduleAnchor configured(AnchorSource::setting, kTagIndex, static_cast<std::int64_t>(kTagTimeNs));
        expect(configured.armed()) << "a configured anchor is armed from construction";

        const SampleClock a = tagged.clock(48'000ULL, 1ULL);
        const SampleClock b = configured.clock(48'000ULL, 1ULL);
        expect(a == b) << "the two clocks are the same value";

        const FrequencySchedule   schedule = passSchedule();
        const std::vector<double> fromTag  = incrementsOn(a, schedule, kSamples);
        const std::vector<double> fromSet  = incrementsOn(b, schedule, kSamples);
        expect(that % std::ranges::equal(fromTag, fromSet)) << "and the schedule they drive is identical to the bit";
    };

    "trigger_offset moves the anchor forward, at the resolution a float has there"_test = [] {
        // The sign is the tree's own: `blocks/basic/Trigger.hpp` publishes `trigger_time = now - relOffset`
        // beside `trigger_offset = relOffset`, so the tagged sample's time is the sum of the two.
        ScheduleAnchor milli(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(milli.onTrigger(0ULL, 1'000'000'000ULL, 1e-3f) == ScheduleAnchor::Response::armed);
        expect(milli.anchorNs() == 1'001'000'000LL) << "1 ms is exact: a float's spacing at 1 ms is 0.116 ns";

        ScheduleAnchor second(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(second.onTrigger(0ULL, 1'000'000'000ULL, 1.f) == ScheduleAnchor::Response::armed);
        // A float's spacing at 1 s is 119.2 ns, so the figure below is the representation and not a defect.
        expect(second.anchorNs() == 1'000'000'000LL + std::llround(static_cast<double>(1.f) * 1e9));
        expect(second.anchorNs() == 2'000'000'000LL);

        ScheduleAnchor negative(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(negative.onTrigger(0ULL, 1'000'000'000ULL, -2.5e-3f) == ScheduleAnchor::Response::armed);
        expect(negative.anchorNs() == 997'500'000LL) << "a negative offset moves the anchor back by the same rule";

        ScheduleAnchor ignoring(AnchorSource::first_trigger, 0ULL, 0LL, false);
        expect(ignoring.onTrigger(0ULL, 1'000'000'000ULL, 1e-3f) == ScheduleAnchor::Response::armed);
        expect(ignoring.anchorNs() == 1'000'000'000LL) << "honor_trigger_offset off leaves the tag time alone";
    };

    "setting ignores every tag, and counts every one"_test = [] {
        ScheduleAnchor anchor(AnchorSource::setting, 7ULL, 700LL);
        for (std::uint64_t i = 0ULL; i < 3ULL; ++i) {
            expect(anchor.onTrigger(100ULL + i, kTagTimeNs) == ScheduleAnchor::Response::ignored);
        }
        expect(anchor.anchorIndex() == 7ULL) << "the configured anchor stands";
        expect(anchor.anchorNs() == 700LL);
        expect(anchor.nIgnoredAnchors() == 3ULL) << "a graph wired two ways can be seen to be wired two ways";
        expect(anchor.nReanchors() == 0ULL);
    };

    "first_trigger honors one tag and counts the rest"_test = [] {
        ScheduleAnchor anchor(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(anchor.onTrigger(kTagIndex, kTagTimeNs) == ScheduleAnchor::Response::armed);
        expect(anchor.onTrigger(kTagIndex + 1'000ULL, kTagTimeNs + 1'000'000'000ULL) == ScheduleAnchor::Response::ignored);
        expect(anchor.onTrigger(kTagIndex + 2'000ULL, kTagTimeNs + 2'000'000'000ULL) == ScheduleAnchor::Response::ignored);

        expect(anchor.anchorIndex() == kTagIndex) << "the anchor does not move under a later tag";
        expect(anchor.anchorNs() == static_cast<std::int64_t>(kTagTimeNs));
        expect(anchor.nIgnoredAnchors() == 2ULL);
        expect(anchor.nReanchors() == 0ULL);

        // The reason it does not move, as a number: re-anchoring 1 s later on a schedule reading 10 kHz would
        // step the accumulated phase by 2*pi*10^4 rad, which is what a coherent demodulator downstream rides.
        const FrequencySchedule schedule = passSchedule();
        const SampleClock       held     = anchor.clock(48'000ULL, 1ULL);
        const SampleClock       moved(kTagIndex + 1'000ULL, static_cast<std::int64_t>(kTagTimeNs) + 1'000'000'000LL, 48'000ULL, 1ULL);
        std::vector<double>     one(4'096UZ);
        std::vector<double>     two(4'096UZ);
        schedule.phaseIncrementsFor(held, 0ULL, one);
        schedule.phaseIncrementsFor(moved, 0ULL, two);
        expect(that % !std::ranges::equal(one, two)) << "moving the anchor really does move the schedule under the signal";
    };

    "every_trigger re-anchors, and declares the discontinuity"_test = [] {
        ScheduleAnchor anchor(AnchorSource::every_trigger, 0ULL, 0LL);
        expect(!anchor.armed());
        expect(anchor.onTrigger(kTagIndex, kTagTimeNs) == ScheduleAnchor::Response::armed) << "the first tag arms rather than re-anchors";
        expect(anchor.nReanchors() == 0ULL);

        expect(anchor.onTrigger(kTagIndex + 5'000ULL, kTagTimeNs + 5'000'000'000ULL) == ScheduleAnchor::Response::reanchored);
        expect(anchor.anchorIndex() == kTagIndex + 5'000ULL) << "the anchor moves to the new burst";
        expect(anchor.anchorNs() == static_cast<std::int64_t>(kTagTimeNs) + 5'000'000'000LL);
        expect(anchor.nReanchors() == 1ULL) << "and the discontinuity is counted, not hidden";
        expect(anchor.nIgnoredAnchors() == 0ULL);
    };

    "a trigger_time that is not a nanosecond count is refused, in every mode"_test = [] {
        constexpr std::uint64_t kPastAxis = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1ULL;

        for (const AnchorSource source : {AnchorSource::setting, AnchorSource::first_trigger, AnchorSource::every_trigger}) {
            ScheduleAnchor anchor(source, 7ULL, 700LL);
            expect(anchor.onTrigger(1ULL, kPastAxis) == ScheduleAnchor::Response::refused) << anchorSourceName(source);
            expect(anchor.nRefusedAnchors() == 1ULL) << anchorSourceName(source);
            expect(anchor.nIgnoredAnchors() == 0ULL) << "a refusal is not an ignoral" << anchorSourceName(source);
            expect(anchor.anchorIndex() == 7ULL) << "the anchor stays where it was" << anchorSourceName(source);
            expect(anchor.anchorNs() == 700LL) << anchorSourceName(source);
            expect(anchor.armed() == (source == AnchorSource::setting)) << anchorSourceName(source);
        }

        // The bound itself is admitted: 2^63 - 1 ns is 2262-04-11T23:47:16.854775807Z, which no honest producer
        // reaches and a producer with something other than a nanosecond count in the field does.
        ScheduleAnchor edge(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(edge.onTrigger(1ULL, ScheduleAnchor::kMaxTriggerTimeNs) == ScheduleAnchor::Response::armed);
        expect(edge.anchorNs() == std::numeric_limits<std::int64_t>::max());

        ScheduleAnchor overflow(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(overflow.onTrigger(1ULL, ScheduleAnchor::kMaxTriggerTimeNs, 1.f) == ScheduleAnchor::Response::refused) << "an offset that carries the sum off the axis is refused, never wrapped";

        ScheduleAnchor nonFinite(AnchorSource::first_trigger, 0ULL, 0LL);
        expect(nonFinite.onTrigger(1ULL, 1'000ULL, std::numeric_limits<float>::quiet_NaN()) == ScheduleAnchor::Response::refused);
        expect(nonFinite.onTrigger(1ULL, 1'000ULL, std::numeric_limits<float>::infinity()) == ScheduleAnchor::Response::refused);
        expect(nonFinite.nRefusedAnchors() == 2ULL);
        expect(!nonFinite.armed());
    };

    "reset returns the configured anchor and forgets the tags"_test = [] {
        ScheduleAnchor anchor(AnchorSource::first_trigger, 7ULL, 700LL);
        (void)anchor.onTrigger(kTagIndex, kTagTimeNs);
        (void)anchor.onTrigger(kTagIndex + 1ULL, kTagTimeNs);
        expect(anchor.armed() && anchor.nIgnoredAnchors() == 1ULL);

        anchor.reset();
        expect(!anchor.armed()) << "first_trigger is unarmed again";
        expect(anchor.anchorIndex() == 7ULL);
        expect(anchor.anchorNs() == 700LL);
        expect(anchor.nIgnoredAnchors() == 0ULL);
        expect(anchor.nReanchors() == 0ULL);
        expect(anchor.nRefusedAnchors() == 0ULL);
    };
};

int main() { /* not needed for UT */ }
