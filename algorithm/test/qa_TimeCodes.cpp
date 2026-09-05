#include <boost/ut.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/ccsds/TimeCodes.hpp>

/*
 * One instant, 1988-01-18T17:20:43.123456Z, is 569 524 843 123 456 000 ns since the Unix epoch, and
 * the same instant is written four ways in CCSDS 301.0-B-4: as an ASCII calendar string in either
 * variation, as a CDS day 10974 with 62 443 123 ms of day and 456 us of millisecond, as a CCS calendar
 * code, and as a CUC coarse count of 948 216 043 seconds from the 1958 epoch on the TAI scale. Every
 * one of those numbers is asserted below against the same nanosecond value, so a code that reads a
 * field one octet wide too many fails here rather than in a mission.
 *
 * The epoch constant is recomputed from the calendar rather than asserted as a literal: 1958 to 1970
 * is twelve years, 4380 days, plus the leap days of 1960, 1964 and 1968, which is 4383. The two axis
 * bounds are asserted at their exact thresholds -- 9 602 063 236 coarse seconds and 111 134 days for
 * the 1958 epoch -- with the value at the bound passing and its successor refused, because a bound
 * that is only approximately right is a bound that is wrong on one value.
 *
 * The CUC round trip has an exact boundary and it is arithmetic: distinct fraction fields map to
 * distinct nanoseconds only while the least significant bit is at or above one nanosecond, that is
 * 8m <= log2(10^9) = 29.897, that is m <= 3. Both halves are asserted, the failure at m = 4 as a
 * measurement rather than as a warning.
 */
namespace {

using namespace gr::ccsds;

/// The worked instant, computed for this file from the calendar and checked four ways below.
constexpr std::int64_t kWorkedNs = 569'524'843'123'456'000LL;

std::uint64_t rng = 0x853C49E6748FEA9BULL;

std::uint64_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng >> 11U;
}

Layout cucLayout(std::uint8_t coarse, std::uint8_t fine) {
    Layout layout{};
    layout.kind          = TimeCodeKind::cuc;
    layout.coarse_octets = coarse;
    layout.fine_octets   = fine;
    return layout;
}

Layout cdsLayout(std::uint8_t dayOctets, std::uint8_t submilliOctets, bool customEpoch = false) {
    Layout layout{};
    layout.kind            = TimeCodeKind::cds;
    layout.day_octets      = dayOctets;
    layout.submilli_octets = submilliOctets;
    layout.custom_epoch    = customEpoch;
    return layout;
}

Layout ccsLayout(std::uint8_t subsecondOctets, bool dayOfYear = false) {
    Layout layout{};
    layout.kind             = TimeCodeKind::ccs;
    layout.subsecond_octets = subsecondOctets;
    layout.day_of_year      = dayOfYear;
    return layout;
}

/// A CUC T-field with a one-octet coarse count of zero and the given fraction, so what comes back is
/// the fraction's own contribution to the axis and nothing else.
std::int64_t fractionNs(std::uint8_t fine, std::uint64_t fraction, std::uint32_t& subNsPs) {
    std::vector<std::uint8_t> tfield(1UZ + fine, 0U);
    gr::ccsds::detail::writeBigEndian(tfield, 1UZ, fine, fraction);
    Instant instant{};
    boost::ut::expect(decode(tfield, cucLayout(1U, fine), 0, 0, instant) == TimeStatus::ok);
    subNsPs = instant.sub_ns_ps;
    return instant.ns + kEpoch1958Ns;
}

/// `fraction -> nanoseconds -> fraction`, through the kernel in both directions.
std::uint64_t roundTripFraction(std::uint8_t fine, std::uint64_t fraction) {
    std::vector<std::uint8_t> tfield(1UZ + fine, 0U);
    gr::ccsds::detail::writeBigEndian(tfield, 1UZ, fine, fraction);
    Instant instant{};
    boost::ut::expect(decode(tfield, cucLayout(1U, fine), 0, 0, instant) == TimeStatus::ok);

    std::vector<std::uint8_t> again(1UZ + fine, 0U);
    boost::ut::expect(encode(instant, cucLayout(1U, fine), 0, 0, again) == TimeStatus::ok);
    return gr::ccsds::detail::readBigEndian(again, 1UZ, fine);
}

std::string asciiOf(const Instant& instant, TimeCodeKind kind, std::uint8_t digits, bool terminator) {
    std::array<char, 40UZ> buffer{};
    std::size_t            written = 0UZ;
    boost::ut::expect(encodeAscii(instant, kind, digits, terminator, buffer, written) == TimeStatus::ok);
    return std::string{buffer.data(), written};
}

} // namespace

const boost::ut::suite<"ccsds time codes"> timeCodeTests = [] {
    using namespace boost::ut;
    using namespace std::chrono;

    "1. the 1958 epoch, recomputed from the calendar rather than asserted"_test = [] {
        const std::int64_t days = (sys_days{1970y / January / 1} - sys_days{1958y / January / 1}).count();
        expect(eq(days, std::int64_t{4383}));
        expect(eq(kEpoch1958Days, std::int64_t{4383}));
        expect(eq(kEpoch1958Secs, std::int64_t{378'691'200}));
        expect(eq(kEpoch1958Ns, std::int64_t{378'691'200'000'000'000}));

        // The decomposition, so the test says which part is wrong if the rule is ever mis-stated.
        int leaps = 0;
        for (int y = 1958; y < 1970; ++y) {
            if (year{y}.is_leap()) {
                ++leaps;
            }
        }
        expect(eq(leaps, 3)) << "1960, 1964 and 1968, and no century falls in the span";
        expect(eq(12 * 365 + leaps, 4383));
    };

    "2. an agency-defined epoch, worked at 1950"_test = [] {
        // 301.0-B-4 B3.2: the 1958 and 1950 epochs differ by exactly 2922.0 days, recomputed here.
        const std::int64_t difference = (sys_days{1958y / January / 1} - sys_days{1950y / January / 1}).count();
        expect(eq(difference, std::int64_t{2922}));
        expect(eq((kEpoch1958Days + difference) * 86400, std::int64_t{631'152'000}));

        constexpr std::int64_t        kEpoch1950Ns = -631'152'000LL * 1'000'000'000LL;
        std::array<std::uint8_t, 6UZ> tfield{};
        Instant                       instant{};
        expect(decode(tfield, cdsLayout(2U, 0U, true), kEpoch1950Ns, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns, kEpoch1950Ns)) << "day 0 of a 1950-epoch CDS code is 1950-01-01T00:00:00Z";

        // The same day count against the Level-1 epoch is 1958, which is what says the epoch is read
        // from the layout and not assumed.
        expect(decode(tfield, cdsLayout(2U, 0U, false), kEpoch1950Ns, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns, -kEpoch1958Ns));
    };

    "3. every legal P-field, and the T-field length each one declares"_test = [] {
        for (unsigned octet = 0U; octet < 256U; ++octet) {
            const std::array<std::uint8_t, 1UZ> pfield{static_cast<std::uint8_t>(octet)};
            Layout                              layout{};
            const TimeStatus                    status = parsePField(pfield, layout);
            const unsigned                      code   = (octet >> 4U) & 0x07U;
            const bool                          extend = (octet & 0x80U) != 0U;

            if (code == 0U || code == 3U || code == 7U) {
                expect(status == TimeStatus::reserved_code) << "identification " << code;
                continue;
            }
            if (code == 6U) {
                expect(status == TimeStatus::no_epoch_defined);
                expect(eq(layout.t_field_octets, std::size_t{(octet & 0x0FU) + 1U})) << "3.6.2's octets minus one, still returned so a caller can skip the field";
                continue;
            }
            if (code == 1U || code == 2U) {
                if (extend) {
                    expect(status == TimeStatus::short_field) << "the extension flag declares a second octet";
                    continue;
                }
                expect(status == TimeStatus::ok);
                expect(eq(static_cast<unsigned>(layout.coarse_octets), ((octet >> 2U) & 0x03U) + 1U));
                expect(eq(static_cast<unsigned>(layout.fine_octets), octet & 0x03U));
                expect(eq(layout.t_field_octets, std::size_t{layout.coarse_octets} + std::size_t{layout.fine_octets}));
                expect(eq(layout.custom_epoch, code == 2U));
                continue;
            }
            if (extend) {
                expect(status == TimeStatus::bad_p_field) << "neither 3.3.2 nor 3.4.2 defines a second octet";
                continue;
            }
            if (code == 4U) {
                if ((octet & 0x03U) == 3U) {
                    expect(status == TimeStatus::reserved_resolution);
                    continue;
                }
                expect(status == TimeStatus::ok);
                expect(eq(static_cast<unsigned>(layout.day_octets), (octet & 0x04U) != 0U ? 3U : 2U));
                expect(layout.t_field_octets == 6UZ || layout.t_field_octets == 7UZ || layout.t_field_octets == 8UZ || layout.t_field_octets == 9UZ || layout.t_field_octets == 10UZ || layout.t_field_octets == 11UZ);
                continue;
            }
            if ((octet & 0x07U) == 7U) {
                expect(status == TimeStatus::reserved_resolution);
                continue;
            }
            expect(status == TimeStatus::ok);
            expect(eq(layout.t_field_octets, 7UZ + std::size_t{layout.subsecond_octets})) << "both CCS variations have the same seven-octet base";
            expect(ge(layout.t_field_octets, 7UZ));
            expect(le(layout.t_field_octets, 13UZ));
        }

        // The CUC extension octet, over its whole cross product.
        std::size_t maximum = 0UZ;
        for (unsigned first = 0x80U | 0x10U; first <= (0x80U | 0x1FU); ++first) {
            for (unsigned second = 0U; second < 128U; ++second) {
                const std::array<std::uint8_t, 2UZ> pfield{static_cast<std::uint8_t>(first), static_cast<std::uint8_t>(second)};
                Layout                              layout{};
                expect(parsePField(pfield, layout) == TimeStatus::ok);
                const unsigned coarse = ((first >> 2U) & 0x03U) + 1U + ((second >> 5U) & 0x03U);
                const unsigned fine   = (first & 0x03U) + ((second >> 2U) & 0x07U);
                expect(eq(static_cast<unsigned>(layout.coarse_octets), coarse));
                expect(eq(static_cast<unsigned>(layout.fine_octets), fine));
                expect(eq(layout.p_field_octets, 2UZ));
                maximum = layout.t_field_octets > maximum ? layout.t_field_octets : maximum;
            }
        }
        expect(eq(maximum, 17UZ)) << "seven coarse octets and ten fine ones";

        // A second octet that sets its own extension flag declares a third, which 3.2.2 does not define.
        const std::array<std::uint8_t, 2UZ> extended{0x9CU, 0x80U};
        Layout                              layout{};
        expect(parsePField(extended, layout) == TimeStatus::bad_p_field);
    };

    "4. the minus-one that is only in one of two adjacent fields"_test = [] {
        // Bits 4-5 '00' and bits 6-7 '00' declare one coarse octet and zero fractional ones.
        const std::array<std::uint8_t, 1UZ> minimal{0x10U};
        Layout                              layout{};
        expect(parsePField(minimal, layout) == TimeStatus::ok);
        expect(eq(static_cast<unsigned>(layout.coarse_octets), 1U));
        expect(eq(static_cast<unsigned>(layout.fine_octets), 0U));
        expect(eq(layout.t_field_octets, 1UZ));

        // Bits 4-5 '11' and bits 6-7 '11' declare four and three, which is a seven-octet T-field.
        const std::array<std::uint8_t, 1UZ> maximal{0x1FU};
        expect(parsePField(maximal, layout) == TimeStatus::ok);
        expect(eq(static_cast<unsigned>(layout.coarse_octets), 4U));
        expect(eq(static_cast<unsigned>(layout.fine_octets), 3U));
        expect(eq(layout.t_field_octets, 7UZ)) << "reading the fractional count as a minus-one too gives four";

        // And back out again, so the two directions agree on the asymmetry.
        std::array<std::uint8_t, 2UZ> out{};
        std::size_t                   written = 0UZ;
        expect(writePField(cucLayout(4U, 3U), out, written) == TimeStatus::ok);
        expect(eq(written, 1UZ));
        expect(eq(static_cast<unsigned>(out[0]), 0x1FU));
        expect(writePField(cucLayout(1U, 0U), out, written) == TimeStatus::ok);
        expect(eq(static_cast<unsigned>(out[0]), 0x10U));
        expect(writePField(cucLayout(7U, 10U), out, written) == TimeStatus::ok);
        expect(eq(written, 2UZ));
        Layout back{};
        expect(parsePField(std::span<const std::uint8_t>{out}.first(written), back) == TimeStatus::ok);
        expect(eq(static_cast<unsigned>(back.coarse_octets), 7U));
        expect(eq(static_cast<unsigned>(back.fine_octets), 10U));
    };

    "5. the CUC round trip, and its exact boundary at m = 3"_test = [] {
        const bool longRun = std::getenv("ENABLE_LONG_TESTS") != nullptr;

        // m <= 3: the map is injective and the nearest-rounding inverse recovers every fraction.
        for (std::uint8_t m = 1U; m <= 3U; ++m) {
            const std::uint64_t span    = 1ULL << (8U * m);
            const std::uint64_t stride  = (m == 3U && !longRun) ? 211ULL : 1ULL;
            std::uint64_t       checked = 0ULL;
            for (std::uint64_t f = 0ULL; f < span; f += stride) {
                expect(eq(roundTripFraction(m, f), f)) << "m " << m << " fraction " << f;
                ++checked;
            }
            expect(gt(checked, 0ULL));
        }

        // The minimum spacing between adjacent fractions at m = 3 is 59 ns, which is what makes the
        // map injective at that width and not at the next.
        std::int64_t  minimumStep = kNsPerSecond;
        std::uint32_t discard     = 0U;
        for (std::uint64_t f = 0ULL; f + 1ULL < (1ULL << 24U); f += 9973ULL) {
            const std::int64_t step = fractionNs(3U, f + 1ULL, discard) - fractionNs(3U, f, discard);
            minimumStep             = step < minimumStep ? step : minimumStep;
        }
        expect(eq(minimumStep, std::int64_t{59})) << "2^-24 s is 59.6 ns, so adjacent fractions differ by 59 or 60";

        // m = 4: distinct fractions share a nanosecond, so no inverse can separate them. Measured.
        std::size_t sharing = 0UZ;
        for (std::uint64_t f = 0ULL; f < 8ULL; ++f) {
            if (fractionNs(4U, f, discard) == 0) {
                ++sharing;
            }
        }
        expect(ge(sharing, 4UZ)) << "at least four distinct fractions map to nanosecond zero: " << sharing;
        expect(neq(roundTripFraction(4U, 3ULL), 3ULL)) << "the round trip cannot be exact at m = 4";

        // The boundary's derivation, as the inequality it is: 8m <= log2(10^9).
        expect(lt(8U * 3U, 30U));
        expect(gt(8U * 4U, 30U));
        expect(gt((1ULL << 30U), 1'000'000'000ULL));
        expect(lt((1ULL << 29U), 1'000'000'000ULL));
        std::println("cuc round trip: m = 3 arm {}", longRun ? "exhaustive over 2^24 (ENABLE_LONG_TESTS)" : "sampled at stride 211");
    };

    "6. exactness where it is claimed, and the double that loses it"_test = [] {
        // m = 1: every fraction is exact, and the multiplier is 10^9 / 2^8.
        std::uint32_t discard = 0U;
        for (std::uint64_t f = 0ULL; f < 256ULL; ++f) {
            expect(eq(fractionNs(1U, f, discard), static_cast<std::int64_t>(f) * 3'906'250LL));
            expect(eq(discard, 0U)) << "nothing falls below a nanosecond at m = 1";
        }
        expect(eq(1'000'000'000LL / 256LL, std::int64_t{3'906'250}));

        // An instant past 2^53 ns round trips exactly here, and the same value through a double does
        // not -- the failure is shown rather than asserted about.
        constexpr std::int64_t kPast2To53 = 9'007'199'254'740'993LL;
        expect(neq(static_cast<std::int64_t>(static_cast<double>(kPast2To53)), kPast2To53)) << "53 significand bits are not 63";

        const Layout              layout = cdsLayout(3U, 4U);
        std::vector<std::uint8_t> tfield(timeFieldOctets(layout));
        Instant                   source{};
        source.ns = kPast2To53;
        expect(encode(source, layout, 0, 0, tfield) == TimeStatus::ok);
        Instant back{};
        expect(decode(tfield, layout, 0, 0, back) == TimeStatus::ok);
        expect(eq(back.ns, kPast2To53)) << "carried exactly, in integers, both ways";
    };

    "7. CDS, exact where it is exact and truncating where it is not"_test = [] {
        // A 16-bit submillisecond segment counts microseconds of millisecond, 0 to 999 (Annex A).
        std::array<std::uint8_t, 8UZ> tfield{};
        gr::ccsds::detail::writeBigEndian(tfield, 6UZ, 2UZ, 456ULL);
        Instant instant{};
        expect(decode(tfield, cdsLayout(2U, 2U), 0, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns + kEpoch1958Ns, std::int64_t{456'000}));
        expect(eq(instant.sub_ns_ps, 0U)) << "a microsecond is a thousand nanoseconds and nothing is lost";

        gr::ccsds::detail::writeBigEndian(tfield, 6UZ, 2UZ, 1000ULL);
        expect(decode(tfield, cdsLayout(2U, 2U), 0, 0, instant) == TimeStatus::segment_out_of_range) << "0 to 999, not 0 to 65 535";

        // A 32-bit segment counts picoseconds of millisecond, 0 to 999 999 999.
        std::array<std::uint8_t, 10UZ> wide{};
        gr::ccsds::detail::writeBigEndian(wide, 6UZ, 4UZ, 456'789'012ULL);
        expect(decode(wide, cdsLayout(2U, 4U), 0, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns + kEpoch1958Ns, std::int64_t{456'789}));
        expect(eq(instant.sub_ns_ps, 12U)) << "the residue below a nanosecond, returned rather than discarded";

        gr::ccsds::detail::writeBigEndian(wide, 6UZ, 4UZ, 1'000'000'000ULL);
        expect(decode(wide, cdsLayout(2U, 4U), 0, 0, instant) == TimeStatus::segment_out_of_range);
        expect(lt(999'999'999ULL, 1ULL << 32U)) << "the reading under which the unit and the width agree";

        // The reserved resolution names no width, so the T-field's length cannot be computed.
        const std::array<std::uint8_t, 1UZ> reserved{0x43U};
        Layout                              layout{};
        expect(parsePField(reserved, layout) == TimeStatus::reserved_resolution);

        // The leap millisecond: converted by the same expression, counted, and flagged.
        std::array<std::uint8_t, 6UZ> leapField{};
        gr::ccsds::detail::writeBigEndian(leapField, 2UZ, 4UZ, 86'400'123ULL);
        expect(decode(leapField, cdsLayout(2U, 0U), 0, 0, instant) == TimeStatus::ok);
        expect(instant.leap) << "23:59:60 has no representation on a POSIX axis, so it is flagged";
        expect(eq(instant.ns + kEpoch1958Ns, std::int64_t{86'400'123} * 1'000'000LL));
        gr::ccsds::detail::writeBigEndian(leapField, 2UZ, 4UZ, 86'401'000ULL);
        expect(decode(leapField, cdsLayout(2U, 0U), 0, 0, instant) == TimeStatus::segment_out_of_range) << "past Annex A's widest range";
        gr::ccsds::detail::writeBigEndian(leapField, 2UZ, 4UZ, 86'400'999ULL);
        expect(decode(leapField, cdsLayout(2U, 0U), 0, 0, instant) == TimeStatus::ok);
    };

    "8. the worked instant, in four codes and one number"_test = [] {
        // The number itself, from the calendar: 6591 days, then 62 443 seconds into the day.
        const std::int64_t days = (sys_days{1988y / January / 18} - sys_days{1970y / January / 1}).count();
        expect(eq(days, std::int64_t{6591}));
        expect(eq(days * 86400 + 17 * 3600 + 20 * 60 + 43, std::int64_t{569'524'843}));
        expect(eq(569'524'843LL * 1'000'000'000LL + 123'456'000LL, kWorkedNs));

        // ASCII A and B, 301.0-B-4 3.5.1.1's and 3.5.1.2's own examples.
        Instant instant{};
        expect(decodeAscii("1988-01-18T17:20:43.123456Z", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, kWorkedNs));
        expect(decodeAscii("1988-018T17:20:43.123456Z", TimeCodeKind::ascii_b, instant) == TimeStatus::ok);
        expect(eq(instant.ns, kWorkedNs));

        // CDS: day 10974 from the 1958 epoch, 62 443 123 ms of day, 456 us of millisecond.
        expect(eq(days + kEpoch1958Days, std::int64_t{10974}));
        std::array<std::uint8_t, 8UZ> cds{};
        gr::ccsds::detail::writeBigEndian(cds, 0UZ, 2UZ, 10974ULL);
        gr::ccsds::detail::writeBigEndian(cds, 2UZ, 4UZ, 62'443'123ULL);
        gr::ccsds::detail::writeBigEndian(cds, 6UZ, 2UZ, 456ULL);
        expect(decode(cds, cdsLayout(2U, 2U), 0, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns, kWorkedNs));

        // CCS with three subsecond octets: six decimal places, so 123456 in units of 10^-6 s.
        std::array<std::uint8_t, 10UZ> ccs{0x19U, 0x88U, 0x01U, 0x18U, 0x17U, 0x20U, 0x43U, 0x12U, 0x34U, 0x56U};
        expect(decode(ccs, ccsLayout(3U), 0, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns, kWorkedNs));

        // And the day-of-year variation of the same calendar code: 1988-018.
        std::array<std::uint8_t, 10UZ> ccsDoy{0x19U, 0x88U, 0x00U, 0x18U, 0x17U, 0x20U, 0x43U, 0x12U, 0x34U, 0x56U};
        expect(decode(ccsDoy, ccsLayout(3U, true), 0, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns, kWorkedNs));

        // CUC: a coarse count of 948 216 043 s from the 1958 epoch, on the TAI scale. The TAI-to-UTC
        // relation is asserted as an identity for every offset the test supplies -- no table anywhere.
        expect(eq(569'524'843LL + kEpoch1958Secs, std::int64_t{948'216'043}));
        std::array<std::uint8_t, 6UZ> cuc{};
        gr::ccsds::detail::writeBigEndian(cuc, 0UZ, 4UZ, 948'216'043ULL);
        gr::ccsds::detail::writeBigEndian(cuc, 4UZ, 2UZ, 8'091ULL); // 123 456 us to the nearest 2^-16 s
        for (const std::int32_t k : {0, 1, 10, 27, 37, -5}) {
            expect(decode(cuc, cucLayout(4U, 2U), 0, k, instant) == TimeStatus::ok);
            expect(eq(instant.ns, 569'524'843'000'000'000LL - std::int64_t{k} * 1'000'000'000LL + 123'458'862LL)) << "offset " << k;
            expect(instant.tai == (k == 0)) << "the scale is named, never assumed";
        }

        // The four codes and the ASCII encoder agree in the other direction too.
        instant    = Instant{};
        instant.ns = kWorkedNs;
        expect(eq(asciiOf(instant, TimeCodeKind::ascii_a, 6U, true), std::string{"1988-01-18T17:20:43.123456Z"}));
        expect(eq(asciiOf(instant, TimeCodeKind::ascii_b, 6U, true), std::string{"1988-018T17:20:43.123456Z"}));

        std::array<std::uint8_t, 8UZ> cdsBack{};
        expect(encode(instant, cdsLayout(2U, 2U), 0, 0, cdsBack) == TimeStatus::ok);
        expect(cdsBack == cds);

        std::array<std::uint8_t, 10UZ> ccsBack{};
        expect(encode(instant, ccsLayout(3U), 0, 0, ccsBack) == TimeStatus::ok);
        expect(ccsBack == ccs);
    };

    "9. CCS binary coded decimal, its ranges, and its resolution boundary"_test = [] {
        // Every subsecond octet count: k <= 4 is exact on the axis and k >= 5 truncates.
        for (std::uint8_t k = 0U; k <= 6U; ++k) {
            const Layout              layout = ccsLayout(k);
            std::vector<std::uint8_t> tfield(timeFieldOctets(layout));
            Instant                   source{};
            source.ns        = kWorkedNs;
            source.sub_ns_ps = 789U;
            expect(encode(source, layout, 0, 0, tfield) == TimeStatus::ok);
            Instant back{};
            expect(decode(tfield, layout, 0, 0, back) == TimeStatus::ok);

            const std::int64_t resolution = k <= 4U ? gr::ccsds::detail::powerOfTen(9U - 2U * k) : 1;
            expect(eq(back.ns, (kWorkedNs / resolution) * resolution)) << "k " << static_cast<unsigned>(k);
            if (k == 4U) {
                expect(eq(resolution, std::int64_t{10})) << "10^-8 s is 10 ns, still above a nanosecond";
            }
            if (k >= 5U) {
                expect(eq(back.ns, kWorkedNs)) << "exact on the axis, with the residue below it carried separately";
            }
        }
        expect(eq(gr::ccsds::detail::powerOfTen(1U), std::int64_t{10}));

        // A nibble above 9 is refused at every one of the thirteen octet positions.
        const Layout                   layout = ccsLayout(6U);
        std::array<std::uint8_t, 13UZ> base{0x19U, 0x88U, 0x01U, 0x18U, 0x17U, 0x20U, 0x43U, 0x12U, 0x34U, 0x56U, 0x00U, 0x00U, 0x00U};
        Instant                        instant{};
        expect(decode(base, layout, 0, 0, instant) == TimeStatus::ok);
        for (std::size_t i = 0UZ; i < base.size(); ++i) {
            std::array<std::uint8_t, 13UZ> bad = base;
            bad[i]                             = static_cast<std::uint8_t>(bad[i] | 0x0AU);
            expect(decode(bad, layout, 0, 0, instant) == TimeStatus::bad_bcd) << "position " << i;
        }

        // Annex A's ranges, each at its boundary and one past it.
        struct Row {
            std::array<std::uint8_t, 7UZ> field;
            bool                          admitted;
            const char*                   what;
        };
        const std::array<Row, 14UZ> rows{
            Row{{0x19U, 0x88U, 0x12U, 0x01U, 0x00U, 0x00U, 0x00U}, true, "month 12"},                //
            Row{{0x19U, 0x88U, 0x13U, 0x01U, 0x00U, 0x00U, 0x00U}, false, "month 13"},               //
            Row{{0x19U, 0x88U, 0x01U, 0x31U, 0x00U, 0x00U, 0x00U}, true, "31 January"},              //
            Row{{0x19U, 0x88U, 0x01U, 0x32U, 0x00U, 0x00U, 0x00U}, false, "32 January"},             //
            Row{{0x19U, 0x88U, 0x04U, 0x30U, 0x00U, 0x00U, 0x00U}, true, "30 April"},                //
            Row{{0x19U, 0x88U, 0x04U, 0x31U, 0x00U, 0x00U, 0x00U}, false, "31 April"},               //
            Row{{0x20U, 0x00U, 0x02U, 0x29U, 0x00U, 0x00U, 0x00U}, true, "29 February 2000"},        //
            Row{{0x20U, 0x00U, 0x02U, 0x30U, 0x00U, 0x00U, 0x00U}, false, "30 February 2000"},       //
            Row{{0x19U, 0x00U, 0x02U, 0x28U, 0x00U, 0x00U, 0x00U}, true, "28 February 1900"},        //
            Row{{0x19U, 0x00U, 0x02U, 0x29U, 0x00U, 0x00U, 0x00U}, false, "29 February 1900"},       //
            Row{{0x19U, 0x88U, 0x01U, 0x01U, 0x23U, 0x00U, 0x00U}, true, "hour 23"},                 //
            Row{{0x19U, 0x88U, 0x01U, 0x01U, 0x24U, 0x00U, 0x00U}, false, "hour 24"},                //
            Row{{0x19U, 0x88U, 0x01U, 0x01U, 0x00U, 0x00U, 0x60U}, true, "second 60, the leap one"}, //
            Row{{0x19U, 0x88U, 0x01U, 0x01U, 0x00U, 0x00U, 0x61U}, false, "second 61"},              //
        };
        for (const Row& row : rows) {
            const TimeStatus status = decode(row.field, ccsLayout(0U), 0, 0, instant);
            expect((status == TimeStatus::ok) == row.admitted) << row.what;
        }
        // The century rule, both ways, is the standard library's and not a hand-written one.
        expect(year{2000}.is_leap());
        expect(!year{1900}.is_leap());

        // The day-of-year variation's ranges.
        const std::array<std::uint8_t, 7UZ> leapDoy{0x19U, 0x88U, 0x03U, 0x66U, 0x00U, 0x00U, 0x00U};
        expect(decode(leapDoy, ccsLayout(0U, true), 0, 0, instant) == TimeStatus::ok) << "day 366 of a leap year";
        const std::array<std::uint8_t, 7UZ> pastLeapDoy{0x19U, 0x88U, 0x03U, 0x67U, 0x00U, 0x00U, 0x00U};
        expect(decode(pastLeapDoy, ccsLayout(0U, true), 0, 0, instant) == TimeStatus::segment_out_of_range);
        const std::array<std::uint8_t, 7UZ> commonDoy{0x19U, 0x89U, 0x03U, 0x65U, 0x00U, 0x00U, 0x00U};
        expect(decode(commonDoy, ccsLayout(0U, true), 0, 0, instant) == TimeStatus::ok) << "day 365 of a common year";
        const std::array<std::uint8_t, 7UZ> pastCommonDoy{0x19U, 0x89U, 0x03U, 0x66U, 0x00U, 0x00U, 0x00U};
        expect(decode(pastCommonDoy, ccsLayout(0U, true), 0, 0, instant) == TimeStatus::segment_out_of_range);
    };

    "10. the ASCII grammar, its subsets, and its fraction"_test = [] {
        expect(eq(std::string_view{"1988-01-18T17:20:43"}.size(), 19UZ)) << "form A's base length, counted";
        expect(eq(std::string_view{"1988-018T17:20:43"}.size(), 17UZ)) << "form B's";

        Instant instant{};
        expect(decodeAscii("1988-01-18T17:20:43", TimeCodeKind::ascii_a, instant) == TimeStatus::ok) << "the terminator is optional";
        expect(eq(instant.ns, 569'524'843'000'000'000LL));
        expect(decodeAscii("1988-01-18T17:20:43Z", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, 569'524'843'000'000'000LL));

        // Right truncation: the omitted subfields take their zero values.
        expect(decodeAscii("1988-01-18", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, 6591LL * 86400LL * 1'000'000'000LL)) << "that day's midnight";
        expect(decodeAscii("1988-01-18T17:20", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, (6591LL * 86400LL + 17 * 3600 + 20 * 60) * 1'000'000'000LL));

        // Left truncation and a time-only subset name no instant at all.
        expect(decodeAscii("17:20:43", TimeCodeKind::ascii_a, instant) == TimeStatus::incomplete_calendar);
        expect(decodeAscii("-01-18T17:20:43", TimeCodeKind::ascii_a, instant) == TimeStatus::incomplete_calendar);
        expect(decodeAscii("Z", TimeCodeKind::ascii_a, instant) == TimeStatus::incomplete_calendar);

        // Rule (d): a subset may not be a partial subfield.
        expect(decodeAscii("1988-1-18T17:20:43", TimeCodeKind::ascii_a, instant) == TimeStatus::bad_character);
        expect(decodeAscii("1988-01-18X17:20:43", TimeCodeKind::ascii_a, instant) == TimeStatus::bad_character);

        // The fraction is an integer parse: fifteen digits keep nine, then three, then count the rest.
        expect(decodeAscii("1988-01-18T17:20:43.123456789012345", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, 569'524'843'123'456'789LL));
        expect(eq(instant.sub_ns_ps, 12U));
        expect(eq(static_cast<unsigned>(instant.precision_discarded_digits), 3U));

        expect(decodeAscii("1988-01-18T17:20:43.999999999999", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, 569'524'843'999'999'999LL)) << "which a double path does not reliably produce";
        expect(eq(instant.sub_ns_ps, 999U));

        // A short fraction pads on the right, which is what a decimal fraction means.
        expect(decodeAscii("1988-01-18T17:20:43.5", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, 569'524'843'500'000'000LL));
        expect(decodeAscii("1988-01-18T17:20:43.", TimeCodeKind::ascii_a, instant) == TimeStatus::bad_character);

        // Both full forms round trip through the encoder over a seeded sweep.
        for (std::size_t trial = 0UZ; trial < 300UZ; ++trial) {
            Instant source{};
            source.ns = static_cast<std::int64_t>(next() % 4'000'000'000'000'000'000ULL);
            for (const TimeCodeKind kind : {TimeCodeKind::ascii_a, TimeCodeKind::ascii_b}) {
                const std::string text = asciiOf(source, kind, 9U, true);
                Instant           back{};
                expect(decodeAscii(text, kind, back) == TimeStatus::ok) << text;
                expect(eq(back.ns, source.ns)) << text;
            }
        }
    };

    "11. one purpose-built input per status, and the two bounds at their exact thresholds"_test = [] {
        Instant instant{};
        Layout  layout{};

        // short_field, both directions.
        const std::array<std::uint8_t, 3UZ> truncated{};
        expect(decode(truncated, cucLayout(4U, 2U), 0, 0, instant) == TimeStatus::short_field);
        expect(parsePField(std::span<const std::uint8_t>{}, layout) == TimeStatus::short_field);

        // bad_p_field: a second octet that declares a third.
        const std::array<std::uint8_t, 2UZ> thirdOctet{0x9FU, 0x80U};
        expect(parsePField(thirdOctet, layout) == TimeStatus::bad_p_field);

        // reserved_code, and no_epoch_defined with its length still returned.
        for (const unsigned code : {0U, 3U, 7U}) {
            const std::array<std::uint8_t, 1UZ> pfield{static_cast<std::uint8_t>(code << 4U)};
            expect(parsePField(pfield, layout) == TimeStatus::reserved_code) << "identification " << code;
        }
        const std::array<std::uint8_t, 1UZ> agency{0x6FU};
        expect(parsePField(agency, layout) == TimeStatus::no_epoch_defined);
        expect(eq(layout.t_field_octets, 16UZ)) << "so a caller can skip the field and keep its place";

        // reserved_resolution, both codes.
        const std::array<std::uint8_t, 1UZ> cdsReserved{0x43U};
        expect(parsePField(cdsReserved, layout) == TimeStatus::reserved_resolution);
        const std::array<std::uint8_t, 1UZ> ccsReserved{0x57U};
        expect(parsePField(ccsReserved, layout) == TimeStatus::reserved_resolution);

        // bad_bcd, bad_character, incomplete_calendar.
        const std::array<std::uint8_t, 7UZ> badNibble{0x1AU, 0x88U, 0x01U, 0x18U, 0x00U, 0x00U, 0x00U};
        expect(decode(badNibble, ccsLayout(0U), 0, 0, instant) == TimeStatus::bad_bcd);
        expect(decodeAscii("1988/01/18T17:20:43", TimeCodeKind::ascii_a, instant) == TimeStatus::bad_character);
        expect(decodeAscii("01-18T17:20:43", TimeCodeKind::ascii_a, instant) == TimeStatus::incomplete_calendar) << "a form with no year names no instant";

        // segment_out_of_range.
        const std::array<std::uint8_t, 7UZ> month13{0x19U, 0x88U, 0x13U, 0x01U, 0x00U, 0x00U, 0x00U};
        expect(decode(month13, ccsLayout(0U), 0, 0, instant) == TimeStatus::segment_out_of_range);

        // axis_overflow: the CUC bound, at the threshold and one past it.
        expect(eq(maxCoarseSeconds(-kEpoch1958Ns), 9'602'063'236ULL)) << "(2^63 - 1)/10^9 + 378 691 200";
        std::array<std::uint8_t, 5UZ> coarse{};
        gr::ccsds::detail::writeBigEndian(coarse, 0UZ, 5UZ, 9'602'063'236ULL);
        expect(decode(coarse, cucLayout(5U, 0U), 0, 0, instant) == TimeStatus::ok);
        gr::ccsds::detail::writeBigEndian(coarse, 0UZ, 5UZ, 9'602'063'237ULL);
        expect(decode(coarse, cucLayout(5U, 0U), 0, 0, instant) == TimeStatus::axis_overflow);
        gr::ccsds::detail::writeBigEndian(coarse, 0UZ, 5UZ, (1ULL << 40U) - 1ULL);
        expect(decode(coarse, cucLayout(5U, 0U), 0, 0, instant) == TimeStatus::axis_overflow);
        // A four-octet coarse count can never fail the test, which is why the check exists at five.
        expect(lt(std::uint64_t{0xFFFFFFFFULL}, maxCoarseSeconds(-kEpoch1958Ns)));

        // axis_overflow: the CDS day bound, likewise.
        expect(eq(maxDayCount(-kEpoch1958Ns), 111'134ULL)) << "106 751 + 4383";
        std::array<std::uint8_t, 7UZ> day{};
        gr::ccsds::detail::writeBigEndian(day, 0UZ, 3UZ, 111'134ULL);
        expect(decode(day, cdsLayout(3U, 0U), 0, 0, instant) == TimeStatus::ok);
        gr::ccsds::detail::writeBigEndian(day, 0UZ, 3UZ, 111'135ULL);
        expect(decode(day, cdsLayout(3U, 0U), 0, 0, instant) == TimeStatus::axis_overflow);
        expect(eq(kMaxAxisSeconds / 86400, std::int64_t{106'751}));
    };

    "the calendar codes at the axis's two ends, to the second"_test = [] {
        // The last day the axis holds is only partly held: 106 751 whole days is 9 223 286 400 s and
        // the axis stops 85 636 s later, at 23:47:16 of that day. A bound on the day count alone lets
        // the rest of the day through, and what comes back from a wrapped sum is a negative instant.
        Instant instant{};
        expect(eq(106'751LL * 86400LL + 23 * 3600 + 47 * 60 + 16, kMaxAxisSeconds));
        expect(decodeAscii("2262-04-11T23:47:16Z", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, std::int64_t{9'223'372'036'000'000'000}));
        expect(decodeAscii("2262-04-11T23:47:17Z", TimeCodeKind::ascii_a, instant) == TimeStatus::axis_overflow);
        expect(decodeAscii("2262-04-11T23:59:59Z", TimeCodeKind::ascii_a, instant) == TimeStatus::axis_overflow) << "the same day, past the end, and never a negative result";

        // The fraction is part of the same sum, so the threshold is a nanosecond and not a second.
        expect(decodeAscii("2262-04-11T23:47:16.854775807", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, std::numeric_limits<std::int64_t>::max()));
        expect(decodeAscii("2262-04-11T23:47:16.854775808", TimeCodeKind::ascii_a, instant) == TimeStatus::axis_overflow);

        // The other end, bounded at the whole second: the axis's floor falls 763 s into 1677-09-21,
        // and the second that contains it is refused with it rather than admitted in part.
        expect(decodeAscii("1677-09-21T00:12:44Z", TimeCodeKind::ascii_a, instant) == TimeStatus::ok);
        expect(eq(instant.ns, std::int64_t{-9'223'372'036'000'000'000}));
        expect(decodeAscii("1677-09-21T00:12:43Z", TimeCodeKind::ascii_a, instant) == TimeStatus::axis_overflow);

        // The calendar code takes the same path, so the same two instants land the same way.
        const std::array<std::uint8_t, 7UZ> ccsInside{0x22U, 0x62U, 0x04U, 0x11U, 0x23U, 0x47U, 0x16U};
        expect(decode(ccsInside, ccsLayout(0U), 0, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns, std::int64_t{9'223'372'036'000'000'000}));
        const std::array<std::uint8_t, 7UZ> ccsPast{0x22U, 0x62U, 0x04U, 0x11U, 0x23U, 0x59U, 0x59U};
        expect(decode(ccsPast, ccsLayout(0U), 0, 0, instant) == TimeStatus::axis_overflow);

        // Year 9999 is four thousand years past the axis and is refused, not wrapped.
        const std::array<std::uint8_t, 7UZ> ccsFar{0x99U, 0x99U, 0x12U, 0x31U, 0x00U, 0x00U, 0x00U};
        expect(decode(ccsFar, ccsLayout(0U), 0, 0, instant) == TimeStatus::axis_overflow);
        expect(decodeAscii("0001-01-01T00:00:00Z", TimeCodeKind::ascii_a, instant) == TimeStatus::axis_overflow);
    };

    "a CUC fraction field wider than the conversion carries"_test = [] {
        // The conversion uses four fractional octets, so a wider field holds the value in its top four
        // and zero below them. Writing it into the whole field would need a shift as wide as the field.
        Instant source{};
        source.ns = -kEpoch1958Ns + 123'456'789LL;
        for (const std::uint8_t fine : {std::uint8_t{4U}, std::uint8_t{5U}, std::uint8_t{9U}, std::uint8_t{10U}}) {
            const Layout              layout = cucLayout(1U, fine);
            std::vector<std::uint8_t> tfield(timeFieldOctets(layout));
            expect(encode(source, layout, 0, 0, tfield) == TimeStatus::ok) << "fine " << static_cast<unsigned>(fine);
            expect(gt(gr::ccsds::detail::readBigEndian(tfield, 1UZ, 4UZ), 0ULL)) << "the value is in the top four octets";
            for (std::size_t i = 5UZ; i < tfield.size(); ++i) {
                expect(eq(static_cast<unsigned>(tfield[i]), 0U)) << "fine " << static_cast<unsigned>(fine) << " octet " << i;
            }
            Instant back{};
            expect(decode(tfield, layout, 0, 0, back) == TimeStatus::ok);
            const std::int64_t difference = back.ns > source.ns ? back.ns - source.ns : source.ns - back.ns;
            expect(le(difference, std::int64_t{1})) << "fine " << static_cast<unsigned>(fine) << " difference " << difference;
        }
    };

    "the TAI offset is bounded before it is applied, in both directions"_test = [] {
        // The last coarse count the axis holds, then an offset that would carry it past the end.
        Instant                       instant{};
        std::array<std::uint8_t, 5UZ> coarse{};
        gr::ccsds::detail::writeBigEndian(coarse, 0UZ, 5UZ, 9'602'063'236ULL);
        expect(decode(coarse, cucLayout(5U, 0U), 0, 0, instant) == TimeStatus::ok);
        expect(decode(coarse, cucLayout(5U, 0U), 0, -32, instant) == TimeStatus::axis_overflow) << "a negative offset adds, and there is no room above";
        expect(decode(coarse, cucLayout(5U, 0U), 0, 37, instant) == TimeStatus::ok);
        expect(eq(instant.ns, std::int64_t{9'223'372'036'000'000'000} - 37'000'000'000LL));

        // The same relation on the encoder's side.
        Instant far{};
        far.ns = std::numeric_limits<std::int64_t>::max() - 1;
        std::array<std::uint8_t, 5UZ> out{};
        expect(encode(far, cucLayout(5U, 0U), 0, 32, out) == TimeStatus::axis_overflow);
        far.ns = std::numeric_limits<std::int64_t>::min() + 1;
        expect(encode(far, cucLayout(5U, 0U), 0, -32, out) == TimeStatus::axis_overflow);
    };

    "an epoch far below the axis leaves more room than a count can carry"_test = [] {
        // The room above INT64_MIN is very nearly 2^64 ns, and a coarse count that large would wrap the
        // unsigned the elapsed time is carried in before any range check saw it. The two bounds are
        // therefore capped, far above anything the axis itself admits, so no reachable case changes.
        constexpr std::int64_t kFarBelow = std::numeric_limits<std::int64_t>::min();
        expect(eq(gr::ccsds::detail::roomAbove(kFarBelow), std::numeric_limits<std::uint64_t>::max()));
        expect(eq(maxCoarseSeconds(kFarBelow), 18'446'744'072ULL));
        expect(eq(maxDayCount(kFarBelow), 213'501ULL));
        expect(eq(maxCoarseSeconds(-kEpoch1958Ns), 9'602'063'236ULL)) << "the 1958 bound is far below the cap and is untouched by it";
        expect(eq(maxDayCount(-kEpoch1958Ns), 111'134ULL));

        Instant                       instant{};
        std::array<std::uint8_t, 7UZ> wide{};
        Layout                        farLayout = cucLayout(7U, 0U);
        farLayout.custom_epoch                  = true;
        gr::ccsds::detail::writeBigEndian(wide, 0UZ, 7UZ, 18'446'744'072ULL);
        expect(decode(wide, farLayout, kFarBelow, 0, instant) == TimeStatus::ok);
        expect(eq(instant.ns, std::int64_t{9'223'372'035'145'224'192})) << "the sum is carried unsigned and converted once, never wrapped";
        gr::ccsds::detail::writeBigEndian(wide, 0UZ, 7UZ, 18'446'744'073ULL);
        expect(decode(wide, farLayout, kFarBelow, 0, instant) == TimeStatus::axis_overflow);

        Layout farDays{};
        farDays.kind            = TimeCodeKind::cds;
        farDays.day_octets      = 3U;
        farDays.submilli_octets = 0U;
        farDays.custom_epoch    = true;
        std::array<std::uint8_t, 7UZ> dayField{};
        gr::ccsds::detail::writeBigEndian(dayField, 0UZ, 3UZ, 213'501ULL);
        gr::ccsds::detail::writeBigEndian(dayField, 3UZ, 4UZ, 86'400'999ULL);
        expect(decode(dayField, farDays, kFarBelow, 0, instant) == TimeStatus::ok) << "the widest day and the widest millisecond of day together still fit";
        gr::ccsds::detail::writeBigEndian(dayField, 0UZ, 3UZ, 213'502ULL);
        expect(decode(dayField, farDays, kFarBelow, 0, instant) == TimeStatus::axis_overflow);
    };

    "a seeded round trip through every binary layout"_test = [] {
        for (std::size_t trial = 0UZ; trial < 400UZ; ++trial) {
            const std::int64_t ns = static_cast<std::int64_t>(next() % 1'000'000'000'000'000'000ULL);

            for (const std::uint8_t submilli : {std::uint8_t{0U}, std::uint8_t{2U}, std::uint8_t{4U}}) {
                for (const std::uint8_t days : {std::uint8_t{2U}, std::uint8_t{3U}}) {
                    const Layout layout = cdsLayout(days, submilli);
                    Instant      source{};
                    source.ns = ns;
                    std::vector<std::uint8_t> tfield(timeFieldOctets(layout));
                    const TimeStatus          written = encode(source, layout, 0, 0, tfield);
                    if (written != TimeStatus::ok) {
                        continue; // a two-octet day count cannot reach every instant, and says so
                    }
                    Instant back{};
                    expect(decode(tfield, layout, 0, 0, back) == TimeStatus::ok);
                    const std::int64_t resolution = submilli == 0U ? 1'000'000LL : (submilli == 2U ? 1'000LL : 1LL);
                    expect(eq(back.ns, (ns / resolution) * resolution)) << "submilli " << static_cast<unsigned>(submilli) << " days " << static_cast<unsigned>(days) << " ns " << ns;
                }
            }

            for (const std::uint8_t fine : {std::uint8_t{0U}, std::uint8_t{1U}, std::uint8_t{2U}, std::uint8_t{3U}}) {
                const Layout layout = cucLayout(4U, fine);
                Instant      source{};
                source.ns = ns;
                std::vector<std::uint8_t> tfield(timeFieldOctets(layout));
                if (encode(source, layout, 0, 0, tfield) != TimeStatus::ok) {
                    continue;
                }
                Instant back{};
                expect(decode(tfield, layout, 0, 0, back) == TimeStatus::ok);
                const std::int64_t difference = back.ns > ns ? back.ns - ns : ns - back.ns;
                const std::int64_t lsbNs      = fine == 0U ? 1'000'000'000LL : (1'000'000'000LL >> (8U * fine));
                expect(le(difference, lsbNs)) << "fine octets " << static_cast<unsigned>(fine) << " difference " << difference;
            }
        }
    };
};

int main() { /* tests are automatically registered and run */ }
