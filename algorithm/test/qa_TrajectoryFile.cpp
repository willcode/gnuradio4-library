#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/timing/TrajectoryFile.hpp>

using namespace boost::ut;
using gr::timing::delayFor;
using gr::timing::FrequencySchedule;
using gr::timing::kSpeedOfLight;
using gr::timing::loadTrajectory;
using gr::timing::loadTrajectoryFile;
using gr::timing::offsetFor;
using gr::timing::Trajectory;
using gr::timing::worstRangeRateMismatch;

namespace {

constexpr double kCarrierHz = 437e6;

/// The header block every fixture below opens with, so that a fixture testing one refusal cannot fail on a
/// second one it did not mean to exercise. Its four lines put the first knot on line 5.
[[nodiscard]] std::string wellFormed(std::string_view columns, std::string_view timeScale, std::string_view knots) { return std::format("#!gr4-trajectory 1\ncolumns {}\ntime_scale {}\ncarrier_hz 437000000\n{}", columns, timeScale, knots); }

/// The exception's type, and the line number in its message. Every refusal names a line, and the line number
/// is the only part of a parse failure an operator can act on, so it is what the test reads.
void refuses(std::string_view text, std::size_t line, std::string_view label) {
    bool        threw = false;
    std::string message;
    try {
        (void)loadTrajectory(text);
    } catch (const std::invalid_argument& error) {
        threw   = true;
        message = error.what();
    }
    expect(threw) << label << "must refuse";
    if (threw) {
        expect(that % message.contains(std::format("line {}:", line))) << label << "must name line" << line << "but said:" << message;
    }
}

/// The ISO-8601 form of an instant, written from `std::chrono` independently of the loader's own parse.
[[nodiscard]] std::string isoOf(std::int64_t secondsSinceEpoch, std::string_view fraction = "") {
    const std::chrono::sys_seconds                    point{std::chrono::seconds{secondsSinceEpoch}};
    const std::chrono::sys_days                       day = std::chrono::floor<std::chrono::days>(point);
    const std::chrono::year_month_day                 ymd{day};
    const std::chrono::hh_mm_ss<std::chrono::seconds> hms{point - day};
    return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}Z", static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()), static_cast<unsigned>(ymd.day()), hms.hours().count(), hms.minutes().count(), hms.seconds().count(), fraction);
}

/// A pass table: a quadratic range and the range rate that is its exact derivative, so that the two columns
/// are consistent by construction and the consistency check's residual is the discretization and nothing else.
struct Pass {
    std::int64_t t0Seconds   = 1'788'000'000LL;
    std::int64_t stepSeconds = 20LL;
    std::size_t  knots       = 200UZ;
    double       range0      = 2'500'000.;
    double       rate0       = -7'000.;
    double       accel       = 12.; ///< m/s^2, so the secant residual is accel*step/2 = 120 m/s

    [[nodiscard]] double       seconds(std::size_t i) const { return static_cast<double>(stepSeconds) * static_cast<double>(i); }
    [[nodiscard]] double       rangeAt(std::size_t i) const { return range0 + rate0 * seconds(i) + 0.5 * accel * seconds(i) * seconds(i); }
    [[nodiscard]] double       rateAt(std::size_t i) const { return rate0 + accel * seconds(i); }
    [[nodiscard]] std::int64_t nsAt(std::size_t i) const { return (t0Seconds + stepSeconds * static_cast<std::int64_t>(i)) * 1'000'000'000LL; }
    [[nodiscard]] std::string  isoAt(std::size_t i) const { return isoOf(t0Seconds + stepSeconds * static_cast<std::int64_t>(i)); }
};

} // namespace

const boost::ut::suite<"trajectory file"> _trajectory_file = [] {
    "round trip, all four column forms and both time scales"_test = [] {
        const Pass pass{.knots = 1'000UZ};

        for (const bool iso : {false, true}) {
            const std::string_view scale = iso ? "utc_iso8601" : "unix_ns";

            std::string offsetOnly = std::format("#!gr4-trajectory 1\ncolumns time offset_hz\ntime_scale {}\n", scale);
            std::string rangeOnly  = std::format("#!gr4-trajectory 1\ncolumns time range_m\ntime_scale {}\n", scale);
            std::string rateOnly   = std::format("#!gr4-trajectory 1\ncolumns time range_rate_m_s\ntime_scale {}\ncarrier_hz 437000000\n", scale);
            std::string bothCols   = std::format("#!gr4-trajectory 1\ncolumns time range_m range_rate_m_s\ntime_scale {}\ncarrier_hz 437000000\n", scale);

            std::vector<double> offsets(pass.knots);
            for (std::size_t i = 0UZ; i < pass.knots; ++i) {
                const std::string when = iso ? pass.isoAt(i) : std::format("{}", pass.nsAt(i));
                offsets[i]             = -1'234.5 + 3.25 * static_cast<double>(i);
                offsetOnly += std::format("{} {:.17g}\n", when, offsets[i]);
                rangeOnly += std::format("{} {:.17g}\n", when, pass.rangeAt(i));
                rateOnly += std::format("{} {:.17g}\n", when, pass.rateAt(i));
                bothCols += std::format("{} {:.17g} {:.17g}\n", when, pass.rangeAt(i), pass.rateAt(i));
            }

            const Trajectory a = loadTrajectory(offsetOnly);
            const Trajectory b = loadTrajectory(rangeOnly);
            const Trajectory c = loadTrajectory(rateOnly);
            const Trajectory d = loadTrajectory(bothCols);

            expect(a.frequency.has_value() && !a.delay.has_value()) << "offset_hz alone makes a frequency schedule";
            expect(!b.frequency.has_value() && b.delay.has_value()) << "range_m alone makes a delay schedule";
            expect(c.frequency.has_value() && !c.delay.has_value()) << "range_rate_m_s alone makes a frequency schedule";
            expect(d.frequency.has_value() && d.delay.has_value()) << "both columns make both schedules";

            std::size_t wrong = 0UZ;
            for (std::size_t i = 0UZ; i < pass.knots; ++i) {
                wrong += (a.frequency->times()[i] != pass.nsAt(i)) ? 1UZ : 0UZ;
                wrong += (a.frequency->offsets()[i] != offsets[i]) ? 1UZ : 0UZ;
                wrong += (b.delay->delays()[i] != delayFor(pass.rangeAt(i))) ? 1UZ : 0UZ;
                wrong += (c.frequency->offsets()[i] != offsetFor(pass.rateAt(i), kCarrierHz)) ? 1UZ : 0UZ;
                wrong += (d.delay->delays()[i] != b.delay->delays()[i]) ? 1UZ : 0UZ;
                wrong += (d.frequency->offsets()[i] != c.frequency->offsets()[i]) ? 1UZ : 0UZ;
            }
            expect(wrong == 0UZ) << "times and values round trip to the last bit under" << scale << "-- mismatches:" << wrong;
            expect(std::isnan(b.carrier_hz)) << "a file with no carrier_hz reports NaN and not a zero to divide by";
            expect(c.carrier_hz == kCarrierHz) << "the carrier is carried";
        }
    };

    "the fractional second is an integer of nanoseconds, exactly"_test = [] {
        constexpr std::int64_t kBase = 1'788'000'000LL;

        struct Case {
            std::string_view fraction;
            std::int64_t     ns;
        };
        constexpr std::array<Case, 5> cases{{{"", 0LL}, {".5", 500'000'000LL}, {".000000001", 1LL}, {".123456789", 123'456'789LL}, {".999999999", 999'999'999LL}}};

        std::string text = "#!gr4-trajectory 1\ncolumns time offset_hz\ntime_scale utc_iso8601\n";
        for (std::size_t i = 0UZ; i < cases.size(); ++i) {
            text += std::format("{} {}\n", isoOf(kBase + static_cast<std::int64_t>(i), cases[i].fraction), i);
        }

        const Trajectory loaded = loadTrajectory(text);
        expect(loaded.frequency.has_value());
        for (std::size_t i = 0UZ; i < cases.size(); ++i) {
            const std::int64_t want = (kBase + static_cast<std::int64_t>(i)) * 1'000'000'000LL + cases[i].ns;
            expect(loaded.frequency->times()[i] == want) << "the fraction" << cases[i].fraction << "is exact to the nanosecond";
        }
    };

    "unix_ns is read as a signed integer, sign and all"_test = [] {
        const Trajectory t = loadTrajectory(wellFormed("time offset_hz", "unix_ns", "-1000000000 1.0\n+0 2.0\n250 3.0\n"));
        expect(t.frequency->times()[0] == -1'000'000'000LL);
        expect(t.frequency->times()[1] == 0LL);
        expect(t.frequency->times()[2] == 250LL);
    };

    "comments, blank lines and a source header"_test = [] {
        const std::string text = "#!gr4-trajectory 1\n"
                                 "# a comment before the headers\n"
                                 "\n"
                                 "columns  time   range_m\n"
                                 "  # an indented comment\n"
                                 "time_scale\tunix_ns\n"
                                 "source   sgp4 over a 2500 km pass, two words and a comma\n"
                                 "\n"
                                 "0            2500000.0\n"
                                 "# a comment between knots\n"
                                 "20000000000  2360000.0\n";
        const Trajectory  t    = loadTrajectory(text);
        expect(t.source == "sgp4 over a 2500 km pass, two words and a comma") << "the source runs to the end of the line, got:" << t.source;
        expect(t.delay->size() == 2UZ);
        expect(t.delay->delays()[0] == delayFor(2'500'000.));
    };

    "every refusal fires and names its line"_test = [] {
        // the version line
        refuses("columns time offset_hz\n", 1UZ, "no version line");
        refuses("", 1UZ, "an empty file");
        refuses("#!gr4-trajectory\n", 1UZ, "a version line with no number");
        refuses("#!gr4-trajectory x\n", 1UZ, "a non-numeric version");
        refuses("#!gr4-trajectory 1 \n", 1UZ, "a trailing space on the version line");
        refuses("#!gr4-trajectory 2\ncolumns time offset_hz\ntime_scale unix_ns\n0 1.0\n1 2.0\n", 1UZ, "format version 2");

        // a byte outside the permitted set, on the line that carries it
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n1 \x01 2.0\n"), 6UZ, "a control byte");
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n1 2.0\r5 3.0\n"), 6UZ, "a carriage return inside a line");

        // the header keys themselves
        refuses("#!gr4-trajectory 1\ncolumns time offset_hz\ntime_scale unix_ns\nfrequency 437\n0 1.0\n1 2.0\n", 4UZ, "an unknown header key");
        refuses("#!gr4-trajectory 1\ncolumns time offset_hz\ntime_scale unix_ns\ncolumns time range_m\n0 1.0\n1 2.0\n", 4UZ, "a duplicate header key");

        // the required keys, named at the line the knots began on
        refuses("#!gr4-trajectory 1\ntime_scale unix_ns\n0 1.0\n1 2.0\n", 3UZ, "no columns key");
        refuses("#!gr4-trajectory 1\ncolumns time offset_hz\n0 1.0\n1 2.0\n", 3UZ, "no time_scale key");
        refuses("#!gr4-trajectory 1\ncolumns time range_rate_m_s\ntime_scale unix_ns\n0 1.0\n1 2.0\n", 4UZ, "range_rate_m_s without a carrier");

        // the columns list
        refuses("#!gr4-trajectory 1\ncolumns offset_hz time\ntime_scale unix_ns\n0 1.0\n", 2UZ, "columns not beginning with time");
        refuses("#!gr4-trajectory 1\ncolumns time range_m range_m\ntime_scale unix_ns\n0 1.0\n", 2UZ, "a quantity named twice");
        refuses("#!gr4-trajectory 1\ncolumns time offset_hz range_rate_m_s\ntime_scale unix_ns\ncarrier_hz 437e6\n0 1.0\n", 2UZ, "offset_hz beside range_rate_m_s");
        refuses("#!gr4-trajectory 1\ncolumns time azimuth_deg\ntime_scale unix_ns\n0 1.0\n", 2UZ, "an unknown quantity");
        refuses("#!gr4-trajectory 1\ncolumns time\ntime_scale unix_ns\n0\n1\n", 2UZ, "no quantity beyond time");

        // the time scale
        refuses("#!gr4-trajectory 1\ncolumns time offset_hz\ntime_scale gps_week\n0 1.0\n", 3UZ, "an unknown time scale");

        // the field count
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n1 2.0 3.0\n"), 6UZ, "one field too many");
        refuses(wellFormed("time range_m range_rate_m_s", "unix_ns", "0 1.0 2.0\n1 2.0\n"), 6UZ, "one field too few");

        // the ISO-8601 time
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-02X10:00:00Z 1.0\n"), 5UZ, "no literal T");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-02T10:00:00 1.0\n"), 5UZ, "no trailing Z");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-02T10:00:00+02:00 1.0\n"), 5UZ, "an offset form");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-9-02T10:00:00Z 1.0\n"), 5UZ, "a one-digit month");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-02T10:00:00.Z 1.0\n"), 5UZ, "a point with no digits");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-02T10:00:00.1234567890Z 1.0\n"), 5UZ, "ten fractional digits");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-02T23:59:60Z 1.0\n"), 5UZ, "a leap second");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-13-02T10:00:00Z 1.0\n"), 5UZ, "month 13");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-32T10:00:00Z 1.0\n"), 5UZ, "day 32");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "2026-09-02T24:00:00Z 1.0\n"), 5UZ, "hour 24");
        refuses(wellFormed("time offset_hz", "utc_iso8601", "9999-09-02T10:00:00Z 1.0\n"), 5UZ, "an instant past the nanosecond axis");

        // the unix_ns time
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n1.5 2.0\n"), 6UZ, "a decimal point in unix_ns");
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n99999999999999999999 2.0\n"), 6UZ, "a unix_ns overflow");

        // monotonicity
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n10 2.0\n10 3.0\n"), 7UZ, "a repeated time");
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n10 2.0\n5 3.0\n"), 7UZ, "a time that goes backwards");

        // the values
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n10 1.0kHz\n"), 6UZ, "trailing characters in a value");
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n10 1.0e\n"), 6UZ, "an incomplete exponent");
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n10 nan\n"), 6UZ, "a non-finite value");
        refuses(wellFormed("time range_m", "unix_ns", "0 1.0\n10 -1.0\n"), 6UZ, "a negative range");
        refuses("#!gr4-trajectory 1\ncolumns time range_rate_m_s\ntime_scale unix_ns\ncarrier_hz -437e6\n0 1.0\n10 2.0\n", 4UZ, "a negative carrier");
        refuses("#!gr4-trajectory 1\ncolumns time range_rate_m_s\ntime_scale unix_ns\ncarrier_hz 0\n0 1.0\n10 2.0\n", 4UZ, "a zero carrier");

        // the knot count
        refuses(wellFormed("time offset_hz", "unix_ns", "0 1.0\n"), 5UZ, "one knot");
        refuses(wellFormed("time offset_hz", "unix_ns", ""), 4UZ, "no knots at all");
    };

    "a refusal leaves nothing partial behind"_test = [] {
        // The schedules are constructed only after the whole file has parsed, so a file that refuses on its
        // last line cannot have handed a caller a table of everything before it.
        expect(throws<std::invalid_argument>([] { (void)loadTrajectory(wellFormed("time offset_hz", "unix_ns", "0 1.0\n10 2.0\n20 3.0\n5 4.0\n")); }));
    };

    "the class-mismatch rule: a range table is not a Doppler table"_test = [] {
        const Trajectory t = loadTrajectory(wellFormed("time range_m", "unix_ns", "0 2500000.0\n20000000000 2360000.0\n"));
        expect(!t.frequency.has_value()) << "a range column never yields a frequency schedule: differentiating a piecewise-linear table leaves the class it lives in";
        expect(t.delay.has_value());

        const Trajectory u = loadTrajectory(wellFormed("time range_rate_m_s", "unix_ns", "0 -7000.0\n20000000000 -6900.0\n"));
        expect(!u.delay.has_value()) << "a range-rate column never yields a delay schedule";
        expect(u.frequency.has_value());
    };

    "a two-knot range ramp is a delay ramp"_test = [] {
        constexpr double kRange0 = 2'500'000.;
        constexpr double kRate   = -7'000.;
        constexpr double kSpan   = 300.;

        const Trajectory t = loadTrajectory(wellFormed("time range_m", "unix_ns", std::format("0 {:.17g}\n{} {:.17g}\n", kRange0, static_cast<std::int64_t>(kSpan) * 1'000'000'000LL, kRange0 + kRate * kSpan)));

        double worst = 0.;
        for (std::size_t i = 0UZ; i <= 10UZ; ++i) {
            const double       seconds = kSpan * static_cast<double>(i) / 10.;
            const std::int64_t t_ns    = static_cast<std::int64_t>(seconds * 1e9);
            worst                      = std::max(worst, std::abs(t.delay->delayAt(t_ns) - (kRange0 + kRate * seconds) / kSpeedOfLight));
        }
        expect(worst < 1e-15) << "the delay follows the range line, worst deviation" << worst << "s";
        expect(t.delay->delayAt(-1'000'000'000LL) == t.delay->delays().front()) << "the first value holds before the table";
        expect(t.delay->delayAt(700'000'000'000LL) == t.delay->delays().back()) << "the last value holds after it";
        expect(t.delay->maxValue() == delayFor(kRange0)) << "maxValue is the largest delay the table reaches";
        expect(t.delay->minValue() == delayFor(kRange0 + kRate * kSpan));
    };

    "the consistency check measures, and its residual is the discretization"_test = [] {
        const Pass pass;

        std::string text = "#!gr4-trajectory 1\ncolumns time range_m range_rate_m_s\ntime_scale unix_ns\ncarrier_hz 437000000\n";
        for (std::size_t i = 0UZ; i < pass.knots; ++i) {
            text += std::format("{} {:.17g} {:.17g}\n", pass.nsAt(i), pass.rangeAt(i), pass.rateAt(i));
        }

        const Trajectory t = loadTrajectory(text);
        expect(t.frequency.has_value() && t.delay.has_value());

        // For a quadratic range sampled every h seconds the secant slope differs from the true derivative by
        // exactly a*h/2, so the figure is arithmetic rather than a tolerance.
        const double bound = pass.accel * static_cast<double>(pass.stepSeconds) / 2.;
        const double worst = worstRangeRateMismatch(*t.delay, *t.frequency, kCarrierHz);
        std::println("worstRangeRateMismatch on a consistent quadratic pass: {:.9f} m/s against the a*h/2 figure of {:.9f}", worst, bound);
        expect(std::abs(worst - bound) <= bound * 1e-9) << "a consistent pair reads a*h/2 and no more";

        // Perturbing one knot away from the residual's own sign raises the figure by exactly the perturbation,
        // which is what says the function measures rather than returning a constant.
        std::vector<double> rates(t.frequency->offsets().begin(), t.frequency->offsets().end());
        rates[pass.knots / 2UZ] -= offsetFor(1., kCarrierHz);
        const FrequencySchedule perturbed(t.frequency->times(), rates);
        const double            after = worstRangeRateMismatch(*t.delay, perturbed, kCarrierHz);
        std::println("one knot moved by 1 m/s: {:.9f} m/s", after);
        expect(std::abs(after - (worst + 1.)) < 1e-6) << "the figure rises by exactly the perturbation";

        expect(throws<std::invalid_argument>([&] { (void)worstRangeRateMismatch(*t.delay, *t.frequency, 0.); })) << "a non-positive carrier refuses";
    };

    "loadTrajectoryFile names the path"_test = [] {
        const std::filesystem::path path = std::filesystem::temp_directory_path() / "qa_TrajectoryFile_scratch.trj";
        {
            std::ofstream out(path, std::ios::binary);
            out << wellFormed("time range_m", "unix_ns", "0 2500000.0\n20000000000 2360000.0\n");
        }
        const Trajectory t = loadTrajectoryFile(path);
        expect(t.delay.has_value() && t.delay->size() == 2UZ);
        std::filesystem::remove(path);

        bool named = false;
        try {
            (void)loadTrajectoryFile(path);
        } catch (const std::invalid_argument& error) {
            named = std::string(error.what()).contains(path.string());
        }
        expect(named) << "an unreadable file names the path it tried";
    };
};

int main() { /* not needed for UT */ }
