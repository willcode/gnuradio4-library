#include <benchmark.hpp>

#include <gnuradio-4.0/algorithm/timing/TrajectoryFile.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>

namespace {

/// The table's own ceiling, which is the only size worth measuring: the loader runs once at staging, so what a
/// caller needs to know is what the worst file costs, not what a typical one does.
constexpr std::size_t kKnots   = std::size_t{1} << 20;
constexpr std::size_t kRepeats = 3UZ;

/// A two-column pass table in one of the two time scales. The unix_ns arm is the cheap one by construction --
/// one `from_chars` against a fixed-width calendar parse -- so the pair is what says what the calendar costs.
[[nodiscard]] std::string passFile(bool iso) {
    constexpr std::int64_t kEpochSeconds = 1'788'000'000LL;

    std::string text = std::format("#!gr4-trajectory 1\ncolumns time range_m range_rate_m_s\ntime_scale {}\ncarrier_hz 437000000\n", iso ? "utc_iso8601" : "unix_ns");
    text.reserve(kKnots * 48UZ);
    for (std::size_t i = 0UZ; i < kKnots; ++i) {
        const std::int64_t seconds = kEpochSeconds + static_cast<std::int64_t>(i);
        const double       range   = 2'500'000. - 1.9073486e0 * static_cast<double>(i);
        const double       rate    = -7'000. + 1.3351e-2 * static_cast<double>(i);
        if (iso) {
            const std::chrono::sys_days                       day = std::chrono::floor<std::chrono::days>(std::chrono::sys_seconds{std::chrono::seconds{seconds}});
            const std::chrono::year_month_day                 ymd{day};
            const std::chrono::hh_mm_ss<std::chrono::seconds> hms{std::chrono::sys_seconds{std::chrono::seconds{seconds}} - day};
            text += std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.123456789Z {:.6f} {:.6f}\n", static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()), static_cast<unsigned>(ymd.day()), hms.hours().count(), hms.minutes().count(), hms.seconds().count(), range, rate);
        } else {
            text += std::format("{} {:.6f} {:.6f}\n", seconds * 1'000'000'000LL + 123'456'789LL, range, rate);
        }
    }
    return text;
}

} // namespace

/// The loader is the one component of this group that is off the sample path, so what it owes is a bound rather
/// than a rate: one pass over the text with no backtracking, at a cost the file's own size sets.
void benchTrajectoryLoader() {
    using namespace benchmark;
    using boost::ut::operator""_test;

    "loadTrajectory: ns/knot at the table's ceiling"_test = [] {
        for (const bool iso : {false, true}) {
            const std::string text = passFile(iso);

            const auto name                                                  = std::format("loadTrajectory, {:11}, {} knots, {} MB", iso ? "utc_iso8601" : "unix_ns", kKnots, text.size() >> 20);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kKnots) = [&] {
                gr::timing::Trajectory loaded = gr::timing::loadTrajectory(text);
                ::benchmark::force_to_memory(loaded);
            };
        }
        ::benchmark::results::add_separator();
    };
}

inline const boost::ut::suite<"trajectory loader benchmarks"> _trajectory_file_bm = [] { benchTrajectoryLoader(); };

int main() { /* not needed by the UT framework */ }
