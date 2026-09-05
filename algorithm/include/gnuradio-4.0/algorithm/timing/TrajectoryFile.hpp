#ifndef GNURADIO_ALGORITHM_TRAJECTORY_FILE_HPP
#define GNURADIO_ALGORITHM_TRAJECTORY_FILE_HPP

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <gnuradio-4.0/algorithm/timing/FrequencySchedule.hpp>

/**
 * @brief The version-1 trajectory table: a text file of knots, and the two schedules it turns into.
 *
 * A pass is hundreds to thousands of knots. Paired vector settings are the right primitive and the wrong user
 * interface, so a trajectory arrives from whatever propagated it as a file, and the file says what it is:
 *
 * ```
 * #!gr4-trajectory 1
 * columns    time range_m range_rate_m_s
 * time_scale utc_iso8601
 * carrier_hz 437000000
 * source     sgp4 over a 2500 km pass
 * 2026-09-02T10:00:00Z          2500000.0  -7000.0
 * 2026-09-02T10:00:20.5Z        2360000.0  -6900.0
 * ```
 *
 * The version is in-band, on the first line, and nothing else may occupy it. The `#!` prefix makes the file
 * self-identifying to `file(1)` and to a human who opened it in the wrong directory; the integer after the
 * space is the **format** version, and a reader built for version 1 refuses any other value by number rather
 * than guessing. A later version that only adds optional headers still bumps it, because a reader that
 * silently ignores an unknown header is a reader that silently ignores the one that mattered.
 *
 * Every refusal names the line it happened on, because the line number is the only part of a parse failure an
 * operator can act on. There are twenty-six of them and each is a QA case.
 *
 * **The loader never differentiates and never integrates**, and that is a derivation rather than a policy.
 * Both schedules are piecewise linear and therefore continuous; the derivative of a continuous piecewise-linear
 * function is piecewise *constant* with a jump at every knot, and the integral is piecewise *quadratic*.
 * Neither is in the class `PiecewiseLinearSchedule` represents, so either conversion would have to
 * approximate — and the approximation is not small. Differentiating a range table sampled every 20 s across a
 * pass whose range rate swings +/-7 km/s mis-states the slope by up to half a segment's change in it, which at
 * 437 MHz is tens of hertz: large enough to break a narrow carrier loop and small enough that nobody would
 * notice it was invented. So the file states what it states, the loader converts units and nothing else, and a
 * caller who wants both schedules supplies both columns.
 *
 * The loader is the one component of this group that is **not** hot path: it runs once, at staging, off the
 * sample path, and its cost is bounded by the file's size — one pass over the text with no backtracking and no
 * allocation per line beyond the vectors it is filling.
 */
namespace gr::timing {

/**
 * @brief What one trajectory file carried, in the two forms a graph consumes.
 *
 * A schedule is present exactly when the file named the column that produces it. A `range_m`-only file yields
 * no `frequency`, and a caller that needs one is told which column is missing rather than handed a table
 * derived from the wrong quantity.
 */
struct Trajectory {
    std::optional<FrequencySchedule> frequency;                                            ///< from `offset_hz` verbatim, or from `range_rate_m_s` through `offsetFor`
    std::optional<DelaySchedule>     delay;                                                ///< from `range_m` through `delayFor`, in seconds
    double                           carrier_hz{std::numeric_limits<double>::quiet_NaN()}; ///< the `carrier_hz` header, or NaN when the file stated none
    std::string                      source;                                               ///< the `source` header, or empty
};

namespace detail {

/// The quantities a knot line may carry beyond its time, in the order `columns` named them.
enum class TrajectoryQuantity : std::uint8_t { offset_hz, range_m, range_rate_m_s };

[[nodiscard]] inline std::string_view trajectoryQuantityName(TrajectoryQuantity q) noexcept {
    switch (q) {
    case TrajectoryQuantity::offset_hz: return "offset_hz";
    case TrajectoryQuantity::range_m: return "range_m";
    case TrajectoryQuantity::range_rate_m_s: return "range_rate_m_s";
    }
    return "";
}

[[noreturn]] inline void trajectoryRefuse(std::size_t line, std::string_view what, std::string_view offending) { throw std::invalid_argument(std::format("gr::timing::loadTrajectory: line {}: {} — {}", line, what, offending)); }

/// @brief Lines of the text, `\r` stripped from the end of each so a file written on another platform loads.
///
/// A `\r` in the middle of a line is *not* stripped: it then fails the byte check below, which is the honest
/// reading of a line that is not the text it claims to be.
struct TrajectoryLines {
    std::string_view text;
    std::size_t      pos    = 0UZ;
    std::size_t      number = 0UZ;

    [[nodiscard]] bool next(std::string_view& out) noexcept {
        if (pos >= text.size()) {
            return false;
        }
        ++number;
        const std::size_t nl  = text.find('\n', pos);
        std::string_view  raw = (nl == std::string_view::npos) ? text.substr(pos) : text.substr(pos, nl - pos);
        pos                   = (nl == std::string_view::npos) ? text.size() : nl + 1UZ;
        if (!raw.empty() && raw.back() == '\r') {
            raw.remove_suffix(1UZ);
        }
        out = raw;
        return true;
    }
};

/// @brief Whitespace-separated fields of one line, without allocating.
struct TrajectoryFields {
    std::string_view line;
    std::size_t      pos = 0UZ;

    [[nodiscard]] static bool blank(char c) noexcept { return c == ' ' || c == '\t'; }

    [[nodiscard]] bool next(std::string_view& out) noexcept {
        while (pos < line.size() && blank(line[pos])) {
            ++pos;
        }
        if (pos >= line.size()) {
            return false;
        }
        const std::size_t start = pos;
        while (pos < line.size() && !blank(line[pos])) {
            ++pos;
        }
        out = line.substr(start, pos - start);
        return true;
    }

    /// What remains of the line from the current position, leading blanks dropped: a header value that runs to
    /// the end of the line.
    [[nodiscard]] std::string_view rest() const noexcept {
        std::size_t at = pos;
        while (at < line.size() && blank(line[at])) {
            ++at;
        }
        return line.substr(at);
    }
};

/// The file is US-ASCII. A byte outside the printable range, tab excepted, turns "someone handed me a
/// recording instead of a table" into one refusal instead of ten thousand.
inline void trajectoryCheckBytes(std::size_t line, std::string_view raw) {
    for (const char c : raw) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte != '\t' && (byte < 0x20U || byte > 0x7EU)) {
            trajectoryRefuse(line, std::format("byte 0x{:02X} is outside the printable US-ASCII the format admits", byte), raw);
        }
    }
}

/// A header key begins with a letter or an underscore and a knot's time never does — it begins with a digit or
/// a sign under either time scale. That is what separates the two, and it is why the header block can end at
/// the first knot line while an unknown key on a line that is plainly a header still refuses.
[[nodiscard]] inline bool trajectoryLooksLikeKey(std::string_view token) noexcept {
    if (token.empty()) {
        return false;
    }
    const char c = token.front();
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

[[nodiscard]] inline double trajectoryValue(std::size_t line, std::string_view field, std::string_view what) {
    double out           = 0.;
    const auto [end, ec] = std::from_chars(field.data(), field.data() + field.size(), out, std::chars_format::general);
    if (ec != std::errc{} || end != field.data() + field.size()) {
        trajectoryRefuse(line, std::format("{} is not a number, or carries trailing characters", what), field);
    }
    if (!std::isfinite(out)) {
        trajectoryRefuse(line, std::format("{} is not finite", what), field);
    }
    return out;
}

/// `unix_ns`: an optional sign then decimal digits, and nothing else. A nanosecond count never passes through
/// a floating type, so there is no exponent form and no decimal point here.
[[nodiscard]] inline std::int64_t trajectoryUnixNs(std::size_t line, std::string_view field) {
    std::string_view digits = field;
    if (!digits.empty() && digits.front() == '+') {
        digits.remove_prefix(1UZ);
    }
    std::int64_t out     = 0;
    const auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), out, 10);
    if (ec == std::errc::result_out_of_range) {
        trajectoryRefuse(line, "the unix_ns time overflows the signed 64-bit nanosecond axis", field);
    }
    if (ec != std::errc{} || end != digits.data() + digits.size()) {
        trajectoryRefuse(line, "the unix_ns time is not a decimal integer", field);
    }
    return out;
}

[[nodiscard]] inline bool trajectoryDigits(std::string_view field, std::size_t at, std::size_t count) noexcept {
    if (at + count > field.size()) {
        return false;
    }
    for (std::size_t i = 0UZ; i < count; ++i) {
        const char c = field[at + i];
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline int trajectoryInt(std::string_view field, std::size_t at, std::size_t count) noexcept {
    int out = 0;
    for (std::size_t i = 0UZ; i < count; ++i) {
        out = out * 10 + (field[at + i] - '0');
    }
    return out;
}

/**
 * @brief `YYYY-MM-DDThh:mm:ss[.f{1,9}]Z` to nanoseconds since the Unix epoch, in integers throughout.
 *
 * Fixed widths on every field; `T` and `Z` are required and literal. No timezone offset form is accepted,
 * because honoring one requires a rule about which offsets are legal and this format does not want that rule:
 * a producer that has an offset converts before writing.
 *
 * The fractional part is parsed as an **integer**, never as a fraction: the digits after the point are read as
 * a decimal integer and scaled by `10^(9-d)` for `d` digits, so `.5` is exactly 500000000 ns and `.000000001`
 * is exactly 1 ns. More than nine digits refuses, because truncating them is silent loss.
 *
 * A seconds field of `60` refuses with its reason named: the axis is a plain nanosecond count, and a leap
 * second has no representation on it.
 */
[[nodiscard]] inline std::int64_t trajectoryIso8601Ns(std::size_t line, std::string_view field) {
    constexpr std::size_t kFixed = 19UZ; // YYYY-MM-DDThh:mm:ss

    const bool shaped = field.size() > kFixed                                     //
                        && trajectoryDigits(field, 0UZ, 4UZ) && field[4] == '-'   //
                        && trajectoryDigits(field, 5UZ, 2UZ) && field[7] == '-'   //
                        && trajectoryDigits(field, 8UZ, 2UZ) && field[10] == 'T'  //
                        && trajectoryDigits(field, 11UZ, 2UZ) && field[13] == ':' //
                        && trajectoryDigits(field, 14UZ, 2UZ) && field[16] == ':' && trajectoryDigits(field, 17UZ, 2UZ);
    if (!shaped) {
        trajectoryRefuse(line, "the time is not YYYY-MM-DDThh:mm:ss[.f{1,9}]Z at fixed field widths", field);
    }

    std::int64_t fraction = 0;
    if (field[kFixed] == '.') {
        const std::size_t first = kFixed + 1UZ;
        std::size_t       last  = first;
        while (last < field.size() && field[last] >= '0' && field[last] <= '9') {
            ++last;
        }
        const std::size_t digits = last - first;
        if (digits == 0UZ) {
            trajectoryRefuse(line, "the fractional second carries no digits", field);
        }
        if (digits > 9UZ) {
            trajectoryRefuse(line, std::format("the fractional second carries {} digits, past the nine a nanosecond axis holds", digits), field);
        }
        if (last + 1UZ != field.size() || field[last] != 'Z') {
            trajectoryRefuse(line, "the time does not end in the literal Z, and no offset form is accepted", field);
        }
        fraction = trajectoryInt(field, first, digits);
        for (std::size_t i = digits; i < 9UZ; ++i) {
            fraction *= 10;
        }
    } else if (field[kFixed] != 'Z' || field.size() != kFixed + 1UZ) {
        trajectoryRefuse(line, "the time does not end in the literal Z, and no offset form is accepted", field);
    }

    const int second = trajectoryInt(field, 17UZ, 2UZ);
    if (second == 60) {
        trajectoryRefuse(line, "a seconds field of 60 names a leap second, which a plain nanosecond axis cannot represent", field);
    }

    const int year   = trajectoryInt(field, 0UZ, 4UZ);
    const int month  = trajectoryInt(field, 5UZ, 2UZ);
    const int day    = trajectoryInt(field, 8UZ, 2UZ);
    const int hour   = trajectoryInt(field, 11UZ, 2UZ);
    const int minute = trajectoryInt(field, 14UZ, 2UZ);

    const std::chrono::year_month_day date{std::chrono::year{year}, std::chrono::month{static_cast<unsigned>(month)}, std::chrono::day{static_cast<unsigned>(day)}};
    if (!date.ok() || hour > 23 || minute > 59 || second > 59) {
        trajectoryRefuse(line, "a calendar field is out of range", field);
    }

    // 4-digit years reach 9999 and the axis reaches 2262-04-11; the product below is what runs out first, so it
    // is checked against the axis rather than against the year.
    constexpr std::int64_t kMaxSeconds = std::numeric_limits<std::int64_t>::max() / 1'000'000'000LL;

    const std::int64_t seconds = std::chrono::sys_days(date).time_since_epoch().count() * 86'400LL + hour * 3'600LL + minute * 60LL + second;
    if (seconds > kMaxSeconds || seconds < -kMaxSeconds) {
        trajectoryRefuse(line, std::format("the instant is {} s from the epoch, outside the +/-{} s the signed 64-bit nanosecond axis holds", seconds, kMaxSeconds), field);
    }
    return seconds * 1'000'000'000LL + fraction;
}

} // namespace detail

/**
 * @brief Load a version-1 trajectory table from its text.
 *
 * A pure function of its argument: no global state and no locale dependence — `std::from_chars` and
 * `std::chrono` only, because `strtod` under a comma locale is exactly the class of defect this avoids.
 *
 * Throws `std::invalid_argument` with the line number in the message, matching `FrequencySchedule`'s landed
 * behavior, and nothing partial escapes: the schedules are constructed after the whole file has parsed.
 */
[[nodiscard]] inline Trajectory loadTrajectory(std::string_view text) {
    using detail::TrajectoryQuantity;
    using detail::trajectoryRefuse;

    detail::TrajectoryLines lines{text};
    std::string_view        raw;

    if (!lines.next(raw)) {
        trajectoryRefuse(1UZ, "the file is empty and carries no version line", "");
    }
    detail::trajectoryCheckBytes(1UZ, raw);
    {
        constexpr std::string_view kMagic = "#!gr4-trajectory ";
        if (!raw.starts_with(kMagic)) {
            trajectoryRefuse(1UZ, "the first line is not `#!gr4-trajectory <n>`", raw);
        }
        const std::string_view digits  = raw.substr(kMagic.size());
        int                    version = 0;
        const auto [end, ec]           = std::from_chars(digits.data(), digits.data() + digits.size(), version, 10);
        if (digits.empty() || ec != std::errc{} || end != digits.data() + digits.size()) {
            trajectoryRefuse(1UZ, "the first line is not `#!gr4-trajectory <n>`", raw);
        }
        if (version != 1) {
            trajectoryRefuse(1UZ, std::format("format version {} — this reader reads version 1", version), raw);
        }
    }

    std::optional<std::size_t> columnsLine;
    std::optional<std::size_t> timeScaleLine;
    std::optional<std::size_t> carrierLine;
    std::optional<std::size_t> sourceLine;

    std::vector<TrajectoryQuantity> quantities;
    bool                            isoTime   = false;
    double                          carrierHz = std::numeric_limits<double>::quiet_NaN();
    std::string                     source;

    std::string_view pendingKnot;
    std::size_t      pendingLine = 0UZ;
    bool             havePending = false;

    while (lines.next(raw)) {
        detail::trajectoryCheckBytes(lines.number, raw);

        detail::TrajectoryFields fields{raw};
        std::string_view         key;
        if (!fields.next(key) || key.front() == '#') {
            continue;
        }
        if (!detail::trajectoryLooksLikeKey(key)) {
            pendingKnot = raw;
            pendingLine = lines.number;
            havePending = true;
            break;
        }

        const auto once = [&](std::optional<std::size_t>& seen) {
            if (seen.has_value()) {
                trajectoryRefuse(lines.number, std::format("header key `{}` was already given on line {}", key, *seen), raw);
            }
            seen = lines.number;
        };

        if (key == "columns") {
            once(columnsLine);
            std::string_view first;
            if (!fields.next(first) || first != "time") {
                trajectoryRefuse(lines.number, "`columns` must begin with `time`", raw);
            }
            std::string_view name;
            while (fields.next(name)) {
                TrajectoryQuantity quantity{};
                if (name == "offset_hz") {
                    quantity = TrajectoryQuantity::offset_hz;
                } else if (name == "range_m") {
                    quantity = TrajectoryQuantity::range_m;
                } else if (name == "range_rate_m_s") {
                    quantity = TrajectoryQuantity::range_rate_m_s;
                } else {
                    trajectoryRefuse(lines.number, std::format("`columns` names the unknown quantity `{}`", name), raw);
                }
                if (std::ranges::find(quantities, quantity) != quantities.end()) {
                    trajectoryRefuse(lines.number, std::format("`columns` names `{}` twice", name), raw);
                }
                // The two are the same quantity through `offsetFor`, so a file carrying both carries either a
                // redundancy or a contradiction, and there is no rule that picks the right one.
                const bool clash = (quantity == TrajectoryQuantity::offset_hz && std::ranges::find(quantities, TrajectoryQuantity::range_rate_m_s) != quantities.end()) //
                                   || (quantity == TrajectoryQuantity::range_rate_m_s && std::ranges::find(quantities, TrajectoryQuantity::offset_hz) != quantities.end());
                if (clash) {
                    trajectoryRefuse(lines.number, "`columns` names both `offset_hz` and `range_rate_m_s`, which are one quantity in two units", raw);
                }
                quantities.push_back(quantity);
            }
            if (quantities.empty()) {
                trajectoryRefuse(lines.number, "`columns` names no quantity beyond `time`, so the table carries nothing", raw);
            }
        } else if (key == "time_scale") {
            once(timeScaleLine);
            std::string_view scale;
            if (!fields.next(scale) || (scale != "utc_iso8601" && scale != "unix_ns")) {
                trajectoryRefuse(lines.number, "`time_scale` is `utc_iso8601` or `unix_ns`", raw);
            }
            isoTime = scale == "utc_iso8601";
        } else if (key == "carrier_hz") {
            once(carrierLine);
            std::string_view value;
            if (!fields.next(value)) {
                trajectoryRefuse(lines.number, "`carrier_hz` carries no value", raw);
            }
            carrierHz = detail::trajectoryValue(lines.number, value, "`carrier_hz`");
            if (!(carrierHz > 0.)) {
                trajectoryRefuse(lines.number, std::format("`carrier_hz` is {}, and a carrier is positive", carrierHz), raw);
            }
        } else if (key == "source") {
            once(sourceLine);
            source = std::string(fields.rest());
        } else {
            trajectoryRefuse(lines.number, std::format("unknown header key `{}`", key), raw);
        }
    }

    const std::size_t firstKnotLine = havePending ? pendingLine : lines.number + 1UZ;
    if (!columnsLine.has_value()) {
        trajectoryRefuse(firstKnotLine, "the header block ended without a `columns` key", "");
    }
    if (!timeScaleLine.has_value()) {
        trajectoryRefuse(firstKnotLine, "the header block ended without a `time_scale` key", "");
    }
    const bool needsCarrier = std::ranges::find(quantities, TrajectoryQuantity::range_rate_m_s) != quantities.end();
    if (needsCarrier && !carrierLine.has_value()) {
        trajectoryRefuse(firstKnotLine, "`columns` names `range_rate_m_s`, which needs a `carrier_hz` header to become a frequency", "");
    }

    const std::size_t nFields = quantities.size() + 1UZ;

    std::vector<std::int64_t> times;
    std::vector<double>       offsets;
    std::vector<double>       delays;
    std::size_t               previousLine = 0UZ;

    const auto knot = [&](std::size_t number, std::string_view line) {
        detail::TrajectoryFields fields{line};
        std::string_view         field;
        std::size_t              count = 0UZ;
        std::string_view         seen[4];
        while (fields.next(field)) {
            if (count < 4UZ) {
                seen[count] = field;
            }
            ++count;
        }
        if (count != nFields) {
            trajectoryRefuse(number, std::format("{} fields against the {} `columns` declares", count, nFields), line);
        }

        const std::int64_t t_ns = isoTime ? detail::trajectoryIso8601Ns(number, seen[0]) : detail::trajectoryUnixNs(number, seen[0]);
        if (!times.empty() && t_ns <= times.back()) {
            trajectoryRefuse(number, std::format("the time {} ns is at or before line {}'s {} ns — times must strictly increase", t_ns, previousLine, times.back()), line);
        }
        times.push_back(t_ns);

        for (std::size_t i = 0UZ; i < quantities.size(); ++i) {
            const std::string_view name  = detail::trajectoryQuantityName(quantities[i]);
            const double           value = detail::trajectoryValue(number, seen[i + 1UZ], std::format("`{}`", name));
            switch (quantities[i]) {
            case TrajectoryQuantity::offset_hz: offsets.push_back(value); break;
            case TrajectoryQuantity::range_rate_m_s: offsets.push_back(offsetFor(value, carrierHz)); break;
            case TrajectoryQuantity::range_m:
                if (value < 0.) {
                    trajectoryRefuse(number, std::format("`range_m` is {}, and a slant range is not negative", value), line);
                }
                delays.push_back(delayFor(value));
                break;
            }
        }
        previousLine = number;
    };

    if (havePending) {
        knot(pendingLine, pendingKnot);
    }
    while (lines.next(raw)) {
        detail::trajectoryCheckBytes(lines.number, raw);
        detail::TrajectoryFields fields{raw};
        std::string_view         first;
        if (!fields.next(first) || first.front() == '#') {
            continue;
        }
        knot(lines.number, raw);
    }

    if (times.size() < PiecewiseLinearSchedule::kMinKnots || times.size() > PiecewiseLinearSchedule::kMaxKnots) {
        trajectoryRefuse(lines.number, std::format("{} knots — the table holds between {} and {}", times.size(), PiecewiseLinearSchedule::kMinKnots, PiecewiseLinearSchedule::kMaxKnots), "");
    }

    Trajectory out;
    out.carrier_hz = carrierHz;
    out.source     = std::move(source);
    if (!offsets.empty()) {
        out.frequency.emplace(times, offsets);
    }
    if (!delays.empty()) {
        out.delay.emplace(times, delays);
    }
    return out;
}

/// @brief The same, from a file. The path joins the message so a staging failure names what it tried to open.
[[nodiscard]] inline Trajectory loadTrajectoryFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::invalid_argument(std::format("gr::timing::loadTrajectoryFile: cannot read `{}`", path.string()));
    }
    const std::string text{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    try {
        return loadTrajectory(text);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(std::format("gr::timing::loadTrajectoryFile: `{}`: {}", path.string(), error.what()));
    }
}

/**
 * @brief The largest radial velocity the two halves of one trajectory disagree by, in meters per second.
 *
 * A file may carry both a range column and a range-rate column, which are genuinely independent quantities and
 * can therefore genuinely disagree. The loader does not police them — a parser that half-validates is a parser
 * whose refusals nobody can predict — so the check lives here, as one free function, so that a second
 * implementation of the same arithmetic never appears.
 *
 * Both schedules are evaluated at every knot time of both. The frequency schedule implies `-offset*c/f_c`; the
 * delay schedule implies `c * dtau/dt` on the segment that time falls in. The largest absolute difference is
 * returned and the caller applies whatever tolerance its producer earns.
 *
 * The figure a consistent pair produces is not zero, and its size is arithmetic rather than tolerance: the
 * secant slope of a piecewise-linear table sampled every `h` seconds from a curve whose second derivative is
 * bounded by `a` differs from the true derivative by at most `a*h/2`, so that is what a consistent pair reads
 * and what the QA asserts against.
 */
[[nodiscard]] inline double worstRangeRateMismatch(const DelaySchedule& delay, const FrequencySchedule& frequency, double carrierHz) {
    if (!(carrierHz > 0.) || !std::isfinite(carrierHz)) {
        throw std::invalid_argument(std::format("gr::timing::worstRangeRateMismatch: carrier {} Hz — the carrier must be positive and finite", carrierHz));
    }

    double worst = 0.;
    for (const std::span<const std::int64_t> knots : {delay.times(), frequency.times()}) {
        for (const std::int64_t t_ns : knots) {
            const double fromFrequency = -frequency.offsetAt(t_ns) * kSpeedOfLight / carrierHz;
            const double fromDelay     = kSpeedOfLight * delay.segmentSlopePerSecond(t_ns);
            worst                      = std::max(worst, std::abs(fromFrequency - fromDelay));
        }
    }
    return worst;
}

} // namespace gr::timing

#endif // GNURADIO_ALGORITHM_TRAJECTORY_FILE_HPP
