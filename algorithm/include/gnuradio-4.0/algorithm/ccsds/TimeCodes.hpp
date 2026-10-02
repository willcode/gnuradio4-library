#ifndef GNURADIO_ALGORITHM_CCSDS_TIME_CODES_HPP
#define GNURADIO_ALGORITHM_CCSDS_TIME_CODES_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string_view>

/**
 * @brief The CCSDS time codes of CCSDS 301.0-B-4, both directions, on one nanosecond axis.
 *
 * The time codes form four families on one axis. The families are the unsegmented binary code (CUC), the
 * day-segmented binary code (CDS), the calendar-segmented binary-coded-decimal code (CCS) and the two
 * ASCII calendar codes. The axis is a signed 64-bit count of nanoseconds since the Unix epoch, floored.
 * **No conversion in either direction uses a `double`.** A `double` has 53 significand bits. A
 * nanosecond count needs up to 63. An instant near `2^53` ns comes back wrong from any conversion that
 * passes through a `double`.
 *
 * **The 1958 epoch is derived here.** 301.0-B-4 3.2.1 makes the recommended epoch 1958 January 1 (TAI).
 * 3.3.1 counts CDS days from the same date. 1958 to 1970 is twelve years, `12 * 365 = 4380` days. The
 * leap days of 1960, 1964 and 1968 add three. No century year falls in the span, so Annex A's
 * leap-year rule adds nothing more. The span is **4383 days**, or `378 691 200` seconds.
 *
 * **`epoch_ns` is the code's epoch on the Unix axis.** The Level-1 CUC and CDS epoch is
 * `-kEpoch1958Ns`. An agency-defined 1950 epoch is `-631 152 000 * 10^9`. 301.0-B-4 B3.2 states the
 * 1958-to-1950 difference as exactly 2922 days, and `(4383 + 2922) * 86400 = 631 152 000`. The 1950
 * epoch is one agency's choice and has no constant here. 3.2.2 gives two CUC identifications, `001`
 * for the 1958 epoch and `010` for an Agency-defined epoch. The caller supplies the epoch of a
 * Level-2 code.
 *
 * **Two time scales, and no leap-second table.** CUC is TAI-based. 3.2.1 says "This time code is not
 * UTC-based and leap-second corrections do not apply". The Unix axis is UTC-based. Converting between
 * them needs `TAI - UTC` at the instant in question. That value is an integer that changes at
 * announced dates. It cannot be derived from the code. The caller therefore supplies it, and the
 * conversion is `t_ns_utc = t_ns_tai - tai_utc_offset_s * 10^9`. `Instant::tai` states the scale of
 * the result. With the default offset of zero the result is on the TAI scale, and `Instant::tai` is
 * set. CDS, CCS and ASCII are UTC-based (3.3.1, 3.4.1, 3.5.1). The offset does not apply to them.
 * This kernel holds no settings and does not refuse an offset supplied with those codes. Refusing one
 * is left to the caller.
 *
 * **A leap second is flagged and not resolved.** Annex A admits a millisecond of day up to
 * 86 400 999 and a CCS second of 60. Both name UTC's 23:59:60. POSIX time cannot represent that
 * second. This is a property of the axis and not a defect of the code. The same expression converts
 * such a value. The result is the first instant of the following day or minute. `Instant::leap` is
 * set, and a caller with a leap-second policy can apply it. The kernel spreads no leap second over
 * other seconds, reads no table and applies no policy.
 *
 * **Every length is checked before it is used as an index.** Every function is `noexcept`, and
 * nothing throws. A malformed received code cannot stop the caller. Each conversion returns its value
 * in a `std::expected`, and each malformed input yields its own `TimeStatus` as the error.
 */
namespace gr::ccsds {

/// @brief 1958 January 1 to 1970 January 1, derived above.
inline constexpr std::int64_t kEpoch1958Days = 4383;

/// @brief The same span in seconds, `4383 * 86400`.
inline constexpr std::int64_t kEpoch1958Secs = kEpoch1958Days * 86400;

/// @brief The same span in nanoseconds. The Level-1 epoch's position on the Unix axis is its negation.
inline constexpr std::int64_t kEpoch1958Ns = kEpoch1958Secs * 1'000'000'000;

inline constexpr std::int64_t kNsPerSecond      = 1'000'000'000;
inline constexpr std::int64_t kNsPerMillisecond = 1'000'000;
inline constexpr std::int64_t kNsPerDay         = 86400 * kNsPerSecond;

/// @brief The largest whole second the axis holds, `(2^63 - 1) / 10^9`, about 292 years.
inline constexpr std::int64_t kMaxAxisSeconds = 9'223'372'036;

/// @brief The four families, plus the two ASCII variations 301.0-B-4 3.5.1 defines.
enum class TimeCodeKind { cuc, cds, ccs, ascii_a, ascii_b };

/// @brief The reason a conversion failed. Every value is produced by a distinct, stated input.
enum class TimeStatus {
    short_field,          //!< fewer octets or characters than the declared layout needs
    bad_p_field,          //!< an undefined extension octet, or a P-field where none is allowed (3.1.1)
    reserved_code,        //!< identification 000, 011 or 111 (3.1.1)
    no_epoch_defined,     //!< identification 110, an agency-defined Level 3 or 4 code (3.6)
    reserved_resolution,  //!< CDS bits 6-7 = '11' (3.3.2), or CCS bits 5-7 = '111' (3.4.2)
    bad_bcd,              //!< a CCS nibble above 9 (3.4.1)
    bad_character,        //!< an ASCII character where the grammar allows none (3.5.1)
    incomplete_calendar,  //!< a left-truncated ASCII subset (3.5.1.3)
    segment_out_of_range, //!< a field outside Annex A's range
    axis_overflow         //!< the instant is outside the signed 64-bit nanosecond axis
};

/// @brief The time code's shape, as a P-field declares it or as a caller supplies it where the P-field is implicit.
struct Layout {
    TimeCodeKind kind             = TimeCodeKind::cuc;
    bool         custom_epoch     = false; //!< identification 010 (CUC) or bit 4 set (CDS)
    std::uint8_t coarse_octets    = 1U;    //!< CUC, 1 to 7 (3.2.2)
    std::uint8_t fine_octets      = 0U;    //!< CUC, 0 to 10 (3.2.2)
    std::uint8_t day_octets       = 2U;    //!< CDS, 2 or 3 (3.3.2)
    std::uint8_t submilli_octets  = 0U;    //!< CDS, 0, 2 or 4 (3.3.2)
    bool         day_of_year      = false; //!< CCS, the 3.4.1.2 variation
    std::uint8_t subsecond_octets = 0U;    //!< CCS, 0 to 6 (3.4.2)
    std::size_t  p_field_octets   = 0UZ;   //!< 0, 1 or 2
    std::size_t  t_field_octets   = 0UZ;   //!< derived from the fields above

    [[nodiscard]] bool operator==(const Layout&) const noexcept = default;
};

/// @brief One instant, with the parts that did not fit on the axis kept beside it.
struct Instant {
    std::int64_t  ns                         = 0;     //!< nanoseconds since the Unix epoch, floored
    std::uint32_t sub_ns_ps                  = 0U;    //!< 0 to 999, the resolution below a nanosecond
    bool          leap                       = false; //!< the value named a leap second
    bool          tai                        = false; //!< the value is on the TAI scale, not UTC
    std::uint8_t  precision_discarded_digits = 0U;    //!< ASCII fraction digits past the twelfth

    [[nodiscard]] bool operator==(const Instant&) const noexcept = default;
};

/// @brief The longest P-field, a CUC code with its extension octet (3.2.2).
inline constexpr std::size_t kMaxPFieldOctets = 2UZ;

/// @brief The longest T-field, a CUC code of seven coarse and ten fine octets (3.2.2).
inline constexpr std::size_t kMaxTFieldOctets = 17UZ;

/// @brief The longest ASCII code, form A with nine fraction digits and the terminator (3.5.1).
inline constexpr std::size_t kMaxAsciiCharacters = 30UZ;

/// @brief An encoded field held by value. The first `size` elements of `data` are the field.
template<typename Element, std::size_t Capacity>
struct EncodedField {
    std::array<Element, Capacity> data{};
    std::size_t                   size = 0UZ;

    [[nodiscard]] constexpr std::span<const Element> view() const noexcept { return {data.data(), size}; }
    [[nodiscard]] bool                               operator==(const EncodedField&) const noexcept = default;
};

using PField    = EncodedField<std::uint8_t, kMaxPFieldOctets>;
using TField    = EncodedField<std::uint8_t, kMaxTFieldOctets>;
using AsciiCode = EncodedField<char, kMaxAsciiCharacters>;

namespace detail {

/// @brief `count` octets read most significant first, the numbering of 301.0-B-4 1.5.
[[nodiscard]] inline constexpr std::uint64_t readBigEndian(std::span<const std::uint8_t> octets, std::size_t offset, std::size_t count) noexcept {
    std::uint64_t value = 0ULL;
    for (std::size_t i = 0UZ; i < count; ++i) {
        value = (value << 8U) | octets[offset + i];
    }
    return value;
}

/// @brief The inverse of `readBigEndian`. A field wider than eight octets is zero above the value's
/// eight octets. A shift by 64 or more is undefined and is not made.
inline constexpr void writeBigEndian(std::span<std::uint8_t> octets, std::size_t offset, std::size_t count, std::uint64_t value) noexcept {
    for (std::size_t i = 0UZ; i < count; ++i) {
        octets[offset + count - 1UZ - i] = i < 8UZ ? static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFULL) : std::uint8_t{0U};
    }
}

/// @brief The room the axis leaves above `epochNs`, in nanoseconds, computed without overflow.
[[nodiscard]] inline constexpr std::uint64_t roomAbove(std::int64_t epochNs) noexcept {
    constexpr std::uint64_t kMaxNs = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (epochNs >= 0) {
        return kMaxNs - static_cast<std::uint64_t>(epochNs);
    }
    return kMaxNs + (static_cast<std::uint64_t>(-(epochNs + 1)) + 1ULL);
}

/// @brief The two BCD digits of one octet, or `false` where either nibble is not a digit (3.4.1).
[[nodiscard]] inline constexpr bool bcdOctet(std::uint8_t octet, unsigned& out) noexcept {
    const unsigned high = (octet >> 4U) & 0x0FU;
    const unsigned low  = octet & 0x0FU;
    if (high > 9U || low > 9U) {
        return false;
    }
    out = high * 10U + low;
    return true;
}

[[nodiscard]] inline constexpr std::uint8_t toBcd(unsigned value) noexcept { return static_cast<std::uint8_t>(((value / 10U) << 4U) | (value % 10U)); }

[[nodiscard]] inline constexpr std::int64_t powerOfTen(unsigned exponent) noexcept {
    std::int64_t value = 1;
    for (unsigned i = 0U; i < exponent; ++i) {
        value *= 10;
    }
    return value;
}

/// @brief Days of the given month, Annex A's rule, through the standard library's own leap-year test.
[[nodiscard]] inline unsigned daysInMonth(int year, unsigned month) noexcept {
    static constexpr std::array<unsigned, 12UZ> kLengths{31U, 28U, 31U, 30U, 31U, 30U, 31U, 31U, 30U, 31U, 30U, 31U};
    if (month == 2U && std::chrono::year{year}.is_leap()) {
        return 29U;
    }
    return kLengths[month - 1UZ];
}

/// @brief Days since the Unix epoch for a civil date, exact integer proleptic-Gregorian arithmetic.
[[nodiscard]] inline std::int64_t daysFromCivil(int year, unsigned month, unsigned day) noexcept {
    const std::chrono::sys_days date{std::chrono::year{year} / std::chrono::month{month} / std::chrono::day{day}};
    return date.time_since_epoch().count();
}

/// @brief The civil date of a day count, the inverse of `daysFromCivil`.
inline void civilFromDays(std::int64_t days, int& year, unsigned& month, unsigned& day) noexcept {
    const std::chrono::year_month_day date{std::chrono::sys_days{std::chrono::days{days}}};
    year  = static_cast<int>(date.year());
    month = static_cast<unsigned>(date.month());
    day   = static_cast<unsigned>(date.day());
}

/**
 * @brief `epochNs + elapsed`, refused with a status instead of wrapped. `elapsed` is unsigned.
 *
 * Once the room check passes, the sum is representable. For an epoch far below the axis, `elapsed`
 * itself can exceed the largest signed 64-bit value. The addition is therefore done in unsigned
 * arithmetic and converted back once. Both steps are modular, and neither overflows.
 */
[[nodiscard]] inline constexpr std::expected<std::int64_t, TimeStatus> addToEpoch(std::int64_t epochNs, std::uint64_t elapsedNs) noexcept {
    if (elapsedNs > roomAbove(epochNs)) {
        return std::unexpected(TimeStatus::axis_overflow);
    }
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(epochNs) + elapsedNs);
}

/**
 * @brief `seconds * 10^9 + subNs` on the axis, refused with a status instead of wrapped.
 *
 * The calendar codes reach the axis from a whole second and a part of a second. They have no epoch
 * and elapsed count, and `addToEpoch` does not fit them. `seconds` is signed and is negative for every
 * instant before 1970. Both ends are checked before the multiply. `kMaxAxisSeconds` is the axis
 * ceiling in whole seconds. Its negation is the lower end. `-kMaxAxisSeconds * 10^9` is inside the
 * axis, and one second further is not. The last 0.855 s at the bottom of the axis is refused with
 * the second that contains it. `subNs` is not negative. Only a nonnegative `whole` can carry the sum
 * past the top, and the room above it is checked only then. Subtracting a negative `whole`
 * from the top of the axis would itself leave the axis.
 */
[[nodiscard]] inline constexpr std::expected<std::int64_t, TimeStatus> nsFromSeconds(std::int64_t seconds, std::int64_t subNs) noexcept {
    if (seconds > kMaxAxisSeconds || seconds < -kMaxAxisSeconds) {
        return std::unexpected(TimeStatus::axis_overflow);
    }
    const std::int64_t whole = seconds * kNsPerSecond;
    if (whole >= 0 && subNs > std::numeric_limits<std::int64_t>::max() - whole) {
        return std::unexpected(TimeStatus::axis_overflow);
    }
    return whole + subNs;
}

/// @brief Whether `value - shift` overflows (`subtractOverflows`) and whether `value + shift` overflows
/// (`addOverflows`), each tested before the operation.
[[nodiscard]] inline constexpr bool subtractOverflows(std::int64_t value, std::int64_t shift) noexcept { return shift > 0 ? value < std::numeric_limits<std::int64_t>::min() + shift : value > std::numeric_limits<std::int64_t>::max() + shift; }

[[nodiscard]] inline constexpr bool addOverflows(std::int64_t value, std::int64_t shift) noexcept { return shift > 0 ? value > std::numeric_limits<std::int64_t>::max() - shift : value < std::numeric_limits<std::int64_t>::min() - shift; }

} // namespace detail

/// @brief The epoch's position on the Unix axis for a layout, `epochNs` being the caller's Level-2 value.
[[nodiscard]] inline constexpr std::int64_t timeCodeEpochNs(const Layout& layout, std::int64_t epochNs) noexcept { return layout.custom_epoch ? epochNs : -kEpoch1958Ns; }

/**
 * @brief The CUC coarse count the axis can still hold above an epoch. 9 602 063 236 for the 1958 epoch.
 *
 * An epoch far below the axis leaves more room than the elapsed count can express. At
 * `epoch_ns = INT64_MIN` the room is very nearly `2^64` ns. A coarse count that large, multiplied out
 * and added to its fraction, wraps the `std::uint64_t` that holds it before any range check runs. The
 * count is therefore capped at the largest value that leaves a whole second of headroom,
 * `18 446 744 072`. That cap is far above every bound the axis itself imposes. It changes no
 * reachable case.
 */
[[nodiscard]] inline constexpr std::uint64_t maxCoarseSeconds(std::int64_t epochNs) noexcept {
    constexpr std::uint64_t kNoWrap = (std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(kNsPerSecond)) / static_cast<std::uint64_t>(kNsPerSecond);
    const std::uint64_t     bound   = detail::roomAbove(epochNs) / static_cast<std::uint64_t>(kNsPerSecond);
    return bound < kNoWrap ? bound : kNoWrap;
}

/// @brief The CDS day count the axis can still hold above an epoch. 111 134 for the 1958 epoch.
///
/// The count is capped for the same reason as in `maxCoarseSeconds`, with two days of headroom. The
/// millisecond of day admits a leap second and can carry the sum a little past a whole day.
[[nodiscard]] inline constexpr std::uint64_t maxDayCount(std::int64_t epochNs) noexcept {
    constexpr std::uint64_t kNoWrap = (std::numeric_limits<std::uint64_t>::max() - 2ULL * static_cast<std::uint64_t>(kNsPerDay)) / static_cast<std::uint64_t>(kNsPerDay);
    const std::uint64_t     bound   = detail::roomAbove(epochNs) / static_cast<std::uint64_t>(kNsPerDay);
    return bound < kNoWrap ? bound : kNoWrap;
}

/**
 * @brief Parse an explicit P-field, 301.0-B-4 3.1.1 and each code's own subsection.
 *
 * The mandatory first octet holds an extension flag, a three-bit time code identification and four
 * detail bits. Each code defines its own detail bits. Three of the eight identifications name no
 * format. They are `000`, `011` and `111`, all reserved. A fourth, `110`, is a Level 3 or 4
 * agency-defined code. Its T-field is a binary number with no stated unit and no stated epoch (3.6.1).
 * It holds no instant that can be placed on an axis. The function refuses it as `no_epoch_defined`.
 * `agencyTimeFieldOctets` reads its declared T-field length. A caller can skip the field and keep its
 * place in the payload.
 *
 * The ASCII codes have no P-field (3.1.1, repeated at 3.5.2). This function does not yield an ASCII
 * kind.
 */
[[nodiscard]] inline std::expected<Layout, TimeStatus> parsePField(std::span<const std::uint8_t> pfield) noexcept {
    if (pfield.empty()) {
        return std::unexpected(TimeStatus::short_field);
    }
    const std::uint8_t first     = pfield[0];
    const bool         extension = (first & 0x80U) != 0U;
    const unsigned     code      = (first >> 4U) & 0x07U;
    const unsigned     detail4   = first & 0x0FU;

    Layout out{};
    out.p_field_octets = 1UZ;

    if (code == 0U || code == 3U || code == 7U) {
        return std::unexpected(TimeStatus::reserved_code);
    }
    if (code == 6U) {
        return std::unexpected(TimeStatus::no_epoch_defined);
    }

    if (code == 1U || code == 2U) {
        out.kind         = TimeCodeKind::cuc;
        out.custom_epoch = code == 2U;

        // 3.2.2 is asymmetric. The basic-octet field is a count minus one. The fractional-octet field
        // is a plain count. Their zero values mean different things.
        unsigned coarse = ((detail4 >> 2U) & 0x03U) + 1U;
        unsigned fine   = detail4 & 0x03U;
        if (extension) {
            if (pfield.size() < 2UZ) {
                return std::unexpected(TimeStatus::short_field);
            }
            const std::uint8_t second = pfield[1];
            if ((second & 0x80U) != 0U) {
                return std::unexpected(TimeStatus::bad_p_field); // a third octet, which 3.2.2 does not define
            }
            coarse += (second >> 5U) & 0x03U;
            fine += (second >> 2U) & 0x07U;
            out.p_field_octets = 2UZ;
        }
        out.coarse_octets  = static_cast<std::uint8_t>(coarse);
        out.fine_octets    = static_cast<std::uint8_t>(fine);
        out.t_field_octets = coarse + fine;
        return out;
    }

    if (extension) {
        return std::unexpected(TimeStatus::bad_p_field); // neither 3.3.2 nor 3.4.2 defines a second octet
    }

    if (code == 4U) {
        out.kind         = TimeCodeKind::cds;
        out.custom_epoch = (detail4 & 0x08U) != 0U;
        out.day_octets   = ((detail4 & 0x04U) != 0U) ? std::uint8_t{3U} : std::uint8_t{2U};
        switch (detail4 & 0x03U) {
        case 0U: out.submilli_octets = 0U; break;
        case 1U: out.submilli_octets = 2U; break;
        case 2U: out.submilli_octets = 4U; break;
        default: return std::unexpected(TimeStatus::reserved_resolution);
        }
        out.t_field_octets = std::size_t{out.day_octets} + 4UZ + std::size_t{out.submilli_octets};
        return out;
    }

    out.kind                  = TimeCodeKind::ccs;
    out.day_of_year           = (detail4 & 0x08U) != 0U;
    const unsigned resolution = detail4 & 0x07U;
    if (resolution == 7U) {
        return std::unexpected(TimeStatus::reserved_resolution);
    }
    out.subsecond_octets = static_cast<std::uint8_t>(resolution);
    // 3.4.1.1 is 16 + 8 + 8 + 8 + 8 + 8 bits and 3.4.1.2 is 16 + 16 + 8 + 8 + 8. Both are seven octets.
    out.t_field_octets = 7UZ + std::size_t{out.subsecond_octets};
    return out;
}

/**
 * @brief The T-field length a Level 3 or 4 agency-defined P-field declares, 301.0-B-4 3.6.2.
 *
 * Bits 4-7 hold the length in octets minus one. A value of 0 to 15 means 1 to 16 octets.
 */
[[nodiscard]] inline constexpr std::size_t agencyTimeFieldOctets(std::uint8_t firstOctet) noexcept { return std::size_t{firstOctet & 0x0FU} + 1UZ; }

/// @brief Write a P-field for a layout. One octet, or two where CUC needs the extension.
[[nodiscard]] inline std::expected<PField, TimeStatus> writePField(const Layout& layout) noexcept {
    PField                  field{};
    std::span<std::uint8_t> out{field.data};
    switch (layout.kind) {
    case TimeCodeKind::cuc: {
        if (layout.coarse_octets < 1U || layout.coarse_octets > 7U || layout.fine_octets > 10U) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        const unsigned baseCoarse  = layout.coarse_octets > 4U ? 4U : layout.coarse_octets;
        const unsigned baseFine    = layout.fine_octets > 3U ? 3U : layout.fine_octets;
        const unsigned extraCoarse = layout.coarse_octets - baseCoarse;
        const unsigned extraFine   = layout.fine_octets - baseFine;
        const bool     extension   = extraCoarse != 0U || extraFine != 0U;
        out[0]                     = static_cast<std::uint8_t>((extension ? 0x80U : 0x00U) | ((layout.custom_epoch ? 2U : 1U) << 4U) | ((baseCoarse - 1U) << 2U) | baseFine);
        if (extension) {
            out[1] = static_cast<std::uint8_t>((extraCoarse << 5U) | (extraFine << 2U));
        }
        field.size = extension ? 2UZ : 1UZ;
        return field;
    }
    case TimeCodeKind::cds: {
        unsigned resolution = 0U;
        if (layout.submilli_octets == 2U) {
            resolution = 1U;
        } else if (layout.submilli_octets == 4U) {
            resolution = 2U;
        } else if (layout.submilli_octets != 0U) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        if (layout.day_octets != 2U && layout.day_octets != 3U) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        out[0]     = static_cast<std::uint8_t>((4U << 4U) | (layout.custom_epoch ? 0x08U : 0x00U) | (layout.day_octets == 3U ? 0x04U : 0x00U) | resolution);
        field.size = 1UZ;
        return field;
    }
    case TimeCodeKind::ccs: {
        if (layout.subsecond_octets > 6U) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        out[0]     = static_cast<std::uint8_t>((5U << 4U) | (layout.day_of_year ? 0x08U : 0x00U) | layout.subsecond_octets);
        field.size = 1UZ;
        return field;
    }
    default: return std::unexpected(TimeStatus::bad_p_field); // the ASCII codes have no P-field (3.5.2)
    }
}

/// @brief The T-field length a layout declares, for a caller that assembled the layout by hand.
[[nodiscard]] inline constexpr std::size_t timeFieldOctets(const Layout& layout) noexcept {
    switch (layout.kind) {
    case TimeCodeKind::cuc: return std::size_t{layout.coarse_octets} + std::size_t{layout.fine_octets};
    case TimeCodeKind::cds: return std::size_t{layout.day_octets} + 4UZ + std::size_t{layout.submilli_octets};
    case TimeCodeKind::ccs: return 7UZ + std::size_t{layout.subsecond_octets};
    default: return 0UZ;
    }
}

/**
 * @brief Decode a CUC, CDS or CCS T-field under a resolved layout.
 *
 * The three conversions are integer throughout and each states where it is exact.
 *
 * **CUC.** With `m` fractional octets the fraction field is an integer `F` in `[0, 2^(8m))`. Its value
 * is `F * 2^(-8m)` seconds, the "elapsed binary fraction" of 3.2.1 over the octet cascade. The
 * conversion `frac_ns = floor(F * 10^9 / 2^(8m))` is a right shift. Since `10^9 = 2^9 * 5^9`, the
 * quotient is exact when `2^(8m-9)` divides `F`. At `m = 1` every `F` qualifies, and
 * `frac_ns = F * 3 906 250`. From `m = 2` on, the floor discards less than one nanosecond. The
 * conversion uses **at most the first four fractional octets**. The part below `2^-32` seconds is
 * 0.233 ns, and the axis cannot represent it. The cap keeps every product below `2^62`. No 128-bit
 * arithmetic is needed in this file.
 *
 * **CDS.** The segments are cascaded counters (3.3.1). Each segment's unit is a subdivision of the
 * segment above it. Annex A tabulates the 16-bit submillisecond segment as microseconds of the
 * millisecond, 0 to 999, and not 0 to 65 535. By the same cascade the 32-bit segment counts
 * picoseconds of the millisecond, 0 to 999 999 999. 3.3.2 labels it "(picosecond)". Only under that
 * reading do its unit and its width agree (`10^9 - 1 < 2^32`). Both ranges are checked. A wrong
 * reading of either returns an error and does not carry an overflow. The 16-bit case is exact. The
 * 32-bit case truncates below a nanosecond and returns the residue in `sub_ns_ps`.
 *
 * **CCS.** The code is binary coded decimal, two digits per octet. Every nibble is validated. A nibble
 * of `0xA` is not a digit, and a lenient reading would produce false data. The civil date uses the
 * exact integer proleptic-Gregorian arithmetic of `std::chrono`. Its leap-year rule is exactly Annex
 * A's footnote rule. The subsecond segments come in pairs of decimal digits. The reachable resolutions
 * are therefore `10^-2k`, and `10^-9` is not one of them. Four octets give `10^-8` s = 10 ns and are
 * exact on the axis. Five octets give `10^-10` s and truncate. The CCS limit is `k <= 4`. It differs
 * from the CUC limit `m <= 3`, and for a different reason.
 */
[[nodiscard]] inline std::expected<Instant, TimeStatus> decode(std::span<const std::uint8_t> tfield, const Layout& layout, std::int64_t epochNs, std::int32_t taiUtcOffsetSeconds) noexcept {
    Instant           out{};
    const std::size_t needed = timeFieldOctets(layout);
    if (needed == 0UZ) {
        return std::unexpected(TimeStatus::bad_p_field); // an ASCII kind has no T-field here, `decodeAscii` decodes it
    }
    if (tfield.size() < needed) {
        return std::unexpected(TimeStatus::short_field);
    }
    const std::int64_t epoch = timeCodeEpochNs(layout, epochNs);

    if (layout.kind == TimeCodeKind::cuc) {
        if (layout.coarse_octets < 1U || layout.coarse_octets > 7U || layout.fine_octets > 10U) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        const std::uint64_t coarse = detail::readBigEndian(tfield, 0UZ, layout.coarse_octets);
        if (coarse > maxCoarseSeconds(epoch)) {
            return std::unexpected(TimeStatus::axis_overflow); // checked before the multiply
        }

        const std::size_t   used     = layout.fine_octets > 4U ? 4UZ : std::size_t{layout.fine_octets};
        const std::uint64_t fraction = used == 0UZ ? 0ULL : detail::readBigEndian(tfield, layout.coarse_octets, used);
        const unsigned      shift    = static_cast<unsigned>(8UZ * used);
        std::uint64_t       fracNs   = 0ULL;
        if (used != 0UZ) {
            const std::uint64_t scaled = fraction * static_cast<std::uint64_t>(kNsPerSecond);
            fracNs                     = scaled >> shift;
            const std::uint64_t rest   = scaled - (fracNs << shift);
            out.sub_ns_ps              = static_cast<std::uint32_t>((rest * 1000ULL) >> shift);
        }

        const auto sum = detail::addToEpoch(epoch, coarse * static_cast<std::uint64_t>(kNsPerSecond) + fracNs);
        if (!sum) {
            return std::unexpected(sum.error());
        }
        std::int64_t value = *sum;
        // The code is TAI-based and the axis is UTC-based (3.2.1). The caller's offset relates the two.
        // No table is read. A 32-bit offset in seconds is at most 2.15e18 ns and fits the axis by
        // itself. The difference can leave the axis. The code tests for that before forming it.
        if (taiUtcOffsetSeconds != 0) {
            const std::int64_t offsetNs = std::int64_t{taiUtcOffsetSeconds} * kNsPerSecond;
            if (detail::subtractOverflows(value, offsetNs)) {
                return std::unexpected(TimeStatus::axis_overflow);
            }
            value -= offsetNs;
        }
        out.ns  = value;
        out.tai = taiUtcOffsetSeconds == 0;
        return out;
    }

    if (layout.kind == TimeCodeKind::cds) {
        if ((layout.day_octets != 2U && layout.day_octets != 3U) || (layout.submilli_octets != 0U && layout.submilli_octets != 2U && layout.submilli_octets != 4U)) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        const std::uint64_t day = detail::readBigEndian(tfield, 0UZ, layout.day_octets);
        if (day > maxDayCount(epoch)) {
            return std::unexpected(TimeStatus::axis_overflow);
        }
        const std::uint64_t milliseconds = detail::readBigEndian(tfield, layout.day_octets, 4UZ);
        if (milliseconds >= 86'401'000ULL) {
            return std::unexpected(TimeStatus::segment_out_of_range); // past Annex A's widest range, 86 400 999
        }
        const bool leap = milliseconds >= 86'400'000ULL;

        std::uint64_t subNs = 0ULL;
        if (layout.submilli_octets == 2U) {
            const std::uint64_t microseconds = detail::readBigEndian(tfield, std::size_t{layout.day_octets} + 4UZ, 2UZ);
            if (microseconds >= 1000ULL) {
                return std::unexpected(TimeStatus::segment_out_of_range); // microsecond of millisecond, 0 to 999 (Annex A)
            }
            subNs = microseconds * 1000ULL;
        } else if (layout.submilli_octets == 4U) {
            const std::uint64_t picoseconds = detail::readBigEndian(tfield, std::size_t{layout.day_octets} + 4UZ, 4UZ);
            if (picoseconds >= 1'000'000'000ULL) {
                return std::unexpected(TimeStatus::segment_out_of_range); // picosecond-of-millisecond, 0 to 999 999 999
            }
            subNs         = picoseconds / 1000ULL;
            out.sub_ns_ps = static_cast<std::uint32_t>(picoseconds % 1000ULL);
        }

        const auto sum = detail::addToEpoch(epoch, day * static_cast<std::uint64_t>(kNsPerDay) + milliseconds * static_cast<std::uint64_t>(kNsPerMillisecond) + subNs);
        if (!sum) {
            return std::unexpected(sum.error());
        }
        out.ns   = *sum;
        out.leap = leap;
        return out;
    }

    // CCS. Every nibble is a digit or the code is refused.
    if (layout.subsecond_octets > 6U) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    std::array<unsigned, 13UZ> pairs{};
    for (std::size_t i = 0UZ; i < needed; ++i) {
        if (!detail::bcdOctet(tfield[i], pairs[i])) {
            return std::unexpected(TimeStatus::bad_bcd);
        }
    }
    int         year  = static_cast<int>(pairs[0] * 100U + pairs[1]);
    unsigned    month = 1U;
    unsigned    day   = 1U;
    std::size_t next  = 4UZ;
    if (year < 1 || year > 9999) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    if (layout.day_of_year) {
        // 3.4.1.2 gives sixteen bits. The four most significant bits are unused and set to zero.
        const unsigned dayOfYear = pairs[2] * 100U + pairs[3];
        const unsigned yearDays  = std::chrono::year{year}.is_leap() ? 366U : 365U;
        if (dayOfYear < 1U || dayOfYear > yearDays) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        detail::civilFromDays(detail::daysFromCivil(year, 1U, 1U) + static_cast<std::int64_t>(dayOfYear) - 1, year, month, day);
    } else {
        month = pairs[2];
        day   = pairs[3];
        if (month < 1U || month > 12U || day < 1U || day > detail::daysInMonth(year, month)) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
    }
    const unsigned hour   = pairs[next];
    const unsigned minute = pairs[next + 1UZ];
    const unsigned second = pairs[next + 2UZ];
    if (hour > 23U || minute > 59U || second > 60U) {
        return std::unexpected(TimeStatus::segment_out_of_range); // Annex A admits the leap second's 60
    }

    std::int64_t   subsecondNs = 0;
    const unsigned k           = layout.subsecond_octets;
    if (k != 0U) {
        std::int64_t fraction = 0;
        for (std::size_t i = 0UZ; i < k; ++i) {
            fraction = fraction * 100 + static_cast<std::int64_t>(pairs[next + 3UZ + i]);
        }
        if (k <= 4U) {
            subsecondNs = fraction * detail::powerOfTen(9U - 2U * k);
        } else {
            const std::int64_t divisor = detail::powerOfTen(2U * k - 9U);
            subsecondNs                = fraction / divisor;
            out.sub_ns_ps              = static_cast<std::uint32_t>(((fraction % divisor) * 1000) / divisor);
        }
    }

    // The calendar bounds the day count. Year 9999 is under three million days from 1970. The
    // seconds of the day can be added to it in whole seconds within `std::int64_t`. The sum is the
    // one value that can leave the axis, and it is checked at that point. The axis holds only part of
    // its last day. 106 751 days plus 23:59:59 is past its end. A bound on the day count alone would
    // admit that case.
    const std::int64_t days         = detail::daysFromCivil(year, month, day);
    const std::int64_t withinSecond = static_cast<std::int64_t>(hour) * 3600 + static_cast<std::int64_t>(minute) * 60 + static_cast<std::int64_t>(second);
    const auto         value        = detail::nsFromSeconds(days * 86400 + withinSecond, subsecondNs);
    if (!value) {
        return std::unexpected(value.error());
    }
    out.ns   = *value;
    out.leap = second == 60U;
    return out;
}

/**
 * @brief Decode ASCII Time Code A or B, 301.0-B-4 3.5.1.
 *
 * Form A is `YYYY-MM-DDThh:mm:ss.d->dZ` and form B is `YYYY-DDDThh:mm:ss.d->dZ`. The terminator is
 * optional. The base lengths are nineteen and seventeen characters. Both are counted from the
 * subfields and their separators. There is no P-field (3.5.2).
 *
 * 3.5.1.3 permits subsets. This kernel **accepts right-truncation and refuses left-truncation**. A
 * right-truncated form still names an instant, with the omitted subfields at zero. A form with no
 * year names no instant. A time-only subset names no instant either. Rule (d) forbids partial
 * subfields. `1988-1-18` is therefore a bad character. Its one-digit month is not read leniently.
 *
 * The fraction is parsed **digit by digit into an integer**, with no floating-point conversion.
 * `0.123456789` read as a `double` and multiplied by `10^9` does not reliably give `123456789`. The
 * first nine digits are the nanoseconds. The tenth through twelfth digits are `sub_ns_ps`. Digits
 * past the twelfth are counted in `precision_discarded_digits`. A producer that sends fifteen digits
 * sends more precision than the axis holds. The count reports that to the caller.
 */
[[nodiscard]] inline std::expected<Instant, TimeStatus> decodeAscii(std::string_view text, TimeCodeKind kind) noexcept {
    Instant out{};
    if (kind != TimeCodeKind::ascii_a && kind != TimeCodeKind::ascii_b) {
        return std::unexpected(TimeStatus::bad_character);
    }
    std::string_view body = text;
    if (!body.empty() && body.back() == 'Z') {
        body.remove_suffix(1UZ);
    }
    if (body.empty()) {
        return std::unexpected(TimeStatus::incomplete_calendar);
    }

    std::size_t pos      = 0UZ;
    const auto  digitsAt = [&body](std::size_t at, std::size_t count, unsigned& value) noexcept -> bool {
        if (at + count > body.size()) {
            return false;
        }
        value = 0U;
        for (std::size_t i = 0UZ; i < count; ++i) {
            const char c = body[at + i];
            if (c < '0' || c > '9') {
                return false;
            }
            value = value * 10U + static_cast<unsigned>(c - '0');
        }
        return true;
    };

    // A left-truncated form has no year, and a time-only subset names no date. Neither names an instant.
    if (body.size() >= 3UZ && body[2] == ':') {
        return std::unexpected(TimeStatus::incomplete_calendar);
    }
    unsigned year = 0U;
    if (!digitsAt(0UZ, 4UZ, year)) {
        return std::unexpected(body.find(':') != std::string_view::npos ? TimeStatus::incomplete_calendar : TimeStatus::bad_character);
    }
    pos = 4UZ;

    unsigned month     = 1U;
    unsigned day       = 1U;
    unsigned dayOfYear = 1U;
    unsigned hour      = 0U;
    unsigned minute    = 0U;
    unsigned second    = 0U;

    const auto peek = [&body, &pos](char c) noexcept -> bool { return pos < body.size() && body[pos] == c; };

    if (pos < body.size()) {
        if (!peek('-')) {
            return std::unexpected(TimeStatus::bad_character);
        }
        ++pos;
        if (kind == TimeCodeKind::ascii_a) {
            if (!digitsAt(pos, 2UZ, month)) {
                return std::unexpected(TimeStatus::bad_character);
            }
            pos += 2UZ;
            if (pos < body.size()) {
                if (!peek('-')) {
                    return std::unexpected(TimeStatus::bad_character);
                }
                ++pos;
                if (!digitsAt(pos, 2UZ, day)) {
                    return std::unexpected(TimeStatus::bad_character);
                }
                pos += 2UZ;
            }
        } else {
            if (!digitsAt(pos, 3UZ, dayOfYear)) {
                return std::unexpected(TimeStatus::bad_character);
            }
            pos += 3UZ;
        }
    }

    std::int64_t subsecondNs = 0;
    if (pos < body.size()) {
        if (!peek('T')) {
            return std::unexpected(TimeStatus::bad_character);
        }
        ++pos;
        if (!digitsAt(pos, 2UZ, hour)) {
            return std::unexpected(TimeStatus::bad_character);
        }
        pos += 2UZ;
        if (pos < body.size()) {
            if (!peek(':')) {
                return std::unexpected(TimeStatus::bad_character);
            }
            ++pos;
            if (!digitsAt(pos, 2UZ, minute)) {
                return std::unexpected(TimeStatus::bad_character);
            }
            pos += 2UZ;
        }
        if (pos < body.size() && body[pos] == ':') {
            ++pos;
            if (!digitsAt(pos, 2UZ, second)) {
                return std::unexpected(TimeStatus::bad_character);
            }
            pos += 2UZ;
        }
        if (pos < body.size() && body[pos] == '.') {
            ++pos;
            if (pos >= body.size()) {
                return std::unexpected(TimeStatus::bad_character); // a fraction is one to n characters (3.5.1)
            }
            std::size_t   nanoDigits = 0UZ;
            std::size_t   picoDigits = 0UZ;
            std::uint32_t picoValue  = 0U;
            while (pos < body.size()) {
                const char c = body[pos];
                if (c < '0' || c > '9') {
                    return std::unexpected(TimeStatus::bad_character);
                }
                const unsigned digit = static_cast<unsigned>(c - '0');
                if (nanoDigits < 9UZ) {
                    subsecondNs = subsecondNs * 10 + static_cast<std::int64_t>(digit);
                    ++nanoDigits;
                } else if (picoDigits < 3UZ) {
                    picoValue = picoValue * 10U + digit;
                    ++picoDigits;
                } else if (out.precision_discarded_digits < 255U) {
                    ++out.precision_discarded_digits;
                }
                ++pos;
            }
            for (std::size_t i = nanoDigits; i < 9UZ; ++i) {
                subsecondNs *= 10;
            }
            for (std::size_t i = picoDigits; i < 3UZ; ++i) {
                picoValue *= 10U;
            }
            out.sub_ns_ps = picoValue;
        }
    }
    if (pos != body.size()) {
        return std::unexpected(TimeStatus::bad_character);
    }

    if (year < 1U || year > 9999U) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    const int signedYear = static_cast<int>(year);
    if (kind == TimeCodeKind::ascii_b) {
        const unsigned yearDays = std::chrono::year{signedYear}.is_leap() ? 366U : 365U;
        if (dayOfYear < 1U || dayOfYear > yearDays) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
    } else if (month < 1U || month > 12U || day < 1U || day > detail::daysInMonth(signedYear, month)) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    if (hour > 23U || minute > 59U || second > 60U) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }

    // As in the calendar code above, the day count and the seconds of the day are summed in whole
    // seconds. The calendar bounds them. The axis is tested once on the sum. The axis holds only part
    // of its last day. A bound on the day count alone would admit an instant past its end.
    const std::int64_t days         = kind == TimeCodeKind::ascii_b ? detail::daysFromCivil(signedYear, 1U, 1U) + static_cast<std::int64_t>(dayOfYear) - 1 : detail::daysFromCivil(signedYear, month, day);
    const std::int64_t withinSecond = static_cast<std::int64_t>(hour) * 3600 + static_cast<std::int64_t>(minute) * 60 + static_cast<std::int64_t>(second);
    const auto         value        = detail::nsFromSeconds(days * 86400 + withinSecond, subsecondNs);
    if (!value) {
        return std::unexpected(value.error());
    }
    out.ns   = *value;
    out.leap = second == 60U;
    return out;
}

/**
 * @brief Encode an instant as a CUC, CDS or CCS T-field under a layout.
 *
 * The CUC fraction rounds to nearest and does not floor, `F = (r_ns * 2^(8m) + 5*10^8) / 10^9`. With
 * that rule `code -> ns -> code` reproduces `F` exactly for `m <= 3`. It cannot for `m >= 4`. The
 * limit follows from the arithmetic. The round trip recovers `F` only while distinct `F` map to
 * distinct nanosecond values. That needs `2^(-8m) >= 10^-9`, or `8m <= log2(10^9) = 29.897`, or
 * `m <= 3`. At `m = 4` four distinct `F` share a nanosecond, and no inverse can separate them.
 */
[[nodiscard]] inline std::expected<TField, TimeStatus> encode(const Instant& instant, const Layout& layout, std::int64_t epochNs, std::int32_t taiUtcOffsetSeconds) noexcept {
    const std::size_t needed = timeFieldOctets(layout);
    if (needed == 0UZ) {
        return std::unexpected(TimeStatus::bad_p_field);
    }
    if (needed > kMaxTFieldOctets) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    TField                  field{.data = {}, .size = needed};
    std::span<std::uint8_t> out{field.data.data(), needed};
    const std::int64_t      epoch = timeCodeEpochNs(layout, epochNs);

    if (layout.kind == TimeCodeKind::cuc) {
        if (layout.coarse_octets < 1U || layout.coarse_octets > 7U || layout.fine_octets > 10U) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        std::int64_t value = instant.ns;
        if (taiUtcOffsetSeconds != 0) {
            // The decoder's relation in the other direction. The range is tested before the sum is formed.
            const std::int64_t offsetNs = std::int64_t{taiUtcOffsetSeconds} * kNsPerSecond;
            if (detail::addOverflows(value, offsetNs)) {
                return std::unexpected(TimeStatus::axis_overflow);
            }
            value += offsetNs;
        }
        if (value < epoch) {
            return std::unexpected(TimeStatus::segment_out_of_range); // the code counts forward from its epoch only
        }
        const std::uint64_t elapsed = static_cast<std::uint64_t>(value - epoch);
        std::uint64_t       coarse  = elapsed / static_cast<std::uint64_t>(kNsPerSecond);
        const std::uint64_t restNs  = elapsed % static_cast<std::uint64_t>(kNsPerSecond);

        const std::size_t used     = layout.fine_octets > 4U ? 4UZ : std::size_t{layout.fine_octets};
        std::uint64_t     fraction = 0ULL;
        if (used != 0UZ) {
            const unsigned shift = static_cast<unsigned>(8UZ * used);
            fraction             = ((restNs << shift) + 500'000'000ULL) / static_cast<std::uint64_t>(kNsPerSecond);
            if (fraction >> shift != 0ULL) { // the rounding carried into the next second
                fraction = 0ULL;
                ++coarse;
            }
        }
        if ((coarse >> (8U * layout.coarse_octets)) != 0ULL) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        detail::writeBigEndian(out, 0UZ, layout.coarse_octets, coarse);
        if (layout.fine_octets != 0U) {
            // The fraction occupies the top `used` octets of the field. The octets below them are
            // zero. The part below 2^-32 s is 0.233 ns, and the axis cannot represent it. Writing the
            // value into the whole field would need a shift as wide as the field. At ten octets that
            // shift is undefined.
            detail::writeBigEndian(out, layout.coarse_octets, used, fraction);
            for (std::size_t i = used; i < std::size_t{layout.fine_octets}; ++i) {
                out[std::size_t{layout.coarse_octets} + i] = std::uint8_t{0U};
            }
        }
        return field;
    }

    if (layout.kind == TimeCodeKind::cds) {
        if ((layout.day_octets != 2U && layout.day_octets != 3U) || (layout.submilli_octets != 0U && layout.submilli_octets != 2U && layout.submilli_octets != 4U)) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        if (instant.ns < epoch) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        const std::uint64_t elapsed = static_cast<std::uint64_t>(instant.ns - epoch);
        const std::uint64_t day     = elapsed / static_cast<std::uint64_t>(kNsPerDay);
        const std::uint64_t within  = elapsed % static_cast<std::uint64_t>(kNsPerDay);
        if (day >> (8U * layout.day_octets) != 0ULL) {
            return std::unexpected(TimeStatus::segment_out_of_range);
        }
        const std::uint64_t milliseconds = within / static_cast<std::uint64_t>(kNsPerMillisecond);
        const std::uint64_t subNs        = within % static_cast<std::uint64_t>(kNsPerMillisecond);

        detail::writeBigEndian(out, 0UZ, layout.day_octets, day);
        detail::writeBigEndian(out, layout.day_octets, 4UZ, milliseconds);
        if (layout.submilli_octets == 2U) {
            detail::writeBigEndian(out, std::size_t{layout.day_octets} + 4UZ, 2UZ, subNs / 1000ULL);
        } else if (layout.submilli_octets == 4U) {
            detail::writeBigEndian(out, std::size_t{layout.day_octets} + 4UZ, 4UZ, subNs * 1000ULL + instant.sub_ns_ps);
        }
        return field;
    }

    if (layout.subsecond_octets > 6U) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    // CCS. The day is floored. A negative instant falls on the calendar day that contains it.
    std::int64_t days   = instant.ns / kNsPerDay;
    std::int64_t within = instant.ns % kNsPerDay;
    if (within < 0) {
        --days;
        within += kNsPerDay;
    }
    int      year  = 0;
    unsigned month = 0U;
    unsigned day   = 0U;
    detail::civilFromDays(days, year, month, day);
    if (year < 1 || year > 9999) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    const std::int64_t seconds = within / kNsPerSecond;
    const std::int64_t restNs  = within % kNsPerSecond;

    out[0]           = detail::toBcd(static_cast<unsigned>(year) / 100U);
    out[1]           = detail::toBcd(static_cast<unsigned>(year) % 100U);
    std::size_t next = 0UZ;
    if (layout.day_of_year) {
        const unsigned dayOfYear = static_cast<unsigned>(days - detail::daysFromCivil(year, 1U, 1U)) + 1U;
        out[2]                   = detail::toBcd(dayOfYear / 100U);
        out[3]                   = detail::toBcd(dayOfYear % 100U);
        next                     = 4UZ;
    } else {
        out[2] = detail::toBcd(month);
        out[3] = detail::toBcd(day);
        next   = 4UZ;
    }
    out[next]        = detail::toBcd(static_cast<unsigned>(seconds / 3600));
    out[next + 1UZ]  = detail::toBcd(static_cast<unsigned>((seconds / 60) % 60));
    out[next + 2UZ]  = detail::toBcd(static_cast<unsigned>(seconds % 60));
    const unsigned k = layout.subsecond_octets;
    if (k != 0U) {
        std::int64_t fraction = 0;
        if (k <= 4U) {
            fraction = restNs / detail::powerOfTen(9U - 2U * k);
        } else {
            fraction = restNs * detail::powerOfTen(2U * k - 9U) + static_cast<std::int64_t>(instant.sub_ns_ps) * detail::powerOfTen(2U * k - 9U) / 1000;
        }
        for (std::size_t i = k; i > 0UZ; --i) {
            out[next + 2UZ + i] = detail::toBcd(static_cast<unsigned>(fraction % 100));
            fraction /= 100;
        }
    }
    return field;
}

/**
 * @brief Encode an instant as ASCII Time Code A or B, in the full form only.
 *
 * A subset would need a policy on which fields to keep. Its receiver would also need to be told what
 * it means. 3.5.1.3 permits subsets on the receive side. This encoder does not emit them.
 */
[[nodiscard]] inline std::expected<AsciiCode, TimeStatus> encodeAscii(const Instant& instant, TimeCodeKind kind, std::uint8_t fractionDigits, bool terminator) noexcept {
    if (kind != TimeCodeKind::ascii_a && kind != TimeCodeKind::ascii_b) {
        return std::unexpected(TimeStatus::bad_character);
    }
    if (fractionDigits > 9U) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }
    AsciiCode       code{};
    std::span<char> out{code.data};

    std::int64_t days   = instant.ns / kNsPerDay;
    std::int64_t within = instant.ns % kNsPerDay;
    if (within < 0) {
        --days;
        within += kNsPerDay;
    }
    int      year  = 0;
    unsigned month = 0U;
    unsigned day   = 0U;
    detail::civilFromDays(days, year, month, day);
    if (year < 1 || year > 9999) {
        return std::unexpected(TimeStatus::segment_out_of_range);
    }

    const auto put = [&out](std::size_t at, unsigned value, std::size_t width) noexcept {
        for (std::size_t i = 0UZ; i < width; ++i) {
            out[at + width - 1UZ - i] = static_cast<char>('0' + static_cast<int>(value % 10U));
            value /= 10U;
        }
    };

    std::size_t at = 0UZ;
    put(at, static_cast<unsigned>(year), 4UZ);
    at += 4UZ;
    out[at++] = '-';
    if (kind == TimeCodeKind::ascii_a) {
        put(at, month, 2UZ);
        at += 2UZ;
        out[at++] = '-';
        put(at, day, 2UZ);
        at += 2UZ;
    } else {
        put(at, static_cast<unsigned>(days - detail::daysFromCivil(year, 1U, 1U)) + 1U, 3UZ);
        at += 3UZ;
    }
    out[at++] = 'T';
    put(at, static_cast<unsigned>(within / (3600 * kNsPerSecond)), 2UZ);
    at += 2UZ;
    out[at++] = ':';
    put(at, static_cast<unsigned>((within / (60 * kNsPerSecond)) % 60), 2UZ);
    at += 2UZ;
    out[at++] = ':';
    put(at, static_cast<unsigned>((within / kNsPerSecond) % 60), 2UZ);
    at += 2UZ;
    if (fractionDigits != 0U) {
        out[at++] = '.';
        put(at, static_cast<unsigned>((within % kNsPerSecond) / detail::powerOfTen(9U - fractionDigits)), fractionDigits);
        at += fractionDigits;
    }
    if (terminator) {
        out[at++] = 'Z';
    }
    code.size = at;
    return code;
}

} // namespace gr::ccsds

#endif // GNURADIO_ALGORITHM_CCSDS_TIME_CODES_HPP
