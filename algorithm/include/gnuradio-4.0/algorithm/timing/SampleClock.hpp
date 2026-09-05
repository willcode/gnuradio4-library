#ifndef GNURADIO_ALGORITHM_SAMPLE_CLOCK_HPP
#define GNURADIO_ALGORITHM_SAMPLE_CLOCK_HPP

#include <cmath>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace gr::timing {

namespace detail {

/**
 * @brief A 128-bit unsigned magnitude in two 64-bit limbs.
 *
 * The exact products this header needs run past 64 bits and must not be rounded, and the tree compiles
 * `-Wpedantic -Werror`, where `__int128` is not ISO C++ — `filter/ArbitraryResampler.hpp` states the same
 * constraint and takes the same way out. Only the four operations the clock performs are provided; this is
 * not a general wide-integer type.
 */
struct U128 {
    std::uint64_t hi{0ULL};
    std::uint64_t lo{0ULL};
};

/// The exact 64x64 product, schoolbook over 32-bit halves.
[[nodiscard]] constexpr U128 mul(std::uint64_t a, std::uint64_t b) noexcept {
    constexpr std::uint64_t kMask = 0xFFFFFFFFULL;

    const std::uint64_t aLo = a & kMask;
    const std::uint64_t aHi = a >> 32;
    const std::uint64_t bLo = b & kMask;
    const std::uint64_t bHi = b >> 32;

    const std::uint64_t ll     = aLo * bLo;
    const std::uint64_t lh     = aLo * bHi;
    const std::uint64_t hl     = aHi * bLo;
    const std::uint64_t hh     = aHi * bHi;
    const std::uint64_t middle = (ll >> 32) + (lh & kMask) + (hl & kMask);

    return U128{hh + (lh >> 32) + (hl >> 32) + (middle >> 32), (middle << 32) | (ll & kMask)};
}

[[nodiscard]] constexpr U128 add(U128 a, std::uint64_t b) noexcept {
    const std::uint64_t lo = a.lo + b;
    return U128{a.hi + (lo < a.lo ? 1ULL : 0ULL), lo};
}

/// `a - b`, the caller having established that `a >= b`.
[[nodiscard]] constexpr U128 sub(U128 a, std::uint64_t b) noexcept { return U128{a.hi - (a.lo < b ? 1ULL : 0ULL), a.lo - b}; }

[[nodiscard]] constexpr bool less(U128 a, std::uint64_t b) noexcept { return a.hi == 0ULL && a.lo < b; }

struct DivResult {
    U128          quotient{};
    std::uint64_t remainder{0ULL};
};

/**
 * @brief `a / d` and `a % d`, exact, for `0 < d < 2^63`.
 *
 * The high limb divides first and leaves a remainder below `d`; the low limb then runs a 64-step
 * shift-subtract over that remainder. The shifted remainder needs no 65th bit precisely because `d` is
 * below `2^63`, which is why `SampleClock` validates both of its divisors against that bound. A clock
 * divides once per span rather than once per sample — `NsWalk` is the per-sample path.
 */
[[nodiscard]] constexpr DivResult divide(U128 a, std::uint64_t d) noexcept {
    DivResult     out{};
    std::uint64_t remainder = 0ULL;
    if (a.hi != 0ULL) {
        out.quotient.hi = a.hi / d;
        remainder       = a.hi % d;
    }

    std::uint64_t quotient = 0ULL;
    for (int bit = 63; bit >= 0; --bit) {
        remainder = (remainder << 1) | ((a.lo >> bit) & 1ULL);
        quotient <<= 1;
        if (remainder >= d) {
            remainder -= d;
            quotient |= 1ULL;
        }
    }
    out.quotient.lo = quotient;
    out.remainder   = remainder;
    return out;
}

/// The magnitude of a negative `std::int64_t`, without the overflow that negating the minimum would be.
[[nodiscard]] constexpr std::uint64_t magnitude(std::int64_t negative) noexcept { return static_cast<std::uint64_t>(-(negative + 1LL)) + 1ULL; }

} // namespace detail

/// The whole sample and the exact fraction of one, as `indexOf` reads a time.
struct IndexAt {
    std::uint64_t index{0ULL};         ///< the last sample at or before the given time
    std::uint64_t remainder_num{0ULL}; ///< how far past that sample the time falls, in [0, remainder_den)
    std::uint64_t remainder_den{1ULL}; ///< 10^9 * rate_den, the exact denominator of that fraction

    /// The fraction as a `double`, for a consumer whose own arithmetic is already floating point. It is a
    /// number in [0, 1) rather than an index or a nanosecond count, so it carries none of the range the
    /// integers above exist to protect.
    [[nodiscard]] double fraction() const noexcept { return static_cast<double>(remainder_num) / static_cast<double>(remainder_den); }

    [[nodiscard]] bool operator==(const IndexAt&) const noexcept = default;
};

/**
 * @brief The map between a sample index and a wall-clock time, exact in integers.
 *
 * A stream's sample index is the only thing that counts samples without drifting, and a timestamp is the only
 * thing two streams can be compared on; this type is the conversion between them and holds it exactly. The rate
 * is an exact rational `rate_num/rate_den` — 250 kS/s is `250000/1`, and an audio rate trimmed by 1 ppm is
 * `44100044100/1000000`, which no `double` sample rate expresses. Neither an index nor a nanosecond count ever
 * passes through a `double` here: at `2^53 + 1` a `double` index is already short by one, and every conversion
 * below is integer arithmetic carried in 128 bits.
 *
 * The exact map is
 *
 *     t(index) = anchor_ns + ((index - anchor_index) * 10^9 * rate_den + anchor_rem) / rate_num
 *
 * as a rational, and `timeOf` is its **floor** — the last whole nanosecond at or before that sample, on both
 * sides of the anchor, so a sample one before the anchor at 250 kS/s reads `anchor_ns - 4000` and a rate whose
 * period is not a whole number of nanoseconds rounds down rather than toward the anchor. `indexOf` is the same
 * floor read the other way: the last sample at or before a time, with the sub-sample remainder returned exactly
 * beside it rather than left for the caller to re-derive.
 *
 * `anchor_rem` is the anchor's own position within its nanosecond, a numerator over `rate_num`. It is what
 * makes `rebase` exact: without it, moving the anchor to an index whose time is not a whole nanosecond would
 * shift every later conversion by up to one nanosecond, and `rebase` exists to keep long runs inside the
 * 128-bit domain without changing the map at all. A clock built by hand leaves it zero, which says the anchor
 * sits exactly on its nanosecond.
 *
 * A clock is an immutable value. `rebase` moves the anchor and `withRate` starts a new rate at a stated index,
 * both returning new clocks: a rate change is a new anchor at the change point, which is what lets a
 * discontinuity be represented rather than smeared across the samples on either side of it.
 *
 * Time is nanoseconds since the Unix epoch and nothing else: this kernel never interprets a calendar, so leap
 * seconds, time zones and any smearing policy belong to whatever produced the timestamps and are stated in
 * metadata, not here.
 *
 * Domain. Every product below is exact while `|index - anchor_index| * 10^9 * rate_den` and
 * `|t_ns - anchor_ns| * rate_num` stay under `2^127`, which the rate bounds and the `std::int64_t` nanosecond
 * range together guarantee. The nanosecond result must itself fit `std::int64_t` — about +/-292 years around the
 * epoch — so a run that spans more than that rebases; that is the second reason `rebase` is here.
 */
struct SampleClock {
    /// `10^9 * rate_den` is the numerator of a sample period in nanoseconds and is `indexOf`'s divisor, and
    /// `rate_num` is `timeOf`'s. Both must stay below `2^63` for `detail::divide` to need no 65th bit, which
    /// caps the denominator at `(2^63 - 1) / 10^9`.
    static constexpr std::uint64_t kNsPerSecond = 1'000'000'000ULL;
    static constexpr std::uint64_t kMaxRateNum  = (1ULL << 63) - 1ULL;
    static constexpr std::uint64_t kMaxRateDen  = kMaxRateNum / kNsPerSecond;

    std::uint64_t anchor_index{0ULL}; ///< the sample the anchor time belongs to
    std::int64_t  anchor_ns{0LL};     ///< nanoseconds since the Unix epoch, floored
    std::uint64_t rate_num{1ULL};     ///< samples per second, numerator
    std::uint64_t rate_den{1ULL};     ///< samples per second, denominator
    std::uint64_t anchor_rem{0ULL};   ///< the anchor's position within `anchor_ns`, over `rate_num`; in [0, rate_num)

    constexpr SampleClock() noexcept = default;

    SampleClock(std::uint64_t anchorIndex, std::int64_t anchorNs, std::uint64_t rateNum, std::uint64_t rateDen, std::uint64_t anchorRem = 0ULL) : anchor_index(anchorIndex), anchor_ns(anchorNs), rate_num(rateNum), rate_den(rateDen), anchor_rem(anchorRem) {
        if (rateNum == 0ULL || rateDen == 0ULL) {
            throw std::invalid_argument(std::format("gr::timing::SampleClock: rate {}/{} — neither term may be zero", rateNum, rateDen));
        }
        if (rateNum > kMaxRateNum || rateDen > kMaxRateDen) {
            throw std::invalid_argument(std::format("gr::timing::SampleClock: rate {}/{} is outside the exact domain — the numerator must not exceed {} and the denominator not {}", rateNum, rateDen, kMaxRateNum, kMaxRateDen));
        }
        if (anchorRem >= rateNum) {
            throw std::invalid_argument(std::format("gr::timing::SampleClock: anchor remainder {} must be below the rate numerator {}", anchorRem, rateNum));
        }
    }

    [[nodiscard]] bool operator==(const SampleClock&) const noexcept = default;

    /// The numerator of one sample period in nanoseconds; the denominator is `rate_num`.
    [[nodiscard]] constexpr std::uint64_t periodNsNum() const noexcept { return kNsPerSecond * rate_den; }

    /// The rate as a `double`, for radian arithmetic that is floating point anyway. Never for an index or a
    /// nanosecond count — that is what the rational above is for.
    [[nodiscard]] constexpr double rateHz() const noexcept { return static_cast<double>(rate_num) / static_cast<double>(rate_den); }

    /// The offset from `anchor_ns` split into whole nanoseconds and the exact remainder within the last one.
    struct NsSplit {
        std::int64_t  whole{0LL}; ///< nanoseconds relative to `anchor_ns`, floored
        std::uint64_t rem{0ULL};  ///< the position within that nanosecond, over `rate_num`; in [0, rate_num)
    };

    [[nodiscard]] NsSplit offsetFrom(std::uint64_t index) const noexcept {
        const std::uint64_t period = periodNsNum();
        if (index >= anchor_index) {
            const detail::DivResult split = detail::divide(detail::add(detail::mul(index - anchor_index, period), anchor_rem), rate_num);
            return NsSplit{static_cast<std::int64_t>(split.quotient.lo), split.remainder};
        }

        // Below the anchor the numerator is negative, and a floor is not a truncation: the quotient rounds away
        // from zero whenever the division leaves anything behind, and the remainder is what is left to reach the
        // next whole nanosecond up.
        const detail::U128 product = detail::mul(anchor_index - index, period);
        if (detail::less(product, anchor_rem)) {
            return NsSplit{0LL, anchor_rem - product.lo};
        }
        const detail::DivResult split = detail::divide(detail::sub(product, anchor_rem), rate_num);
        if (split.remainder == 0ULL) {
            return NsSplit{-static_cast<std::int64_t>(split.quotient.lo), 0ULL};
        }
        return NsSplit{-static_cast<std::int64_t>(split.quotient.lo) - 1LL, rate_num - split.remainder};
    }

    /// The last whole nanosecond at or before sample `index`.
    [[nodiscard]] std::int64_t timeOf(std::uint64_t index) const noexcept { return anchor_ns + offsetFrom(index).whole; }

    /// The last sample at or before `t_ns`, with the exact fraction of a sample by which the time overshoots it.
    [[nodiscard]] IndexAt indexOf(std::int64_t t_ns) const noexcept {
        const std::uint64_t period = periodNsNum();
        const std::int64_t  dt     = t_ns - anchor_ns;

        detail::U128 numerator{};
        bool         negative = false;
        if (dt >= 0LL) {
            const detail::U128 scaled = detail::mul(static_cast<std::uint64_t>(dt), rate_num);
            if (detail::less(scaled, anchor_rem)) {
                numerator = detail::U128{0ULL, anchor_rem - scaled.lo};
                negative  = true;
            } else {
                numerator = detail::sub(scaled, anchor_rem);
            }
        } else {
            numerator = detail::add(detail::mul(detail::magnitude(dt), rate_num), anchor_rem);
            negative  = true;
        }

        const detail::DivResult split = detail::divide(numerator, period);
        if (!negative) {
            return IndexAt{anchor_index + split.quotient.lo, split.remainder, period};
        }
        if (split.remainder == 0ULL) {
            return IndexAt{anchor_index - split.quotient.lo, 0ULL, period};
        }
        return IndexAt{anchor_index - split.quotient.lo - 1ULL, period - split.remainder, period};
    }

    /// The same map anchored at `index` instead. Exact, because `anchor_rem` carries what the floor drops.
    [[nodiscard]] SampleClock rebase(std::uint64_t index) const {
        const NsSplit at = offsetFrom(index);
        return SampleClock(index, anchor_ns + at.whole, rate_num, rate_den, at.rem);
    }

    /// @brief A new clock running at `rateNum/rateDen` from sample `atIndex`, anchored at that sample's time.
    ///
    /// The change index is required rather than defaulted: a rate change belongs at the sample it happens on,
    /// and a caller who silently kept the old anchor would smear the change back across every sample since.
    /// The anchor's sub-nanosecond position is re-expressed over the new numerator, so the anchor time moves by
    /// less than `1/rateNum` nanoseconds — the only inexactness anywhere in this type, and it is bounded by one
    /// period of the new rate rather than by a nanosecond.
    [[nodiscard]] SampleClock withRate(std::uint64_t rateNum, std::uint64_t rateDen, std::uint64_t atIndex) const {
        const NsSplit at = offsetFrom(atIndex);
        if (rateNum == 0ULL) {
            throw std::invalid_argument("gr::timing::SampleClock::withRate: the rate numerator may not be zero");
        }
        const std::uint64_t rescaled = detail::divide(detail::mul(at.rem, rateNum), rate_num).quotient.lo;
        return SampleClock(atIndex, anchor_ns + at.whole, rateNum, rateDen, rescaled);
    }

    /**
     * @brief A forward walk over consecutive sample times, exact and division-free.
     *
     * `timeOf` costs a 128-bit division, and a consumer stepping through a span of samples wants the same
     * answers without paying it per sample. Writing the period as `period = whole * rate_num + carry`, the time
     * of sample `index + k` is `timeOf(index) + k * whole` plus one nanosecond for each time an accumulator of
     * `carry` passes `rate_num` — so a step is an addition and at most one subtraction. `advance()` reproduces
     * `timeOf(index + k)` exactly, which the QA pins; a chunked consumer therefore reads the same times as an
     * unchunked one because both are the same function of the absolute index.
     */
    struct NsWalk {
        std::int64_t  t_ns{0LL};     ///< `timeOf` of the sample the walk stands on
        std::uint64_t rem{0ULL};     ///< that sample's position within `t_ns`, over `modulus`
        std::uint64_t whole{0ULL};   ///< whole nanoseconds a sample advances
        std::uint64_t carry{0ULL};   ///< what it advances beyond those, over `modulus`
        std::uint64_t modulus{1ULL}; ///< `rate_num`

        void advance() noexcept {
            t_ns += static_cast<std::int64_t>(whole);
            rem += carry;
            if (rem >= modulus) {
                rem -= modulus;
                ++t_ns;
            }
        }
    };

    [[nodiscard]] NsWalk walkFrom(std::uint64_t index) const noexcept {
        const NsSplit       at     = offsetFrom(index);
        const std::uint64_t period = periodNsNum();
        return NsWalk{anchor_ns + at.whole, at.rem, period / rate_num, period % rate_num, rate_num};
    }
};

static_assert(SampleClock::kMaxRateDen * SampleClock::kNsPerSecond <= SampleClock::kMaxRateNum, "10^9 * rate_den is a divisor and must stay inside the same bound as the numerator");

/**
 * @brief A clock at @p rateHz on the microhertz grid, anchored at (@p anchorIndex, @p anchorNs).
 *
 * The one place a floating rate touches the exact time axis: the rate lands on a 10^-6 Hz grid, whose
 * half-microhertz residual moves a sample's time by under a nanosecond across ten hours of stream —
 * far inside the resolution any trajectory or schedule is stated to. A rate outside
 * `[1e-6, kMaxRateNum / 1e6]` Hz, or one that is not finite and positive, throws
 * `std::invalid_argument`; the callers that hold a settings name wrap the message under it.
 */
[[nodiscard]] inline SampleClock clockForRateHz(double rateHz, std::uint64_t anchorIndex, std::int64_t anchorNs) {
    constexpr std::uint64_t kDenominator = 1'000'000ULL;
    constexpr double        kMaxRateHz   = static_cast<double>(SampleClock::kMaxRateNum / kDenominator);
    if (!(rateHz >= 1. / static_cast<double>(kDenominator)) || !(rateHz <= kMaxRateHz)) {
        throw std::invalid_argument(std::format("gr::timing::clockForRateHz: {} Hz is outside the [{:g}, {:g}] Hz the exact time axis allows", rateHz, 1. / static_cast<double>(kDenominator), kMaxRateHz));
    }
    return SampleClock(anchorIndex, anchorNs, static_cast<std::uint64_t>(std::llround(rateHz * static_cast<double>(kDenominator))), kDenominator);
}

} // namespace gr::timing

#endif // GNURADIO_ALGORITHM_SAMPLE_CLOCK_HPP
