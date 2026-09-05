#ifndef GNURADIO_ALGORITHM_SIGMF_SAMPLE_CODEC_HPP
#define GNURADIO_ALGORITHM_SIGMF_SAMPLE_CODEC_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

/**
 * @brief The SigMF datatype grammar and the sample conversions between a dataset and a stream.
 *
 * The tuple (datatype, stream component type, scaling, endianness) is fixed for a whole recording,
 * so it is resolved once into a single monomorphic conversion held as a plain function pointer. The
 * per-component loop then carries no branch on datatype, scaling or endianness.
 */
namespace gr::sigmf {

// The codec's byte handling assumes a little-endian host: the read path swaps only for a `_be`
// datatype and the write path emits little-endian only. A big-endian port needs the whole of this
// header revisited rather than silently mishandled.
static_assert(std::endian::native == std::endian::little, "gr::sigmf's sample codec requires a little-endian host");

enum class Domain : std::uint8_t { Real, Complex };
enum class SampleKind : std::uint8_t { Float, SignedInt, UnsignedInt };
enum class Scaling : std::uint8_t { Unit, Raw };

/// One parsed `core:datatype` spelling.
struct Datatype {
    Domain     domain{Domain::Real};
    SampleKind kind{SampleKind::Float};
    unsigned   width{32U};       ///< bits per component
    bool       bigEndian{false}; ///< the spelling carried a `_be` suffix
    bool       hasEndianSuffix{false};

    [[nodiscard]] bool operator==(const Datatype&) const noexcept = default;

    [[nodiscard]] constexpr std::size_t bytesPerComponent() const noexcept { return width / 8U; }
    [[nodiscard]] constexpr std::size_t componentsPerSample() const noexcept { return domain == Domain::Complex ? 2UZ : 1UZ; }
    [[nodiscard]] constexpr std::size_t bytesPerSample() const noexcept { return bytesPerComponent() * componentsPerSample(); }
};

/**
 * @brief Parse a `core:datatype` spelling.
 *
 * Returns `datatype_unparsable` on failure, which is the only refusal the grammar has. The
 * 64-bit integers and every 16-bit float spelling are outside the specification's own dataset
 * format production, so they are refused for the reason any unknown spelling is refused.
 */
[[nodiscard]] inline std::expected<Datatype, std::string> parseDatatype(std::string_view spelling) {
    const auto unparsable = [] { return std::unexpected(std::string("datatype_unparsable")); };

    std::size_t cursor = 0UZ;
    if (spelling.size() < 3UZ) {
        return unparsable();
    }
    Datatype result{};
    switch (spelling[cursor]) {
    case 'r': result.domain = Domain::Real; break;
    case 'c': result.domain = Domain::Complex; break;
    default: return unparsable();
    }
    ++cursor;
    switch (spelling[cursor]) {
    case 'f': result.kind = SampleKind::Float; break;
    case 'i': result.kind = SampleKind::SignedInt; break;
    case 'u': result.kind = SampleKind::UnsignedInt; break;
    default: return unparsable();
    }
    ++cursor;

    const std::string_view remainder = spelling.substr(cursor);
    if (remainder.starts_with("8")) {
        result.width = 8U;
        cursor += 1UZ;
    } else if (remainder.starts_with("16")) {
        result.width = 16U;
        cursor += 2UZ;
    } else if (remainder.starts_with("32")) {
        result.width = 32U;
        cursor += 2UZ;
    } else if (remainder.starts_with("64")) {
        result.width = 64U;
        cursor += 2UZ;
    } else {
        return unparsable();
    }
    // The grammar's own productions are `f32`/`f64` for a float and `i8`/`i16`/`i32`/`u8`/`u16`/
    // `u32` for an integer. A 16-bit float and a 64-bit integer are not derivable from either, so
    // neither is a datatype this format defines and both are refused as unknown spellings.
    const bool widthInGrammar = result.kind == SampleKind::Float ? (result.width == 32U || result.width == 64U) : (result.width != 64U);
    if (!widthInGrammar) {
        return unparsable();
    }

    const std::string_view suffix = spelling.substr(cursor);
    if (suffix.empty()) {
        if (result.width > 8U) {
            return unparsable(); // an endianness suffix is required above eight bits
        }
    } else if (suffix == "_le" || suffix == "_be") {
        result.hasEndianSuffix = true;
        result.bigEndian       = suffix == "_be";
    } else {
        return unparsable();
    }
    return result;
}

/// Spell a datatype back. Eight-bit types are spelled without an endianness suffix.
[[nodiscard]] inline std::string spellDatatype(const Datatype& datatype) {
    std::string out;
    out.push_back(datatype.domain == Domain::Complex ? 'c' : 'r');
    out.push_back(datatype.kind == SampleKind::Float ? 'f' : (datatype.kind == SampleKind::SignedInt ? 'i' : 'u'));
    out.append(std::to_string(datatype.width));
    if (datatype.width > 8U) {
        out.append(datatype.bigEndian ? "_be" : "_le");
    }
    return out;
}

/// True for the sixteen spellings this tree writes: little-endian, plus the two suffixless 8-bit ones.
[[nodiscard]] inline bool isWriteLegal(const Datatype& datatype) {
    if (datatype.kind != SampleKind::Float && datatype.width == 64U) {
        return false;
    }
    if (datatype.width > 8U) {
        return !datatype.bigEndian;
    }
    return datatype.kind != SampleKind::Float;
}

namespace detail {

template<typename T>
concept StreamComponent = std::same_as<T, std::uint8_t> || std::same_as<T, std::int16_t> || std::same_as<T, std::int32_t> || std::same_as<T, float>;

/// The four conversion cases of the scaling table, closed. An integer pair needs the same width and
/// signedness; any other integer pair is refused rather than defined.
template<typename TFile, typename TStream>
inline constexpr bool kConvertible = (std::floating_point<TFile> && std::floating_point<TStream>) //
                                     || (std::integral<TFile> && std::floating_point<TStream>)    //
                                     || (std::floating_point<TFile> && std::integral<TStream>)    //
                                     || (std::integral<TFile> && std::integral<TStream> && sizeof(TFile) == sizeof(TStream) && std::is_signed_v<TFile> == std::is_signed_v<TStream>);

template<typename T>
using UnsignedOfSize = std::conditional_t<sizeof(T) == 1UZ, std::uint8_t, std::conditional_t<sizeof(T) == 2UZ, std::uint16_t, std::conditional_t<sizeof(T) == 4UZ, std::uint32_t, std::uint64_t>>>;

template<typename T, bool Swap>
[[nodiscard]] inline T loadValue(const std::byte* source) noexcept {
    T value{};
    if constexpr (sizeof(T) == 1UZ || !Swap) {
        std::memcpy(&value, source, sizeof(T));
    } else {
        UnsignedOfSize<T> raw{};
        std::memcpy(&raw, source, sizeof(raw));
        raw = std::byteswap(raw);
        std::memcpy(&value, &raw, sizeof(value));
    }
    return value;
}

template<typename T, bool Swap>
inline void storeValue(std::byte* destination, T value) noexcept {
    if constexpr (sizeof(T) == 1UZ || !Swap) {
        std::memcpy(destination, &value, sizeof(T));
    } else {
        UnsignedOfSize<T> raw{};
        std::memcpy(&raw, &value, sizeof(raw));
        raw = std::byteswap(raw);
        std::memcpy(destination, &raw, sizeof(raw));
    }
}

/// `2^(w-1)` for an integer type of width `w`, as a `double`.
template<typename TInteger>
inline constexpr double kFullScale = static_cast<double>(std::uint64_t{1} << (8UZ * sizeof(TInteger) - 1UZ));

/// The offset that puts zero at mid-scale for an unsigned integer type, and at zero for a signed one.
template<typename TInteger>
inline constexpr double kZeroOffset = std::is_signed_v<TInteger> ? 0.0 : kFullScale<TInteger>;

/// The saturation limits of the unit path: the signed range of the width, before the offset is added.
template<typename TInteger>
inline constexpr double kUnitLow = -kFullScale<TInteger>;
template<typename TInteger>
inline constexpr double kUnitHigh = kFullScale<TInteger> - 1.0;

/// The saturation limits of the raw path: the integer type's own range.
template<typename TInteger>
inline constexpr double kRawLow = static_cast<double>(std::numeric_limits<TInteger>::lowest());
template<typename TInteger>
inline constexpr double kRawHigh = static_cast<double>(std::numeric_limits<TInteger>::max());

/// Round half away from zero, saturate, and report whether the value was clipped.
[[nodiscard]] inline double roundAndClamp(double value, double low, double high, std::uint64_t& clipped) noexcept {
    double rounded = std::round(value);
    if (!(rounded >= low)) { // also catches a non-finite value, whose comparison is false
        ++clipped;
        return low;
    }
    if (rounded > high) {
        ++clipped;
        return high;
    }
    return rounded;
}

/**
 * @brief Convert `count` components from the dataset's layout into the stream's component type.
 * @return the number of components that saturated on the float-to-integer path.
 */
template<typename TFile, typename TStream, bool Unit, bool Swap>
inline std::uint64_t decodeComponents(const std::byte* source, TStream* destination, std::size_t count) noexcept {
    std::uint64_t clipped = 0U;
    if constexpr (!kConvertible<TFile, TStream>) {
        // unreachable: the selection step refuses this pair by name before a pointer to it is taken
        (void)source;
        (void)destination;
        (void)count;
    } else if constexpr (std::same_as<TFile, TStream> && !Swap) {
        std::memcpy(destination, source, count * sizeof(TStream));
    } else if constexpr (std::integral<TFile> && std::integral<TStream>) {
        for (std::size_t i = 0UZ; i < count; ++i) {
            destination[i] = loadValue<TFile, Swap>(source + i * sizeof(TFile));
        }
    } else if constexpr (std::floating_point<TFile> && std::floating_point<TStream>) {
        for (std::size_t i = 0UZ; i < count; ++i) {
            const TFile value = loadValue<TFile, Swap>(source + i * sizeof(TFile));
            if constexpr (std::same_as<TFile, TStream>) {
                destination[i] = value;
            } else {
                destination[i] = static_cast<TStream>(value);
            }
        }
    } else if constexpr (std::integral<TFile> && std::floating_point<TStream>) {
        constexpr TStream kOffset  = static_cast<TStream>(kZeroOffset<TFile>);
        constexpr TStream kInverse = static_cast<TStream>(1.0 / kFullScale<TFile>);
        for (std::size_t i = 0UZ; i < count; ++i) {
            const TStream raw = static_cast<TStream>(loadValue<TFile, Swap>(source + i * sizeof(TFile)));
            if constexpr (Unit) {
                destination[i] = (raw - kOffset) * kInverse;
            } else {
                destination[i] = raw;
            }
        }
    } else { // a floating-point dataset into an integer stream
        constexpr double kScale  = kFullScale<TStream>;
        constexpr double kOffset = kZeroOffset<TStream>;
        constexpr double kLow    = Unit ? kUnitLow<TStream> : kRawLow<TStream>;
        constexpr double kHigh   = Unit ? kUnitHigh<TStream> : kRawHigh<TStream>;
        for (std::size_t i = 0UZ; i < count; ++i) {
            const double raw     = static_cast<double>(loadValue<TFile, Swap>(source + i * sizeof(TFile)));
            const double scaled  = Unit ? raw * kScale : raw;
            const double clamped = roundAndClamp(scaled, kLow, kHigh, clipped);
            destination[i]       = static_cast<TStream>(Unit ? clamped + kOffset : clamped);
        }
    }
    return clipped;
}

/**
 * @brief Convert `count` components from the stream's component type into the dataset's layout.
 * @return the number of components that saturated on the float-to-integer path.
 */
template<typename TFile, typename TStream, bool Unit, bool Swap>
inline std::uint64_t encodeComponents(const TStream* source, std::byte* destination, std::size_t count) noexcept {
    std::uint64_t clipped = 0U;
    if constexpr (!kConvertible<TFile, TStream>) {
        // unreachable: the selection step refuses this pair by name before a pointer to it is taken
        (void)source;
        (void)destination;
        (void)count;
    } else if constexpr (std::same_as<TFile, TStream> && !Swap) {
        std::memcpy(destination, source, count * sizeof(TStream));
    } else if constexpr (std::integral<TFile> && std::integral<TStream>) {
        for (std::size_t i = 0UZ; i < count; ++i) {
            storeValue<TFile, Swap>(destination + i * sizeof(TFile), source[i]);
        }
    } else if constexpr (std::floating_point<TFile> && std::floating_point<TStream>) {
        for (std::size_t i = 0UZ; i < count; ++i) {
            if constexpr (std::same_as<TFile, TStream>) {
                storeValue<TFile, Swap>(destination + i * sizeof(TFile), source[i]);
            } else {
                storeValue<TFile, Swap>(destination + i * sizeof(TFile), static_cast<TFile>(source[i]));
            }
        }
    } else if constexpr (std::integral<TStream> && std::floating_point<TFile>) {
        constexpr TFile kOffset  = static_cast<TFile>(kZeroOffset<TStream>);
        constexpr TFile kInverse = static_cast<TFile>(1.0 / kFullScale<TStream>);
        for (std::size_t i = 0UZ; i < count; ++i) {
            const TFile raw = static_cast<TFile>(source[i]);
            storeValue<TFile, Swap>(destination + i * sizeof(TFile), Unit ? (raw - kOffset) * kInverse : raw);
        }
    } else { // a floating-point stream into an integer dataset
        constexpr double kScale  = kFullScale<TFile>;
        constexpr double kOffset = kZeroOffset<TFile>;
        constexpr double kLow    = Unit ? kUnitLow<TFile> : kRawLow<TFile>;
        constexpr double kHigh   = Unit ? kUnitHigh<TFile> : kRawHigh<TFile>;
        for (std::size_t i = 0UZ; i < count; ++i) {
            const double raw     = static_cast<double>(source[i]);
            const double scaled  = Unit ? raw * kScale : raw;
            const double clamped = roundAndClamp(scaled, kLow, kHigh, clipped);
            storeValue<TFile, Swap>(destination + i * sizeof(TFile), static_cast<TFile>(Unit ? clamped + kOffset : clamped));
        }
    }
    return clipped;
}

} // namespace detail

template<typename TStream>
using DecodeFn = std::uint64_t (*)(const std::byte*, TStream*, std::size_t) noexcept;

template<typename TStream>
using EncodeFn = std::uint64_t (*)(const TStream*, std::byte*, std::size_t) noexcept;

namespace detail {

template<typename TFile, typename TStream>
[[nodiscard]] std::expected<DecodeFn<TStream>, std::string> pickDecoder(bool unit, bool swap) {
    if constexpr (!kConvertible<TFile, TStream>) {
        return std::unexpected(std::string("integer_width"));
    } else {
        if (unit) {
            return swap ? &decodeComponents<TFile, TStream, true, true> : &decodeComponents<TFile, TStream, true, false>;
        }
        return swap ? &decodeComponents<TFile, TStream, false, true> : &decodeComponents<TFile, TStream, false, false>;
    }
}

template<typename TFile, typename TStream>
[[nodiscard]] std::expected<EncodeFn<TStream>, std::string> pickEncoder(bool unit, bool swap) {
    if constexpr (!kConvertible<TFile, TStream>) {
        return std::unexpected(std::string("integer_width"));
    } else {
        if (unit) {
            return swap ? &encodeComponents<TFile, TStream, true, true> : &encodeComponents<TFile, TStream, true, false>;
        }
        return swap ? &encodeComponents<TFile, TStream, false, true> : &encodeComponents<TFile, TStream, false, false>;
    }
}

} // namespace detail

/// Resolve the dataset-to-stream conversion once, for the whole recording.
template<detail::StreamComponent TStream>
[[nodiscard]] std::expected<DecodeFn<TStream>, std::string> selectDecoder(const Datatype& datatype, Scaling scaling) {
    const bool unit = scaling == Scaling::Unit;
    const bool swap = datatype.bigEndian && datatype.width > 8U;
    switch (datatype.kind) {
    case SampleKind::Float:
        if (datatype.width == 32U) {
            return detail::pickDecoder<float, TStream>(unit, swap);
        }
        return detail::pickDecoder<double, TStream>(unit, swap);
    case SampleKind::SignedInt:
        if (datatype.width == 8U) {
            return detail::pickDecoder<std::int8_t, TStream>(unit, swap);
        }
        if (datatype.width == 16U) {
            return detail::pickDecoder<std::int16_t, TStream>(unit, swap);
        }
        if (datatype.width == 32U) {
            return detail::pickDecoder<std::int32_t, TStream>(unit, swap);
        }
        break;
    case SampleKind::UnsignedInt:
        if (datatype.width == 8U) {
            return detail::pickDecoder<std::uint8_t, TStream>(unit, swap);
        }
        if (datatype.width == 16U) {
            return detail::pickDecoder<std::uint16_t, TStream>(unit, swap);
        }
        if (datatype.width == 32U) {
            return detail::pickDecoder<std::uint32_t, TStream>(unit, swap);
        }
        break;
    }
    // unreachable for a value from `parseDatatype`, which admits no other width; a hand-built
    // value naming one is refused with the grammar's own name
    return std::unexpected(std::string("datatype_unparsable"));
}

/// Resolve the stream-to-dataset conversion once, for the whole recording.
template<detail::StreamComponent TStream>
[[nodiscard]] std::expected<EncodeFn<TStream>, std::string> selectEncoder(const Datatype& datatype, Scaling scaling) {
    const bool unit = scaling == Scaling::Unit;
    const bool swap = datatype.bigEndian && datatype.width > 8U;
    switch (datatype.kind) {
    case SampleKind::Float:
        if (datatype.width == 32U) {
            return detail::pickEncoder<float, TStream>(unit, swap);
        }
        return detail::pickEncoder<double, TStream>(unit, swap);
    case SampleKind::SignedInt:
        if (datatype.width == 8U) {
            return detail::pickEncoder<std::int8_t, TStream>(unit, swap);
        }
        if (datatype.width == 16U) {
            return detail::pickEncoder<std::int16_t, TStream>(unit, swap);
        }
        if (datatype.width == 32U) {
            return detail::pickEncoder<std::int32_t, TStream>(unit, swap);
        }
        break;
    case SampleKind::UnsignedInt:
        if (datatype.width == 8U) {
            return detail::pickEncoder<std::uint8_t, TStream>(unit, swap);
        }
        if (datatype.width == 16U) {
            return detail::pickEncoder<std::uint16_t, TStream>(unit, swap);
        }
        if (datatype.width == 32U) {
            return detail::pickEncoder<std::uint32_t, TStream>(unit, swap);
        }
        break;
    }
    // unreachable for a value from `parseDatatype`, as above
    return std::unexpected(std::string("datatype_unparsable"));
}

/// The stream sample type's own domain: a complex stream needs a complex datatype and the reverse.
template<typename T>
[[nodiscard]] constexpr Domain domainOf() noexcept {
    if constexpr (std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>) {
        return Domain::Complex;
    } else {
        return Domain::Real;
    }
}

/// Number of scalar components one stream item carries.
template<typename T>
inline constexpr std::size_t componentsPerItem = domainOf<T>() == Domain::Complex ? 2UZ : 1UZ;

/// The scalar component type of one stream item.
template<typename T>
using ComponentOf = std::conditional_t<std::same_as<T, std::complex<float>>, float, std::conditional_t<std::same_as<T, std::complex<double>>, double, T>>;

/// The one datatype spelling whose conversion to `T` is a `memcpy` on a little-endian host.
template<typename T>
[[nodiscard]] constexpr std::string_view canonicalDatatypeFor() noexcept {
    if constexpr (std::same_as<T, std::uint8_t>) {
        return "ru8";
    } else if constexpr (std::same_as<T, std::int16_t>) {
        return "ri16_le";
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return "ri32_le";
    } else if constexpr (std::same_as<T, float>) {
        return "rf32_le";
    } else if constexpr (std::same_as<T, std::complex<float>>) {
        return "cf32_le";
    } else {
        return "";
    }
}

} // namespace gr::sigmf

#endif // GNURADIO_ALGORITHM_SIGMF_SAMPLE_CODEC_HPP
