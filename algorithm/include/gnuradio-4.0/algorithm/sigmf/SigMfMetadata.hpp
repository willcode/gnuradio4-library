#ifndef GNURADIO_ALGORITHM_SIGMF_METADATA_HPP
#define GNURADIO_ALGORITHM_SIGMF_METADATA_HPP

#include <gnuradio-4.0/algorithm/sigmf/Json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/**
 * @brief The SigMF metadata document: its three objects, the field mapping, and the validator.
 *
 * Two rules bind the whole type. Every index is a `std::uint64_t` and never passes through a
 * floating-point type, and an absent field is a value distinct from a field written as zero, so the
 * writer's omission rule can tell them apart.
 */
namespace gr::sigmf {

/// The SigMF core specification version this tree writes: the current release, v1.2.6.
inline constexpr std::string_view kWrittenVersion = "1.2.6";

/// The largest array this reader will repair by sorting. Above it an unsorted array is refused:
/// a few out-of-order entries are hand-edited sloppiness, thousands are a malfunctioning producer.
inline constexpr std::size_t kMaxUnsortedEntries = 1024UZ;

/// A refusal: a stable name a test can assert on, plus the offending value in prose.
struct Error {
    std::string code{};
    std::string message{};
};

/// Bounds applied to a metadata document before anything is sized from it.
struct Limits {
    std::uint64_t maxCaptures{1048576U};
    std::uint64_t maxAnnotations{1048576U};
    std::size_t   maxUnsortedEntries{kMaxUnsortedEntries};
};

/// Tolerances exercised while reading a document; every tolerance leaves a count behind.
struct ParseCounters {
    std::uint64_t nIntegralFloatIndices{0U};
    std::uint64_t nCapturesSorted{0U};
    std::uint64_t nAnnotationsSorted{0U};
};

struct Global {
    std::string                  datatype{};
    std::string                  version{};
    std::optional<double>        sampleRate{};
    std::optional<std::uint64_t> numChannels{};
    std::optional<std::uint64_t> offset{};
    std::optional<std::string>   description{};
    std::optional<std::string>   author{};
    std::optional<std::string>   recorder{};
    std::optional<std::string>   hardware{}; ///< the SigMF field is `core:hw`
    std::optional<std::string>   license{};
    std::optional<std::string>   sha512{};
    std::optional<std::string>   dataset{};
    std::optional<std::uint64_t> trailingBytes{};
    std::optional<bool>          metadataOnly{};
    json::Value                  extra{json::Value::makeObject()}; ///< every member outside the list above, verbatim

    [[nodiscard]] bool operator==(const Global&) const = default;
};

struct Capture {
    std::uint64_t                sampleStart{0U};
    std::optional<std::uint64_t> globalIndex{};
    std::optional<std::uint64_t> headerBytes{};
    std::optional<double>        frequency{};
    std::optional<std::string>   datetime{};
    json::Value                  extra{json::Value::makeObject()};

    [[nodiscard]] bool operator==(const Capture&) const = default;
};

struct Annotation {
    std::uint64_t                sampleStart{0U};
    std::optional<std::uint64_t> sampleCount{};
    std::optional<double>        freqLower{};
    std::optional<double>        freqUpper{};
    std::optional<std::string>   label{};
    std::optional<std::string>   comment{};
    std::optional<std::string>   generator{};
    std::optional<std::string>   uuid{};
    json::Value                  extra{json::Value::makeObject()};

    [[nodiscard]] bool operator==(const Annotation&) const = default;
};

struct Metadata {
    Global                  global{};
    std::vector<Capture>    captures{};    ///< sorted ascending by `sampleStart`
    std::vector<Annotation> annotations{}; ///< sorted ascending by `sampleStart`, then by `sampleCount`

    [[nodiscard]] bool operator==(const Metadata&) const = default;
};

namespace detail {

inline constexpr std::array<std::string_view, 14> kGlobalKnownKeys{"core:datatype", "core:version", "core:sample_rate", "core:num_channels", "core:offset", "core:description", "core:author", "core:recorder", "core:hw", "core:license", "core:sha512", "core:dataset", "core:trailing_bytes", "core:metadata_only"};
inline constexpr std::array<std::string_view, 5>  kCaptureKnownKeys{"core:sample_start", "core:global_index", "core:header_bytes", "core:frequency", "core:datetime"};
inline constexpr std::array<std::string_view, 8>  kAnnotationKnownKeys{"core:sample_start", "core:sample_count", "core:freq_lower_edge", "core:freq_upper_edge", "core:label", "core:comment", "core:generator", "core:uuid"};

[[nodiscard]] inline bool isKnown(std::span<const std::string_view> known, std::string_view key) noexcept { return std::ranges::find(known, key) != known.end(); }

/// Read a schema field the specification types as an unsigned integer, applying the integral-float
/// tolerance: `4096.0` is read as `4096` from the token's own digits and counted, while a non-zero
/// fraction or an exponent is refused.
[[nodiscard]] inline std::expected<std::uint64_t, Error> unsignedField(const json::Value& value, std::string_view field, ParseCounters& counters) {
    if (!value.isNumber()) {
        return std::unexpected(Error{"meta_unparsable", std::format("field '{}' is not a number", field)});
    }
    const json::Number& number = value.number();
    if (number.negative) {
        return std::unexpected(Error{"negative_index", std::format("field '{}' carries a negative value", field)});
    }
    if (!number.isInteger && !number.isIntegralFloat) {
        return std::unexpected(Error{"non_integral_index", std::format("field '{}' is not an integer token", field)});
    }
    if (number.magnitudeOverflow) {
        return std::unexpected(Error{"index_overflow", std::format("field '{}' does not fit a 64-bit unsigned integer", field)});
    }
    if (number.isIntegralFloat) {
        ++counters.nIntegralFloatIndices;
    }
    return number.magnitude;
}

[[nodiscard]] inline std::expected<double, Error> doubleField(const json::Value& value, std::string_view field) {
    if (!value.isNumber()) {
        return std::unexpected(Error{"meta_unparsable", std::format("field '{}' is not a number", field)});
    }
    return value.number().real;
}

[[nodiscard]] inline std::expected<std::string, Error> stringField(const json::Value& value, std::string_view field) {
    if (!value.isString()) {
        return std::unexpected(Error{"meta_unparsable", std::format("field '{}' is not a string", field)});
    }
    return value.str();
}

[[nodiscard]] inline std::expected<bool, Error> boolField(const json::Value& value, std::string_view field) {
    if (!value.isBool()) {
        return std::unexpected(Error{"meta_unparsable", std::format("field '{}' is not a boolean", field)});
    }
    return value.boolValue();
}

/// Copy every member outside `known` into a fresh object, keeping the document's own order.
[[nodiscard]] inline json::Value carriedMembers(const json::Value& object, std::span<const std::string_view> known) {
    json::Value carried = json::Value::makeObject();
    for (std::size_t i = 0UZ; i < object.size(); ++i) {
        if (!isKnown(known, object.keyAt(i))) {
            carried.append(object.keyAt(i), object.valueAt(i));
        }
    }
    return carried;
}

/// Append `object`'s members to `out`, ordered lexicographically by key.
inline void appendSorted(json::Value& out, const json::Value& object) {
    std::vector<std::size_t> order(object.size());
    for (std::size_t i = 0UZ; i < order.size(); ++i) {
        order[i] = i;
    }
    std::ranges::sort(order, [&object](std::size_t lhs, std::size_t rhs) { return object.keyAt(lhs) < object.keyAt(rhs); });
    for (const std::size_t index : order) {
        out.append(object.keyAt(index), object.valueAt(index));
    }
}

[[nodiscard]] inline std::optional<Error> checkFinite(const std::optional<double>& value, std::string_view field) {
    if (value.has_value() && !std::isfinite(*value)) {
        return Error{"non_finite_number", std::format("field '{}' is not a finite number", field)};
    }
    return std::nullopt;
}

[[nodiscard]] constexpr std::int64_t daysFromCivil(std::int64_t year, unsigned month, unsigned day) noexcept {
    year -= month <= 2U ? 1 : 0;
    const std::int64_t  era = (year >= 0 ? year : year - 399) / 400;
    const auto          yoe = static_cast<std::uint32_t>(year - era * 400);
    const std::uint32_t doy = (153U * (month + (month > 2U ? 0U : 12U) - 3U) + 2U) / 5U + day - 1U;
    const std::uint32_t doe = yoe * 365U + yoe / 4U - yoe / 100U + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

struct CivilDate {
    std::int64_t  year{};
    std::uint32_t month{};
    std::uint32_t day{};
};

[[nodiscard]] constexpr CivilDate civilFromDays(std::int64_t days) noexcept {
    days += 719468;
    const std::int64_t  era = (days >= 0 ? days : days - 146096) / 146097;
    const auto          doe = static_cast<std::uint32_t>(days - era * 146097);
    const std::uint32_t yoe = (doe - doe / 1460U + doe / 36524U - doe / 146096U) / 365U;
    const std::int64_t  y   = static_cast<std::int64_t>(yoe) + era * 400;
    const std::uint32_t doy = doe - (365U * yoe + yoe / 4U - yoe / 100U);
    const std::uint32_t mp  = (5U * doy + 2U) / 153U;
    const std::uint32_t d   = doy - (153U * mp + 2U) / 5U + 1U;
    const std::uint32_t m   = mp < 10U ? mp + 3U : mp - 9U;
    return CivilDate{y + (m <= 2U ? 1 : 0), m, d};
}

[[nodiscard]] inline bool readDigits(std::string_view text, std::size_t offset, std::size_t count, std::uint32_t& out) noexcept {
    if (offset + count > text.size()) {
        return false;
    }
    std::uint32_t accumulated = 0U;
    for (std::size_t i = 0UZ; i < count; ++i) {
        const char digit = text[offset + i];
        if (digit < '0' || digit > '9') {
            return false;
        }
        accumulated = accumulated * 10U + static_cast<std::uint32_t>(digit - '0');
    }
    out = accumulated;
    return true;
}

} // namespace detail

/**
 * @brief Read an RFC 3339 UTC timestamp with a mandatory `Z` and any number of fractional digits.
 *
 * The specification's `time-secfrac = "." 1*DIGIT` puts no bound on the fractional part, so every
 * digit is consumed and the first nine are kept; a tenth and beyond are below the resolution of
 * the returned value and are ignored rather than refused.
 *
 * @return nanoseconds since the Unix epoch, or nothing when the text is not of that form or names
 *         an instant before the epoch.
 */
[[nodiscard]] inline std::optional<std::uint64_t> parseDatetimeNs(std::string_view text) {
    if (text.size() < 20UZ || text[4] != '-' || text[7] != '-' || text[10] != 'T' || text[13] != ':' || text[16] != ':' || text.back() != 'Z') {
        return std::nullopt;
    }
    std::uint32_t year = 0U, month = 0U, day = 0U, hour = 0U, minute = 0U, second = 0U;
    if (!detail::readDigits(text, 0UZ, 4UZ, year) || !detail::readDigits(text, 5UZ, 2UZ, month) || !detail::readDigits(text, 8UZ, 2UZ, day) //
        || !detail::readDigits(text, 11UZ, 2UZ, hour) || !detail::readDigits(text, 14UZ, 2UZ, minute) || !detail::readDigits(text, 17UZ, 2UZ, second)) {
        return std::nullopt;
    }
    if (month < 1U || month > 12U || day < 1U || day > 31U || hour > 23U || minute > 59U || second > 60U) {
        return std::nullopt;
    }

    std::uint64_t fractionNs = 0U;
    if (text.size() > 20UZ) {
        if (text[19] != '.') {
            return std::nullopt;
        }
        const std::string_view fraction = text.substr(20UZ, text.size() - 21UZ);
        if (fraction.empty()) {
            return std::nullopt;
        }
        std::uint64_t scale = 1000000000U;
        for (const char digit : fraction) {
            if (digit < '0' || digit > '9') {
                return std::nullopt;
            }
            scale /= 10U; // reaches zero past the ninth digit, so the surplus contributes nothing
            fractionNs += static_cast<std::uint64_t>(digit - '0') * scale;
        }
    } else if (text.size() != 20UZ) {
        return std::nullopt;
    }

    const std::int64_t days    = detail::daysFromCivil(static_cast<std::int64_t>(year), month, day);
    const std::int64_t seconds = days * 86400 + static_cast<std::int64_t>(hour) * 3600 + static_cast<std::int64_t>(minute) * 60 + static_cast<std::int64_t>(second);
    if (seconds < 0) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(seconds) * 1000000000U + fractionNs;
}

/// Format nanoseconds since the Unix epoch as `YYYY-MM-DDTHH:MM:SS.ffffffZ`, with six fractional digits.
[[nodiscard]] inline std::string formatDatetimeNs(std::uint64_t nanoseconds) {
    const std::uint64_t totalSeconds = nanoseconds / 1000000000U;
    const std::uint64_t microseconds = (nanoseconds % 1000000000U) / 1000U;
    const std::int64_t  days         = static_cast<std::int64_t>(totalSeconds / 86400U);
    const std::uint64_t timeOfDay    = totalSeconds % 86400U;
    const auto          date         = detail::civilFromDays(days);
    return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z", date.year, date.month, date.day, timeOfDay / 3600U, (timeOfDay / 60U) % 60U, timeOfDay % 60U, microseconds);
}

/// Section 1.3's table: `core:version` governs parsing, so a higher major version is refused.
[[nodiscard]] inline std::optional<Error> checkVersion(std::string_view version) {
    if (version.empty()) {
        return Error{"missing_version", "the document carries no 'core:version'"};
    }
    std::uint32_t major  = 0U;
    std::size_t   digits = 0UZ;
    while (digits < version.size() && version[digits] >= '0' && version[digits] <= '9') {
        major = major * 10U + static_cast<std::uint32_t>(version[digits] - '0');
        ++digits;
    }
    if (digits == 0UZ) {
        return Error{"missing_version", std::format("'core:version' is '{}', which has no major version", version)};
    }
    if (major > 1U) {
        return Error{"future_version", std::format("'core:version' is '{}', which this reader does not describe", version)};
    }
    if (major < 1U) {
        return Error{"legacy_version", std::format("'core:version' is '{}'; the v0.x drafts are not described", version)};
    }
    return std::nullopt;
}

/// Build the JSON document for `metadata`, in section 2.3's fixed key order.
[[nodiscard]] inline std::expected<json::Value, Error> toJson(const Metadata& metadata) {
    if (auto refusal = detail::checkFinite(metadata.global.sampleRate, "core:sample_rate"); refusal) {
        return std::unexpected(*refusal);
    }

    json::Value global = json::Value::makeObject();
    global.append("core:datatype", json::Value::fromString(metadata.global.datatype));
    global.append("core:version", json::Value::fromString(metadata.global.version));
    if (metadata.global.sampleRate) {
        global.append("core:sample_rate", json::Value::fromDouble(*metadata.global.sampleRate));
    }
    if (metadata.global.numChannels && *metadata.global.numChannels != 1U) { // writing a stated default states nothing
        global.append("core:num_channels", json::Value::fromUnsigned(*metadata.global.numChannels));
    }
    if (metadata.global.offset && *metadata.global.offset != 0U) {
        global.append("core:offset", json::Value::fromUnsigned(*metadata.global.offset));
    }
    const auto appendString = [](json::Value& object, std::string_view key, const std::optional<std::string>& text) {
        if (text) {
            object.append(std::string(key), json::Value::fromString(*text));
        }
    };
    appendString(global, "core:description", metadata.global.description);
    appendString(global, "core:author", metadata.global.author);
    appendString(global, "core:recorder", metadata.global.recorder);
    appendString(global, "core:hw", metadata.global.hardware);
    appendString(global, "core:license", metadata.global.license);
    appendString(global, "core:sha512", metadata.global.sha512);
    appendString(global, "core:dataset", metadata.global.dataset);
    if (metadata.global.trailingBytes) {
        global.append("core:trailing_bytes", json::Value::fromUnsigned(*metadata.global.trailingBytes));
    }
    if (metadata.global.metadataOnly) {
        global.append("core:metadata_only", json::Value::fromBool(*metadata.global.metadataOnly));
    }
    detail::appendSorted(global, metadata.global.extra);

    json::Value captures = json::Value::makeArray();
    for (const Capture& capture : metadata.captures) {
        if (auto refusal = detail::checkFinite(capture.frequency, "core:frequency"); refusal) {
            return std::unexpected(*refusal);
        }
        json::Value entry = json::Value::makeObject();
        entry.append("core:sample_start", json::Value::fromUnsigned(capture.sampleStart));
        if (capture.globalIndex) {
            entry.append("core:global_index", json::Value::fromUnsigned(*capture.globalIndex));
        }
        if (capture.headerBytes) {
            entry.append("core:header_bytes", json::Value::fromUnsigned(*capture.headerBytes));
        }
        if (capture.frequency) {
            entry.append("core:frequency", json::Value::fromDouble(*capture.frequency));
        }
        appendString(entry, "core:datetime", capture.datetime);
        detail::appendSorted(entry, capture.extra);
        captures.push(std::move(entry));
    }

    json::Value annotations = json::Value::makeArray();
    for (const Annotation& annotation : metadata.annotations) {
        if (auto refusal = detail::checkFinite(annotation.freqLower, "core:freq_lower_edge"); refusal) {
            return std::unexpected(*refusal);
        }
        if (auto refusal = detail::checkFinite(annotation.freqUpper, "core:freq_upper_edge"); refusal) {
            return std::unexpected(*refusal);
        }
        json::Value entry = json::Value::makeObject();
        entry.append("core:sample_start", json::Value::fromUnsigned(annotation.sampleStart));
        if (annotation.sampleCount) {
            entry.append("core:sample_count", json::Value::fromUnsigned(*annotation.sampleCount));
        }
        if (annotation.freqLower) {
            entry.append("core:freq_lower_edge", json::Value::fromDouble(*annotation.freqLower));
        }
        if (annotation.freqUpper) {
            entry.append("core:freq_upper_edge", json::Value::fromDouble(*annotation.freqUpper));
        }
        appendString(entry, "core:label", annotation.label);
        appendString(entry, "core:comment", annotation.comment);
        appendString(entry, "core:generator", annotation.generator);
        appendString(entry, "core:uuid", annotation.uuid);
        detail::appendSorted(entry, annotation.extra);
        annotations.push(std::move(entry));
    }

    json::Value document = json::Value::makeObject();
    document.append("global", std::move(global));
    document.append("captures", std::move(captures));
    document.append("annotations", std::move(annotations));
    return document;
}

/// Serialize `metadata`. The output is a pure function of the value: equal values give equal bytes.
[[nodiscard]] inline std::expected<std::string, Error> write(const Metadata& metadata) {
    auto document = toJson(metadata);
    if (!document) {
        return std::unexpected(document.error());
    }
    return json::write(*document);
}

namespace detail {

/// Map a JSON refusal onto the source's own vocabulary; the rest become `meta_unparsable`.
[[nodiscard]] inline Error fromParseError(const json::ParseError& error) {
    const bool        named = error.message == "duplicate_key" || error.message == "nesting_too_deep" || error.message == "bad_escape" || error.message == "index_overflow";
    const std::string where = error.detail.empty() ? std::format("at line {}, column {}", error.line, error.column) : std::format("at line {}, column {} ('{}')", error.line, error.column, error.detail);
    if (named) {
        return Error{error.message, where};
    }
    return Error{"meta_unparsable", std::format("{} {}", error.message, where)};
}

[[nodiscard]] inline std::expected<Capture, Error> captureFromJson(const json::Value& object, ParseCounters& counters) {
    if (!object.isObject()) {
        return std::unexpected(Error{"meta_unparsable", "a 'captures' entry is not an object"});
    }
    Capture            capture{};
    const json::Value* sampleStart = object.find("core:sample_start");
    if (sampleStart == nullptr) {
        return std::unexpected(Error{"meta_unparsable", "a 'captures' entry has no 'core:sample_start'"});
    }
    auto start = unsignedField(*sampleStart, "core:sample_start", counters);
    if (!start) {
        return std::unexpected(start.error());
    }
    capture.sampleStart = *start;

    if (const json::Value* member = object.find("core:global_index"); member != nullptr) {
        auto parsed = unsignedField(*member, "core:global_index", counters);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        capture.globalIndex = *parsed;
    }
    if (const json::Value* member = object.find("core:header_bytes"); member != nullptr) {
        auto parsed = unsignedField(*member, "core:header_bytes", counters);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        capture.headerBytes = *parsed;
    }
    if (const json::Value* member = object.find("core:frequency"); member != nullptr) {
        auto parsed = doubleField(*member, "core:frequency");
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        capture.frequency = *parsed;
    }
    if (const json::Value* member = object.find("core:datetime"); member != nullptr) {
        auto parsed = stringField(*member, "core:datetime");
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        capture.datetime = *parsed;
    }
    capture.extra = carriedMembers(object, kCaptureKnownKeys);
    return capture;
}

[[nodiscard]] inline std::expected<Annotation, Error> annotationFromJson(const json::Value& object, ParseCounters& counters) {
    if (!object.isObject()) {
        return std::unexpected(Error{"meta_unparsable", "an 'annotations' entry is not an object"});
    }
    Annotation         annotation{};
    const json::Value* sampleStart = object.find("core:sample_start");
    if (sampleStart == nullptr) {
        return std::unexpected(Error{"meta_unparsable", "an 'annotations' entry has no 'core:sample_start'"});
    }
    auto start = unsignedField(*sampleStart, "core:sample_start", counters);
    if (!start) {
        return std::unexpected(start.error());
    }
    annotation.sampleStart = *start;

    if (const json::Value* member = object.find("core:sample_count"); member != nullptr) {
        auto parsed = unsignedField(*member, "core:sample_count", counters);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        annotation.sampleCount = *parsed;
    }
    const auto readDouble = [&object](std::string_view key, std::optional<double>& slot) -> std::optional<Error> {
        if (const json::Value* member = object.find(key); member != nullptr) {
            auto parsed = doubleField(*member, key);
            if (!parsed) {
                return parsed.error();
            }
            slot = *parsed;
        }
        return std::nullopt;
    };
    if (auto refusal = readDouble("core:freq_lower_edge", annotation.freqLower); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readDouble("core:freq_upper_edge", annotation.freqUpper); refusal) {
        return std::unexpected(*refusal);
    }
    const auto readString = [&object](std::string_view key, std::optional<std::string>& slot) -> std::optional<Error> {
        if (const json::Value* member = object.find(key); member != nullptr) {
            auto parsed = stringField(*member, key);
            if (!parsed) {
                return parsed.error();
            }
            slot = *parsed;
        }
        return std::nullopt;
    };
    if (auto refusal = readString("core:label", annotation.label); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:comment", annotation.comment); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:generator", annotation.generator); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:uuid", annotation.uuid); refusal) {
        return std::unexpected(*refusal);
    }
    annotation.extra = carriedMembers(object, kAnnotationKnownKeys);
    return annotation;
}

[[nodiscard]] inline std::expected<Global, Error> globalFromJson(const json::Value& object, ParseCounters& counters) {
    Global global{};
    if (const json::Value* member = object.find("core:datatype"); member != nullptr) {
        auto parsed = stringField(*member, "core:datatype");
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        global.datatype = *parsed;
    }
    if (const json::Value* member = object.find("core:version"); member != nullptr) {
        auto parsed = stringField(*member, "core:version");
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        global.version = *parsed;
    }
    if (const json::Value* member = object.find("core:sample_rate"); member != nullptr) {
        auto parsed = doubleField(*member, "core:sample_rate");
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        global.sampleRate = *parsed;
    }
    const auto readUnsigned = [&object, &counters](std::string_view key, std::optional<std::uint64_t>& slot) -> std::optional<Error> {
        if (const json::Value* member = object.find(key); member != nullptr) {
            auto parsed = unsignedField(*member, key, counters);
            if (!parsed) {
                return parsed.error();
            }
            slot = *parsed;
        }
        return std::nullopt;
    };
    if (auto refusal = readUnsigned("core:num_channels", global.numChannels); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readUnsigned("core:offset", global.offset); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readUnsigned("core:trailing_bytes", global.trailingBytes); refusal) {
        return std::unexpected(*refusal);
    }
    const auto readString = [&object](std::string_view key, std::optional<std::string>& slot) -> std::optional<Error> {
        if (const json::Value* member = object.find(key); member != nullptr) {
            auto parsed = stringField(*member, key);
            if (!parsed) {
                return parsed.error();
            }
            slot = *parsed;
        }
        return std::nullopt;
    };
    if (auto refusal = readString("core:description", global.description); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:author", global.author); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:recorder", global.recorder); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:hw", global.hardware); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:license", global.license); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:sha512", global.sha512); refusal) {
        return std::unexpected(*refusal);
    }
    if (auto refusal = readString("core:dataset", global.dataset); refusal) {
        return std::unexpected(*refusal);
    }
    if (const json::Value* member = object.find("core:metadata_only"); member != nullptr) {
        auto parsed = boolField(*member, "core:metadata_only");
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        global.metadataOnly = *parsed;
    }
    global.extra = carriedMembers(object, kGlobalKnownKeys);
    return global;
}

} // namespace detail

/**
 * @brief Parse a SigMF metadata document, bounding every count before anything is sized from it.
 *
 * An array that arrived unsorted within `limits.maxUnsortedEntries` is sorted, counted and reported;
 * a larger unsorted array is refused, because thousands of out-of-order machine-generated entries
 * mean the producer is malfunctioning and sorting them would launder its output.
 */
[[nodiscard]] inline std::expected<Metadata, Error> parse(std::string_view text, const Limits& limits, ParseCounters& counters) {
    auto document = json::parse(text);
    if (!document) {
        return std::unexpected(detail::fromParseError(document.error()));
    }
    if (!document->isObject()) {
        return std::unexpected(Error{"meta_not_object", "the top-level value is not an object"});
    }
    const json::Value* globalObject = document->find("global");
    if (globalObject == nullptr || !globalObject->isObject()) {
        return std::unexpected(Error{"missing_global", "the document has no 'global' object"});
    }
    const json::Value* capturesArray = document->find("captures");
    if (capturesArray == nullptr || !capturesArray->isArray()) {
        return std::unexpected(Error{"missing_captures", "the document has no 'captures' array"});
    }
    const json::Value* annotationsArray = document->find("annotations");
    if (annotationsArray != nullptr && !annotationsArray->isArray()) {
        return std::unexpected(Error{"meta_unparsable", "'annotations' is not an array"});
    }

    if (capturesArray->size() > limits.maxCaptures) {
        return std::unexpected(Error{"too_many_captures", std::format("{} capture segments exceed the bound of {}", capturesArray->size(), limits.maxCaptures)});
    }
    const std::size_t annotationCount = annotationsArray == nullptr ? 0UZ : annotationsArray->size();
    if (annotationCount > limits.maxAnnotations) {
        return std::unexpected(Error{"too_many_annotations", std::format("{} annotations exceed the bound of {}", annotationCount, limits.maxAnnotations)});
    }

    Metadata metadata{};
    auto     global = detail::globalFromJson(*globalObject, counters);
    if (!global) {
        return std::unexpected(global.error());
    }
    metadata.global = std::move(*global);

    metadata.captures.reserve(capturesArray->size());
    for (std::size_t i = 0UZ; i < capturesArray->size(); ++i) {
        auto capture = detail::captureFromJson(capturesArray->at(i), counters);
        if (!capture) {
            return std::unexpected(capture.error());
        }
        metadata.captures.push_back(std::move(*capture));
    }
    metadata.annotations.reserve(annotationCount);
    for (std::size_t i = 0UZ; i < annotationCount; ++i) {
        auto annotation = detail::annotationFromJson(annotationsArray->at(i), counters);
        if (!annotation) {
            return std::unexpected(annotation.error());
        }
        metadata.annotations.push_back(std::move(*annotation));
    }

    const auto captureLess    = [](const Capture& lhs, const Capture& rhs) { return lhs.sampleStart < rhs.sampleStart; };
    const auto annotationLess = [](const Annotation& lhs, const Annotation& rhs) { return std::pair{lhs.sampleStart, lhs.sampleCount.value_or(0U)} < std::pair{rhs.sampleStart, rhs.sampleCount.value_or(0U)}; };

    if (!std::ranges::is_sorted(metadata.captures, captureLess)) {
        if (metadata.captures.size() > limits.maxUnsortedEntries) {
            return std::unexpected(Error{"captures_unsorted", std::format("{} unsorted capture segments exceed the repair bound of {}", metadata.captures.size(), limits.maxUnsortedEntries)});
        }
        std::ranges::stable_sort(metadata.captures, captureLess);
        counters.nCapturesSorted = 1U;
    }
    if (!std::ranges::is_sorted(metadata.annotations, annotationLess)) {
        if (metadata.annotations.size() > limits.maxUnsortedEntries) {
            return std::unexpected(Error{"annotations_unsorted", std::format("{} unsorted annotations exceed the repair bound of {}", metadata.annotations.size(), limits.maxUnsortedEntries)});
        }
        std::ranges::stable_sort(metadata.annotations, annotationLess);
        counters.nAnnotationsSorted = 1U;
    }

    if (metadata.captures.empty()) {
        // An empty `captures` array is not an error: the specification gives it a meaning, which is
        // one capture segment beginning at sample zero. Materializing that segment here is what
        // lets every step downstream assume the array holds at least one.
        metadata.captures.emplace_back();
    }
    return metadata;
}

/**
 * @brief The document-level checks a recording must pass before a source streams it.
 *
 * Every failure is a named refusal: a recording this tree cannot read faithfully is one it refuses
 * to read at all. `parse` has already materialized the segment an empty `captures` array implies,
 * so the array is non-empty by the time this runs. The bound of every index against `core:offset`
 * is a SHOULD in the specification and a refusal here, which is this tree's own strictness rather
 * than a conformance check.
 */
[[nodiscard]] inline std::optional<Error> validateForStreaming(const Metadata& metadata) {
    if (auto refusal = checkVersion(metadata.global.version); refusal) {
        return refusal;
    }
    if (metadata.global.metadataOnly.value_or(false)) {
        return Error{"metadata_only", "'core:metadata_only' is true, so there is no dataset to stream"};
    }
    if (metadata.global.numChannels.value_or(1U) == 0U) {
        return Error{"num_channels_zero", "'core:num_channels' is zero, which makes the item stride zero"};
    }

    const std::uint64_t offset = metadata.global.offset.value_or(0U);
    for (const Capture& capture : metadata.captures) {
        if (capture.sampleStart < offset) {
            return Error{"index_below_offset", std::format("a capture segment starts at {}, below 'core:offset' {}", capture.sampleStart, offset)};
        }
        if (capture.headerBytes.value_or(0U) != 0U) {
            return Error{"header_bytes_unsupported", std::format("a capture segment states 'core:header_bytes' {}", *capture.headerBytes)};
        }
    }
    for (const Annotation& annotation : metadata.annotations) {
        if (annotation.sampleStart < offset) {
            return Error{"index_below_offset", std::format("an annotation starts at {}, below 'core:offset' {}", annotation.sampleStart, offset)};
        }
    }
    for (std::size_t i = 1UZ; i < metadata.captures.size(); ++i) {
        const Capture& previous = metadata.captures[i - 1UZ];
        const Capture& current  = metadata.captures[i];
        if (!previous.globalIndex || !current.globalIndex) {
            continue; // a recording that does not state the counter is not asserting continuity
        }
        const std::uint64_t expected = *previous.globalIndex + (current.sampleStart - previous.sampleStart);
        if (*current.globalIndex < expected) {
            return Error{"global_index_regression", std::format("'core:global_index' {} at sample {} is below the expected {}", *current.globalIndex, current.sampleStart, expected)};
        }
    }
    return std::nullopt;
}

/**
 * @brief The number of samples the hardware lost between two consecutive capture segments.
 * @return the drop count, or nothing when either segment does not state `core:global_index`.
 */
[[nodiscard]] inline std::optional<std::uint64_t> droppedBetween(const Capture& previous, const Capture& current) {
    if (!previous.globalIndex || !current.globalIndex) {
        return std::nullopt;
    }
    const std::uint64_t expected = *previous.globalIndex + (current.sampleStart - previous.sampleStart);
    if (*current.globalIndex < expected) {
        return std::nullopt;
    }
    return *current.globalIndex - expected;
}

} // namespace gr::sigmf

#endif // GNURADIO_ALGORITHM_SIGMF_METADATA_HPP
