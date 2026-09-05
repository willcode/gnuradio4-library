#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/sigmf/Json.hpp>
#include <gnuradio-4.0/algorithm/sigmf/SampleCodec.hpp>
#include <gnuradio-4.0/algorithm/sigmf/SigMfMetadata.hpp>

// SigMF is an interchange format, so the assertions below are byte assertions wherever another program would see
// bytes: the canonical recording's metadata derived by hand and compared in full, the scaling tables compared as
// dataset bytes, every refusal the document layer can express, and the datatype alphabet generated from the grammar
// rather than listed. Nothing here opens a file or instantiates a block — Part I is testable on its own.

namespace {

using namespace boost::ut;
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace sigmf = gr::sigmf;
namespace json  = gr::sigmf::json;

/// Anchor A: eight complex samples at 48 kHz, one capture segment, one annotation.
constexpr std::array<std::complex<float>, 8> kAnchorSamples{
    std::complex<float>{0.0f, 0.0f},   //
    std::complex<float>{1.0f, 0.0f},   //
    std::complex<float>{0.0f, 1.0f},   //
    std::complex<float>{-1.0f, 0.0f},  //
    std::complex<float>{0.0f, -1.0f},  //
    std::complex<float>{0.5f, 0.5f},   //
    std::complex<float>{-0.5f, 0.5f},  //
    std::complex<float>{0.25f, -0.75f} //
};

constexpr std::string_view kAnchorMetadata = R"({
  "global": {
    "core:datatype": "cf32_le",
    "core:version": "1.2.6",
    "core:sample_rate": 48000,
    "core:recorder": "gnuradio4"
  },
  "captures": [
    {
      "core:sample_start": 0,
      "core:frequency": 433921337,
      "core:datetime": "2026-08-26T12:00:00.000000Z"
    }
  ],
  "annotations": [
    {
      "core:sample_start": 4,
      "core:sample_count": 2,
      "core:label": "burst"
    }
  ]
}
)"sv;

[[nodiscard]] sigmf::Metadata anchorMetadata() {
    sigmf::Metadata metadata;
    metadata.global.datatype   = "cf32_le";
    metadata.global.version    = "1.2.6";
    metadata.global.sampleRate = 48000.0;
    metadata.global.recorder   = "gnuradio4";

    sigmf::Capture capture;
    capture.sampleStart = 0U;
    capture.frequency   = 433921337.0;
    capture.datetime    = "2026-08-26T12:00:00.000000Z";
    metadata.captures.push_back(std::move(capture));

    sigmf::Annotation annotation;
    annotation.sampleStart = 4U;
    annotation.sampleCount = 2U;
    annotation.label       = "burst";
    metadata.annotations.push_back(std::move(annotation));
    return metadata;
}

/// The byte offset and length of every line of `text`, for locating a byte-comparison failure.
[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> lineTable(std::string_view text) {
    std::vector<std::pair<std::size_t, std::size_t>> lines;
    std::size_t                                      start = 0UZ;
    for (std::size_t i = 0UZ; i < text.size(); ++i) {
        if (text[i] == '\n') {
            lines.emplace_back(start, i - start + 1UZ);
            start = i + 1UZ;
        }
    }
    return lines;
}

/// Wrap `body` as the `global` object of an otherwise minimal document.
[[nodiscard]] std::string documentWithGlobal(std::string_view body) { return std::format(R"({{"global": {{{}}}, "captures": [{{"core:sample_start": 0}}], "annotations": []}})", body); }

[[nodiscard]] std::string documentWithCaptures(std::string_view captures) { return std::format(R"({{"global": {{"core:datatype": "cf32_le", "core:version": "1.2.0"}}, "captures": {}, "annotations": []}})", captures); }

/// Parse `text` and return the refusal code, or "ok" when it is accepted.
[[nodiscard]] std::string refusalOf(std::string_view text, const sigmf::Limits& limits = sigmf::Limits{}) {
    sigmf::ParseCounters counters;
    auto                 parsed = sigmf::parse(text, limits, counters);
    if (!parsed) {
        return parsed.error().code;
    }
    if (auto refused = sigmf::validateForStreaming(*parsed); refused) {
        return refused->code;
    }
    return "ok";
}

[[nodiscard]] std::string parseRefusal(std::string_view text) {
    auto document = json::parse(text);
    return document ? std::string("ok") : document.error().message;
}

[[nodiscard]] std::string hexOf(std::span<const std::byte> bytes) {
    std::string out;
    for (const std::byte value : bytes) {
        out.append(std::format("{:02X}", std::to_integer<unsigned>(value)));
    }
    return out;
}

const suite<"SigMF metadata document"> _metadata = [] {
    "the golden metadata, byte for byte, and as a parse"_test = [] {
        expect(eq(sigmf::kWrittenVersion, "1.2.6"sv)) << "the writer states the current release, and the anchor carries that string";

        const sigmf::Metadata metadata = anchorMetadata();
        const auto            text     = sigmf::write(metadata);
        expect(text.has_value());
        expect(eq(text->size(), 422UZ)) << "anchor A is 422 bytes";
        expect(eq(*text, std::string(kAnchorMetadata)));

        const auto lines = lineTable(*text);
        expect(eq(lines.size(), 22UZ));
        expect(eq(lines[2].first, 16UZ)) << "\"core:datatype\" begins at byte 16";
        expect(eq(lines[2].second, 32UZ));
        expect(eq(lines[10].first, 198UZ)) << "\"core:frequency\" begins at byte 198";
        expect(eq(lines[10].second, 35UZ));
        expect(eq(lines[11].first, 233UZ)) << "\"core:datetime\" begins at byte 233";
        expect(eq(lines[18].first, 382UZ)) << "\"core:label\" begins at byte 382";

        sigmf::ParseCounters counters;
        auto                 back = sigmf::parse(*text, sigmf::Limits{}, counters);
        expect(back.has_value());
        expect(*back == metadata) << "field for field, and optional for optional";
        expect(!back->global.numChannels.has_value()) << "a default is omitted, and absent is not one";
        expect(!back->global.offset.has_value());

        const auto again = sigmf::write(*back);
        expect(again.has_value());
        expect(eq(*again, *text)) << "write(parse(write(m))) == write(m)";
    };

    "determinism is a property, not a habit"_test = [] {
        sigmf::Metadata forward = anchorMetadata();
        forward.global.extra    = json::Value::makeObject();
        forward.global.extra.append("antenna:gain", json::Value::fromDouble(12.5));
        forward.global.extra.append("aardvark:count", json::Value::fromUnsigned(3U));
        forward.captures.push_back([] {
            sigmf::Capture later;
            later.sampleStart = 4U;
            return later;
        }());

        sigmf::Metadata reversed = anchorMetadata();
        reversed.global.extra    = json::Value::makeObject();
        reversed.global.extra.append("aardvark:count", json::Value::fromUnsigned(3U));
        reversed.global.extra.append("antenna:gain", json::Value::fromDouble(12.5));
        reversed.captures.insert(reversed.captures.begin(), [] {
            sigmf::Capture later;
            later.sampleStart = 4U;
            return later;
        }());
        std::ranges::stable_sort(reversed.captures, [](const auto& lhs, const auto& rhs) { return lhs.sampleStart < rhs.sampleStart; });

        const auto forwardText  = sigmf::write(forward);
        const auto reversedText = sigmf::write(reversed);
        expect(forwardText.has_value() && reversedText.has_value());
        expect(eq(*forwardText, *reversedText)) << "two equal documents serialize to identical bytes";
        expect(forwardText->find("\"aardvark:count\"") < forwardText->find("\"antenna:gain\"")) << "carried keys are written in lexicographic byte order";
    };

    "float emission is shortest round-trip, asserted as exactness"_test = [] {
        for (const double frequency : {100000000.0, 145825000.0, 433920000.0, 433921337.0, 1090000000.0, 2400000001.0}) {
            sigmf::Metadata metadata       = anchorMetadata();
            metadata.captures[0].frequency = frequency;
            const auto text                = sigmf::write(metadata);
            expect(text.has_value());
            sigmf::ParseCounters counters;
            const auto           back = sigmf::parse(*text, sigmf::Limits{}, counters);
            expect(back.has_value());
            expect(back->captures[0].frequency.has_value());
            expect(std::bit_cast<std::uint64_t>(*back->captures[0].frequency) == std::bit_cast<std::uint64_t>(frequency)) << std::format("{} did not survive bit-exactly", frequency);
        }

        expect(eq(json::write(json::Value::fromDouble(433921337.0)), "433921337\n"s)) << "fixed notation is shorter here";
        expect(eq(json::write(json::Value::fromDouble(100000000.0)), "1e+08\n"s)) << "the exponent form is pinned as intended";
        expect(eq(json::write(json::Value::fromDouble(48000.0)), "48000\n"s));
        expect(eq(json::write(json::Value::fromDouble(61440000.0)), "61440000\n"s));
    };

    "the integer rule, all arms"_test = [] {
        sigmf::ParseCounters counters;
        auto                 exact = sigmf::parse(documentWithCaptures(R"([{"core:sample_start": 9007199254740993}])"), sigmf::Limits{}, counters);
        expect(exact.has_value());
        expect(eq(exact->captures[0].sampleStart, std::uint64_t{9007199254740993U})) << "2^53 + 1 is not 2^53";

        expect(eq(refusalOf(documentWithCaptures(R"([{"core:sample_start": 18446744073709551616}])")), "index_overflow"s));
        expect(eq(refusalOf(documentWithCaptures(R"([{"core:sample_start": -1}])")), "negative_index"s));
        expect(eq(refusalOf(documentWithCaptures(R"([{"core:sample_start": 4096.5}])")), "non_integral_index"s));
        expect(eq(refusalOf(documentWithCaptures(R"([{"core:sample_start": 4.096e3}])")), "non_integral_index"s));

        for (const std::string_view token : {"4096.0"sv, "4096.00"sv}) {
            sigmf::ParseCounters tolerated;
            const auto           parsed = sigmf::parse(documentWithCaptures(std::format(R"([{{"core:sample_start": {}}}])", token)), sigmf::Limits{}, tolerated);
            expect(parsed.has_value());
            expect(eq(parsed->captures[0].sampleStart, std::uint64_t{4096U}));
            expect(eq(tolerated.nIntegralFloatIndices, std::uint64_t{1U})) << "every tolerance leaves a count behind";
        }

        sigmf::ParseCounters wide;
        const auto           large = sigmf::parse(documentWithCaptures(R"([{"core:sample_start": 9007199254740993.0}])"), sigmf::Limits{}, wide);
        expect(large.has_value());
        expect(eq(large->captures[0].sampleStart, std::uint64_t{9007199254740993U})) << "the text path never touched a double";
        expect(eq(wide.nIntegralFloatIndices, std::uint64_t{1U}));
    };

    "every refusal the document layer can express"_test = [] {
        expect(eq(refusalOf("{"), "meta_unparsable"s));
        expect(eq(refusalOf("[1, 2]"), "meta_not_object"s));
        expect(eq(refusalOf(R"({"captures": [], "annotations": []})"), "missing_global"s));
        expect(eq(refusalOf(R"({"global": {}, "annotations": []})"), "missing_captures"s));
        expect(eq(refusalOf(documentWithGlobal(R"("core:datatype": "cf32_le")")), "missing_version"s));
        expect(eq(refusalOf(documentWithGlobal(R"("core:version": "2.0.0")")), "future_version"s));
        expect(eq(refusalOf(documentWithGlobal(R"("core:version": "0.0.2")")), "legacy_version"s));
        expect(eq(refusalOf(R"({"global": {"core:version": "1.2.0", "core:version": "1.2.0"}, "captures": [], "annotations": []})"), "duplicate_key"s));
        expect(eq(refusalOf(documentWithGlobal(R"("core:version": "1.2.0", "core:metadata_only": true)")), "metadata_only"s));
        expect(eq(refusalOf(R"({"global": {"core:version": "1.2.0", "core:num_channels": 0}, "captures": [{"core:sample_start": 0}], "annotations": []})"), "num_channels_zero"s));
        expect(eq(refusalOf(R"({"global": {"core:version": "1.2.0", "core:offset": 16}, "captures": [{"core:sample_start": 8}], "annotations": []})"), "index_below_offset"s));
        expect(eq(refusalOf(R"({"global": {"core:version": "1.2.0"}, "captures": [{"core:sample_start": 0, "core:header_bytes": 4}], "annotations": []})"), "header_bytes_unsupported"s));
        expect(eq(refusalOf(R"({"global": {"core:version": "1.2.0"}, "captures": [{"core:sample_start": 0, "core:global_index": 1000}, {"core:sample_start": 4096, "core:global_index": 5000}], "annotations": []})"), "global_index_regression"s));

        const sigmf::Limits tight{4U, 4U, sigmf::kMaxUnsortedEntries};
        expect(eq(refusalOf(R"({"global": {"core:version": "1.2.0"}, "captures": [{"core:sample_start": 0},{"core:sample_start": 1},{"core:sample_start": 2},{"core:sample_start": 3},{"core:sample_start": 4}], "annotations": []})", tight), "too_many_captures"s));
        expect(eq(refusalOf(R"({"global": {"core:version": "1.2.0"}, "captures": [{"core:sample_start": 0}], "annotations": [{"core:sample_start": 0},{"core:sample_start": 1},{"core:sample_start": 2},{"core:sample_start": 3},{"core:sample_start": 4}]})", tight), "too_many_annotations"s));
    };

    "an empty captures array carries its implied segment"_test = [] {
        constexpr std::string_view kEmpty = R"({"global": {"core:datatype": "cf32_le", "core:version": "1.2.6"}, "captures": [], "annotations": []})";

        sigmf::ParseCounters counters;
        const auto           parsed = sigmf::parse(kEmpty, sigmf::Limits{}, counters);
        expect(parsed.has_value());
        expect(eq(parsed->captures.size(), 1UZ)) << "an empty array means one segment, not none";
        expect(eq(parsed->captures[0].sampleStart, std::uint64_t{0U}));
        expect(!parsed->captures[0].datetime.has_value()) << "the implied segment states nothing the document did not";
        expect(eq(refusalOf(kEmpty), "ok"s)) << "and it streams rather than refusing";
    };

    "unsorted arrays are repaired below the bound and refused above it"_test = [] {
        const auto unsortedCaptures = [](std::size_t count) {
            std::string body = "[";
            for (std::size_t i = 0UZ; i < count; ++i) {
                body.append(std::format(R"({}{{"core:sample_start": {}}})", i == 0UZ ? "" : ", ", count - i));
            }
            body.append("]");
            return documentWithCaptures(body);
        };

        sigmf::ParseCounters repaired;
        auto                 sorted = sigmf::parse(unsortedCaptures(sigmf::kMaxUnsortedEntries), sigmf::Limits{}, repaired);
        expect(sorted.has_value());
        expect(eq(repaired.nCapturesSorted, std::uint64_t{1U}));
        expect(std::ranges::is_sorted(sorted->captures, [](const auto& lhs, const auto& rhs) { return lhs.sampleStart < rhs.sampleStart; }));

        expect(eq(refusalOf(unsortedCaptures(sigmf::kMaxUnsortedEntries + 1UZ)), "captures_unsorted"s)) << "a producer emitting thousands out of order is malfunctioning, not sloppy";

        std::string ascending = "[";
        for (std::size_t i = 0UZ; i < sigmf::kMaxUnsortedEntries + 64UZ; ++i) {
            ascending.append(std::format(R"({}{{"core:sample_start": {}}})", i == 0UZ ? "" : ", ", i));
        }
        ascending.append("]");
        sigmf::ParseCounters untouched;
        expect(sigmf::parse(documentWithCaptures(ascending), sigmf::Limits{}, untouched).has_value());
        expect(eq(untouched.nCapturesSorted, std::uint64_t{0U})) << "a sorted array of any size passes untouched";
    };

    "unsorted annotations are repaired at their own boundary"_test = [] {
        const auto unsortedAnnotations = [](std::size_t count) {
            std::string body = R"({"global": {"core:version": "1.2.0"}, "captures": [{"core:sample_start": 0}], "annotations": [)";
            for (std::size_t i = 0UZ; i < count; ++i) {
                body.append(std::format(R"({}{{"core:sample_start": {}}})", i == 0UZ ? "" : ", ", count - i));
            }
            body.append("]}");
            return body;
        };
        sigmf::ParseCounters repaired;
        auto                 sorted = sigmf::parse(unsortedAnnotations(sigmf::kMaxUnsortedEntries), sigmf::Limits{}, repaired);
        expect(sorted.has_value());
        expect(eq(repaired.nAnnotationsSorted, std::uint64_t{1U}));
        expect(eq(refusalOf(unsortedAnnotations(sigmf::kMaxUnsortedEntries + 1UZ)), "annotations_unsorted"s));
    };
};

const suite<"SigMF JSON layer"> _json = [] {
    "the parser refuses what it says it does"_test = [] {
        expect(eq(parseRefusal(R"({"a": 1,})"), "trailing_comma"s));
        expect(eq(parseRefusal(R"({"a": /* note */ 1})"), "comment_not_allowed"s));
        expect(eq(parseRefusal(R"({a: 1})"), "unquoted_key"s));
        expect(eq(parseRefusal("{'a': 1}"), "single_quote"s));
        expect(eq(parseRefusal(R"({"a": +1})"), "leading_plus"s));
        expect(eq(parseRefusal(R"({"a": NaN})"), "bare_literal"s));
        expect(eq(parseRefusal(R"({"a": 0x10})"), "hex_number"s));
        expect(eq(parseRefusal(R"({"a": 01})"), "leading_zero"s));
        expect(eq(parseRefusal(R"({"a": 1, "a": 2})"), "duplicate_key"s));
        expect(eq(parseRefusal(R"({"a": "x)"), "unterminated_string"s));
        expect(eq(parseRefusal(R"({"a": 1} {})"), "trailing_content"s));
        expect(eq(parseRefusal("{\"a\": \"\xFF\"}"), "invalid_utf8"s));
        expect(eq(parseRefusal(R"({"a": "\uD83D"})"), "bad_escape"s));

        std::string deep(json::kMaxDepth + 1UZ, '[');
        deep.append("1");
        deep.append(json::kMaxDepth + 1UZ, ']');
        expect(eq(parseRefusal(deep), "nesting_too_deep"s));

        std::string atLimit(json::kMaxDepth, '[');
        atLimit.append("1");
        atLimit.append(json::kMaxDepth, ']');
        expect(eq(parseRefusal(atLimit), "ok"s)) << "sixty-four levels are legal";
    };

    "a refusal names the offending byte"_test = [] {
        const auto document = json::parse("{\n  \"a\": 1,\n  \"a\": 2\n}");
        expect(!document.has_value());
        expect(eq(document.error().message, "duplicate_key"s));
        expect(eq(document.error().detail, "a"s)) << "the refusal names the key";
        expect(eq(document.error().line, 3UZ));
        expect(eq(document.error().column, 3UZ));
    };

    "non-ASCII text survives, and an escape resolves to UTF-8"_test = [] {
        const auto document = json::parse(R"({"core:author": "René Maßmann", "core:license": "Björn"})");
        expect(document.has_value());
        const json::Value* author = document->find("core:author");
        expect(author != nullptr);
        expect(eq(author->str(), "René Maßmann"s)) << "the escape is resolved to UTF-8";
        expect(eq(json::write(*document), "{\n  \"core:author\": \"René Maßmann\",\n  \"core:license\": \"Björn\"\n}\n"s)) << "and is never re-escaped";
    };

    "verbatim carriage round-trips"_test = [] {
        constexpr std::string_view kCarried = R"({
  "global": {
    "core:datatype": "cf32_le",
    "core:version": "1.2.0",
    "antenna:gain": 12.5,
    "capture_details:source_file": "a.bin",
    "core:extensions": [
      {
        "name": "antenna",
        "version": "1.0.0",
        "optional": false
      }
    ],
    "core:geolocation": {
      "type": "Point",
      "coordinates": [
        -71.5,
        42.3
      ]
    },
    "vendor:nested": {
      "one": {
        "two": {
          "three": 3
        }
      }
    }
  },
  "captures": [
    {
      "core:sample_start": 0
    }
  ],
  "annotations": []
}
)"sv;

        sigmf::ParseCounters counters;
        const auto           parsed = sigmf::parse(kCarried, sigmf::Limits{}, counters);
        expect(parsed.has_value());
        expect(eq(parsed->global.extra.size(), 5UZ)) << "every unconsumed member is carried, and nothing else";
        expect(parsed->global.extra.contains("core:extensions")) << "the extensions array travels with the fields it declares";
        expect(parsed->global.extra.contains("core:geolocation"));

        const auto rewritten = sigmf::write(*parsed);
        expect(rewritten.has_value());
        expect(eq(*rewritten, std::string(kCarried))) << "the carried members are re-emitted byte-identically";

        const std::string extras = json::write(parsed->global.extra);
        const auto        again  = json::parse(extras);
        expect(again.has_value());
        expect(*again == parsed->global.extra) << "a carriage document read back by the same parser is exact";
    };
};

const suite<"SigMF sample codec"> _codec = [] {
    "the datatype alphabet, counted and exhaustive"_test = [] {
        std::vector<std::string> spellings;
        for (const std::string_view domain : {"r"sv, "c"sv}) {
            for (const std::string_view base : {"f32"sv, "f64"sv, "i8"sv, "i16"sv, "i32"sv, "u8"sv, "u16"sv, "u32"sv}) {
                if (base == "i8" || base == "u8") {
                    spellings.push_back(std::format("{}{}", domain, base));
                    continue;
                }
                spellings.push_back(std::format("{}{}_le", domain, base));
                spellings.push_back(std::format("{}{}_be", domain, base));
            }
        }
        expect(eq(spellings.size(), 28UZ)) << "two domains, eight base types, an endianness suffix above eight bits";

        std::size_t accepted   = 0UZ;
        std::size_t writeLegal = 0UZ;
        for (const std::string& spelling : spellings) {
            const auto parsed = sigmf::parseDatatype(spelling);
            expect(parsed.has_value()) << spelling << " is in the grammar and is read";
            if (parsed) {
                ++accepted;
                if (sigmf::isWriteLegal(*parsed)) {
                    ++writeLegal;
                    expect(eq(sigmf::spellDatatype(*parsed), spelling)) << spelling;
                }
            }
        }
        expect(eq(accepted, 28UZ)) << "the whole spelled grammar is read, with nothing left refused";
        expect(eq(writeLegal, 16UZ));

        expect(eq(sigmf::canonicalDatatypeFor<std::uint8_t>(), "ru8"sv));
        expect(eq(sigmf::canonicalDatatypeFor<std::int16_t>(), "ri16_le"sv));
        expect(eq(sigmf::canonicalDatatypeFor<std::int32_t>(), "ri32_le"sv));
        expect(eq(sigmf::canonicalDatatypeFor<float>(), "rf32_le"sv));
        expect(eq(sigmf::canonicalDatatypeFor<std::complex<float>>(), "cf32_le"sv));

        // The published schema's validation pattern accepts an endianness suffix on an 8-bit spelling
        // although its own ABNF derives none. A suffix that cannot reorder a single byte is read as
        // the no-op it is, and is dropped on the way back out.
        for (const std::string_view tolerated : {"ri8_le"sv, "cu8_be"sv}) {
            const auto parsed = sigmf::parseDatatype(tolerated);
            expect(parsed.has_value()) << tolerated;
            expect(eq(parsed->bytesPerComponent(), 1UZ)) << tolerated;
            expect(eq(sigmf::spellDatatype(*parsed), std::string(tolerated.substr(0UZ, 3UZ)))) << tolerated;
        }

        // The 64-bit integers and the 16-bit float are not productions of the dataset-format grammar,
        // so they are refused for the reason any unknown spelling is, and not by a name of their own.
        for (const std::string_view outside : {"ri64_le"sv, "ri64_be"sv, "ci64_le"sv, "cu64_le"sv, "ru64_be"sv, "cu64_be"sv, "rf16_le"sv, "cf16_be"sv, "rf16"sv}) {
            const auto parsed = sigmf::parseDatatype(outside);
            expect(!parsed.has_value()) << outside;
            expect(eq(parsed.error(), "datatype_unparsable"s)) << outside;
        }

        for (const std::string_view malformed : {"cf8"sv, "cf32"sv, "ci16"sv, "cx32_le"sv, "cf32_me"sv, "rf128_le"sv, "f32_le"sv, ""sv}) {
            const auto parsed = sigmf::parseDatatype(malformed);
            expect(!parsed.has_value()) << malformed;
            expect(eq(parsed.error(), "datatype_unparsable"s)) << malformed;
        }
    };

    "the scaling tables, exactly"_test = [] {
        const auto toBytes = [](std::string_view spelling, std::span<const std::complex<float>> samples, std::uint64_t& clipped) {
            const auto datatype = sigmf::parseDatatype(spelling);
            expect(datatype.has_value());
            const auto encode = sigmf::selectEncoder<float>(*datatype, sigmf::Scaling::Unit);
            expect(encode.has_value());
            std::vector<std::byte> out(samples.size() * datatype->bytesPerSample());
            clipped = (*encode)(reinterpret_cast<const float*>(samples.data()), out.data(), samples.size() * 2UZ);
            return out;
        };

        std::uint64_t clippedI16 = 0U;
        const auto    bytesI16   = toBytes("ci16_le", kAnchorSamples, clippedI16);
        expect(eq(bytesI16.size(), 32UZ));
        expect(eq(hexOf(bytesI16), "00000000FF7F00000000FF7F00800000000000800040004000C00040002000A0"s));
        expect(eq(clippedI16, std::uint64_t{2U})) << "+1.0 is not representable at either width";

        std::uint64_t clippedU8 = 0U;
        const auto    bytesU8   = toBytes("cu8", kAnchorSamples, clippedU8);
        expect(eq(bytesU8.size(), 16UZ));
        expect(eq(hexOf(bytesU8), "8080FF8080FF00808000C0C040C0A020"s));
        expect(eq(clippedU8, std::uint64_t{2U}));

        const auto decodeBack = [](std::string_view spelling, std::span<const std::byte> bytes, std::size_t sampleCount) {
            const auto datatype = sigmf::parseDatatype(spelling);
            expect(datatype.has_value());
            const auto decode = sigmf::selectDecoder<float>(*datatype, sigmf::Scaling::Unit);
            expect(decode.has_value());
            std::vector<std::complex<float>> out(sampleCount);
            std::ignore = (*decode)(bytes.data(), reinterpret_cast<float*>(out.data()), sampleCount * 2UZ);
            return out;
        };

        const auto fromI16 = decodeBack("ci16_le", bytesI16, kAnchorSamples.size());
        expect(eq(fromI16[3].real(), -1.0f)) << "-1.0 returns exactly";
        expect(eq(fromI16[1].real(), 0.999969482421875f)) << "+1.0 returns 32767/32768";
        expect(eq(fromI16[5].real(), 0.5f));
        expect(eq(fromI16[7].imag(), -0.75f));

        const auto fromU8 = decodeBack("cu8", bytesU8, kAnchorSamples.size());
        expect(eq(fromU8[1].real(), 0.9921875f)) << "+1.0 returns 127/128";
        expect(eq(fromU8[3].real(), -1.0f));
        expect(eq(fromU8[7].real(), 0.25f));
        expect(eq(fromU8[7].imag(), -0.75f));

        // round half away from zero, pinned so a later change to std::lrint breaks the test rather than the data
        const auto               datatype = sigmf::parseDatatype("ri16_le");
        const auto               encode   = sigmf::selectEncoder<float>(*datatype, sigmf::Scaling::Unit);
        std::array<float, 2>     half{0.5f / 32768.0f, -0.5f / 32768.0f};
        std::array<std::byte, 4> encoded{};
        std::ignore = (*encode)(half.data(), encoded.data(), half.size());
        std::array<std::int16_t, 2> values{};
        std::memcpy(values.data(), encoded.data(), encoded.size());
        expect(eq(values[0], std::int16_t{1}));
        expect(eq(values[1], std::int16_t{-1}));
    };

    "the identity pairs move bytes and nothing else"_test = [] {
        const auto datatype = sigmf::parseDatatype("cf32_le");
        expect(datatype.has_value());
        const auto decode = sigmf::selectDecoder<float>(*datatype, sigmf::Scaling::Unit);
        expect(decode.has_value());

        std::array<std::byte, 64> raw{};
        std::memcpy(raw.data(), kAnchorSamples.data(), raw.size());
        std::vector<std::complex<float>> out(kAnchorSamples.size());
        const std::uint64_t              clipped = (*decode)(raw.data(), reinterpret_cast<float*>(out.data()), out.size() * 2UZ);
        expect(eq(clipped, std::uint64_t{0U}));
        expect(std::ranges::equal(out, kAnchorSamples)) << "cf32_le into complex<float> does no arithmetic at all";
    };

    "a big-endian dataset is one swap in the same pass"_test = [] {
        const auto little = sigmf::parseDatatype("ri16_le");
        const auto big    = sigmf::parseDatatype("ri16_be");
        expect(little.has_value() && big.has_value());
        const auto decodeLittle = sigmf::selectDecoder<float>(*little, sigmf::Scaling::Unit);
        const auto decodeBig    = sigmf::selectDecoder<float>(*big, sigmf::Scaling::Unit);
        expect(decodeLittle.has_value() && decodeBig.has_value());

        const std::array<std::byte, 4> littleBytes{std::byte{0x00}, std::byte{0x40}, std::byte{0x00}, std::byte{0xC0}};
        const std::array<std::byte, 4> bigBytes{std::byte{0x40}, std::byte{0x00}, std::byte{0xC0}, std::byte{0x00}};
        std::array<float, 2>           fromLittle{};
        std::array<float, 2>           fromBig{};
        std::ignore = (*decodeLittle)(littleBytes.data(), fromLittle.data(), fromLittle.size());
        std::ignore = (*decodeBig)(bigBytes.data(), fromBig.data(), fromBig.size());
        expect(eq(fromLittle[0], 0.5f));
        expect(eq(fromLittle[1], -0.5f));
        expect(std::ranges::equal(fromLittle, fromBig)) << "read accepts both byte orders";
    };

    "an integer pair of another width or signedness is refused by name"_test = [] {
        const auto sixteen = sigmf::parseDatatype("ri16_le");
        expect(sixteen.has_value());
        const auto wrongWidth = sigmf::selectDecoder<std::int32_t>(*sixteen, sigmf::Scaling::Unit);
        expect(!wrongWidth.has_value());
        expect(eq(wrongWidth.error(), "integer_width"s));

        const auto unsignedEight = sigmf::parseDatatype("ru8");
        expect(unsignedEight.has_value());
        expect(sigmf::selectDecoder<std::uint8_t>(*unsignedEight, sigmf::Scaling::Unit).has_value());
        expect(!sigmf::selectDecoder<std::int16_t>(*unsignedEight, sigmf::Scaling::Unit).has_value());
    };

    "the raw scaling passes the container's own numbers"_test = [] {
        const auto datatype = sigmf::parseDatatype("ri16_le");
        expect(datatype.has_value());
        const auto decode = sigmf::selectDecoder<float>(*datatype, sigmf::Scaling::Raw);
        expect(decode.has_value());
        const std::array<std::byte, 4> bytes{std::byte{0x00}, std::byte{0x40}, std::byte{0x00}, std::byte{0xC0}};
        std::array<float, 2>           out{};
        std::ignore = (*decode)(bytes.data(), out.data(), out.size());
        expect(eq(out[0], 16384.0f));
        expect(eq(out[1], -16384.0f));
    };
};

const suite<"SigMF derived quantities"> _derived = [] {
    "the drop arithmetic, worked"_test = [] {
        sigmf::Capture first;
        first.sampleStart = 0U;
        first.globalIndex = 1000U;
        sigmf::Capture second;
        second.sampleStart = 4096U;
        second.globalIndex = 5300U;
        expect(sigmf::droppedBetween(first, second).value_or(0U) == 204U);

        second.globalIndex = 5096U;
        expect(sigmf::droppedBetween(first, second).value_or(1U) == 0U) << "contiguous segments derive nothing";

        second.globalIndex.reset();
        expect(!sigmf::droppedBetween(first, second).has_value()) << "a recording that does not state the counter is not asserting continuity";
    };

    "the timestamp profile, both directions"_test = [] {
        expect(sigmf::parseDatetimeNs("2026-08-26T12:00:00.000000Z").value_or(0U) == 1787745600000000000ULL);
        expect(eq(sigmf::formatDatetimeNs(1787745600000000000ULL), "2026-08-26T12:00:00.000000Z"s));
        expect(sigmf::parseDatetimeNs("1970-01-01T00:00:00Z").value_or(1U) == 0U) << "the fractional part is optional on read";
        expect(sigmf::parseDatetimeNs("2026-08-26T12:00:00.123456789Z").value_or(0U) == 1787745600123456789ULL);
        expect(sigmf::parseDatetimeNs("2026-08-26T12:00:00.123456789012Z").value_or(0U) == 1787745600123456789ULL) << "the ABNF bounds the fractional part at one digit and not above; a surplus truncates to nanoseconds";
        expect(sigmf::parseDatetimeNs("2026-08-26T12:00:00.5Z").value_or(0U) == 1787745600500000000ULL) << "one digit is enough";
        expect(!sigmf::parseDatetimeNs("2026-08-26T12:00:00.Z").has_value()) << "and the point without a digit is not";
        expect(!sigmf::parseDatetimeNs("2026-08-26T12:00:00.12345678x012Z").has_value()) << "every character past the point is still a digit";
        expect(!sigmf::parseDatetimeNs("2026-08-26T12:00:00").has_value()) << "the trailing Z is mandatory";
        expect(!sigmf::parseDatetimeNs("2026-08-26 12:00:00Z").has_value());
    };

    "a non-finite number is refused on write"_test = [] {
        sigmf::Metadata metadata       = anchorMetadata();
        metadata.captures[0].frequency = std::numeric_limits<double>::infinity();
        const auto text                = sigmf::write(metadata);
        expect(!text.has_value());
        expect(eq(text.error().code, "non_finite_number"s));
    };
};

} // namespace

int main() { /* not needed for UT */ }
