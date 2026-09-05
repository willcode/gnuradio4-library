#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>

#include <gnuradio-4.0/algorithm/network/PacketEnvelope.hpp>

// The envelope is a format, and a format is only worth the bytes it pins. So the assertions below are byte
// assertions wherever a peer would see bytes: one fully specified message derived by hand, every refusal a header
// alone can express, and the item-type table checked against the framework enumeration it borrows at compile time.
// Nothing here opens a socket — the format is testable with the transport that carries it switched off.

namespace {

using namespace gr::network;

/// @brief The metadata map of the golden message: four keys, deliberately no floating-point value, so the pinned
/// bytes survive any later change to how the emitter prints a float.
[[nodiscard]] gr::property_map goldenMap() {
    gr::property_map map;
    map.insert_or_assign(gr::property_map::key_type("crc_ok"), gr::pmt::Value(true));
    map.insert_or_assign(gr::property_map::key_type("protocol"), gr::pmt::Value(std::string("ax25")));
    map.insert_or_assign(gr::property_map::key_type("sample_start"), gr::pmt::Value(static_cast<std::uint64_t>(4096)));
    map.insert_or_assign(gr::property_map::key_type("sequence"), gr::pmt::Value(static_cast<std::uint64_t>(7)));
    return map;
}

constexpr std::array<std::uint8_t, 32> kGoldenHeader{0x47U, 0x52U, 0x34U, 0x50U, 0x01U, 0x00U, 0x00U, 0x20U, //
    0x06U, 0x00U, 0x01U, 0x01U, 0x05U, 0x00U, 0x00U, 0x00U,                                                  //
    0x05U, 0x00U, 0x00U, 0x00U, 0x58U, 0x00U, 0x00U, 0x00U,                                                  //
    0x00U, 0x00U, 0x00U, 0x00U, 0x45U, 0xD6U, 0xE3U, 0xBAU};

constexpr std::string_view kGoldenMetadata = "\ncrc_ok: !!bool true\nprotocol:  \"ax25\"\nsample_start: !!uint64 4096\nsequence: !!uint64 7\n";

[[nodiscard]] EnvelopeHeader goldenHeaderFields() {
    EnvelopeHeader header;
    header.item_type     = 6U; // UInt8
    header.item_size     = 1U;
    header.item_count    = 5U;
    header.payload_bytes = 5U;
    header.meta_bytes    = static_cast<std::uint32_t>(kGoldenMetadata.size());
    return header;
}

/// @brief A header whose CRC is recomputed after @p corrupt has edited the covered bytes, so a field-value refusal is
/// reached rather than the CRC catching the edit first.
template<typename F>
[[nodiscard]] std::array<std::uint8_t, 32> resealed(F&& corrupt) {
    std::array<std::uint8_t, 32> bytes = kGoldenHeader;
    corrupt(bytes);
    const std::uint32_t crc = static_cast<std::uint32_t>(kHeaderCrc.compute(std::span<const std::uint8_t>(bytes.data(), 28UZ)));
    bytes[28UZ]             = static_cast<std::uint8_t>(crc & 0xFFU);
    bytes[29UZ]             = static_cast<std::uint8_t>((crc >> 8U) & 0xFFU);
    bytes[30UZ]             = static_cast<std::uint8_t>((crc >> 16U) & 0xFFU);
    bytes[31UZ]             = static_cast<std::uint8_t>((crc >> 24U) & 0xFFU);
    return bytes;
}

[[nodiscard]] EnvelopeError refusalOf(std::span<const std::uint8_t> bytes) {
    const auto decoded = decodeHeader(bytes);
    boost::ut::expect(!decoded.has_value()) << "a corrupted header was accepted";
    return decoded.has_value() ? EnvelopeError::BadMagic : decoded.error();
}

/// @brief Encode one packet's three described frames, the way a transport sink would.
template<typename T>
struct Message {
    std::array<std::uint8_t, 32> header{};
    std::string                  metadata{};
    std::vector<std::uint8_t>    payload{};
};

template<typename T>
[[nodiscard]] Message<T> encodeMessage(std::span<const T> items, const gr::property_map& map) {
    Message<T> message;
    message.metadata = gr::pmt::yaml::serialize(map);
    const auto raw   = std::as_bytes(items);
    message.payload.resize(raw.size());
    std::memcpy(message.payload.data(), raw.data(), raw.size());

    EnvelopeHeader header;
    header.item_type     = kItemTypeCode<T>;
    header.item_size     = static_cast<std::uint8_t>(sizeof(T));
    header.item_count    = static_cast<std::uint32_t>(items.size());
    header.payload_bytes = static_cast<std::uint32_t>(raw.size());
    header.meta_bytes    = static_cast<std::uint32_t>(message.metadata.size());
    message.header       = encodeHeader(header);
    return message;
}

/// @brief Decode the three frames back into items and a map, as a transport source would.
template<typename T>
[[nodiscard]] std::pair<std::vector<T>, gr::property_map> decodeMessage(const Message<T>& message) {
    const auto header = decodeHeader(message.header);
    boost::ut::expect(header.has_value()) << "a self-encoded header was refused";
    if (!header.has_value()) {
        return {};
    }
    boost::ut::expect(header->payload_bytes == message.payload.size());
    boost::ut::expect(header->meta_bytes == message.metadata.size());

    std::vector<T> items(header->item_count);
    if (!items.empty()) {
        std::memcpy(items.data(), message.payload.data(), message.payload.size());
    }
    gr::property_map map;
    if (header->meta_bytes != 0U) { // the empty map short-circuits, so the reader never asks what deserialize("") does
        const auto parsed = gr::pmt::yaml::deserialize(message.metadata);
        boost::ut::expect(parsed.has_value()) << "the metadata frame did not parse";
        if (parsed.has_value()) {
            map = *parsed;
        }
    }
    return {std::move(items), std::move(map)};
}

template<typename T>
void roundTripType(std::string_view label) {
    using namespace boost::ut;
    for (const std::size_t nItems : {1UZ, 2UZ, 1500UZ}) {
        std::vector<T> items(nItems);
        for (std::size_t i = 0UZ; i < nItems; ++i) {
            if constexpr (gr::meta::complex_like<T>) {
                items[i] = T(static_cast<typename T::value_type>(i + 1UZ), static_cast<typename T::value_type>(-static_cast<double>(i + 1UZ)));
            } else {
                items[i] = static_cast<T>(i + 1UZ);
            }
        }
        gr::property_map map;
        map.insert_or_assign(gr::property_map::key_type("sequence"), gr::pmt::Value(static_cast<std::uint64_t>(nItems)));
        map.insert_or_assign(gr::property_map::key_type("protocol"), gr::pmt::Value(std::string(label)));

        const Message<T> message   = encodeMessage(std::span<const T>(items), map);
        const auto [back, backMap] = decodeMessage(message);
        expect(eq(back.size(), items.size())) << label << " item count";
        expect(std::ranges::equal(back, items)) << label << " payload bytes";
        expect(backMap == map) << label << " metadata map";
    }
}

} // namespace

// clang-format off
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int8)           ==  2U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int16)          ==  3U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int32)          ==  4U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int64)          ==  5U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt8)          ==  6U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt16)         ==  7U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt32)         ==  8U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt64)         ==  9U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Float32)        == 10U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Float64)        == 11U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::ComplexFloat32) == 12U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::ComplexFloat64) == 13U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Monostate)      ==  0U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Bool)           ==  1U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::String)         == 14U);
static_assert(static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Value)          == 15U);

static_assert(kItemTypeCode<std::int8_t>          == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int8));
static_assert(kItemTypeCode<std::int16_t>         == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int16));
static_assert(kItemTypeCode<std::int32_t>         == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int32));
static_assert(kItemTypeCode<std::int64_t>         == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Int64));
static_assert(kItemTypeCode<std::uint8_t>         == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt8));
static_assert(kItemTypeCode<std::uint16_t>        == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt16));
static_assert(kItemTypeCode<std::uint32_t>        == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt32));
static_assert(kItemTypeCode<std::uint64_t>        == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::UInt64));
static_assert(kItemTypeCode<float>                == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Float32));
static_assert(kItemTypeCode<double>               == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::Float64));
static_assert(kItemTypeCode<std::complex<float>>  == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::ComplexFloat32));
static_assert(kItemTypeCode<std::complex<double>> == static_cast<std::uint16_t>(gr::pmt::Value::ValueType::ComplexFloat64));
// a framework renumbering therefore breaks this build rather than a peer at run time
// clang-format on

const boost::ut::suite<"packet envelope"> envelopeTests = [] {
    using namespace boost::ut;

    "the golden header, byte for byte"_test = [] {
        const std::array<std::uint8_t, 32> encoded = encodeHeader(goldenHeaderFields());
        expect(std::ranges::equal(encoded, kGoldenHeader)) << "encodeHeader did not reproduce the specified 32 bytes";

        const auto decoded = decodeHeader(kGoldenHeader);
        expect(decoded.has_value()) << "the specified header was refused";
        if (decoded.has_value()) {
            expect(eq(decoded->wire_version, std::uint16_t{1U}));
            expect(eq(decoded->byte_order, std::uint8_t{0U}));
            expect(eq(decoded->header_bytes, std::uint8_t{32U}));
            expect(eq(decoded->item_type, std::uint16_t{6U}));
            expect(eq(decoded->item_size, std::uint8_t{1U}));
            expect(eq(decoded->meta_encoding, std::uint8_t{1U}));
            expect(eq(decoded->item_count, std::uint32_t{5U}));
            expect(eq(decoded->payload_bytes, std::uint32_t{5U}));
            expect(eq(decoded->meta_bytes, std::uint32_t{88U}));
            expect(eq(decoded->flags, std::uint32_t{0U}));
            expect(eq(decoded->header_crc, std::uint32_t{0xBAE3D645U}));
        }

        // the catalog check value confirms the parameter set, and the residue gives a reader a one-line validation
        const std::string_view check = "123456789";
        expect(eq(static_cast<std::uint32_t>(kHeaderCrc.compute(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(check.data()), check.size()))), std::uint32_t{0xCBF43926U}));
        expect(eq(static_cast<std::uint32_t>(kHeaderCrc.compute(kGoldenHeader)), kHeaderCrcResidue));
    };

    // an emitter pin, not an interop contract: a deliberate change to how gr::pmt::yaml writes a map is a deliberate
    // change to this assertion, and the name says so
    "the golden metadata, byte for byte (emitter pin)"_test = [] {
        const std::string serialized = gr::pmt::yaml::serialize(goldenMap());
        expect(eq(serialized.size(), kGoldenMetadata.size()));
        expect(eq(serialized, std::string(kGoldenMetadata)));
    };

    "the golden metadata, as a parse"_test = [] {
        const auto parsed = gr::pmt::yaml::deserialize(kGoldenMetadata);
        expect(parsed.has_value()) << "the specified metadata frame did not parse";
        if (parsed.has_value()) {
            const gr::property_map expected = goldenMap();
            expect(eq(parsed->size(), expected.size()));
            expect(*parsed == expected) << "the parsed map differs from the one that was serialized";
            expect(parsed->at("crc_ok").get_if<bool>() != nullptr) << "crc_ok is not a bool";
            expect(parsed->at("protocol").get_if<std::pmr::string>() != nullptr) << "protocol is not a string";
            expect(parsed->at("sample_start").get_if<std::uint64_t>() != nullptr) << "sample_start is not a uint64";
            expect(parsed->at("sequence").get_if<std::uint64_t>() != nullptr) << "sequence is not a uint64";
        }
    };

    "the item-type table"_test = [] {
        constexpr std::array<std::pair<std::uint16_t, std::uint8_t>, 12> legal{{{2U, 1U}, {3U, 2U}, {4U, 4U}, {5U, 8U}, {6U, 1U}, {7U, 2U}, {8U, 4U}, {9U, 8U}, {10U, 4U}, {11U, 8U}, {12U, 8U}, {13U, 16U}}};
        for (const auto& [code, size] : legal) {
            const std::optional<std::uint8_t> got = itemSizeOf(code);
            expect(got.has_value()) << "item type" << code << "should be legal";
            if (got.has_value()) {
                expect(eq(*got, size)) << "item type" << code << "size";
            }
        }
        for (const std::uint16_t refused : {std::uint16_t{0U}, std::uint16_t{1U}, std::uint16_t{14U}, std::uint16_t{15U}, std::uint16_t{16U}, std::uint16_t{256U}}) {
            expect(!itemSizeOf(refused).has_value()) << "item type" << refused << "should be refused";
        }
        // five registered sample types inside twelve legal codes inside sixteen enumerators
        expect(eq(legal.size(), 12UZ));
        expect(eq(static_cast<std::size_t>(kLastItemTypeCode - kFirstItemTypeCode + 1U), 12UZ));
    };

    // The emitter writes the shortest representation that parses back to the same value, so the round trip is exact
    // rather than bounded. The bound is asserted too, because a regression to a fixed precision would still satisfy
    // it and the two assertions then say which of the two happened.
    "float fidelity"_test = [] {
        for (const double hz : {100'000'000.0, 145'825'000.0, 433'920'000.0, 433'921'337.0, 1'090'000'000.0, 2'400'000'001.0}) {
            gr::property_map map;
            map.insert_or_assign(gr::property_map::key_type("frequency"), gr::pmt::Value(hz));
            const auto parsed = gr::pmt::yaml::deserialize(gr::pmt::yaml::serialize(map));
            expect(parsed.has_value());
            if (!parsed.has_value()) {
                continue;
            }
            const double back = parsed->at("frequency").value_or<double>(0.0);
            expect(std::abs(back - hz) <= 5e-6 * hz) << "frequency" << hz << "outside six significant digits";
            expect(eq(back, hz)) << "frequency" << hz << "did not round-trip exactly";
        }
        for (const float rate : {8'000.f, 48'000.f, 192'000.f, 1'000'000.f, 2'048'000.f, 61'440'000.f, 61'440'004.f}) {
            gr::property_map map;
            map.insert_or_assign(gr::property_map::key_type("sample_rate"), gr::pmt::Value(rate));
            const auto parsed = gr::pmt::yaml::deserialize(gr::pmt::yaml::serialize(map));
            expect(parsed.has_value());
            if (parsed.has_value()) {
                expect(eq(parsed->at("sample_rate").value_or<float>(0.f), rate)) << "sample_rate" << rate << "did not round-trip exactly";
            }
        }
    };

    // Every refusal a header alone can express, over a 32-byte span and no more, so nothing can have been sized from
    // a field before it was checked.
    "every kernel refusal"_test = [] {
        expect(refusalOf(resealed([](auto& b) { b[0UZ] = 0x67U; })) == EnvelopeError::BadMagic);
        expect(refusalOf(resealed([](auto& b) { b[4UZ] = 0U; })) == EnvelopeError::BadVersion);
        expect(refusalOf(resealed([](auto& b) { b[4UZ] = 2U; })) == EnvelopeError::FutureVersion);
        expect(refusalOf(resealed([](auto& b) { b[5UZ] = 1U; })) == EnvelopeError::FutureVersion); // 0x0100 = 256
        expect(refusalOf(resealed([](auto& b) { b[6UZ] = 1U; })) == EnvelopeError::ByteOrder);
        expect(refusalOf(resealed([](auto& b) { b[7UZ] = 31U; })) == EnvelopeError::BadHeaderBytes);
        expect(refusalOf(resealed([](auto& b) { b[7UZ] = 33U; })) == EnvelopeError::BadHeaderBytes);
        for (const std::uint16_t code : {std::uint16_t{0U}, std::uint16_t{1U}, std::uint16_t{14U}, std::uint16_t{15U}, std::uint16_t{16U}}) {
            expect(refusalOf(resealed([code](auto& b) {
                b[8UZ] = static_cast<std::uint8_t>(code & 0xFFU);
                b[9UZ] = static_cast<std::uint8_t>(code >> 8U);
            })) == EnvelopeError::UnsupportedItemType)
                << "item type" << code;
        }
        expect(refusalOf(resealed([](auto& b) { b[10UZ] = 2U; })) == EnvelopeError::BadItemSize);
        expect(refusalOf(resealed([](auto& b) { b[16UZ] = 6U; })) == EnvelopeError::PayloadLengthField);
        expect(refusalOf(resealed([](auto& b) { b[11UZ] = 2U; })) == EnvelopeError::UnknownMetaEncoding);
        for (std::size_t bit = 0UZ; bit < 32UZ; ++bit) {
            expect(refusalOf(resealed([bit](auto& b) { b[24UZ + bit / 8UZ] = static_cast<std::uint8_t>(1U << (bit % 8UZ)); })) == EnvelopeError::UnknownFlags) << "flags bit" << bit;
        }
        expect(refusalOf(std::span<const std::uint8_t>(kGoldenHeader.data(), 31UZ)) == EnvelopeError::BadHeaderBytes);

        // Bytes [8, 28) carry no field a reader consults before the CRC, so a single flipped bit anywhere in them is
        // caught by the CRC alone. Bytes [0, 8) are the four fields that locate the CRC and are named above.
        for (std::size_t bit = 64UZ; bit < 8UZ * 28UZ; ++bit) {
            std::array<std::uint8_t, 32> bytes = kGoldenHeader;
            bytes[bit / 8UZ] ^= static_cast<std::uint8_t>(1U << (bit % 8UZ));
            expect(refusalOf(bytes) == EnvelopeError::BadHeaderCrc) << "flipped bit" << bit << "was not caught by the CRC";
        }
        for (std::size_t bit = 0UZ; bit < 32UZ; ++bit) { // the check value's own bytes are covered by the residue
            std::array<std::uint8_t, 32> bytes = kGoldenHeader;
            bytes[28UZ + bit / 8UZ] ^= static_cast<std::uint8_t>(1U << (bit % 8UZ));
            expect(refusalOf(bytes) == EnvelopeError::BadHeaderCrc) << "flipped check-value bit" << bit;
        }
    };

    "the item-count product cannot wrap"_test = [] {
        const std::array<std::uint8_t, 32> bytes = resealed([](auto& b) {
            b[8UZ]  = 13U; // ComplexFloat64, 16 bytes an item
            b[9UZ]  = 0U;
            b[10UZ] = 16U;
            b[12UZ] = 0xFFU;
            b[13UZ] = 0xFFU;
            b[14UZ] = 0xFFU;
            b[15UZ] = 0xFFU; // item_count = 0xFFFFFFFF, so the product is 6.87e10 and no 32-bit field can equal it
        });
        expect(refusalOf(bytes) == EnvelopeError::PayloadLengthField);
    };

    "round trip over every registered type"_test = [] {
        roundTripType<std::uint8_t>("uint8");
        roundTripType<std::int16_t>("int16");
        roundTripType<std::int32_t>("int32");
        roundTripType<float>("float32");
        roundTripType<std::complex<float>>("complex32");
    };

    "complex items are real then imaginary"_test = [] {
        const std::vector<std::complex<float>> items{{1.f, -2.f}};
        const Message<std::complex<float>>     message = encodeMessage(std::span<const std::complex<float>>(items), gr::property_map{});
        expect(eq(message.payload.size(), 8UZ));
        std::array<std::uint8_t, 8> expected{};
        const float                 re = 1.f;
        const float                 im = -2.f;
        std::memcpy(expected.data(), &re, 4UZ);
        std::memcpy(expected.data() + 4UZ, &im, 4UZ);
        expect(std::ranges::equal(message.payload, expected)) << "interleaved order is not real then imaginary";

        const auto header = decodeHeader(message.header);
        expect(header.has_value());
        if (header.has_value()) {
            expect(eq(header->item_type, std::uint16_t{12U}));
            expect(eq(header->item_size, std::uint8_t{8U}));
        }
    };

    "the empty map and the empty payload"_test = [] {
        // meta_bytes == 0 is the empty map, and a reader short-circuits it rather than asking deserialize what an
        // empty document means
        expect(eq(gr::pmt::yaml::serialize(gr::property_map{}).size(), 0UZ));

        const std::vector<std::uint8_t> none{};
        const Message<std::uint8_t>     message = encodeMessage(std::span<const std::uint8_t>(none), gr::property_map{});
        expect(eq(message.metadata.size(), 0UZ));
        expect(eq(message.payload.size(), 0UZ));

        const auto header = decodeHeader(message.header);
        expect(header.has_value()) << "an empty payload is a stated empty payload, not a refusal";
        if (header.has_value()) {
            expect(eq(header->item_count, std::uint32_t{0U}));
            expect(eq(header->payload_bytes, std::uint32_t{0U}));
            expect(eq(header->meta_bytes, std::uint32_t{0U}));
        }
        const auto [back, backMap] = decodeMessage(message);
        expect(back.empty());
        expect(backMap.empty());
    };
};

int main() { /* not needed for UT */ }
