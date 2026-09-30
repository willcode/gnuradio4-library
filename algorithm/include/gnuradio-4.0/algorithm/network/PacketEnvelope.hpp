#ifndef GNURADIO_PACKETENVELOPE_HPP
#define GNURADIO_PACKETENVELOPE_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

#include <gnuradio-4.0/algorithm/digital/Crc.hpp>

/**
 * @brief The versioned byte envelope a packet takes when it leaves the process.
 *
 * A transport must state three things that a local buffer leaves implicit. They are the width of one payload item,
 * the byte order of an item, and the layout of the whole message. This header states all three in 32 fixed-width
 * little-endian bytes. A peer built from this file's table alone can read them without a framework header. A peer
 * that meets an unknown layout refuses it by name and does not guess.
 *
 * The envelope is transport-neutral. Nothing here refers to a socket. A message has four parts, which are topic,
 * header, metadata and payload. The header describes the last three. It carries two lengths. A byte-stream reader
 * uses them to find the next message when the transport supplies no frame boundaries.
 *
 * Two decisions shape most of the design.
 *
 * **`header_bytes` is self-describing, and `header_crc` is the last four bytes it covers.** A later version may only
 * append. It extends `header_bytes` and moves the CRC to the new end. "A reader accepts any version up to its own"
 * then stays one comparison and needs no compatibility matrix. Bytes `[0, 28)` keep their offsets and meanings in
 * every version. A reader must parse `magic`, `wire_version`, `byte_order` and `header_bytes` before it knows which
 * version it is reading.
 *
 * **The header carries a CRC and the payload does not.** The header holds the sizes. `item_count`, `payload_bytes`
 * and `meta_bytes` drive allocation and indexing. The CRC turns every header this format did not write into one
 * named refusal. It costs a fixed 28 table lookups, outside the payload path. The CRC also lets a byte-stream reader
 * confirm that four bytes matching the magic start a real header and are not a chance match inside a payload. The
 * payload's integrity is checked elsewhere, by an appended CRC and a `crc_ok` verdict over it. A second, weaker
 * check here could disagree with that one and would be worse than none.
 *
 * The CRC residue gives a reader a one-line validation with no field extraction. CRC-32/ISO-HDLC over all 32
 * bytes of an intact header is `0x2144DF1C`.
 *
 * Decoding allocates nothing. `decodeHeader` reads a 32-byte span, validates it in a fixed order and returns
 * `std::expected`. `item_count * item_size` is formed in 64 bits and compared with `payload_bytes`. No product of
 * two received values can wrap. Nothing here throws. A value chosen by a peer cannot stop the caller.
 */
namespace gr::network {

/// @brief The ASCII bytes `G`, `R`, `4`, `P`. As a byte string it reads the same in either byte order, and it
/// cannot serve as a byte-order mark. The magic identifies the format. `byte_order` states the order.
inline constexpr std::array<std::uint8_t, 4> kMagic{0x47U, 0x52U, 0x34U, 0x50U};

inline constexpr std::uint16_t kWireVersion       = 1U;          ///< the layout this file writes and the highest it accepts
inline constexpr std::uint8_t  kHeaderBytesV1     = 32U;         ///< header length in wire version 1
inline constexpr std::uint8_t  kByteOrderLittle   = 0U;          ///< the legal `byte_order` in wire version 1
inline constexpr std::uint8_t  kMetaEncodingYaml  = 1U;          ///< metadata frame is a gr::pmt::yaml document
inline constexpr std::uint32_t kHeaderCrcResidue  = 0x2144DF1CU; ///< CRC-32/ISO-HDLC over all bytes of an intact header
inline constexpr std::uint16_t kFirstItemTypeCode = 2U;          ///< `gr::pmt::Value::ValueType::Int8`
inline constexpr std::uint16_t kLastItemTypeCode  = 13U;         ///< `gr::pmt::Value::ValueType::ComplexFloat64`

/// @brief Every reason a reader can refuse a header from the 32 bytes alone.
///
/// The enumeration holds only what the header can express by itself. A refusal that needs the transport's frame
/// sizes, the reader's item type, its size bound or a metadata parser is the caller's. This split keeps the file
/// usable from a socket, a byte stream and a file alike.
enum class EnvelopeError : std::uint8_t {
    BadMagic,            ///< the first four bytes are not `GR4P`
    BadVersion,          ///< `wire_version` is zero, not a valid version
    FutureVersion,       ///< `wire_version` is above this reader's, and the layout is unknown
    ByteOrder,           ///< `byte_order` is not little, the order wire version 1 defines
    BadHeaderBytes,      ///< `header_bytes` disagrees with the version, or the span is shorter than a header
    BadHeaderCrc,        ///< the covered bytes do not match `header_crc`
    UnsupportedItemType, ///< `item_type` is not one of the twelve fixed-width numeric codes
    BadItemSize,         ///< `item_size` disagrees with the size `item_type` fixes
    PayloadLengthField,  ///< `payload_bytes` is not `item_count * item_size`
    UnknownMetaEncoding, ///< `meta_encoding` names an encoding this reader does not implement
    UnknownFlags         ///< a reserved `flags` bit is set, and each `flags` bit must be understood by the reader
};

/// @brief The `discard_reason` spelling of a refusal.
[[nodiscard]] inline constexpr std::string_view discardReason(EnvelopeError error) noexcept {
    switch (error) {
    case EnvelopeError::BadMagic: return "bad_magic";
    case EnvelopeError::BadVersion: return "bad_version";
    case EnvelopeError::FutureVersion: return "future_version";
    case EnvelopeError::ByteOrder: return "byte_order";
    case EnvelopeError::BadHeaderBytes: return "bad_header_bytes";
    case EnvelopeError::BadHeaderCrc: return "bad_header_crc";
    case EnvelopeError::UnsupportedItemType: return "unsupported_item_type";
    case EnvelopeError::BadItemSize: return "bad_item_size";
    case EnvelopeError::PayloadLengthField: return "payload_length_field";
    case EnvelopeError::UnknownMetaEncoding: return "unknown_meta_encoding";
    case EnvelopeError::UnknownFlags: return "unknown_flags";
    }
    return "bad_magic";
}

/// @brief The header's eleven fields, in wire order after the fixed magic.
///
/// Every field is an unsigned integer of its wire width. The struct and the bytes cannot differ in type. They may
/// differ in layout. The codec decides the offsets, not the compiler.
struct EnvelopeHeader {
    std::uint16_t wire_version  = kWireVersion;
    std::uint8_t  byte_order    = kByteOrderLittle;
    std::uint8_t  header_bytes  = kHeaderBytesV1;
    std::uint16_t item_type     = 0U;
    std::uint8_t  item_size     = 0U;
    std::uint8_t  meta_encoding = kMetaEncodingYaml;
    std::uint32_t item_count    = 0U;
    std::uint32_t payload_bytes = 0U;
    std::uint32_t meta_bytes    = 0U;
    std::uint32_t flags         = 0U;
    std::uint32_t header_crc    = 0U; ///< filled by `encodeHeader`, which overwrites any value the caller sets

    [[nodiscard]] bool operator==(const EnvelopeHeader&) const noexcept = default;
};

/// @brief CRC-32/ISO-HDLC, the catalog's `CRC-32`, with check value `0xCBF43926` and residue `0x2144DF1C`.
///
/// It is a namespace-scope `inline const` and not a function-local `static`. The per-message path then has no
/// thread-safe-initialization guard. The constructor throws on an invalid width. These parameters are compile-time
/// constants that pass. The throw is therefore unreachable, and stating that costs less than wrapping it.
inline const gr::digital::Crc kHeaderCrc{32U, 0x04C11DB7ULL, 0xFFFFFFFFULL, 0xFFFFFFFFULL, true, true};

/// @brief Bytes per payload item for an `item_type`, or nothing when the code is not a legal payload item type.
///
/// The codes are the numbering of `gr::pmt::Value::ValueType`. That numbering is safe on a wire here. The
/// enumeration is stored in a four-bit field and fills its sixteen values exactly. A renumbering would therefore
/// change the storage layout of `pmt::Value`. Four of the sixteen are refused, `Monostate`, `Bool`, `String` and
/// `Value`. None of them is a fixed-width numeric item. Codes 14 upwards are reserved.
[[nodiscard]] inline constexpr std::optional<std::uint8_t> itemSizeOf(std::uint16_t itemType) noexcept {
    switch (itemType) {
    case 2U: return std::uint8_t{1U};   // Int8
    case 3U: return std::uint8_t{2U};   // Int16
    case 4U: return std::uint8_t{4U};   // Int32
    case 5U: return std::uint8_t{8U};   // Int64
    case 6U: return std::uint8_t{1U};   // UInt8
    case 7U: return std::uint8_t{2U};   // UInt16
    case 8U: return std::uint8_t{4U};   // UInt32
    case 9U: return std::uint8_t{8U};   // UInt64
    case 10U: return std::uint8_t{4U};  // Float32
    case 11U: return std::uint8_t{8U};  // Float64
    case 12U: return std::uint8_t{8U};  // ComplexFloat32, two binary32, real then imaginary
    case 13U: return std::uint8_t{16U}; // ComplexFloat64, two binary64, real then imaginary
    default: return std::nullopt;
    }
}

/// @brief The `item_type` code of a C++ payload item type, as a compile-time lookup.
///
/// The specializations restate the table of `itemSizeOf` in the other direction. They do not call
/// `gr::pmt::Value::get_value_type<T>()`, which is a private member. The mapping here also keeps the kernel free of
/// any dependency on the interface of `pmt::Value`. `static_assert`s pin the two numberings to each other. A
/// renumbering in the framework then fails the build and does not fail a peer at run time. The primary template is
/// left undefined on purpose. An item type this envelope cannot carry is a compile error and not a run-time zero.
template<typename T>
struct ItemTypeCode;

// clang-format off
template<> struct ItemTypeCode<std::int8_t>            { static constexpr std::uint16_t value =  2U; };
template<> struct ItemTypeCode<std::int16_t>           { static constexpr std::uint16_t value =  3U; };
template<> struct ItemTypeCode<std::int32_t>           { static constexpr std::uint16_t value =  4U; };
template<> struct ItemTypeCode<std::int64_t>           { static constexpr std::uint16_t value =  5U; };
template<> struct ItemTypeCode<std::uint8_t>           { static constexpr std::uint16_t value =  6U; };
template<> struct ItemTypeCode<std::uint16_t>          { static constexpr std::uint16_t value =  7U; };
template<> struct ItemTypeCode<std::uint32_t>          { static constexpr std::uint16_t value =  8U; };
template<> struct ItemTypeCode<std::uint64_t>          { static constexpr std::uint16_t value =  9U; };
template<> struct ItemTypeCode<float>                  { static constexpr std::uint16_t value = 10U; };
template<> struct ItemTypeCode<double>                 { static constexpr std::uint16_t value = 11U; };
template<> struct ItemTypeCode<std::complex<float>>    { static constexpr std::uint16_t value = 12U; };
template<> struct ItemTypeCode<std::complex<double>>   { static constexpr std::uint16_t value = 13U; };
// clang-format on

template<typename T>
inline constexpr std::uint16_t kItemTypeCode = ItemTypeCode<T>::value;

/// @brief True when `T` is a payload item type this envelope carries.
template<typename T>
concept EnvelopeItem = requires { ItemTypeCode<T>::value; } && std::is_trivially_copyable_v<T> && itemSizeOf(ItemTypeCode<T>::value) == sizeof(T);

namespace detail {

[[nodiscard]] inline constexpr std::uint16_t readU16(std::span<const std::uint8_t> bytes, std::size_t at) noexcept { //
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[at]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[at + 1UZ]) << 8U));
}

[[nodiscard]] inline constexpr std::uint32_t readU32(std::span<const std::uint8_t> bytes, std::size_t at) noexcept { return static_cast<std::uint32_t>(bytes[at]) | (static_cast<std::uint32_t>(bytes[at + 1UZ]) << 8U) | (static_cast<std::uint32_t>(bytes[at + 2UZ]) << 16U) | (static_cast<std::uint32_t>(bytes[at + 3UZ]) << 24U); }

inline constexpr void writeU16(std::span<std::uint8_t> bytes, std::size_t at, std::uint16_t value) noexcept {
    bytes[at]       = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[at + 1UZ] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
}

inline constexpr void writeU32(std::span<std::uint8_t> bytes, std::size_t at, std::uint32_t value) noexcept {
    bytes[at]       = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[at + 1UZ] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    bytes[at + 2UZ] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
    bytes[at + 3UZ] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
}

} // namespace detail

/// @brief Serialize a header into its 32 wire bytes, computing `header_crc` over everything before it.
///
/// The caller's `header_crc` is ignored. A header and its check value must not disagree, and only the header is an
/// input. `header_bytes` is written as `kHeaderBytesV1` for the same reason. The codec writes the layout it
/// implements.
[[nodiscard]] inline std::array<std::uint8_t, kHeaderBytesV1> encodeHeader(const EnvelopeHeader& header) noexcept {
    std::array<std::uint8_t, kHeaderBytesV1> bytes{};
    const std::span<std::uint8_t>            out(bytes);
    std::ranges::copy(kMagic, bytes.begin());
    detail::writeU16(out, 4UZ, header.wire_version);
    bytes[6UZ] = header.byte_order;
    bytes[7UZ] = kHeaderBytesV1;
    detail::writeU16(out, 8UZ, header.item_type);
    bytes[10UZ] = header.item_size;
    bytes[11UZ] = header.meta_encoding;
    detail::writeU32(out, 12UZ, header.item_count);
    detail::writeU32(out, 16UZ, header.payload_bytes);
    detail::writeU32(out, 20UZ, header.meta_bytes);
    detail::writeU32(out, 24UZ, header.flags);
    detail::writeU32(out, 28UZ, static_cast<std::uint32_t>(kHeaderCrc.compute(std::span<const std::uint8_t>(bytes.data(), kHeaderBytesV1 - 4UZ))));
    return bytes;
}

/// @brief Validate 32 wire bytes and return the fields they state, or the first refusal.
///
/// The order is fixed for a reason. `magic` comes first. Foreign traffic is then named as foreign and not as
/// corruption. A silent filter would report nothing, and a magic mismatch is one refusal to count. `wire_version`,
/// `byte_order` and `header_bytes` come next, because they locate the CRC. The CRC is checked before any other field
/// is trusted. Every other single-bit corruption in the covered bytes then yields one status. It does not depend on
/// which field held the flipped bit.
[[nodiscard]] inline std::expected<EnvelopeHeader, EnvelopeError> decodeHeader(std::span<const std::uint8_t> bytes) noexcept {
    if (bytes.size() < kHeaderBytesV1) {
        return std::unexpected(EnvelopeError::BadHeaderBytes); // a span shorter than one header
    }
    if (!std::ranges::equal(bytes.first(kMagic.size()), kMagic)) {
        return std::unexpected(EnvelopeError::BadMagic);
    }

    EnvelopeHeader header;
    header.wire_version = detail::readU16(bytes, 4UZ);
    if (header.wire_version == 0U) {
        return std::unexpected(EnvelopeError::BadVersion);
    }
    if (header.wire_version > kWireVersion) {
        return std::unexpected(EnvelopeError::FutureVersion);
    }
    header.byte_order = bytes[6UZ];
    if (header.byte_order != kByteOrderLittle) {
        return std::unexpected(EnvelopeError::ByteOrder);
    }
    header.header_bytes = bytes[7UZ];
    if (header.header_bytes != kHeaderBytesV1) {
        return std::unexpected(EnvelopeError::BadHeaderBytes);
    }
    if (static_cast<std::uint32_t>(kHeaderCrc.compute(bytes.first(kHeaderBytesV1))) != kHeaderCrcResidue) {
        return std::unexpected(EnvelopeError::BadHeaderCrc); // the residue identity, with no field extracted to check the CRC
    }

    header.item_type                           = detail::readU16(bytes, 8UZ);
    const std::optional<std::uint8_t> declared = itemSizeOf(header.item_type);
    if (!declared.has_value()) {
        return std::unexpected(EnvelopeError::UnsupportedItemType);
    }
    header.item_size = bytes[10UZ];
    if (header.item_size != *declared) {
        return std::unexpected(EnvelopeError::BadItemSize);
    }
    header.meta_encoding = bytes[11UZ];
    header.item_count    = detail::readU32(bytes, 12UZ);
    header.payload_bytes = detail::readU32(bytes, 16UZ);
    // formed in 64 bits and compared, not multiplied in the 32-bit width of the field
    if (static_cast<std::uint64_t>(header.item_count) * static_cast<std::uint64_t>(header.item_size) != static_cast<std::uint64_t>(header.payload_bytes)) {
        return std::unexpected(EnvelopeError::PayloadLengthField);
    }
    if (header.meta_encoding != kMetaEncodingYaml) {
        return std::unexpected(EnvelopeError::UnknownMetaEncoding);
    }
    header.meta_bytes = detail::readU32(bytes, 20UZ);
    header.flags      = detail::readU32(bytes, 24UZ);
    if (header.flags != 0U) {
        return std::unexpected(EnvelopeError::UnknownFlags);
    }
    header.header_crc = detail::readU32(bytes, 28UZ);
    return header;
}

} // namespace gr::network

#endif // GNURADIO_PACKETENVELOPE_HPP
