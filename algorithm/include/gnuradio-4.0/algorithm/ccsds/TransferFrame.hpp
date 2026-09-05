#ifndef GNURADIO_ALGORITHM_CCSDS_TRANSFER_FRAME_HPP
#define GNURADIO_ALGORITHM_CCSDS_TRANSFER_FRAME_HPP

#include <cstddef>
#include <cstdint>
#include <span>

/**
 * @brief The three CCSDS transfer-frame headers, the CLCW and the M_PDU header, as value types over spans.
 *
 * A transfer frame is a fixed-or-declared-length container whose header names a spacecraft, a virtual
 * channel and a frame count. Three standards define three of them and they are close enough to be
 * confused and far enough apart that confusing them is silent: CCSDS 132.0-B-3 (TM), 732.0-B-4 (AOS) and
 * 232.0-B-4 (TC). Each gets its own type here rather than one parameterized header, because the three
 * differences below are what a shared parser gets wrong.
 *
 *   - The version number is `'00'` for TM and TC and `'01'` for AOS (132.0-B-3 4.1.2.2.2.2,
 *     732.0-B-4 4.1.2.2.2.2). It is the field that says which of the three these octets are, so a
 *     mismatch is fatal and every other status here is not.
 *   - TM's spacecraft identifier is ten bits and its virtual channel identifier three; AOS's are eight
 *     and six. The boundary between the two fields therefore sits two bits apart in the two headers,
 *     and one spacecraft identifier of 42 writes `02 A3` in TM and `4A BF` in AOS.
 *   - TM signals the operational control field's presence with a header bit (4.1.2.4.2); AOS has no such
 *     bit anywhere and establishes presence by management (732.0-B-4 4.1.1.1). So for TM the field's
 *     presence is a read and for AOS it is a setting, and no amount of parsing recovers it.
 *
 * Bit numbering is the standards' own (132.0-B-3 1.6.2, and the identical convention in the other
 * three): bit 0 is the first bit transmitted and is the most significant bit of the field it starts.
 * Every multi-bit field is big-endian in that sense and nothing anywhere is reflected.
 *
 * Two conventions govern the status values. `bad_version` is fatal because the version is the field
 * that identifies the frame. `reserved_violation` never is: it reports a reserved-spare field holding
 * something other than the value the standard mandates, the parsed header is complete and usable
 * beside it, and a caller counts it and carries on — a reserved bit set wrong is a transmitter's
 * business and refusing the frame would discard data over a bit that names nothing.
 *
 * Nothing here allocates, nothing throws, and no parse indexes a span before its length is checked.
 */
namespace gr::ccsds {

/// @brief What a header parse reports. `reserved_violation` is informational; only `bad_version` is fatal.
enum class ParseStatus { ok, short_span, bad_version, reserved_violation };

/// @brief What a header build reports.
enum class WriteStatus { ok, short_span, field_out_of_range };

inline constexpr std::size_t kTmPrimaryHeaderSize      = 6UZ; //!< 132.0-B-3 4.1.2.1
inline constexpr std::size_t kTmSecondaryHeaderIdSize  = 1UZ; //!< 132.0-B-3 4.1.3.1.2, the identification octet
inline constexpr std::size_t kAosPrimaryHeaderSize     = 6UZ; //!< 732.0-B-4 4.1.2.1, without the frame header error control
inline constexpr std::size_t kAosPrimaryHeaderFhecSize = 8UZ; //!< 732.0-B-4 4.1.2.6, with it
inline constexpr std::size_t kTcPrimaryHeaderSize      = 5UZ; //!< 232.0-B-4 4.1.2.1
inline constexpr std::size_t kMpduHeaderSize           = 2UZ; //!< 732.0-B-4 4.1.4.2.1.3
inline constexpr std::size_t kOcfSize                  = 4UZ; //!< 132.0-B-3 4.1.5.1
inline constexpr std::size_t kFecfSize                 = 2UZ; //!< 132.0-B-3 4.1.6.1.1
inline constexpr std::size_t kClcwSize                 = 4UZ; //!< 232.0-B-4 4.2.1.1.2

/// @brief No packet starts in this zone (132.0-B-3 4.1.2.7.6.4, 732.0-B-4 4.1.4.2.3.4). Derived: `2^11 - 1`.
inline constexpr std::uint16_t kFhpNoPacketStart = 2047U;

/// @brief The zone holds only idle data (132.0-B-3 4.1.2.7.6.5, 732.0-B-4 4.1.4.2.3.5). Derived: `2047 - 1`.
inline constexpr std::uint16_t kFhpOnlyIdleData = 2046U;

/// @brief TM's master and virtual channel frame counts are modulo 256 (132.0-B-3 4.1.2.5.2, 4.1.2.6.2).
inline constexpr std::uint32_t kTmCountModulus = 256U;

/// @brief AOS's virtual channel frame count is twenty-four bits (732.0-B-4 4.1.2.4.2). Derived: `2^24`.
inline constexpr std::uint32_t kAosCountModulus = 1U << 24U;

/// @brief The same count extended by the four-bit cycle field (732.0-B-4 4.1.2.5.5.2 NOTE). Derived: `2^28`.
inline constexpr std::uint32_t kAosCycleCountModulus = 1U << 28U;

namespace detail {

/// @brief The `width` bits starting at `bitOffset`, bit 0 being the most significant bit of octet 0.
///
/// The caller has already established that the span holds those bits; every parse below checks its
/// span's length once, against the header size, before it reads a field.
[[nodiscard]] inline constexpr std::uint32_t readField(std::span<const std::uint8_t> octets, std::size_t bitOffset, std::size_t width) noexcept {
    std::uint32_t value = 0U;
    for (std::size_t i = 0UZ; i < width; ++i) {
        const std::size_t bit = bitOffset + i;
        value                 = (value << 1U) | ((octets[bit / 8UZ] >> (7UZ - (bit % 8UZ))) & 1U);
    }
    return value;
}

/// @brief The inverse of `readField`, leaving every bit outside `[bitOffset, bitOffset + width)` untouched.
inline constexpr void writeField(std::span<std::uint8_t> octets, std::size_t bitOffset, std::size_t width, std::uint32_t value) noexcept {
    for (std::size_t i = 0UZ; i < width; ++i) {
        const std::size_t  bit   = bitOffset + i;
        const std::uint8_t mask  = static_cast<std::uint8_t>(1U << (7UZ - (bit % 8UZ)));
        const bool         isSet = ((value >> (width - 1UZ - i)) & 1U) != 0U;
        octets[bit / 8UZ]        = static_cast<std::uint8_t>(isSet ? (octets[bit / 8UZ] | mask) : (octets[bit / 8UZ] & static_cast<std::uint8_t>(~mask)));
    }
}

/// @brief Whether `value` fits `width` bits, which is what `field_out_of_range` means.
[[nodiscard]] inline constexpr bool fits(std::uint32_t value, std::size_t width) noexcept { return width >= 32UZ || value < (1U << width); }

} // namespace detail

/// @brief The TM transfer frame primary header, 132.0-B-3 4.1.2.1, six octets.
struct TmPrimaryHeader {
    std::uint8_t  version              = 0U;    //!< bits 0-1, `'00'` (4.1.2.2.2.2)
    std::uint16_t spacecraft_id        = 0U;    //!< bits 2-11, ten bits (4.1.2.2.3)
    std::uint8_t  virtual_channel      = 0U;    //!< bits 12-14, three bits (4.1.2.3)
    bool          ocf_present          = false; //!< bit 15 (4.1.2.4.2)
    std::uint8_t  master_frame_count   = 0U;    //!< bits 16-23, modulo 256 (4.1.2.5.2)
    std::uint8_t  vc_frame_count       = 0U;    //!< bits 24-31, modulo 256 (4.1.2.6.2)
    bool          secondary_header     = false; //!< bit 32 (4.1.2.7.2.2)
    bool          sync_flag            = false; //!< bit 33; `0` packets or idle data, `1` a VCA_SDU (4.1.2.7.3.2)
    bool          packet_order         = false; //!< bit 34; reserved `0` when the sync flag is `0` (4.1.2.7.4)
    std::uint8_t  segment_length_id    = 0U;    //!< bits 35-36; `'11'` when the sync flag is `0` (4.1.2.7.5.2)
    std::uint16_t first_header_pointer = 0U;    //!< bits 37-47, eleven bits (4.1.2.7.6)

    [[nodiscard]] bool operator==(const TmPrimaryHeader&) const noexcept = default;
};

/**
 * @brief Parse a TM primary header.
 *
 * The segment length identifier is read and reported and refuses nothing. 4.1.2.7.5.2 requires `'11'`
 * whenever the synchronization flag is `0`, and NOTE 1 says why: the identifier existed for Source
 * Packet Segments, those are no longer defined, and `'11'` is the value that denoted non-use of them.
 * So it is not a length and it selects nothing — a constant with a historical name — and any other
 * value under a zero synchronization flag is `reserved_violation` and nothing more. Under a set
 * synchronization flag it is undefined (NOTE 2) and is not tested at all.
 */
[[nodiscard]] inline constexpr ParseStatus parseTmPrimaryHeader(std::span<const std::uint8_t> octets, TmPrimaryHeader& out) noexcept {
    if (octets.size() < kTmPrimaryHeaderSize) {
        return ParseStatus::short_span;
    }
    out.version              = static_cast<std::uint8_t>(detail::readField(octets, 0UZ, 2UZ));
    out.spacecraft_id        = static_cast<std::uint16_t>(detail::readField(octets, 2UZ, 10UZ));
    out.virtual_channel      = static_cast<std::uint8_t>(detail::readField(octets, 12UZ, 3UZ));
    out.ocf_present          = detail::readField(octets, 15UZ, 1UZ) != 0U;
    out.master_frame_count   = static_cast<std::uint8_t>(detail::readField(octets, 16UZ, 8UZ));
    out.vc_frame_count       = static_cast<std::uint8_t>(detail::readField(octets, 24UZ, 8UZ));
    out.secondary_header     = detail::readField(octets, 32UZ, 1UZ) != 0U;
    out.sync_flag            = detail::readField(octets, 33UZ, 1UZ) != 0U;
    out.packet_order         = detail::readField(octets, 34UZ, 1UZ) != 0U;
    out.segment_length_id    = static_cast<std::uint8_t>(detail::readField(octets, 35UZ, 2UZ));
    out.first_header_pointer = static_cast<std::uint16_t>(detail::readField(octets, 37UZ, 11UZ));

    if (out.version != 0U) {
        return ParseStatus::bad_version;
    }
    if (!out.sync_flag && (out.segment_length_id != 3U || out.packet_order)) {
        return ParseStatus::reserved_violation;
    }
    return ParseStatus::ok;
}

[[nodiscard]] inline constexpr WriteStatus writeTmPrimaryHeader(const TmPrimaryHeader& header, std::span<std::uint8_t> octets) noexcept {
    if (octets.size() < kTmPrimaryHeaderSize) {
        return WriteStatus::short_span;
    }
    if (!detail::fits(header.version, 2UZ) || !detail::fits(header.spacecraft_id, 10UZ) || !detail::fits(header.virtual_channel, 3UZ) || !detail::fits(header.segment_length_id, 2UZ) || !detail::fits(header.first_header_pointer, 11UZ)) {
        return WriteStatus::field_out_of_range;
    }
    detail::writeField(octets, 0UZ, 2UZ, header.version);
    detail::writeField(octets, 2UZ, 10UZ, header.spacecraft_id);
    detail::writeField(octets, 12UZ, 3UZ, header.virtual_channel);
    detail::writeField(octets, 15UZ, 1UZ, header.ocf_present ? 1U : 0U);
    detail::writeField(octets, 16UZ, 8UZ, header.master_frame_count);
    detail::writeField(octets, 24UZ, 8UZ, header.vc_frame_count);
    detail::writeField(octets, 32UZ, 1UZ, header.secondary_header ? 1U : 0U);
    detail::writeField(octets, 33UZ, 1UZ, header.sync_flag ? 1U : 0U);
    detail::writeField(octets, 34UZ, 1UZ, header.packet_order ? 1U : 0U);
    detail::writeField(octets, 35UZ, 2UZ, header.segment_length_id);
    detail::writeField(octets, 37UZ, 11UZ, header.first_header_pointer);
    return WriteStatus::ok;
}

/// @brief The TM transfer frame secondary header identification octet, 132.0-B-3 4.1.3.2.
struct TmSecondaryHeaderId {
    std::uint8_t version = 0U; //!< bits 0-1, `'00'` (4.1.3.2.2)
    std::uint8_t length  = 0U; //!< bits 2-7, AS TRANSMITTED: one fewer than the header's total octets (4.1.3.2.3.2)

    [[nodiscard]] bool operator==(const TmSecondaryHeaderId&) const noexcept = default;
};

/**
 * @brief The secondary header's total length in octets, 132.0-B-3 4.1.3.2.3.2's minus-one.
 *
 * `length` holds the wire value and the `+ 1` lives here and nowhere else, which is the same rule
 * `totalPacketOctets` states for the space packet and `totalTcFrameOctets` for the TC frame. The three
 * are one convention and getting any of them wrong slips a boundary by one octet on every frame.
 */
[[nodiscard]] inline constexpr std::size_t tmSecondaryHeaderOctets(const TmSecondaryHeaderId& header) noexcept { return std::size_t{header.length} + 1UZ; }

/**
 * @brief Whether the identification octet describes a secondary header that can exist.
 *
 * 4.1.3.1.3 makes the data field 1 to 63 octets, so the total is 2 to 64 and the field value is 1 to
 * 63. A field value of 0 would describe a one-octet secondary header with no data field, which the
 * standard does not admit; it is a refusal for the caller to count rather than a parse status, because
 * the octets themselves parsed cleanly and it is the described geometry that is impossible.
 */
[[nodiscard]] inline constexpr bool tmSecondaryHeaderUsable(const TmSecondaryHeaderId& header) noexcept { return header.length != 0U; }

[[nodiscard]] inline constexpr ParseStatus parseTmSecondaryHeaderId(std::span<const std::uint8_t> octets, TmSecondaryHeaderId& out) noexcept {
    if (octets.size() < kTmSecondaryHeaderIdSize) {
        return ParseStatus::short_span;
    }
    out.version = static_cast<std::uint8_t>(detail::readField(octets, 0UZ, 2UZ));
    out.length  = static_cast<std::uint8_t>(detail::readField(octets, 2UZ, 6UZ));
    return out.version != 0U ? ParseStatus::bad_version : ParseStatus::ok;
}

[[nodiscard]] inline constexpr WriteStatus writeTmSecondaryHeaderId(const TmSecondaryHeaderId& header, std::span<std::uint8_t> octets) noexcept {
    if (octets.size() < kTmSecondaryHeaderIdSize) {
        return WriteStatus::short_span;
    }
    if (!detail::fits(header.version, 2UZ) || !detail::fits(header.length, 6UZ)) {
        return WriteStatus::field_out_of_range;
    }
    detail::writeField(octets, 0UZ, 2UZ, header.version);
    detail::writeField(octets, 2UZ, 6UZ, header.length);
    return WriteStatus::ok;
}

/// @brief The AOS transfer frame primary header, 732.0-B-4 4.1.2.1, six octets or eight with the FHEC.
struct AosPrimaryHeader {
    std::uint8_t  version                    = 1U;    //!< bits 0-1, `'01'` for AOS (4.1.2.2.2.2)
    std::uint8_t  spacecraft_id              = 0U;    //!< bits 2-9, eight bits, not TM's ten (4.1.2.2.3)
    std::uint8_t  virtual_channel            = 0U;    //!< bits 10-15, six bits; all ones is the OID channel (4.1.2.3.2)
    std::uint32_t vc_frame_count             = 0U;    //!< bits 16-39, modulo 16 777 216 (4.1.2.4.2)
    bool          replay                     = false; //!< bit 40 (4.1.2.5.2.3)
    bool          vc_count_cycle_used        = false; //!< bit 41 (4.1.2.5.3.2)
    std::uint8_t  reserved                   = 0U;    //!< bits 42-43, `'00'` (4.1.2.5.4)
    std::uint8_t  vc_count_cycle             = 0U;    //!< bits 44-47, four bits (4.1.2.5.5)
    bool          has_fhec                   = false; //!< whether bits 48-63 are present; by management (4.1.2.6.2)
    std::uint16_t frame_header_error_control = 0U;    //!< bits 48-63, when present (4.1.2.6)

    [[nodiscard]] bool operator==(const AosPrimaryHeader&) const noexcept = default;
};

/// @brief Six octets, or eight where management says the frame header error control is present (4.1.2.6.2).
[[nodiscard]] inline constexpr std::size_t aosPrimaryHeaderOctets(bool hasFhec) noexcept { return hasFhec ? kAosPrimaryHeaderFhecSize : kAosPrimaryHeaderSize; }

/**
 * @brief The virtual channel frame count widened by the cycle field, 732.0-B-4 4.1.2.5.5.2 NOTE.
 *
 * `24 + 4 = 28` bits, so the extended modulus is `2^28`. When the use flag is clear the cycle field is
 * `'0000'` (4.1.2.5.5.3) and shall be ignored by the receiver (4.1.2.5.3.2 a), so the count stays
 * twenty-four bits and gap detection uses the smaller modulus.
 */
[[nodiscard]] inline constexpr std::uint32_t aosWidenedFrameCount(const AosPrimaryHeader& header) noexcept { return header.vc_count_cycle_used ? ((std::uint32_t{header.vc_count_cycle} << 24U) | header.vc_frame_count) : header.vc_frame_count; }

/// @brief The modulus the widened count runs at, which is the one `frameGap` must be given.
[[nodiscard]] inline constexpr std::uint32_t aosCountModulus(const AosPrimaryHeader& header) noexcept { return header.vc_count_cycle_used ? kAosCycleCountModulus : kAosCountModulus; }

[[nodiscard]] inline constexpr ParseStatus parseAosPrimaryHeader(std::span<const std::uint8_t> octets, bool hasFhec, AosPrimaryHeader& out) noexcept {
    if (octets.size() < aosPrimaryHeaderOctets(hasFhec)) {
        return ParseStatus::short_span;
    }
    out.version                    = static_cast<std::uint8_t>(detail::readField(octets, 0UZ, 2UZ));
    out.spacecraft_id              = static_cast<std::uint8_t>(detail::readField(octets, 2UZ, 8UZ));
    out.virtual_channel            = static_cast<std::uint8_t>(detail::readField(octets, 10UZ, 6UZ));
    out.vc_frame_count             = detail::readField(octets, 16UZ, 24UZ);
    out.replay                     = detail::readField(octets, 40UZ, 1UZ) != 0U;
    out.vc_count_cycle_used        = detail::readField(octets, 41UZ, 1UZ) != 0U;
    out.reserved                   = static_cast<std::uint8_t>(detail::readField(octets, 42UZ, 2UZ));
    out.vc_count_cycle             = static_cast<std::uint8_t>(detail::readField(octets, 44UZ, 4UZ));
    out.has_fhec                   = hasFhec;
    out.frame_header_error_control = hasFhec ? static_cast<std::uint16_t>(detail::readField(octets, 48UZ, 16UZ)) : std::uint16_t{0U};

    if (out.version != 1U) {
        return ParseStatus::bad_version;
    }
    if (out.reserved != 0U || (!out.vc_count_cycle_used && out.vc_count_cycle != 0U)) {
        return ParseStatus::reserved_violation;
    }
    return ParseStatus::ok;
}

[[nodiscard]] inline constexpr WriteStatus writeAosPrimaryHeader(const AosPrimaryHeader& header, std::span<std::uint8_t> octets) noexcept {
    if (octets.size() < aosPrimaryHeaderOctets(header.has_fhec)) {
        return WriteStatus::short_span;
    }
    if (!detail::fits(header.version, 2UZ) || !detail::fits(header.virtual_channel, 6UZ) || !detail::fits(header.vc_frame_count, 24UZ) || !detail::fits(header.reserved, 2UZ) || !detail::fits(header.vc_count_cycle, 4UZ)) {
        return WriteStatus::field_out_of_range;
    }
    detail::writeField(octets, 0UZ, 2UZ, header.version);
    detail::writeField(octets, 2UZ, 8UZ, header.spacecraft_id);
    detail::writeField(octets, 10UZ, 6UZ, header.virtual_channel);
    detail::writeField(octets, 16UZ, 24UZ, header.vc_frame_count);
    detail::writeField(octets, 40UZ, 1UZ, header.replay ? 1U : 0U);
    detail::writeField(octets, 41UZ, 1UZ, header.vc_count_cycle_used ? 1U : 0U);
    detail::writeField(octets, 42UZ, 2UZ, header.reserved);
    detail::writeField(octets, 44UZ, 4UZ, header.vc_count_cycle);
    if (header.has_fhec) {
        detail::writeField(octets, 48UZ, 16UZ, header.frame_header_error_control);
    }
    return WriteStatus::ok;
}

/// @brief The M_PDU header that heads a packet-carrying AOS data field, 732.0-B-4 4.1.4.2.1.4.
struct MpduHeader {
    std::uint8_t  reserved             = 0U; //!< bits 0-4, `'00000'` (4.1.4.2.2)
    std::uint16_t first_header_pointer = 0U; //!< bits 5-15, eleven bits (4.1.4.2.3)

    [[nodiscard]] bool operator==(const MpduHeader&) const noexcept = default;
};

/**
 * @brief Parse an M_PDU header.
 *
 * The pointer is the same field as TM's with the same width and the same two reserved values, over a
 * different zone: 4.1.4.2.3.3 numbers the packet zone's octets from 0, and the packet zone is the data
 * field less this two-octet header. That off-by-two is the whole difference between AOS extraction and
 * TM extraction, which is why `PacketExtractor` takes a zone and a pointer rather than a frame.
 */
[[nodiscard]] inline constexpr ParseStatus parseMpduHeader(std::span<const std::uint8_t> octets, MpduHeader& out) noexcept {
    if (octets.size() < kMpduHeaderSize) {
        return ParseStatus::short_span;
    }
    out.reserved             = static_cast<std::uint8_t>(detail::readField(octets, 0UZ, 5UZ));
    out.first_header_pointer = static_cast<std::uint16_t>(detail::readField(octets, 5UZ, 11UZ));
    return out.reserved != 0U ? ParseStatus::reserved_violation : ParseStatus::ok;
}

[[nodiscard]] inline constexpr WriteStatus writeMpduHeader(const MpduHeader& header, std::span<std::uint8_t> octets) noexcept {
    if (octets.size() < kMpduHeaderSize) {
        return WriteStatus::short_span;
    }
    if (!detail::fits(header.reserved, 5UZ) || !detail::fits(header.first_header_pointer, 11UZ)) {
        return WriteStatus::field_out_of_range;
    }
    detail::writeField(octets, 0UZ, 5UZ, header.reserved);
    detail::writeField(octets, 5UZ, 11UZ, header.first_header_pointer);
    return WriteStatus::ok;
}

/// @brief The four combined states of TC's bypass and control command flags, 232.0-B-4 table 4-1.
enum class TcFrameType { type_ad, reserved, type_bd, type_bc };

/// @brief The TC transfer frame primary header, 232.0-B-4 4.1.2.1, five octets.
struct TcPrimaryHeader {
    std::uint8_t  version         = 0U;    //!< bits 0-1, `'00'` (4.1.2.2.2)
    bool          bypass          = false; //!< bit 2; `0` Type-A, `1` Type-B (4.1.2.3.1.2)
    bool          control_command = false; //!< bit 3; `0` data, `1` control commands (4.1.2.3.2.2)
    std::uint8_t  reserved        = 0U;    //!< bits 4-5, `'00'` (4.1.2.4)
    std::uint16_t spacecraft_id   = 0U;    //!< bits 6-15, ten bits (4.1.2.5)
    std::uint8_t  virtual_channel = 0U;    //!< bits 16-21, six bits (4.1.2.6)
    std::uint16_t frame_length    = 0U;    //!< bits 22-31, AS TRANSMITTED: one fewer than the frame's octets (4.1.2.7.2)
    std::uint8_t  sequence_number = 0U;    //!< bits 32-39, N(S) (4.1.2.8)

    [[nodiscard]] bool operator==(const TcPrimaryHeader&) const noexcept = default;
};

/**
 * @brief The frame's total length in octets, 232.0-B-4 4.1.2.7.2's minus-one.
 *
 * `C = (total octets) - 1` is the standard's own expression, measured from the first bit of the primary
 * header to the last bit of the frame error control field where one is present and to the last bit of
 * the data field otherwise (4.1.2.7.3). The field is ten bits, so the frame is bounded at `2^10 = 1024`
 * octets, which leaves `1024 - 5 = 1019` octets of data field with no error control field and
 * `1024 - 5 - 2 = 1017` with one — exactly what 4.1.1.1 b) states.
 */
[[nodiscard]] inline constexpr std::size_t totalTcFrameOctets(const TcPrimaryHeader& header) noexcept { return std::size_t{header.frame_length} + 1UZ; }

/**
 * @brief The frame type the two flags name together, 232.0-B-4 table 4-1.
 *
 * `01` is reserved for future application. It is reported rather than refused: an unknown combination
 * is still a frame, and a caller that does not know what to do with one can say so with a counter.
 */
[[nodiscard]] inline constexpr TcFrameType tcFrameType(const TcPrimaryHeader& header) noexcept {
    if (!header.bypass) {
        return header.control_command ? TcFrameType::reserved : TcFrameType::type_ad;
    }
    return header.control_command ? TcFrameType::type_bc : TcFrameType::type_bd;
}

[[nodiscard]] inline constexpr ParseStatus parseTcPrimaryHeader(std::span<const std::uint8_t> octets, TcPrimaryHeader& out) noexcept {
    if (octets.size() < kTcPrimaryHeaderSize) {
        return ParseStatus::short_span;
    }
    out.version         = static_cast<std::uint8_t>(detail::readField(octets, 0UZ, 2UZ));
    out.bypass          = detail::readField(octets, 2UZ, 1UZ) != 0U;
    out.control_command = detail::readField(octets, 3UZ, 1UZ) != 0U;
    out.reserved        = static_cast<std::uint8_t>(detail::readField(octets, 4UZ, 2UZ));
    out.spacecraft_id   = static_cast<std::uint16_t>(detail::readField(octets, 6UZ, 10UZ));
    out.virtual_channel = static_cast<std::uint8_t>(detail::readField(octets, 16UZ, 6UZ));
    out.frame_length    = static_cast<std::uint16_t>(detail::readField(octets, 22UZ, 10UZ));
    out.sequence_number = static_cast<std::uint8_t>(detail::readField(octets, 32UZ, 8UZ));

    if (out.version != 0U) {
        return ParseStatus::bad_version;
    }
    return out.reserved != 0U ? ParseStatus::reserved_violation : ParseStatus::ok;
}

[[nodiscard]] inline constexpr WriteStatus writeTcPrimaryHeader(const TcPrimaryHeader& header, std::span<std::uint8_t> octets) noexcept {
    if (octets.size() < kTcPrimaryHeaderSize) {
        return WriteStatus::short_span;
    }
    if (!detail::fits(header.version, 2UZ) || !detail::fits(header.reserved, 2UZ) || !detail::fits(header.spacecraft_id, 10UZ) || !detail::fits(header.virtual_channel, 6UZ) || !detail::fits(header.frame_length, 10UZ)) {
        return WriteStatus::field_out_of_range;
    }
    detail::writeField(octets, 0UZ, 2UZ, header.version);
    detail::writeField(octets, 2UZ, 1UZ, header.bypass ? 1U : 0U);
    detail::writeField(octets, 3UZ, 1UZ, header.control_command ? 1U : 0U);
    detail::writeField(octets, 4UZ, 2UZ, header.reserved);
    detail::writeField(octets, 6UZ, 10UZ, header.spacecraft_id);
    detail::writeField(octets, 16UZ, 6UZ, header.virtual_channel);
    detail::writeField(octets, 22UZ, 10UZ, header.frame_length);
    detail::writeField(octets, 32UZ, 8UZ, header.sequence_number);
    return WriteStatus::ok;
}

/// @brief What the operational control field's first two bits say it holds, 132.0-B-3 4.1.5.4 and 4.1.5.5.
enum class OcfReportType { type_1_clcw, type_2_project, type_2_sdls };

/// @brief Read the report type without interpreting the rest of the field. The span must hold four octets.
[[nodiscard]] inline constexpr OcfReportType ocfReportType(std::span<const std::uint8_t> octets) noexcept {
    if (detail::readField(octets, 0UZ, 1UZ) == 0U) {
        return OcfReportType::type_1_clcw;
    }
    return detail::readField(octets, 1UZ, 1UZ) == 0U ? OcfReportType::type_2_project : OcfReportType::type_2_sdls;
}

/**
 * @brief The Communications Link Control Word, 232.0-B-4 4.2.1.1.2, thirty-two bits.
 *
 * The five flags of 4.2.1.8.1 are five booleans and not a packed integer, because each one names an
 * operational condition an operator reads independently: packed, "no bit lock" is invisible unless
 * somebody knows which bit it is.
 */
struct Clcw {
    bool         control_word_type = false; //!< bit 0, always `'0'` for a CLCW (4.2.1.2)
    std::uint8_t version           = 0U;    //!< bits 1-2, `'00'` (4.2.1.3)
    std::uint8_t status            = 0U;    //!< bits 3-5 (4.2.1.4)
    std::uint8_t cop_in_effect     = 0U;    //!< bits 6-7 (4.2.1.5)
    std::uint8_t virtual_channel   = 0U;    //!< bits 8-13, six bits (4.2.1.6)
    std::uint8_t reserved          = 0U;    //!< bits 14-15 (4.2.1.7)
    bool         no_rf_available   = false; //!< bit 16 (4.2.1.8.2)
    bool         no_bit_lock       = false; //!< bit 17 (4.2.1.8.3)
    bool         lockout           = false; //!< bit 18 (4.2.1.8.4)
    bool         wait              = false; //!< bit 19 (4.2.1.8.5)
    bool         retransmit        = false; //!< bit 20 (4.2.1.8.6)
    std::uint8_t farm_b_counter    = 0U;    //!< bits 21-22 (4.2.1.9)
    std::uint8_t reserved_bit      = 0U;    //!< bit 23, `'0'` (4.2.1.10)
    std::uint8_t report_value      = 0U;    //!< bits 24-31, N(R) (4.2.1.11)

    [[nodiscard]] bool operator==(const Clcw&) const noexcept = default;
};

/**
 * @brief Parse a CLCW.
 *
 * The control word type and the CLCW version number together are what say these four octets are a
 * CLCW: 4.2.1.2 fixes the type at `'0'` and 4.2.1.3 the version at `'00'`, and a field carrying
 * anything else is a Type-2 report or a control word of another kind. Both are therefore
 * `bad_version`, which is the one fatal status here, and the caller reads `ocfReportType` first if it
 * wants to know which of the two it has.
 */
[[nodiscard]] inline constexpr ParseStatus parseClcw(std::span<const std::uint8_t> octets, Clcw& out) noexcept {
    if (octets.size() < kClcwSize) {
        return ParseStatus::short_span;
    }
    out.control_word_type = detail::readField(octets, 0UZ, 1UZ) != 0U;
    out.version           = static_cast<std::uint8_t>(detail::readField(octets, 1UZ, 2UZ));
    out.status            = static_cast<std::uint8_t>(detail::readField(octets, 3UZ, 3UZ));
    out.cop_in_effect     = static_cast<std::uint8_t>(detail::readField(octets, 6UZ, 2UZ));
    out.virtual_channel   = static_cast<std::uint8_t>(detail::readField(octets, 8UZ, 6UZ));
    out.reserved          = static_cast<std::uint8_t>(detail::readField(octets, 14UZ, 2UZ));
    out.no_rf_available   = detail::readField(octets, 16UZ, 1UZ) != 0U;
    out.no_bit_lock       = detail::readField(octets, 17UZ, 1UZ) != 0U;
    out.lockout           = detail::readField(octets, 18UZ, 1UZ) != 0U;
    out.wait              = detail::readField(octets, 19UZ, 1UZ) != 0U;
    out.retransmit        = detail::readField(octets, 20UZ, 1UZ) != 0U;
    out.farm_b_counter    = static_cast<std::uint8_t>(detail::readField(octets, 21UZ, 2UZ));
    out.reserved_bit      = static_cast<std::uint8_t>(detail::readField(octets, 23UZ, 1UZ));
    out.report_value      = static_cast<std::uint8_t>(detail::readField(octets, 24UZ, 8UZ));

    if (out.control_word_type || out.version != 0U) {
        return ParseStatus::bad_version;
    }
    return out.reserved_bit != 0U ? ParseStatus::reserved_violation : ParseStatus::ok;
}

[[nodiscard]] inline constexpr WriteStatus writeClcw(const Clcw& clcw, std::span<std::uint8_t> octets) noexcept {
    if (octets.size() < kClcwSize) {
        return WriteStatus::short_span;
    }
    if (!detail::fits(clcw.version, 2UZ) || !detail::fits(clcw.status, 3UZ) || !detail::fits(clcw.cop_in_effect, 2UZ) || !detail::fits(clcw.virtual_channel, 6UZ) || !detail::fits(clcw.reserved, 2UZ) || !detail::fits(clcw.farm_b_counter, 2UZ) || !detail::fits(clcw.reserved_bit, 1UZ)) {
        return WriteStatus::field_out_of_range;
    }
    detail::writeField(octets, 0UZ, 1UZ, clcw.control_word_type ? 1U : 0U);
    detail::writeField(octets, 1UZ, 2UZ, clcw.version);
    detail::writeField(octets, 3UZ, 3UZ, clcw.status);
    detail::writeField(octets, 6UZ, 2UZ, clcw.cop_in_effect);
    detail::writeField(octets, 8UZ, 6UZ, clcw.virtual_channel);
    detail::writeField(octets, 14UZ, 2UZ, clcw.reserved);
    detail::writeField(octets, 16UZ, 1UZ, clcw.no_rf_available ? 1U : 0U);
    detail::writeField(octets, 17UZ, 1UZ, clcw.no_bit_lock ? 1U : 0U);
    detail::writeField(octets, 18UZ, 1UZ, clcw.lockout ? 1U : 0U);
    detail::writeField(octets, 19UZ, 1UZ, clcw.wait ? 1U : 0U);
    detail::writeField(octets, 20UZ, 1UZ, clcw.retransmit ? 1U : 0U);
    detail::writeField(octets, 21UZ, 2UZ, clcw.farm_b_counter);
    detail::writeField(octets, 23UZ, 1UZ, clcw.reserved_bit);
    detail::writeField(octets, 24UZ, 8UZ, clcw.report_value);
    return WriteStatus::ok;
}

/// @brief What one step of a frame count says about the frames between it and the last one.
struct SequenceGap {
    std::uint32_t lost       = 0U;    //!< `gap - 1`, the number of frames that did not arrive
    bool          duplicate  = false; //!< `gap == 0`: this frame repeats the last one
    bool          continuous = false; //!< `gap == 1`: nothing was lost

    [[nodiscard]] bool operator==(const SequenceGap&) const noexcept = default;
};

/**
 * @brief The gap between two frame counts, with the wrap.
 *
 * `gap = (count - last) mod modulus` in unsigned arithmetic is the whole of the wrap handling — there
 * is no branch on it anywhere. At TM's modulus a count of 3 after a count of 254 gives
 * `(3 - 254) mod 256 = 5`, so four frames were lost across the wrap and the arithmetic never noticed
 * the wrap happened. The modulus is the field's: 256 for TM's two counts, `2^24` for AOS's, and
 * `2^28` for AOS's count extended by its cycle field.
 */
[[nodiscard]] inline constexpr SequenceGap frameGap(std::uint32_t count, std::uint32_t last, std::uint32_t modulus) noexcept {
    const std::uint32_t gap = ((count % modulus) + modulus - (last % modulus)) % modulus;
    return SequenceGap{gap > 1U ? gap - 1U : 0U, gap == 0U, gap == 1U};
}

/**
 * @brief The Only Idle Data fill sequence, 132.0-B-3 4.1.4.6.2 and 732.0-B-4 4.1.4.1.5.2.
 *
 * The standard specifies the 32-cell shift register with `h(x) = x^32 + x^22 + x^2 + x + 1`, the
 * Fibonacci form seeded all ones and never restarted (4.1.4.6.2.1), and publishes the first ten octets
 * of the result in the NOTE to 4.1.4.6.2.2: `FF FF FF FF 6D B6 D8 61 45 1F`. Those ten octets are
 * reproduced by the register below and by nothing else, which is what makes them the test.
 *
 * The register holds stage 1 in the most significant bit and stage 32 in the least. The output is
 * stage 32 and the feedback is the exclusive-or of stages 1, 2, 22 and 32, most significant bit first,
 * which is the standard's `D0 + D1 + D2 + D22 + D32` read in the transmission direction. Written as a
 * recurrence over the output sequence that is `x^32 + x^31 + x^30 + x^10 + 1`, the reciprocal of the
 * stated polynomial, and the reciprocal is what the standard's own anchor selects.
 *
 * 4.1.4.6.2.1's "shall not be restarted" is a transmitter property: the generator runs across frames
 * rather than resetting per frame, which is why the state lives in the object and `next` is not pure.
 * A receiver never needs the sequence at all — an OID frame is identified by its pointer, not by its
 * contents — so this is a transmit-side type only.
 */
class OidFill {
public:
    static constexpr std::uint32_t kSeed = 0xFFFFFFFFU; //!< 4.1.4.6.2.1, all ones

    /// @brief The Galois form of 4.1.4.6.2.2, published as `00000000001111111111111111111101`.
    static constexpr std::uint32_t kGaloisSeed = 0x003FFFFDU;

    /// @brief The Galois register's feedback mask, the stated polynomial's reciprocal without its top term.
    static constexpr std::uint32_t kGaloisMask = 0x80200003U;

    constexpr void reset() noexcept { state_ = kSeed; }

    /// @brief Fill `out` with the next octets of the sequence, carrying the register across calls.
    constexpr void next(std::span<std::uint8_t> out) noexcept {
        for (std::uint8_t& octet : out) {
            std::uint32_t value = 0U;
            for (std::size_t bit = 0UZ; bit < 8UZ; ++bit) {
                const std::uint32_t output   = state_ & 1U;
                const std::uint32_t feedback = ((state_ >> 31U) ^ (state_ >> 30U) ^ (state_ >> 10U) ^ state_) & 1U;
                state_                       = (state_ >> 1U) | (feedback << 31U);
                value                        = (value << 1U) | output;
            }
            octet = static_cast<std::uint8_t>(value);
        }
    }

    [[nodiscard]] constexpr std::uint32_t state() const noexcept { return state_; }

private:
    std::uint32_t state_ = kSeed;
};

} // namespace gr::ccsds

#endif // GNURADIO_ALGORITHM_CCSDS_TRANSFER_FRAME_HPP
