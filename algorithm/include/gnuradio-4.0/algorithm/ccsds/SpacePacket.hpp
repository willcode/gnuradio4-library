#ifndef GNURADIO_ALGORITHM_CCSDS_SPACE_PACKET_HPP
#define GNURADIO_ALGORITHM_CCSDS_SPACE_PACKET_HPP

#include <cstddef>
#include <cstdint>
#include <span>

#include <gnuradio-4.0/algorithm/ccsds/TransferFrame.hpp>

/**
 * @brief The CCSDS space packet primary header, CCSDS 133.0-B-2 4.1.3.1, six octets.
 *
 * The header carries a length count that is **one fewer** than the packet data field's octets
 * (4.1.3.5.2, and 4.1.3.5.3 writes it out as `C = (Total Number of Octets in the Packet Data Field)
 * - 1`). That convention is the whole of what this header is easy to get wrong, and it is handled here
 * by holding the wire value in the struct and putting the `+ 1` in exactly one function:
 *
 *     packet_data_field_octets = data_length + 1
 *     total_packet_octets      = 6 + data_length + 1 = data_length + 7
 *
 * The `+ 7`, not `+ 6`. A field named `data_length` holding a payload length would be a second
 * convention with the same name, and a builder written against the wrong one produces a packet one
 * octet longer than it claims on every packet — after which the receiver's next packet starts one
 * octet early and every packet in the stream is misparsed with nothing to say so. `headerForPayload`
 * is the other direction and is a separate function for the same reason: a caller cannot pick up the
 * convention from the wrong end.
 *
 * Derived bounds, from the sixteen-bit field: the data field is 1 to 65 536 octets, which is the range
 * 133.0-B-2 figure 4-1 gives, and the total packet is 7 to 65 542 octets. 4.1.4.1.2 makes the data
 * field at least one octet, so a length field of 0 means a one-octet data field and is legal; there is
 * no encoding for an empty data field, which is why the minus-one exists at all.
 *
 * Bit numbering and the status enumerations are `TransferFrame.hpp`'s, unchanged.
 */
namespace gr::ccsds {

/// @brief Reserved for idle packets, 133.0-B-2 4.1.3.3.4.4. Derived: `'11111111111'` = `2^11 - 1`.
inline constexpr std::uint16_t kIdleApid = 2047U;

/// @brief 133.0-B-2 4.1.3.1.
inline constexpr std::size_t kSpacePacketHeaderSize = 6UZ;

/// @brief The largest space packet the sixteen-bit length field can describe. Derived: `6 + 65535 + 1`.
inline constexpr std::size_t kMaxPacketOctets = 65542UZ;

/// @brief The largest packet data field. Derived: `65535 + 1`.
inline constexpr std::size_t kMaxPacketDataOctets = 65536UZ;

/// @brief 133.0-B-2 4.1.3.4.2.2's four sequence-flag values.
enum class SequenceFlags : std::uint8_t { continuation = 0U, first = 1U, last = 2U, unsegmented = 3U };

struct SpacePacketHeader {
    std::uint8_t  version          = 0U;    //!< bits 0-2, `'000'` (4.1.3.2.2)
    bool          type             = false; //!< bit 3; `0` telemetry, `1` telecommand (4.1.3.3.2.3)
    bool          secondary_header = false; //!< bit 4; `0` for idle packets (4.1.3.3.3.4)
    std::uint16_t apid             = 0U;    //!< bits 5-15, eleven bits (4.1.3.3.4)
    std::uint8_t  sequence_flags   = 0U;    //!< bits 16-17 (4.1.3.4.2.2)
    std::uint16_t sequence_count   = 0U;    //!< bits 18-31, modulo 16384 (4.1.3.4.3.4)
    std::uint16_t data_length      = 0U;    //!< bits 32-47, AS TRANSMITTED: one fewer than the data field (4.1.3.5.2)

    [[nodiscard]] bool operator==(const SpacePacketHeader&) const noexcept = default;
};

/// @brief The packet data field's length in octets. 133.0-B-2 4.1.3.5.3's `C + 1`, in one place.
[[nodiscard]] inline constexpr std::size_t packetDataOctets(const SpacePacketHeader& header) noexcept { return std::size_t{header.data_length} + 1UZ; }

/// @brief The whole packet's length in octets, header included. Derived: `6 + data_length + 1`.
[[nodiscard]] inline constexpr std::size_t totalPacketOctets(const SpacePacketHeader& header) noexcept { return kSpacePacketHeaderSize + packetDataOctets(header); }

/// @brief Whether this packet is an idle packet, 133.0-B-2 4.1.3.3.4.4.
[[nodiscard]] inline constexpr bool isIdlePacket(const SpacePacketHeader& header) noexcept { return header.apid == kIdleApid; }

[[nodiscard]] inline constexpr ParseStatus parseSpacePacketHeader(std::span<const std::uint8_t> octets, SpacePacketHeader& out) noexcept {
    if (octets.size() < kSpacePacketHeaderSize) {
        return ParseStatus::short_span;
    }
    out.version          = static_cast<std::uint8_t>(detail::readField(octets, 0UZ, 3UZ));
    out.type             = detail::readField(octets, 3UZ, 1UZ) != 0U;
    out.secondary_header = detail::readField(octets, 4UZ, 1UZ) != 0U;
    out.apid             = static_cast<std::uint16_t>(detail::readField(octets, 5UZ, 11UZ));
    out.sequence_flags   = static_cast<std::uint8_t>(detail::readField(octets, 16UZ, 2UZ));
    out.sequence_count   = static_cast<std::uint16_t>(detail::readField(octets, 18UZ, 14UZ));
    out.data_length      = static_cast<std::uint16_t>(detail::readField(octets, 32UZ, 16UZ));

    return out.version != 0U ? ParseStatus::bad_version : ParseStatus::ok;
}

[[nodiscard]] inline constexpr WriteStatus writeSpacePacketHeader(const SpacePacketHeader& header, std::span<std::uint8_t> octets) noexcept {
    if (octets.size() < kSpacePacketHeaderSize) {
        return WriteStatus::short_span;
    }
    if (!detail::fits(header.version, 3UZ) || !detail::fits(header.apid, 11UZ) || !detail::fits(header.sequence_flags, 2UZ) || !detail::fits(header.sequence_count, 14UZ)) {
        return WriteStatus::field_out_of_range;
    }
    detail::writeField(octets, 0UZ, 3UZ, header.version);
    detail::writeField(octets, 3UZ, 1UZ, header.type ? 1U : 0U);
    detail::writeField(octets, 4UZ, 1UZ, header.secondary_header ? 1U : 0U);
    detail::writeField(octets, 5UZ, 11UZ, header.apid);
    detail::writeField(octets, 16UZ, 2UZ, header.sequence_flags);
    detail::writeField(octets, 18UZ, 14UZ, header.sequence_count);
    detail::writeField(octets, 32UZ, 16UZ, header.data_length);
    return WriteStatus::ok;
}

/**
 * @brief Build a header from a payload length rather than from a wire value.
 *
 * The one place in the tree that turns a payload length into the field, and the inverse of the one
 * place that turns the field into a length. A payload of zero octets is refused because 4.1.4.1.2
 * makes the data field at least one octet, and a payload above 65 536 octets is refused because the
 * field cannot describe it.
 */
[[nodiscard]] inline constexpr WriteStatus headerForPayload(std::uint16_t apid, bool type, bool secondaryHeader, std::uint8_t sequenceFlags, std::uint16_t sequenceCount, std::size_t payloadOctets, SpacePacketHeader& out) noexcept {
    if (payloadOctets == 0UZ || payloadOctets > kMaxPacketDataOctets) {
        return WriteStatus::field_out_of_range;
    }
    if (!detail::fits(apid, 11UZ) || !detail::fits(sequenceFlags, 2UZ) || !detail::fits(sequenceCount, 14UZ)) {
        return WriteStatus::field_out_of_range;
    }
    out.version          = 0U;
    out.type             = type;
    out.secondary_header = secondaryHeader;
    out.apid             = apid;
    out.sequence_flags   = sequenceFlags;
    out.sequence_count   = sequenceCount;
    out.data_length      = static_cast<std::uint16_t>(payloadOctets - 1UZ);
    return WriteStatus::ok;
}

} // namespace gr::ccsds

#endif // GNURADIO_ALGORITHM_CCSDS_SPACE_PACKET_HPP
