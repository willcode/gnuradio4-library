#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/ccsds/SpacePacket.hpp>

/*
 * The space packet primary header is six octets and one arithmetic convention. That convention is the subject of this
 * file. 133.0-B-2 4.1.3.5.3 states it as an equation, C = (octets in the packet data field) - 1. These tests assert the
 * numbers it produces and the numbers the other reading produces. They do not restate the rule.
 *
 *     data field 1 octet      -> field 0      -> total 7
 *     data field 256 octets   -> field 255    -> total 262
 *     data field 65 536       -> field 65535  -> total 65 542
 *
 * A builder that wrote the payload length into the field would produce a packet one octet longer than it claims, on
 * every packet. The test asserts the consequence at the octet level. The next packet's header is then read one octet
 * early, and it does not parse as a version 0 header.
 */
namespace {

using namespace gr::ccsds;

std::uint64_t rng = 0x243F6A8885A308D3ULL;

std::uint32_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(rng >> 32U);
}

/// A whole packet. The primary header is written from a payload length and followed by the payload.
std::vector<std::uint8_t> makePacket(std::uint16_t apid, std::uint16_t sequenceCount, std::size_t payloadOctets, std::uint8_t firstByte) {
    const auto header = headerForPayload(apid, false, false, static_cast<std::uint8_t>(SequenceFlags::unsegmented), sequenceCount, payloadOctets);
    boost::ut::expect(header.has_value());
    const auto octets = writeSpacePacketHeader(header.value());
    boost::ut::expect(octets.has_value());

    std::vector<std::uint8_t> packet(totalPacketOctets(header.value()));
    std::ranges::copy(octets.value(), packet.begin());
    for (std::size_t i = 0UZ; i < payloadOctets; ++i) {
        packet[kSpacePacketHeaderSize + i] = static_cast<std::uint8_t>(firstByte + i);
    }
    return packet;
}

// The three functions evaluate at compile time.
static_assert(parseSpacePacketHeader(writeSpacePacketHeader(headerForPayload(100U, false, false, 3U, 5U, 1UZ).value()).value()).value().apid == 100U);
static_assert(headerForPayload(100U, false, false, 3U, 5U, 0UZ).error() == WriteStatus::field_out_of_range);

} // namespace

const boost::ut::suite<"ccsds space packets"> spacePacketTests = [] {
    using namespace boost::ut;

    "2. the primary header octets, assembled by hand from 133.0-B-2 4.1.3.1"_test = [] {
        // APID 100, sequence flags '11', sequence count 5 and data length field 0 make a seven-octet packet.
        SpacePacketHeader header{};
        header.apid           = 100U;
        header.sequence_flags = static_cast<std::uint8_t>(SequenceFlags::unsegmented);
        header.sequence_count = 5U;
        header.data_length    = 0U;
        const auto octets     = writeSpacePacketHeader(header);
        expect(octets.has_value());
        expect(octets.value() == std::array<std::uint8_t, 6UZ>{0x00U, 0x64U, 0xC0U, 0x05U, 0x00U, 0x00U});
        expect(eq(totalPacketOctets(header), 7UZ));

        const auto back = parseSpacePacketHeader(octets.value());
        expect(back.has_value());
        expect(back.value() == header);
    };

    "3. the minus-one, in numbers"_test = [] {
        struct Row {
            std::size_t   payload;
            std::uint16_t field;
            std::size_t   total;
        };
        constexpr std::array<Row, 3UZ> rows{Row{1UZ, 0U, 7UZ}, Row{256UZ, 255U, 262UZ}, Row{65536UZ, 65535U, 65542UZ}};

        for (const Row& row : rows) {
            const auto header = headerForPayload(100U, false, false, 3U, 0U, row.payload);
            expect(header.has_value());
            expect(eq(static_cast<std::size_t>(header.value().data_length), static_cast<std::size_t>(row.field)));
            expect(eq(totalPacketOctets(header.value()), row.total));
            expect(eq(packetDataOctets(header.value()), row.payload));

            SpacePacketHeader fromWire{};
            fromWire.data_length = row.field;
            expect(eq(totalPacketOctets(fromWire), row.total));
        }

        expect(eq(kMaxPacketOctets, 65542UZ)) << "6 + 65535 + 1, derived from a sixteen-bit field";
        expect(eq(kSpacePacketHeaderSize + kMaxPacketDataOctets, kMaxPacketOctets));

        // The failure of the other reading, at the octet level. Two packets with 100-octet payloads are laid end to
        // end. Read with the correct convention, the second header parses. Read with a header whose field holds the
        // payload length, the second header starts one octet early.
        const std::vector<std::uint8_t> first  = makePacket(100U, 0U, 100UZ, 0x10U);
        const std::vector<std::uint8_t> second = makePacket(101U, 1U, 100UZ, 0x40U);
        std::vector<std::uint8_t>       stream = first;
        stream.insert(stream.end(), second.begin(), second.end());

        const SpacePacketHeader head = parseSpacePacketHeader(stream).value();
        expect(eq(totalPacketOctets(head), 106UZ)) << "6 + 100, and the field holds 99";
        expect(eq(static_cast<unsigned>(head.data_length), 99U));

        const auto nextHeader = parseSpacePacketHeader(std::span{stream}.subspan(totalPacketOctets(head)));
        expect(nextHeader.has_value());
        expect(eq(static_cast<unsigned>(nextHeader.value().apid), 101U));

        // The other convention writes the payload length into the field and makes the total 107. The walk then lands
        // one octet past the second header and reads the payload as a header.
        SpacePacketHeader wrong = head;
        wrong.data_length       = 100U;
        expect(eq(totalPacketOctets(wrong), 107UZ)) << "one octet too many, every packet, silently";
        const auto misread = parseSpacePacketHeader(std::span{stream}.subspan(totalPacketOctets(wrong)));
        expect(!misread.has_value() || misread.value().apid != 101U) << "the second packet is read as something else entirely";
    };

    "4. a legal data length field of zero, and the payload builder's refusals"_test = [] {
        // 4.1.4.1.2 makes the data field at least one octet long. A field of 0 is therefore legal and describes a
        // one-octet data field. No encoding exists for an empty data field, and that is why the minus-one exists.
        const auto header = headerForPayload(1U, false, false, 3U, 0U, 1UZ);
        expect(header.has_value());
        expect(eq(static_cast<unsigned>(header.value().data_length), 0U));

        const auto refusedBy = [](std::uint16_t apid, std::uint8_t sequenceFlags, std::uint16_t sequenceCount, std::size_t payloadOctets) {
            const auto refused = headerForPayload(apid, false, false, sequenceFlags, sequenceCount, payloadOctets);
            return !refused.has_value() && refused.error() == WriteStatus::field_out_of_range;
        };
        expect(refusedBy(1U, 3U, 0U, 0UZ)) << "a zero-octet payload has no encoding";
        expect(refusedBy(1U, 3U, 0U, kMaxPacketDataOctets + 1UZ));
        expect(refusedBy(2048U, 3U, 0U, 10UZ)) << "eleven bits of APID";
        expect(refusedBy(1U, 4U, 0U, 10UZ)) << "two bits of sequence flags";
        expect(refusedBy(1U, 3U, 16384U, 10UZ)) << "fourteen bits of sequence count";
    };

    "the header round trips over every field's declared range"_test = [] {
        for (std::size_t trial = 0UZ; trial < 5000UZ; ++trial) {
            SpacePacketHeader header{};
            header.type             = (next() & 1U) != 0U;
            header.secondary_header = (next() & 1U) != 0U;
            header.apid             = static_cast<std::uint16_t>(next() % 2048U);
            header.sequence_flags   = static_cast<std::uint8_t>(next() % 4U);
            header.sequence_count   = static_cast<std::uint16_t>(next() % 16384U);
            header.data_length      = static_cast<std::uint16_t>(next() % 65536U);

            const auto octets = writeSpacePacketHeader(header);
            expect(octets.has_value());
            const auto back = parseSpacePacketHeader(octets.value());
            expect(back.has_value());
            expect(back.value() == header);

            const auto again = writeSpacePacketHeader(back.value());
            expect(again.has_value());
            expect(again.value() == octets.value());
        }

        SpacePacketHeader maxima{};
        maxima.apid           = 2047U;
        maxima.sequence_flags = 3U;
        maxima.sequence_count = 16383U;
        maxima.data_length    = 65535U;
        const auto at         = writeSpacePacketHeader(maxima);
        expect(at.has_value());
        const auto back = parseSpacePacketHeader(at.value());
        expect(back.has_value());
        expect(back.value() == maxima);
        expect(isIdlePacket(back.value())) << "APID 2047 is the reserved idle APID of 4.1.3.3.4.4";

        for (const SpacePacketHeader& over : {SpacePacketHeader{.version = 8U}, SpacePacketHeader{.apid = 2048U}, SpacePacketHeader{.sequence_flags = 4U}, SpacePacketHeader{.sequence_count = 16384U}}) {
            const auto refused = writeSpacePacketHeader(over);
            expect(!refused.has_value() && refused.error() == WriteStatus::field_out_of_range);
        }

        const std::array<std::uint8_t, kSpacePacketHeaderSize> full       = at.value();
        const auto                                             shortParse = parseSpacePacketHeader(std::span{full}.first(kSpacePacketHeaderSize - 1UZ));
        expect(!shortParse.has_value() && shortParse.error() == ParseStatus::short_span);

        // A non-zero version number is fatal. 4.1.3.2.2 fixes it at '000'.
        constexpr std::array<std::uint8_t, kSpacePacketHeaderSize> versioned{0x20U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U};
        const auto                                                 versionedParse = parseSpacePacketHeader(versioned);
        expect(!versionedParse.has_value() && versionedParse.error() == ParseStatus::bad_version);
    };
};

int main() { /* tests are automatically registered and run */ }
