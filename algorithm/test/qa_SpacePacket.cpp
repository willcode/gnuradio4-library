#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/ccsds/SpacePacket.hpp>

/*
 * The space packet primary header is six octets and one arithmetic convention, and the convention is
 * the whole of what this file is for. 133.0-B-2 4.1.3.5.3 states it as an equation --
 * C = (octets in the packet data field) - 1 -- so these tests assert the numbers it produces and the
 * numbers the other reading produces, rather than restating the rule:
 *
 *     data field 1 octet      -> field 0      -> total 7
 *     data field 256 octets   -> field 255    -> total 262
 *     data field 65 536       -> field 65535  -> total 65 542
 *
 * A builder that wrote the payload length into the field instead would produce a packet one octet
 * longer than it claims on every packet. The consequence is asserted here at the octet level: the
 * next packet's header is then read one octet early, and what comes back is not a version 0 header.
 */
namespace {

using namespace gr::ccsds;

std::uint64_t rng = 0x243F6A8885A308D3ULL;

std::uint32_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(rng >> 32U);
}

/// A whole packet: the primary header written from a payload length, then the payload itself.
std::vector<std::uint8_t> makePacket(std::uint16_t apid, std::uint16_t sequenceCount, std::size_t payloadOctets, std::uint8_t firstByte) {
    SpacePacketHeader header{};
    const WriteStatus built = headerForPayload(apid, false, false, static_cast<std::uint8_t>(SequenceFlags::unsegmented), sequenceCount, payloadOctets, header);
    boost::ut::expect(built == WriteStatus::ok);

    std::vector<std::uint8_t> packet(totalPacketOctets(header));
    boost::ut::expect(writeSpacePacketHeader(header, packet) == WriteStatus::ok);
    for (std::size_t i = 0UZ; i < payloadOctets; ++i) {
        packet[kSpacePacketHeaderSize + i] = static_cast<std::uint8_t>(firstByte + i);
    }
    return packet;
}

} // namespace

const boost::ut::suite<"ccsds space packets"> spacePacketTests = [] {
    using namespace boost::ut;

    "2. the primary header octets, assembled by hand from 133.0-B-2 4.1.3.1"_test = [] {
        // APID 100, sequence flags '11', sequence count 5, data length field 0 -- a seven-octet packet.
        SpacePacketHeader header{};
        header.apid           = 100U;
        header.sequence_flags = static_cast<std::uint8_t>(SequenceFlags::unsegmented);
        header.sequence_count = 5U;
        header.data_length    = 0U;
        std::array<std::uint8_t, kSpacePacketHeaderSize> octets{};
        expect(writeSpacePacketHeader(header, octets) == WriteStatus::ok);
        expect(octets == std::array<std::uint8_t, 6UZ>{0x00U, 0x64U, 0xC0U, 0x05U, 0x00U, 0x00U});
        expect(eq(totalPacketOctets(header), 7UZ));

        SpacePacketHeader back{};
        expect(parseSpacePacketHeader(octets, back) == ParseStatus::ok);
        expect(back == header);
    };

    "3. the minus-one, in numbers"_test = [] {
        struct Row {
            std::size_t   payload;
            std::uint16_t field;
            std::size_t   total;
        };
        constexpr std::array<Row, 3UZ> rows{Row{1UZ, 0U, 7UZ}, Row{256UZ, 255U, 262UZ}, Row{65536UZ, 65535U, 65542UZ}};

        for (const Row& row : rows) {
            SpacePacketHeader header{};
            expect(headerForPayload(100U, false, false, 3U, 0U, row.payload, header) == WriteStatus::ok);
            expect(eq(static_cast<std::size_t>(header.data_length), static_cast<std::size_t>(row.field)));
            expect(eq(totalPacketOctets(header), row.total));
            expect(eq(packetDataOctets(header), row.payload));

            SpacePacketHeader fromWire{};
            fromWire.data_length = row.field;
            expect(eq(totalPacketOctets(fromWire), row.total));
        }

        expect(eq(kMaxPacketOctets, 65542UZ)) << "6 + 65535 + 1, derived from a sixteen-bit field";
        expect(eq(kSpacePacketHeaderSize + kMaxPacketDataOctets, kMaxPacketOctets));

        // The failure of the other reading, at the octet level. Two 100-octet-payload packets laid end
        // to end: read with the correct convention the second header parses; read with a header whose
        // field was set to the payload length the second header starts one octet early.
        const std::vector<std::uint8_t> first  = makePacket(100U, 0U, 100UZ, 0x10U);
        const std::vector<std::uint8_t> second = makePacket(101U, 1U, 100UZ, 0x40U);
        std::vector<std::uint8_t>       stream = first;
        stream.insert(stream.end(), second.begin(), second.end());

        SpacePacketHeader head{};
        expect(parseSpacePacketHeader(stream, head) == ParseStatus::ok);
        expect(eq(totalPacketOctets(head), 106UZ)) << "6 + 100, and the field holds 99";
        expect(eq(static_cast<unsigned>(head.data_length), 99U));

        SpacePacketHeader nextHeader{};
        expect(parseSpacePacketHeader(std::span{stream}.subspan(totalPacketOctets(head)), nextHeader) == ParseStatus::ok);
        expect(eq(static_cast<unsigned>(nextHeader.apid), 101U));

        // The surveyed convention: the field written as the payload length makes the total 107, so the
        // walk lands one octet past the second header and reads the payload as a header.
        SpacePacketHeader wrong = head;
        wrong.data_length       = 100U;
        expect(eq(totalPacketOctets(wrong), 107UZ)) << "one octet too many, every packet, silently";
        SpacePacketHeader misread{};
        const ParseStatus misreadStatus = parseSpacePacketHeader(std::span{stream}.subspan(totalPacketOctets(wrong)), misread);
        expect(misreadStatus != ParseStatus::ok || misread.apid != 101U) << "the second packet is read as something else entirely: status " << static_cast<int>(misreadStatus) << " apid " << misread.apid;
    };

    "4. a legal data length field of zero, and the payload builder's refusals"_test = [] {
        // 4.1.4.1.2 makes the data field at least one octet, so a field of 0 is legal and describes a
        // one-octet data field; there is no encoding for an empty one, which is why the minus-one exists.
        SpacePacketHeader header{};
        expect(headerForPayload(1U, false, false, 3U, 0U, 1UZ, header) == WriteStatus::ok);
        expect(eq(static_cast<unsigned>(header.data_length), 0U));

        expect(headerForPayload(1U, false, false, 3U, 0U, 0UZ, header) == WriteStatus::field_out_of_range) << "a zero-octet payload has no encoding";
        expect(headerForPayload(1U, false, false, 3U, 0U, kMaxPacketDataOctets + 1UZ, header) == WriteStatus::field_out_of_range);
        expect(headerForPayload(2048U, false, false, 3U, 0U, 10UZ, header) == WriteStatus::field_out_of_range) << "eleven bits of APID";
        expect(headerForPayload(1U, false, false, 4U, 0U, 10UZ, header) == WriteStatus::field_out_of_range) << "two bits of sequence flags";
        expect(headerForPayload(1U, false, false, 3U, 16384U, 10UZ, header) == WriteStatus::field_out_of_range) << "fourteen bits of sequence count";
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

            std::array<std::uint8_t, kSpacePacketHeaderSize> octets{};
            expect(writeSpacePacketHeader(header, octets) == WriteStatus::ok);
            SpacePacketHeader back{};
            expect(parseSpacePacketHeader(octets, back) == ParseStatus::ok);
            expect(back == header);

            std::array<std::uint8_t, kSpacePacketHeaderSize> again{};
            expect(writeSpacePacketHeader(back, again) == WriteStatus::ok);
            expect(again == octets);
        }

        SpacePacketHeader maxima{};
        maxima.apid           = 2047U;
        maxima.sequence_flags = 3U;
        maxima.sequence_count = 16383U;
        maxima.data_length    = 65535U;
        std::array<std::uint8_t, kSpacePacketHeaderSize> at{};
        expect(writeSpacePacketHeader(maxima, at) == WriteStatus::ok);
        SpacePacketHeader back{};
        expect(parseSpacePacketHeader(at, back) == ParseStatus::ok);
        expect(back == maxima);
        expect(isIdlePacket(back)) << "APID 2047 is the reserved idle APID of 4.1.3.3.4.4";

        for (const SpacePacketHeader& over : {SpacePacketHeader{.version = 8U}, SpacePacketHeader{.apid = 2048U}, SpacePacketHeader{.sequence_flags = 4U}, SpacePacketHeader{.sequence_count = 16384U}}) {
            std::array<std::uint8_t, kSpacePacketHeaderSize> octets{};
            expect(writeSpacePacketHeader(over, octets) == WriteStatus::field_out_of_range);
        }

        std::array<std::uint8_t, kSpacePacketHeaderSize - 1UZ> shortSpan{};
        expect(writeSpacePacketHeader(maxima, shortSpan) == WriteStatus::short_span);
        expect(parseSpacePacketHeader(shortSpan, back) == ParseStatus::short_span);

        // A non-zero version number is fatal: 4.1.3.2.2 fixes it at '000'.
        std::array<std::uint8_t, kSpacePacketHeaderSize> versioned{0x20U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U};
        expect(parseSpacePacketHeader(versioned, back) == ParseStatus::bad_version);
    };
};

int main() { /* tests are automatically registered and run */ }
