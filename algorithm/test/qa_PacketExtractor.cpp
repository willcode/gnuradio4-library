#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/ccsds/PacketExtractor.hpp>

/*
 * The packet extraction function is one sentence of the standard turned into a state machine, and the
 * sentence is 132.0-B-3 4.3.2.4: when the held fragment's own length disagrees with the first header
 * pointer, the pointer is correct. Everything here tests the consequence of that rule rather than the
 * rule, and the criterion the group exists for is the third test below -- one zone lost between two
 * fragments of a spanning packet costs exactly one packet, not the rest of the pass.
 *
 * The negative control is in the same file and is the point of it: a reassembler that ignores the
 * pointer and simply concatenates, fed the same three zones, is asserted to produce a *wrong* second
 * packet. That is what the pointer buys, stated as a difference rather than as a warning.
 */
namespace {

using namespace gr::ccsds;

std::uint64_t rng = 0xB5026F5AA96619E9ULL;

std::uint32_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(rng >> 32U);
}

std::vector<std::uint8_t> makePacket(std::uint16_t apid, std::uint16_t sequenceCount, std::size_t payloadOctets, std::uint8_t seed) {
    SpacePacketHeader header{};
    boost::ut::expect(headerForPayload(apid, false, false, static_cast<std::uint8_t>(SequenceFlags::unsegmented), sequenceCount, payloadOctets, header) == WriteStatus::ok);
    std::vector<std::uint8_t> packet(totalPacketOctets(header));
    boost::ut::expect(writeSpacePacketHeader(header, packet) == WriteStatus::ok);
    std::uint8_t value = seed;
    for (std::size_t i = 0UZ; i < payloadOctets; ++i) {
        value                              = static_cast<std::uint8_t>(value * 31U + 17U);
        packet[kSpacePacketHeaderSize + i] = value;
    }
    return packet;
}

/// One virtual channel's packet zones, as a transmitter would lay them: packets end to end, one
/// pointer per zone naming the first packet that starts in it, idle packets padding the tail so no
/// zone is short and no octet of fill is ever walked as a header.
struct Segmented {
    std::vector<std::vector<std::uint8_t>> zones;
    std::vector<std::uint16_t>             pointers;
    std::size_t                            idle_packets = 0UZ;
};

Segmented segment(const std::vector<std::vector<std::uint8_t>>& packets, std::size_t zoneLength) {
    std::vector<std::uint8_t> stream;
    std::vector<std::size_t>  starts;
    for (const std::vector<std::uint8_t>& packet : packets) {
        starts.push_back(stream.size());
        stream.insert(stream.end(), packet.begin(), packet.end());
    }

    Segmented         out{};
    const std::size_t remainder = stream.size() % zoneLength;
    if (remainder != 0UZ) {
        std::size_t padTotal = zoneLength - remainder;
        if (padTotal < kSpacePacketHeaderSize + 1UZ) {
            padTotal += zoneLength;
        }
        starts.push_back(stream.size());
        const std::vector<std::uint8_t> idle = makePacket(kIdleApid, 0U, padTotal - kSpacePacketHeaderSize, 0x5AU);
        stream.insert(stream.end(), idle.begin(), idle.end());
        ++out.idle_packets;
    }

    std::size_t nextStart = 0UZ;
    for (std::size_t base = 0UZ; base < stream.size(); base += zoneLength) {
        out.zones.emplace_back(stream.begin() + static_cast<std::ptrdiff_t>(base), stream.begin() + static_cast<std::ptrdiff_t>(base + zoneLength));
        while (nextStart < starts.size() && starts[nextStart] < base) {
            ++nextStart;
        }
        if (nextStart < starts.size() && starts[nextStart] < base + zoneLength) {
            out.pointers.push_back(static_cast<std::uint16_t>(starts[nextStart] - base));
        } else {
            out.pointers.push_back(kFhpNoPacketStart);
        }
    }
    return out;
}

/// Compared through a function rather than in the assertion, because the unit-test framework's own
/// equality expression wants to print what it compared and a vector of vectors is not streamable.
bool same(const std::vector<std::vector<std::uint8_t>>& a, const std::vector<std::vector<std::uint8_t>>& b) { return a == b; }

} // namespace

const boost::ut::suite<"ccsds packet extraction"> packetExtractorTests = [] {
    using namespace boost::ut;

    "5. segment then extract is the identity, at every zone length the pointer can address"_test = [] {
        // 2046 rather than 2048: eleven bits with two reserved values name positions 0 to 2045, so a
        // zone longer than 2046 octets has positions no first header pointer can express. That bound is
        // asserted on its own in the test below.
        for (const std::size_t zoneLength : {223UZ, 1115UZ, 2046UZ}) {
            for (const std::size_t packetOctets : {7UZ, 8UZ, 100UZ, 1000UZ, 2047UZ, 4096UZ}) {
                std::vector<std::vector<std::uint8_t>> packets;
                for (std::size_t i = 0UZ; i < 9UZ; ++i) {
                    packets.push_back(makePacket(static_cast<std::uint16_t>(100U + i), static_cast<std::uint16_t>(i), packetOctets - kSpacePacketHeaderSize, static_cast<std::uint8_t>(i + 1UZ)));
                }
                const Segmented laid = segment(packets, zoneLength);

                PacketExtractor                        extractor;
                std::vector<std::vector<std::uint8_t>> got;
                std::uint32_t                          count = 0U;
                for (std::size_t z = 0UZ; z < laid.zones.size(); ++z) {
                    extractor.feed(laid.zones[z], laid.pointers[z], count, [&got](std::span<const std::uint8_t> packet) { got.emplace_back(packet.begin(), packet.end()); });
                    count = (count + 1U) % kTmCountModulus;
                }

                expect(eq(got.size(), packets.size())) << "zone " << zoneLength << " packet " << packetOctets;
                expect(same(got, packets)) << "every packet byte for byte, in order";

                const PacketExtractor::Counters& counters = extractor.counters();
                expect(eq(counters.packets, packets.size()));
                expect(eq(counters.idle_packets, laid.idle_packets));
                expect(eq(counters.frames_lost, 0ULL));
                expect(eq(counters.duplicate_frames, 0ULL));
                expect(eq(counters.fragments_dropped, 0ULL));
                expect(eq(counters.pointer_mismatch, 0ULL));
                expect(eq(counters.bad_pointer, 0ULL));
                expect(eq(counters.orphan_octets, 0ULL));
                expect(eq(counters.oversize_dropped, 0ULL));
                expect(eq(counters.idle_frames, 0ULL));
            }
        }

        // A packet boundary exactly at a zone boundary, and one octet either side of it.
        for (const std::size_t firstPacket : {223UZ - 1UZ, 223UZ, 223UZ + 1UZ}) {
            const std::vector<std::vector<std::uint8_t>> packets{makePacket(1U, 0U, firstPacket - kSpacePacketHeaderSize, 0x11U), makePacket(2U, 1U, 300UZ, 0x22U), makePacket(3U, 2U, 40UZ, 0x33U)};
            const Segmented                              laid = segment(packets, 223UZ);

            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            std::uint32_t                          count = 0U;
            for (std::size_t z = 0UZ; z < laid.zones.size(); ++z) {
                extractor.feed(laid.zones[z], laid.pointers[z], count, [&got](std::span<const std::uint8_t> packet) { got.emplace_back(packet.begin(), packet.end()); });
                count = (count + 1U) % kTmCountModulus;
            }
            expect(same(got, packets)) << "a packet ending at " << firstPacket;
        }
    };

    "6. idle frames and idle packets, counted and silent"_test = [] {
        // A hundred only-idle-data zones yield nothing, and nothing else moves.
        {
            PacketExtractor           extractor;
            OidFill                   fill;
            std::vector<std::uint8_t> zone(223UZ);
            std::size_t               emitted = 0UZ;
            for (std::uint32_t i = 0U; i < 100U; ++i) {
                fill.next(zone);
                extractor.feed(zone, kFhpOnlyIdleData, i % kTmCountModulus, [&emitted](std::span<const std::uint8_t>) { ++emitted; });
            }
            expect(eq(emitted, 0UZ));
            expect(eq(extractor.counters().idle_frames, 100ULL));
            expect(eq(extractor.counters().packets, 0ULL));
            expect(eq(extractor.counters().fragments_dropped, 0ULL));
            expect(eq(extractor.counters().orphan_octets, 0ULL));
        }

        // Three real packets and two idle packets in one zone: three out, two counted.
        {
            const std::vector<std::vector<std::uint8_t>> packets{makePacket(10U, 0U, 20UZ, 0x01U), makePacket(kIdleApid, 0U, 15UZ, 0x02U), makePacket(11U, 1U, 30UZ, 0x03U), makePacket(kIdleApid, 0U, 9UZ, 0x04U), makePacket(12U, 2U, 25UZ, 0x05U)};
            std::vector<std::uint8_t>                    zone;
            for (const std::vector<std::uint8_t>& packet : packets) {
                zone.insert(zone.end(), packet.begin(), packet.end());
            }

            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            extractor.feed(zone, 0U, 0U, [&got](std::span<const std::uint8_t> packet) { got.emplace_back(packet.begin(), packet.end()); });
            expect(eq(got.size(), 3UZ));
            expect(eq(extractor.counters().idle_packets, 2ULL));
            expect(got[0] == packets[0]);
            expect(got[1] == packets[2]);
            expect(got[2] == packets[4]);
        }

        // An only-idle-data zone in the middle of a spanning packet drops the fragment once, per
        // 4.1.4.6.3 NOTE 3, and the packet that starts in the next real zone comes out whole.
        {
            const std::vector<std::uint8_t> spanning = makePacket(20U, 0U, 500UZ, 0x10U);
            const std::vector<std::uint8_t> follower = makePacket(21U, 1U, 60UZ, 0x20U);

            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            const auto                             emit = [&got](std::span<const std::uint8_t> packet) { got.emplace_back(packet.begin(), packet.end()); };

            std::vector<std::uint8_t> zone1(spanning.begin(), spanning.begin() + 223);
            extractor.feed(zone1, 0U, 0U, emit);
            expect(eq(extractor.fragment().size(), 223UZ));

            std::vector<std::uint8_t> idleZone(223UZ, 0xAAU);
            extractor.feed(idleZone, kFhpOnlyIdleData, 1U, emit);
            expect(eq(extractor.counters().fragments_dropped, 1ULL));
            expect(eq(extractor.counters().idle_frames, 1ULL));
            expect(extractor.fragment().empty());

            std::vector<std::uint8_t> zone3 = follower;
            zone3.resize(223UZ, 0x00U);
            extractor.feed(std::span{zone3}.first(follower.size()), 0U, 2U, emit);
            expect(eq(got.size(), 1UZ));
            expect(got[0] == follower) << "the next packet is recovered from that zone's pointer alone";
        }
    };

    "7. a lost zone between two fragments costs one packet and no more"_test = [] {
        // A packet long enough to span three zones, with three packets behind it.
        constexpr std::size_t                        kZone = 223UZ;
        const std::vector<std::vector<std::uint8_t>> packets{makePacket(30U, 0U, 500UZ, 0x40U), makePacket(31U, 1U, 90UZ, 0x60U), makePacket(32U, 2U, 90UZ, 0x62U), makePacket(33U, 3U, 90UZ, 0x64U)};
        const Segmented                              laid = segment(packets, kZone);
        expect(eq(laid.zones.size(), 4UZ));
        expect(eq(static_cast<unsigned>(laid.pointers[1]), static_cast<unsigned>(kFhpNoPacketStart))) << "no packet starts in the second zone";

        const std::vector<std::vector<std::uint8_t>> tail{packets[1], packets[2], packets[3]};

        // Zone 2 dropped, and its frame count with it.
        {
            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            const auto                             emit = [&got](std::span<const std::uint8_t> packet) { got.emplace_back(packet.begin(), packet.end()); };
            extractor.feed(laid.zones[0], laid.pointers[0], 0U, emit);
            for (std::size_t z = 2UZ; z < laid.zones.size(); ++z) {
                extractor.feed(laid.zones[z], laid.pointers[z], static_cast<std::uint32_t>(z), emit);
            }
            expect(eq(extractor.counters().frames_lost, 1ULL));
            expect(eq(extractor.counters().fragments_dropped, 1ULL));
            expect(same(got, tail)) << "the spanning packet is lost and every packet behind it is not";
        }

        // The count continuous but the octets wrong: the zone that arrives in the gap's slot carries
        // the following zone's octets. The fragment then disagrees with the pointer, and 4.3.2.4 says
        // the pointer is correct.
        {
            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            const auto                             emit = [&got](std::span<const std::uint8_t> packet) { got.emplace_back(packet.begin(), packet.end()); };
            extractor.feed(laid.zones[0], laid.pointers[0], 0U, emit);
            extractor.feed(laid.zones[2], laid.pointers[2], 1U, emit);
            extractor.feed(laid.zones[3], laid.pointers[3], 2U, emit);
            expect(eq(extractor.counters().frames_lost, 0ULL));
            expect(eq(extractor.counters().pointer_mismatch, 1ULL));
            expect(eq(extractor.counters().fragments_dropped, 1ULL));
            expect(same(got, tail));
        }

        // The negative control: a reassembler that concatenates the surviving zones and cuts by the
        // first packet's own length puts its next packet boundary 60 octets away from where the second
        // packet actually starts, and never recovers.
        {
            std::vector<std::uint8_t> concatenated;
            for (const std::size_t z : {0UZ, 2UZ, 3UZ}) {
                concatenated.insert(concatenated.end(), laid.zones[z].begin(), laid.zones[z].end());
            }
            SpacePacketHeader head{};
            expect(parseSpacePacketHeader(concatenated, head) == ParseStatus::ok);
            const std::size_t firstTotal = totalPacketOctets(head);
            expect(eq(firstTotal, 506UZ));
            expect(lt(firstTotal, concatenated.size()));
            expect(!std::equal(packets[1].begin(), packets[1].begin() + 6, concatenated.begin() + static_cast<std::ptrdiff_t>(firstTotal))) << "ignoring the pointer loses the rest of the pass, not one packet";
        }
    };

    "a zone longer than 2046 octets has positions the pointer cannot name"_test = [] {
        // Derived in 132.0-B-3 4.1.2.7.6 from the eleven-bit width and its two reserved values: the
        // addressable positions are 0 to 2045. A packet starting at 2046 or 2047 of a 2048-octet zone
        // cannot be pointed at, and a pointer at or beyond the zone is the refusal that covers it.
        expect(eq(static_cast<unsigned>(kFhpOnlyIdleData), 2046U));
        expect(eq(static_cast<unsigned>(kFhpNoPacketStart), 2047U));

        const std::vector<std::uint8_t> packet = makePacket(90U, 0U, 100UZ, 0x77U);
        std::vector<std::uint8_t>       zone(2048UZ, 0xAAU);
        std::copy_n(packet.begin(), zone.size() - 2046UZ, zone.begin() + 2046);

        PacketExtractor                        extractor;
        std::vector<std::vector<std::uint8_t>> got;
        const auto                             emit = [&got](std::span<const std::uint8_t> p) { got.emplace_back(p.begin(), p.end()); };

        // 2046 is the only-idle-data value and 2047 says no packet starts here; neither can mean 2046.
        extractor.feed(zone, 2046U, 0U, emit);
        expect(eq(extractor.counters().idle_frames, 1ULL));
        expect(got.empty()) << "the position is unreachable, not misread";

        // A pointer of 2048 into a 2048-octet zone is the refusal the derivation predicts.
        extractor.feed(zone, 2048U, 1U, emit);
        expect(eq(extractor.counters().bad_pointer, 1ULL));
        expect(got.empty());
    };

    "8. a duplicate zone is not processed twice"_test = [] {
        const std::vector<std::vector<std::uint8_t>> packets{makePacket(40U, 0U, 60UZ, 0x70U), makePacket(41U, 1U, 60UZ, 0x80U)};
        const Segmented                              laid = segment(packets, 223UZ);

        PacketExtractor extractor;
        std::size_t     emitted = 0UZ;
        const auto      emit    = [&emitted](std::span<const std::uint8_t>) { ++emitted; };
        extractor.feed(laid.zones[0], laid.pointers[0], 7U, emit);
        const std::size_t afterFirst = emitted;
        extractor.feed(laid.zones[0], laid.pointers[0], 7U, emit);
        expect(eq(emitted, afterFirst)) << "the zone is not re-walked";
        expect(eq(extractor.counters().duplicate_frames, 1ULL));
    };

    "9. a primary header split across a zone boundary, at each of its five split points"_test = [] {
        for (const std::size_t inFirst : {1UZ, 2UZ, 3UZ, 4UZ, 5UZ}) {
            const std::vector<std::uint8_t> lead  = makePacket(50U, 0U, 40UZ, 0x90U);
            const std::vector<std::uint8_t> split = makePacket(51U, 1U, 70UZ, 0xA0U);

            std::vector<std::uint8_t> zone1 = lead;
            zone1.insert(zone1.end(), split.begin(), split.begin() + static_cast<std::ptrdiff_t>(inFirst));

            const std::vector<std::uint8_t> zone2(split.begin() + static_cast<std::ptrdiff_t>(inFirst), split.end());

            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            const auto                             emit = [&got](std::span<const std::uint8_t> packet) { got.emplace_back(packet.begin(), packet.end()); };

            extractor.feed(zone1, 0U, 0U, emit);
            expect(eq(extractor.fragment().size(), inFirst)) << "the fragment never exceeds five octets before its length is known";
            expect(eq(extractor.expected(), 0UZ)) << "the length is unknown until the sixth octet arrives";

            extractor.feed(zone2, kFhpNoPacketStart, 1U, emit);
            expect(eq(got.size(), 2UZ));
            expect(got[0] == lead);
            expect(got[1] == split) << "reassembled exactly across the split at " << inFirst;
        }
    };

    "10. bad pointers and orphans"_test = [] {
        const std::vector<std::uint8_t> packet = makePacket(60U, 0U, 50UZ, 0xB0U);
        std::vector<std::uint8_t>       zone   = packet;
        zone.resize(200UZ, 0x00U);

        for (const std::uint16_t pointer : {static_cast<std::uint16_t>(200U), static_cast<std::uint16_t>(201U)}) {
            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            const auto                             emit = [&got](std::span<const std::uint8_t> p) { got.emplace_back(p.begin(), p.end()); };
            extractor.feed(zone, pointer, 0U, emit);
            expect(eq(extractor.counters().bad_pointer, 1ULL)) << "pointer " << pointer;
            expect(got.empty());

            extractor.feed(std::span{zone}.first(packet.size()), 0U, 1U, emit);
            expect(eq(got.size(), 1UZ)) << "and the next zone extracts normally";
            expect(got[0] == packet);
        }

        // A nonzero pointer arriving first, with no prior state: the residue is orphaned and counted,
        // and the packet at the pointer comes out.
        {
            std::vector<std::uint8_t> leading(37UZ + packet.size(), 0xCCU);
            std::copy(packet.begin(), packet.end(), leading.begin() + 37);

            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            extractor.feed(leading, 37U, 0U, [&got](std::span<const std::uint8_t> p) { got.emplace_back(p.begin(), p.end()); });
            expect(eq(extractor.counters().orphan_octets, 37ULL));
            expect(eq(got.size(), 1UZ));
            expect(got[0] == packet);
        }
    };

    "11. the reassembly bound is arithmetic, and max_packet_length only tightens it"_test = [] {
        const std::vector<std::uint8_t> largest = makePacket(70U, 0U, kMaxPacketDataOctets, 0xD0U);
        expect(eq(largest.size(), kMaxPacketOctets));

        constexpr std::size_t kZone = 1115UZ;
        {
            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            const auto                             emit = [&got](std::span<const std::uint8_t> p) { got.emplace_back(p.begin(), p.end()); };

            std::uint32_t count = 0U;
            for (std::size_t base = 0UZ; base < largest.size(); base += kZone) {
                const std::size_t         length = std::min(kZone, largest.size() - base);
                std::vector<std::uint8_t> zone(largest.begin() + static_cast<std::ptrdiff_t>(base), largest.begin() + static_cast<std::ptrdiff_t>(base + length));
                extractor.feed(zone, base == 0UZ ? std::uint16_t{0U} : kFhpNoPacketStart, count, emit);
                expect(le(extractor.fragment().size(), extractor.config().max_packet_length)) << "the held fragment never exceeds the bound";
                count = (count + 1U) % kTmCountModulus;
            }
            expect(eq(got.size(), 1UZ));
            expect(eq(got[0].size(), kMaxPacketOctets));
            expect(got[0] == largest);
        }

        // The same input at max_packet_length = 1024: refused once, the rest of the zone discarded,
        // and the next zone's pointer recovering normally.
        {
            PacketExtractor                        extractor{PacketExtractor::Config{.max_packet_length = 1024UZ, .count_modulus = kTmCountModulus}};
            std::vector<std::vector<std::uint8_t>> got;
            const auto                             emit = [&got](std::span<const std::uint8_t> p) { got.emplace_back(p.begin(), p.end()); };

            std::vector<std::uint8_t> zone(largest.begin(), largest.begin() + static_cast<std::ptrdiff_t>(kZone));
            extractor.feed(zone, 0U, 0U, emit);
            expect(eq(extractor.counters().oversize_dropped, 1ULL));
            expect(got.empty());

            const std::vector<std::uint8_t> small = makePacket(71U, 1U, 100UZ, 0xE0U);
            extractor.feed(small, 0U, 1U, emit);
            expect(eq(got.size(), 1UZ));
            expect(got[0] == small);
        }

        // The bound can be tightened and never loosened.
        const PacketExtractor loose{PacketExtractor::Config{.max_packet_length = kMaxPacketOctets * 4UZ, .count_modulus = kTmCountModulus}};
        expect(eq(loose.config().max_packet_length, kMaxPacketOctets));
    };

    "reset returns the machine to its opening state"_test = [] {
        const std::vector<std::uint8_t> packet = makePacket(80U, 0U, 400UZ, 0xF0U);
        PacketExtractor                 extractor;
        std::size_t                     emitted = 0UZ;
        const auto                      emit    = [&emitted](std::span<const std::uint8_t>) { ++emitted; };

        extractor.feed(std::span{packet}.first(200UZ), 0U, 0U, emit);
        expect(!extractor.fragment().empty());
        extractor.reset();
        expect(extractor.fragment().empty());
        expect(extractor.counters() == PacketExtractor::Counters{});

        // After a reset the first zone is the machine's first zone: no gap is reported against a count
        // it never saw.
        extractor.feed(packet, 0U, 200U, emit);
        expect(eq(emitted, 1UZ));
        expect(eq(extractor.counters().frames_lost, 0ULL));
    };

    "a seeded soak: random packet lengths through random zone lengths"_test = [] {
        for (std::size_t trial = 0UZ; trial < 40UZ; ++trial) {
            const std::size_t                      zoneLength = 64UZ + (next() % 1500UZ);
            std::vector<std::vector<std::uint8_t>> packets;
            for (std::size_t i = 0UZ; i < 25UZ; ++i) {
                packets.push_back(makePacket(static_cast<std::uint16_t>(1U + (next() % 2000U)), static_cast<std::uint16_t>(i), 1UZ + (next() % 3000UZ), static_cast<std::uint8_t>(next())));
            }
            const Segmented laid = segment(packets, zoneLength);

            PacketExtractor                        extractor;
            std::vector<std::vector<std::uint8_t>> got;
            std::uint32_t                          count = 0U;
            for (std::size_t z = 0UZ; z < laid.zones.size(); ++z) {
                extractor.feed(laid.zones[z], laid.pointers[z], count, [&got](std::span<const std::uint8_t> p) { got.emplace_back(p.begin(), p.end()); });
                count = (count + 1U) % kTmCountModulus;
            }

            std::vector<std::vector<std::uint8_t>> expected;
            for (const std::vector<std::uint8_t>& packet : packets) {
                SpacePacketHeader header{};
                static_cast<void>(parseSpacePacketHeader(packet, header));
                if (!isIdlePacket(header)) {
                    expected.push_back(packet);
                }
            }
            expect(same(got, expected)) << "zone length " << zoneLength;
        }
    };
};

int main() { /* tests are automatically registered and run */ }
