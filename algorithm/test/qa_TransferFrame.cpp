#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/ccsds/TransferFrame.hpp>

/*
 * The three transfer-frame headers, the CLCW and the M_PDU header are pinned three ways, and none of
 * them is the implementation restated:
 *
 *   - every field round trips through write and parse over its full declared range, and a field at
 *     its maximum leaves its neighbors alone while one past its maximum is refused, which is what
 *     asserts the widths as consequences rather than as comments;
 *   - nine header octet literals assembled by hand from the standards' bit positions, so a boundary
 *     that moves by one bit fails here and nowhere else. The pair that matters most is the first
 *     octet of a TM and an AOS header carrying the same spacecraft identifier of 42: 0x02 against
 *     0x4A, because the AOS version is '01' and its identifier is eight bits where TM's is ten;
 *   - the only-idle-data fill against the ten octets 132.0-B-3 publishes in its own NOTE, generated
 *     from the polynomial and from nothing else, in both the Fibonacci and the Galois realization.
 *
 * The frame-count arithmetic is asserted at the wrap, where the number is not obvious: a count of 3
 * after a count of 254 at modulus 256 is four frames lost, not 251 and not an error.
 */
namespace {

using namespace gr::ccsds;

std::uint64_t rng = 0x9E3779B97F4A7C15ULL;

std::uint32_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(rng >> 32U);
}

/// The Galois realization of 132.0-B-3 4.1.4.6.2.2, written here rather than in the header so the
/// two forms are independent implementations of one sequence.
std::vector<std::uint8_t> galoisOidFill(std::size_t octets) {
    std::vector<std::uint8_t> out(octets);
    std::uint32_t             state = OidFill::kGaloisSeed;
    for (std::uint8_t& octet : out) {
        std::uint32_t value = 0U;
        for (std::size_t bit = 0UZ; bit < 8UZ; ++bit) {
            const std::uint32_t output = state & 1U;
            state >>= 1U;
            if (output != 0U) {
                state ^= OidFill::kGaloisMask;
            }
            value = (value << 1U) | output;
        }
        octet = static_cast<std::uint8_t>(value);
    }
    return out;
}

} // namespace

const boost::ut::suite<"ccsds transfer frames"> transferFrameTests = [] {
    using namespace boost::ut;

    "1a. the TM primary header round trips over every field's declared range"_test = [] {
        for (std::size_t trial = 0UZ; trial < 4000UZ; ++trial) {
            TmPrimaryHeader header{};
            header.spacecraft_id        = static_cast<std::uint16_t>(next() % 1024U);
            header.virtual_channel      = static_cast<std::uint8_t>(next() % 8U);
            header.ocf_present          = (next() & 1U) != 0U;
            header.master_frame_count   = static_cast<std::uint8_t>(next() % 256U);
            header.vc_frame_count       = static_cast<std::uint8_t>(next() % 256U);
            header.secondary_header     = (next() & 1U) != 0U;
            header.sync_flag            = true; // leaves the data field status bits unconstrained
            header.packet_order         = (next() & 1U) != 0U;
            header.segment_length_id    = static_cast<std::uint8_t>(next() % 4U);
            header.first_header_pointer = static_cast<std::uint16_t>(next() % 2048U);

            std::array<std::uint8_t, kTmPrimaryHeaderSize> octets{};
            expect(writeTmPrimaryHeader(header, octets) == WriteStatus::ok);
            TmPrimaryHeader   back{};
            const ParseStatus status = parseTmPrimaryHeader(octets, back);
            expect(status == ParseStatus::ok || status == ParseStatus::reserved_violation);
            expect(back == header);

            std::array<std::uint8_t, kTmPrimaryHeaderSize> again{};
            expect(writeTmPrimaryHeader(back, again) == WriteStatus::ok);
            expect(again == octets);
        }

        // The extremes, and one past each of them.
        TmPrimaryHeader maxima{};
        maxima.spacecraft_id        = 1023U;
        maxima.virtual_channel      = 7U;
        maxima.segment_length_id    = 3U;
        maxima.first_header_pointer = 2047U;
        maxima.master_frame_count   = 255U;
        maxima.vc_frame_count       = 255U;
        std::array<std::uint8_t, kTmPrimaryHeaderSize> at{};
        expect(writeTmPrimaryHeader(maxima, at) == WriteStatus::ok);
        TmPrimaryHeader back{};
        expect(parseTmPrimaryHeader(at, back) != ParseStatus::bad_version);
        expect(back == maxima) << "a field at its maximum must not disturb its neighbors";

        for (const TmPrimaryHeader& over : {TmPrimaryHeader{.spacecraft_id = 1024U}, TmPrimaryHeader{.virtual_channel = 8U}, TmPrimaryHeader{.segment_length_id = 4U}, TmPrimaryHeader{.first_header_pointer = 2048U}, TmPrimaryHeader{.version = 4U}}) {
            std::array<std::uint8_t, kTmPrimaryHeaderSize> octets{};
            expect(writeTmPrimaryHeader(over, octets) == WriteStatus::field_out_of_range);
        }
        std::array<std::uint8_t, kTmPrimaryHeaderSize - 1UZ> shortSpan{};
        expect(writeTmPrimaryHeader(maxima, shortSpan) == WriteStatus::short_span);
        expect(parseTmPrimaryHeader(shortSpan, back) == ParseStatus::short_span);
    };

    "1b. the AOS primary header round trips, with and without the frame header error control"_test = [] {
        for (std::size_t trial = 0UZ; trial < 4000UZ; ++trial) {
            AosPrimaryHeader header{};
            header.spacecraft_id              = static_cast<std::uint8_t>(next() % 256U);
            header.virtual_channel            = static_cast<std::uint8_t>(next() % 64U);
            header.vc_frame_count             = next() % (1U << 24U);
            header.replay                     = (next() & 1U) != 0U;
            header.vc_count_cycle_used        = true;
            header.vc_count_cycle             = static_cast<std::uint8_t>(next() % 16U);
            header.has_fhec                   = (next() & 1U) != 0U;
            header.frame_header_error_control = header.has_fhec ? static_cast<std::uint16_t>(next() % 65536U) : std::uint16_t{0U};

            std::array<std::uint8_t, kAosPrimaryHeaderFhecSize> octets{};
            const std::size_t                                   size = aosPrimaryHeaderOctets(header.has_fhec);
            expect(writeAosPrimaryHeader(header, std::span{octets}.first(size)) == WriteStatus::ok);
            AosPrimaryHeader back{};
            expect(parseAosPrimaryHeader(std::span<const std::uint8_t>{octets}.first(size), header.has_fhec, back) == ParseStatus::ok);
            expect(back == header);
        }

        AosPrimaryHeader maxima{};
        maxima.spacecraft_id       = 255U;
        maxima.virtual_channel     = 63U;
        maxima.vc_frame_count      = (1U << 24U) - 1U;
        maxima.vc_count_cycle_used = true;
        maxima.vc_count_cycle      = 15U;
        std::array<std::uint8_t, kAosPrimaryHeaderFhecSize> buf{};
        expect(writeAosPrimaryHeader(maxima, buf) == WriteStatus::ok);
        AosPrimaryHeader back{};
        expect(parseAosPrimaryHeader(std::span<const std::uint8_t>{buf}.first(kAosPrimaryHeaderSize), false, back) == ParseStatus::ok);
        expect(eq(back.vc_frame_count, maxima.vc_frame_count));
        expect(eq(static_cast<unsigned>(back.virtual_channel), 63U));

        for (const AosPrimaryHeader& over : {AosPrimaryHeader{.version = 4U}, AosPrimaryHeader{.virtual_channel = 64U}, AosPrimaryHeader{.vc_frame_count = 1U << 24U}, AosPrimaryHeader{.reserved = 4U}, AosPrimaryHeader{.vc_count_cycle = 16U}}) {
            std::array<std::uint8_t, kAosPrimaryHeaderFhecSize> octets{};
            expect(writeAosPrimaryHeader(over, octets) == WriteStatus::field_out_of_range);
        }
    };

    "1c. the TC primary header, the CLCW and the M_PDU header round trip"_test = [] {
        for (std::size_t trial = 0UZ; trial < 3000UZ; ++trial) {
            TcPrimaryHeader header{};
            header.bypass          = (next() & 1U) != 0U;
            header.control_command = (next() & 1U) != 0U;
            header.spacecraft_id   = static_cast<std::uint16_t>(next() % 1024U);
            header.virtual_channel = static_cast<std::uint8_t>(next() % 64U);
            header.frame_length    = static_cast<std::uint16_t>(next() % 1024U);
            header.sequence_number = static_cast<std::uint8_t>(next() % 256U);

            std::array<std::uint8_t, kTcPrimaryHeaderSize> octets{};
            expect(writeTcPrimaryHeader(header, octets) == WriteStatus::ok);
            TcPrimaryHeader back{};
            expect(parseTcPrimaryHeader(octets, back) == ParseStatus::ok);
            expect(back == header);

            Clcw clcw{};
            clcw.status          = static_cast<std::uint8_t>(next() % 8U);
            clcw.cop_in_effect   = static_cast<std::uint8_t>(next() % 4U);
            clcw.virtual_channel = static_cast<std::uint8_t>(next() % 64U);
            clcw.reserved        = static_cast<std::uint8_t>(next() % 4U);
            clcw.no_rf_available = (next() & 1U) != 0U;
            clcw.no_bit_lock     = (next() & 1U) != 0U;
            clcw.lockout         = (next() & 1U) != 0U;
            clcw.wait            = (next() & 1U) != 0U;
            clcw.retransmit      = (next() & 1U) != 0U;
            clcw.farm_b_counter  = static_cast<std::uint8_t>(next() % 4U);
            clcw.report_value    = static_cast<std::uint8_t>(next() % 256U);

            std::array<std::uint8_t, kClcwSize> clcwOctets{};
            expect(writeClcw(clcw, clcwOctets) == WriteStatus::ok);
            Clcw clcwBack{};
            expect(parseClcw(clcwOctets, clcwBack) == ParseStatus::ok);
            expect(clcwBack == clcw);
            expect(ocfReportType(clcwOctets) == OcfReportType::type_1_clcw);

            MpduHeader mpdu{};
            mpdu.first_header_pointer = static_cast<std::uint16_t>(next() % 2048U);
            std::array<std::uint8_t, kMpduHeaderSize> mpduOctets{};
            expect(writeMpduHeader(mpdu, mpduOctets) == WriteStatus::ok);
            MpduHeader mpduBack{};
            expect(parseMpduHeader(mpduOctets, mpduBack) == ParseStatus::ok);
            expect(mpduBack == mpdu);
        }

        std::array<std::uint8_t, kTcPrimaryHeaderSize> tcBuf{};
        expect(writeTcPrimaryHeader(TcPrimaryHeader{.frame_length = 1024U}, tcBuf) == WriteStatus::field_out_of_range);
        expect(writeTcPrimaryHeader(TcPrimaryHeader{.spacecraft_id = 1024U}, tcBuf) == WriteStatus::field_out_of_range);
        std::array<std::uint8_t, kMpduHeaderSize> mpduBuf{};
        expect(writeMpduHeader(MpduHeader{.first_header_pointer = 2048U}, mpduBuf) == WriteStatus::field_out_of_range);
        expect(writeMpduHeader(MpduHeader{.reserved = 32U}, mpduBuf) == WriteStatus::field_out_of_range);
    };

    "2. the octet literals, assembled by hand from the standards' bit positions"_test = [] {
        // TM: TFVN 00, SCID 42, VCID 1, OCF set, MC count 200, VC count 17, TFSH clear, sync clear,
        // packet order clear, segment length '11', FHP 0.
        TmPrimaryHeader tm{};
        tm.spacecraft_id      = 42U;
        tm.virtual_channel    = 1U;
        tm.ocf_present        = true;
        tm.master_frame_count = 200U;
        tm.vc_frame_count     = 17U;
        tm.segment_length_id  = 3U;
        std::array<std::uint8_t, kTmPrimaryHeaderSize> tmOctets{};
        expect(writeTmPrimaryHeader(tm, tmOctets) == WriteStatus::ok);
        expect(tmOctets == std::array<std::uint8_t, 6UZ>{0x02U, 0xA3U, 0xC8U, 0x11U, 0x18U, 0x00U});

        tm.first_header_pointer = kFhpNoPacketStart;
        expect(writeTmPrimaryHeader(tm, tmOctets) == WriteStatus::ok);
        expect(tmOctets == std::array<std::uint8_t, 6UZ>{0x02U, 0xA3U, 0xC8U, 0x11U, 0x1FU, 0xFFU});

        tm.first_header_pointer = kFhpOnlyIdleData;
        expect(writeTmPrimaryHeader(tm, tmOctets) == WriteStatus::ok);
        expect(tmOctets == std::array<std::uint8_t, 6UZ>{0x02U, 0xA3U, 0xC8U, 0x11U, 0x1FU, 0xFEU});

        // AOS: TFVN 01, SCID 42, VCID 63, VC count 0xFFFFFF, replay set, cycle use set, cycle 15.
        AosPrimaryHeader aos{};
        aos.spacecraft_id       = 42U;
        aos.virtual_channel     = 63U;
        aos.vc_frame_count      = 0xFFFFFFU;
        aos.replay              = true;
        aos.vc_count_cycle_used = true;
        aos.vc_count_cycle      = 15U;
        std::array<std::uint8_t, kAosPrimaryHeaderSize> aosOctets{};
        expect(writeAosPrimaryHeader(aos, aosOctets) == WriteStatus::ok);
        expect(aosOctets == std::array<std::uint8_t, 6UZ>{0x4AU, 0xBFU, 0xFFU, 0xFFU, 0xFFU, 0xCFU});
        expect(eq(static_cast<unsigned>(tmOctets[0]), 0x02U)) << "the same spacecraft identifier of 42";
        expect(eq(static_cast<unsigned>(aosOctets[0]), 0x4AU)) << "one octet is the mistake a shared parser makes";

        // M_PDU header with FHP 2047: the five reserved bits sit above the pointer.
        std::array<std::uint8_t, kMpduHeaderSize> mpduOctets{};
        expect(writeMpduHeader(MpduHeader{.first_header_pointer = kFhpNoPacketStart}, mpduOctets) == WriteStatus::ok);
        expect(mpduOctets == std::array<std::uint8_t, 2UZ>{0x07U, 0xFFU});

        // TC: SCID 42, VCID 1, frame length field 999, sequence number 7 -- a 1000-octet frame.
        TcPrimaryHeader tc{};
        tc.spacecraft_id   = 42U;
        tc.virtual_channel = 1U;
        tc.frame_length    = 999U;
        tc.sequence_number = 7U;
        std::array<std::uint8_t, kTcPrimaryHeaderSize> tcOctets{};
        expect(writeTcPrimaryHeader(tc, tcOctets) == WriteStatus::ok);
        expect(tcOctets == std::array<std::uint8_t, 5UZ>{0x00U, 0x2AU, 0x07U, 0xE7U, 0x07U});
        expect(eq(totalTcFrameOctets(tc), 1000UZ));

        // A TM secondary header identification octet with version '00' and length field 63.
        std::array<std::uint8_t, kTmSecondaryHeaderIdSize> tfshOctets{};
        expect(writeTmSecondaryHeaderId(TmSecondaryHeaderId{.length = 63U}, tfshOctets) == WriteStatus::ok);
        expect(eq(static_cast<unsigned>(tfshOctets[0]), 0x3FU));

        // A CLCW with COP in effect 1, VCID 1, every flag set, FARM-B 3 and report value 255.
        Clcw clcw{};
        clcw.cop_in_effect   = 1U;
        clcw.virtual_channel = 1U;
        clcw.no_rf_available = true;
        clcw.no_bit_lock     = true;
        clcw.lockout         = true;
        clcw.wait            = true;
        clcw.retransmit      = true;
        clcw.farm_b_counter  = 3U;
        clcw.report_value    = 255U;
        std::array<std::uint8_t, kClcwSize> clcwOctets{};
        expect(writeClcw(clcw, clcwOctets) == WriteStatus::ok);
        expect(clcwOctets == std::array<std::uint8_t, 4UZ>{0x01U, 0x04U, 0xFEU, 0xFFU});
    };

    "4a. the two minus-one fields this header carries"_test = [] {
        // 232.0-B-4 4.1.2.7.2: a 1000-octet TC frame carries a frame length field of 999.
        expect(eq(totalTcFrameOctets(TcPrimaryHeader{.frame_length = 999U}), 1000UZ));
        expect(eq(totalTcFrameOctets(TcPrimaryHeader{.frame_length = 0U}), 1UZ));
        expect(eq(totalTcFrameOctets(TcPrimaryHeader{.frame_length = 1023U}), 1024UZ)) << "ten bits bound the frame at 2^10 octets";
        expect(eq(1024UZ - kTcPrimaryHeaderSize, 1019UZ)) << "232.0-B-4 4.1.1.1 b) with no error control field";
        expect(eq(1024UZ - kTcPrimaryHeaderSize - kFecfSize, 1017UZ)) << "and with one";

        // The off-by-one variant: a field of 1000 describes a 1001-octet frame, one octet past the
        // frame that was built, and the next frame boundary is then one octet late.
        expect(eq(totalTcFrameOctets(TcPrimaryHeader{.frame_length = 1000U}), 1001UZ));

        // 132.0-B-3 4.1.3.2.3.2: a 64-octet secondary header carries a length field of 63.
        expect(eq(tmSecondaryHeaderOctets(TmSecondaryHeaderId{.length = 63U}), 64UZ));
        expect(eq(tmSecondaryHeaderOctets(TmSecondaryHeaderId{.length = 1U}), 2UZ)) << "the shortest header the standard admits";
        expect(!tmSecondaryHeaderUsable(TmSecondaryHeaderId{.length = 0U})) << "a one-octet secondary header with no data field cannot exist";
        expect(tmSecondaryHeaderUsable(TmSecondaryHeaderId{.length = 1U}));
        expect(eq(tmSecondaryHeaderOctets(TmSecondaryHeaderId{.length = 64U - 1U}) - 1UZ, 63UZ)) << "and the data field is 20 - 1 = 19 octets for a 20-octet header";
        expect(eq(tmSecondaryHeaderOctets(TmSecondaryHeaderId{.length = 19U}), 20UZ));
    };

    "8. frame count gaps, and the wrap that the modular subtraction never notices"_test = [] {
        for (const std::uint32_t modulus : {kTmCountModulus, kAosCountModulus, kAosCycleCountModulus}) {
            expect(frameGap(11U, 10U, modulus).continuous);
            expect(eq(frameGap(11U, 10U, modulus).lost, 0U));
            expect(frameGap(10U, 10U, modulus).duplicate);
            expect(eq(frameGap(15U, 10U, modulus).lost, 4U)) << "a run missing four frames";
            expect(!frameGap(15U, 10U, modulus).continuous);
            // The wrap: the count returns to zero and the arithmetic does not change.
            expect(frameGap(0U, modulus - 1U, modulus).continuous);
            expect(eq(frameGap(4U, modulus - 1U, modulus).lost, 4U));
        }

        // 254, 255, 0, 1 at modulus 256 loses nothing; 254 then 3 loses four.
        std::uint32_t last = 254U;
        for (const std::uint32_t count : {255U, 0U, 1U}) {
            expect(frameGap(count, last, kTmCountModulus).continuous);
            last = count;
        }
        expect(eq(frameGap(3U, 254U, kTmCountModulus).lost, 4U));

        // The AOS cycle. With the use flag set the count is 28 bits wide, so 0xFFFFFF followed by
        // cycle 1 count 0 is continuous; with the flag clear the same two counts are continuous at
        // the 24-bit modulus too, and cycle 2 under the flag is a gap of 2^24.
        AosPrimaryHeader a{};
        a.vc_count_cycle_used = true;
        a.vc_frame_count      = 0xFFFFFFU;
        a.vc_count_cycle      = 0U;
        AosPrimaryHeader b    = a;
        b.vc_frame_count      = 0U;
        b.vc_count_cycle      = 1U;
        expect(eq(aosCountModulus(a), kAosCycleCountModulus));
        expect(frameGap(aosWidenedFrameCount(b), aosWidenedFrameCount(a), aosCountModulus(a)).continuous);

        AosPrimaryHeader plain     = a;
        plain.vc_count_cycle_used  = false;
        plain.vc_count_cycle       = 0U;
        AosPrimaryHeader plainNext = plain;
        plainNext.vc_frame_count   = 0U;
        expect(eq(aosCountModulus(plain), kAosCountModulus));
        expect(frameGap(aosWidenedFrameCount(plainNext), aosWidenedFrameCount(plain), aosCountModulus(plain)).continuous);

        AosPrimaryHeader far = b;
        far.vc_count_cycle   = 2U;
        expect(eq(frameGap(aosWidenedFrameCount(far), aosWidenedFrameCount(a), aosCountModulus(a)).lost, kAosCountModulus)) << "the number the widening exists to produce";
    };

    "12. the only-idle-data fill reproduces the standard's published anchor"_test = [] {
        constexpr std::array<std::uint8_t, 10UZ> anchor{0xFFU, 0xFFU, 0xFFU, 0xFFU, 0x6DU, 0xB6U, 0xD8U, 0x61U, 0x45U, 0x1FU};

        OidFill                   fill;
        std::vector<std::uint8_t> produced(anchor.size());
        fill.next(produced);
        expect(std::equal(anchor.begin(), anchor.end(), produced.begin())) << "132.0-B-3's own NOTE to 4.1.4.6.2.2";

        // The register runs across calls rather than restarting per frame (4.1.4.6.2.1), so the same
        // sequence comes out whether it is drawn in one span or in several.
        OidFill                   piecewise;
        std::vector<std::uint8_t> assembled;
        for (const std::size_t chunk : {3UZ, 1UZ, 6UZ, 22UZ}) {
            std::vector<std::uint8_t> part(chunk);
            piecewise.next(part);
            assembled.insert(assembled.end(), part.begin(), part.end());
        }
        OidFill                   whole;
        std::vector<std::uint8_t> straight(assembled.size());
        whole.next(straight);
        expect(assembled == straight);

        // And the Galois realization of 4.1.4.6.2.2, seeded with the bit pattern that subsection
        // publishes, produces the same sequence -- which is what says the two forms are one code.
        expect(galoisOidFill(assembled.size()) == straight);

        // Reset returns the register to the all-ones seed of 4.1.4.6.2.1.
        piecewise.reset();
        std::vector<std::uint8_t> again(anchor.size());
        piecewise.next(again);
        expect(std::equal(anchor.begin(), anchor.end(), again.begin()));
    };

    "14a. the refusals a decoder is built from, each named"_test = [] {
        // A wrong version number is fatal and is the pair a shared parser confuses.
        std::array<std::uint8_t, kTmPrimaryHeaderSize> octets{};
        AosPrimaryHeader                               aos{};
        aos.spacecraft_id = 42U;
        expect(writeAosPrimaryHeader(aos, octets) == WriteStatus::ok);
        TmPrimaryHeader tm{};
        expect(parseTmPrimaryHeader(octets, tm) == ParseStatus::bad_version) << "AOS octets given to the TM parser";

        TmPrimaryHeader tmSource{};
        tmSource.spacecraft_id = 42U;
        tmSource.sync_flag     = true;
        expect(writeTmPrimaryHeader(tmSource, octets) == WriteStatus::ok);
        AosPrimaryHeader aosBack{};
        expect(parseAosPrimaryHeader(octets, false, aosBack) == ParseStatus::bad_version) << "and TM octets given to the AOS parser";

        // A reserved spare set wrong reports and refuses nothing: the header is complete beside it.
        TmPrimaryHeader reserved{};
        reserved.segment_length_id = 1U; // not '11' under a zero synchronization flag
        expect(writeTmPrimaryHeader(reserved, octets) == WriteStatus::ok);
        TmPrimaryHeader back{};
        expect(parseTmPrimaryHeader(octets, back) == ParseStatus::reserved_violation);
        expect(back == reserved) << "the parsed header is usable beside the violation";

        MpduHeader mpduReserved{};
        mpduReserved.reserved             = 1U;
        mpduReserved.first_header_pointer = 5U;
        std::array<std::uint8_t, kMpduHeaderSize> mpduOctets{};
        expect(writeMpduHeader(mpduReserved, mpduOctets) == WriteStatus::ok);
        MpduHeader mpduBack{};
        expect(parseMpduHeader(mpduOctets, mpduBack) == ParseStatus::reserved_violation);
        expect(eq(static_cast<unsigned>(mpduBack.first_header_pointer), 5U));

        // A Type-2 report is not a CLCW, and the type flag says so before anything else is read.
        std::array<std::uint8_t, kClcwSize> type2{0x80U, 0x00U, 0x00U, 0x00U};
        expect(ocfReportType(type2) == OcfReportType::type_2_project);
        std::array<std::uint8_t, kClcwSize> sdls{0xC0U, 0x00U, 0x00U, 0x00U};
        expect(ocfReportType(sdls) == OcfReportType::type_2_sdls);
        Clcw clcw{};
        expect(parseClcw(type2, clcw) == ParseStatus::bad_version);
    };

    "the TC frame type is reported, and the reserved combination is not refused"_test = [] {
        expect(tcFrameType(TcPrimaryHeader{}) == TcFrameType::type_ad);
        expect(tcFrameType(TcPrimaryHeader{.control_command = true}) == TcFrameType::reserved);
        expect(tcFrameType(TcPrimaryHeader{.bypass = true}) == TcFrameType::type_bd);
        expect(tcFrameType(TcPrimaryHeader{.bypass = true, .control_command = true}) == TcFrameType::type_bc);

        std::array<std::uint8_t, kTcPrimaryHeaderSize> octets{};
        expect(writeTcPrimaryHeader(TcPrimaryHeader{.control_command = true, .frame_length = 9U}, octets) == WriteStatus::ok);
        TcPrimaryHeader back{};
        expect(parseTcPrimaryHeader(octets, back) == ParseStatus::ok) << "an unknown combination is still a frame";
        expect(tcFrameType(back) == TcFrameType::reserved);
    };
};

int main() { /* tests are automatically registered and run */ }
