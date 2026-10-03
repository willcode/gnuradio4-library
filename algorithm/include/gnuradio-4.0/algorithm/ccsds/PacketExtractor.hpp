#ifndef GNURADIO_ALGORITHM_CCSDS_PACKET_EXTRACTOR_HPP
#define GNURADIO_ALGORITHM_CCSDS_PACKET_EXTRACTOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/ccsds/SpacePacket.hpp>
#include <gnuradio-4.0/algorithm/ccsds/TransferFrame.hpp>

/**
 * @brief One virtual channel's Packet Extraction Function, CCSDS 132.0-B-3 4.3.2.
 *
 * Space packets lie end to end through the data fields of the transfer frames. Each frame carries
 * **one pointer** to the start of the first packet that begins in the same frame. 4.1.2.7.6.3 NOTE 1
 * gives the reason for the pointer. It delimits variable-length packets *without* requiring that the
 * previous frame arrived. This class implements that rule.
 *
 * The extractor takes a **zone** and a **pointer** in place of a frame. The two protocols that carry
 * packets this way differ only in those two things. For TM the zone is the transfer frame data field.
 * The pointer is a field of the primary header (4.1.2.7.6). For AOS the zone is the data field less the
 * two-octet M_PDU header. The pointer is a field of that header (732.0-B-4 4.1.4.2.3.3). That clause
 * says "the first octet in this zone is assigned the number 0", which accounts for the offset of two.
 * Each protocol has one adapter, and both use this one extractor.
 *
 * **132.0-B-3 4.3.2.4 is mandatory, and the design follows it.** *"If the calculated location of the
 * beginning of the first Packet is not consistent with the location indicated by the First Header
 * Pointer, then the Packet Extraction Function shall assume that the First Header Pointer is correct,
 * and shall continue the extraction based on that assumption."* The pointer takes precedence over the
 * held fragment's own length. One lost frame therefore loses one packet and not the rest of the pass.
 * After a loss, the next frame whose pointer is neither reserved value sets a new packet boundary from
 * that frame alone.
 *
 * **The state is bounded by construction.** Between calls the held fragment is at most
 * `max_packet_length`. That is at most `kMaxPacketOctets` = 65 542, derived from the sixteen-bit packet
 * data length field. A primary header split across a zone boundary grows a fragment with no known
 * length. That fragment is at most five octets, because a sixth octet completes the header. Within one
 * call, a whole-zone continuation is the exception. It appends the zone before the length that trims it
 * is known. The buffer then peaks below `max_packet_length + zone.size()`. It is trimmed back before the
 * call returns. The buffer is reserved for the settled bound. Only that peak can reallocate. Received
 * data does not change either bound.
 *
 * A completed packet is passed to a callable and not returned. The class builds no record and names no
 * framework type. The buffer is reserved once. An ordinary zone allocates nothing.
 */
namespace gr::ccsds {

class PacketExtractor {
public:
    struct Config {
        /// The longest packet that will be assembled. It defaults to `kMaxPacketOctets` and is clamped to
        /// it. A mission with short packets can lower the bound. It cannot raise it.
        std::size_t max_packet_length = kMaxPacketOctets;

        /// The modulus of the frame count the caller supplies. It is `kTmCountModulus` for TM. For AOS it
        /// is `kAosCountModulus` or `kAosCycleCountModulus`, and `aosCountModulus` picks between them.
        std::uint32_t count_modulus = kTmCountModulus;
    };

    struct Counters {
        std::uint64_t packets           = 0ULL; //!< emitted
        std::uint64_t idle_packets      = 0ULL; //!< APID 2047, recognized and discarded (132.0-B-3 4.3.2.5 NOTE 1)
        std::uint64_t idle_frames       = 0ULL; //!< only-idle-data zones discarded (4.1.4.6)
        std::uint64_t frames_lost       = 0ULL; //!< from the frame count gap
        std::uint64_t duplicate_frames  = 0ULL; //!< a gap of zero, the zone not processed twice
        std::uint64_t fragments_dropped = 0ULL; //!< a held fragment abandoned
        std::uint64_t pointer_mismatch  = 0ULL; //!< the pointer disagreed with the fragment's length (4.3.2.4)
        std::uint64_t bad_pointer       = 0ULL; //!< a pointer at or beyond the zone
        std::uint64_t orphan_octets     = 0ULL; //!< residue with no fragment to complete
        std::uint64_t oversize_dropped  = 0ULL; //!< a packet longer than `max_packet_length`
        std::uint64_t bad_version       = 0ULL; //!< a packet header whose version is not `'000'`, its length not trusted

        [[nodiscard]] bool operator==(const Counters&) const noexcept = default;
    };

    PacketExtractor() : PacketExtractor(Config{}) {}

    explicit PacketExtractor(Config config) : config_{config.max_packet_length == 0UZ ? kMaxPacketOctets : std::min(config.max_packet_length, kMaxPacketOctets), config.count_modulus} { partial_.reserve(config_.max_packet_length); }

    /**
     * @brief Take one zone and the pointer that came with it.
     *
     * `zone` is the TM data field, or the AOS M_PDU packet zone with its header already removed.
     * `fhp` is the first header pointer that arrived with the zone. `count` is the virtual channel
     * frame count, already widened where the AOS cycle field applies. Each completed packet is passed
     * whole to `emit`, with its primary header.
     */
    template<typename Emit>
    void feed(std::span<const std::uint8_t> zone, std::uint16_t fhp, std::uint32_t count, Emit&& emit) {
        // 1. Continuity. The modular subtraction handles the count wrap.
        if (have_count_) {
            const SequenceGap gap = frameGap(count, last_count_, config_.count_modulus);
            if (gap.duplicate) {
                ++counters_.duplicate_frames;
                return; // processing the zone again would emit each of its packets twice
            }
            if (!gap.continuous) {
                counters_.frames_lost += gap.lost;
                dropFragment(); // the octets that complete the fragment were in the lost frames
            }
        }
        last_count_ = count;
        have_count_ = true;

        // 2. Only idle data. 4.1.4.6.3 NOTE 3 permits an OID frame in the middle of a split packet.
        //    The fill does not continue the packet. Appending it would corrupt the packet with no error.
        if (fhp == kFhpOnlyIdleData) {
            ++counters_.idle_frames;
            dropFragment();
            return;
        }

        // 3. No packet starts here. The whole zone continues a packet begun earlier (4.1.2.7.6.4 NOTE).
        if (fhp == kFhpNoPacketStart) {
            if (partial_.empty()) {
                counters_.orphan_octets += zone.size();
                return;
            }
            append(zone); // settleFragment trims this to the length in the fragment's own header
            std::size_t resume = 0UZ;
            if (settleFragment(zone.size(), resume, emit)) {
                walk(zone, resume, emit);
            }
            return;
        }

        // 4a. Out of range. The pointer lies at or past the end of the zone.
        if (fhp >= zone.size()) {
            ++counters_.bad_pointer;
            dropFragment();
            return;
        }

        // 4b. The residue, and the pointer's authority (4.3.2.4).
        if (!partial_.empty()) {
            append(zone.first(fhp));
            const bool resolved = resolveExpected();
            if (partial_.empty()) {
                // The completed header was refused for its version or for a length past
                // `max_packet_length`. That drop is already counted.
            } else if (!resolved || partial_.size() != expected_) {
                // The fragment and the pointer disagree, and the pointer is taken as correct. The
                // fragment is dropped when it is short, from a packet lost between. It is dropped
                // the same way when it is long, from a corrupt length field.
                ++counters_.pointer_mismatch;
                dropFragment();
            } else {
                emitPacket(partial_, emit);
                clearFragment();
            }
        } else if (fhp > 0UZ) {
            counters_.orphan_octets += fhp;
        }

        // 4c. Forward from the pointer.
        walk(zone, fhp, emit);
    }

    void reset() noexcept {
        partial_.clear();
        expected_   = 0UZ;
        last_count_ = 0U;
        have_count_ = false;
        counters_   = Counters{};
    }

    [[nodiscard]] const Counters& counters() const noexcept { return counters_; }
    [[nodiscard]] const Config&   config() const noexcept { return config_; }

    /// @brief The octets currently held from a packet that spans zones. Empty between packets.
    [[nodiscard]] std::span<const std::uint8_t> fragment() const noexcept { return partial_; }

    /// @brief The fragment's target length, or zero while its primary header is still incomplete.
    [[nodiscard]] std::size_t expected() const noexcept { return expected_; }

private:
    void append(std::span<const std::uint8_t> octets) { partial_.insert(partial_.end(), octets.begin(), octets.end()); }

    void clearFragment() noexcept {
        partial_.clear();
        expected_ = 0UZ;
    }

    void dropFragment() noexcept {
        if (!partial_.empty()) {
            ++counters_.fragments_dropped;
        }
        clearFragment();
    }

    /// @brief Resolve the held fragment's target length once its primary header is complete (step 6).
    [[nodiscard]] bool resolveExpected() noexcept {
        if (expected_ != 0UZ) {
            return true;
        }
        if (partial_.size() < kSpacePacketHeaderSize) {
            return false;
        }
        const auto header = parseSpacePacketHeader(partial_);
        if (!header) {
            ++counters_.bad_version;
            clearFragment();
            return false;
        }
        const std::size_t total = totalPacketOctets(*header);
        if (total > config_.max_packet_length) {
            ++counters_.oversize_dropped;
            clearFragment();
            return false;
        }
        expected_ = total;
        return true;
    }

    /**
     * @brief Step 6. Complete a fragment that has just been appended to.
     *
     * Returns true when the fragment completed and left a surplus. `resume` is then the zone offset
     * where the forward walk continues. The surplus belongs to the next packet in the same zone. Only
     * a whole-zone continuation leaves a surplus. The pointer that follows a residue bounds it.
     */
    template<typename Emit>
    [[nodiscard]] bool settleFragment(std::size_t zoneEnd, std::size_t& resume, Emit& emit) {
        if (!resolveExpected() || partial_.size() < expected_) {
            return false;
        }
        const std::size_t surplus = partial_.size() - expected_;
        partial_.resize(expected_);
        emitPacket(partial_, emit);
        clearFragment();
        if (surplus == 0UZ) {
            return false;
        }
        resume = zoneEnd - surplus;
        return true;
    }

    /// @brief Step 4c. Read whole packets forward from `offset` until the zone ends or one spills over.
    template<typename Emit>
    void walk(std::span<const std::uint8_t> zone, std::size_t offset, Emit& emit) {
        std::size_t o = offset;
        while (o < zone.size()) {
            const std::size_t remaining = zone.size() - o;
            if (remaining < kSpacePacketHeaderSize) {
                // The primary header itself is split across the zone boundary. Its length is unknown
                // until the sixth octet arrives. This fragment is therefore at most five octets.
                append(zone.subspan(o));
                expected_ = 0UZ;
                return;
            }
            const auto header = parseSpacePacketHeader(zone.subspan(o, kSpacePacketHeaderSize));
            if (!header) {
                // A header of another version gives no length. The walk stops as for an oversize packet.
                ++counters_.bad_version;
                return;
            }
            const std::size_t total = totalPacketOctets(*header);
            if (total > config_.max_packet_length) {
                // The length is not trusted, and no second pointer in the zone allows recovery.
                ++counters_.oversize_dropped;
                return;
            }
            if (o + total <= zone.size()) {
                emitPacket(zone.subspan(o, total), emit);
                o += total;
                continue;
            }
            append(zone.subspan(o));
            expected_ = total;
            return;
        }
    }

    /// @brief Step 5. An idle packet is counted and discarded. Every other packet is emitted whole.
    /// Every packet that reaches this step has a header that parsed when its length was read.
    template<typename Emit>
    void emitPacket(std::span<const std::uint8_t> packet, Emit& emit) {
        const auto header = parseSpacePacketHeader(packet);
        if (header.has_value() && isIdlePacket(*header)) {
            ++counters_.idle_packets;
            return;
        }
        ++counters_.packets;
        emit(packet);
    }

    Config                    config_{};
    Counters                  counters_{};
    std::vector<std::uint8_t> partial_{};
    std::size_t               expected_   = 0UZ;
    std::uint32_t             last_count_ = 0U;
    bool                      have_count_ = false;
};

} // namespace gr::ccsds

#endif // GNURADIO_ALGORITHM_CCSDS_PACKET_EXTRACTOR_HPP
