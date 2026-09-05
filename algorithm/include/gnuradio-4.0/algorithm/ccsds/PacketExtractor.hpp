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
 * Space packets are laid end to end through the transfer frames' data fields, and each frame carries
 * **one pointer** saying where the first packet that starts in that frame begins. 4.1.2.7.6.3 NOTE 1
 * says why the pointer exists: it delimits variable-length packets *without* requiring that the
 * previous frame arrived. Everything hard about the receive side is that sentence, and this class is
 * its implementation.
 *
 * The machine takes a **zone** and a **pointer** rather than a frame, because that is the only
 * difference between the two protocols that carry packets this way: for TM the zone is the transfer
 * frame data field and the pointer is a field of the primary header (4.1.2.7.6), and for AOS the zone
 * is the data field less the two-octet M_PDU header and the pointer is a field of that header
 * (732.0-B-4 4.1.4.2.3.3, whose "the first octet in this zone is assigned the number 0" is the
 * off-by-two). One adapter each side, one machine.
 *
 * **132.0-B-3 4.3.2.4 is the rule the whole design turns on, and it is not optional:** *"If the
 * calculated location of the beginning of the first Packet is not consistent with the location
 * indicated by the First Header Pointer, then the Packet Extraction Function shall assume that the
 * First Header Pointer is correct, and shall continue the extraction based on that assumption."* The
 * pointer wins over the held fragment's own length, and that is what makes one lost frame cost one
 * packet rather than the rest of the pass. After any loss, the next frame whose pointer is neither
 * reserved value re-establishes a packet boundary from that frame alone.
 *
 * **Bounded state, proved rather than capped.** Between calls the held fragment never exceeds
 * `max_packet_length`, which never exceeds `kMaxPacketOctets` = 65 542, which is derived from the
 * sixteen-bit packet data length field. The one path that could grow a fragment with no length known — a
 * primary header split across a zone boundary — is bounded at five octets, because a sixth completes the
 * header. Inside one call the whole-zone continuation is the exception: it appends the zone before the
 * length that will trim it is known, so the buffer peaks below `max_packet_length + zone.size()` and is
 * trimmed back before the call returns. The buffer is reserved for the settled bound, so only that peak can
 * reallocate, and nothing that arrives on the wire changes either bound.
 *
 * The completed packet is handed to a callable rather than returned, so this kernel builds no record
 * and names no framework type; the buffer is reserved once and an ordinary zone allocates nothing.
 */
namespace gr::ccsds {

class PacketExtractor {
public:
    struct Config {
        /// The longest packet that will be assembled. Derived at `kMaxPacketOctets` and clamped to it:
        /// a mission that knows its packets are short can tighten the bound, never loosen it.
        std::size_t max_packet_length = kMaxPacketOctets;

        /// The modulus of the frame count the caller supplies: `kTmCountModulus` for TM,
        /// `kAosCountModulus` or `kAosCycleCountModulus` for AOS (`aosCountModulus` picks between them).
        std::uint32_t count_modulus = kTmCountModulus;
    };

    struct Counters {
        std::uint64_t packets           = 0ULL; //!< emitted
        std::uint64_t idle_packets      = 0ULL; //!< APID 2047, recognized and discarded (132.0-B-3 4.3.2.5 NOTE 1)
        std::uint64_t idle_frames       = 0ULL; //!< only-idle-data zones discarded (4.1.4.6)
        std::uint64_t frames_lost       = 0ULL; //!< from the frame count gap
        std::uint64_t duplicate_frames  = 0ULL; //!< a gap of zero; the zone is not processed twice
        std::uint64_t fragments_dropped = 0ULL; //!< a held fragment abandoned
        std::uint64_t pointer_mismatch  = 0ULL; //!< 4.3.2.4: the pointer disagreed with the fragment's length
        std::uint64_t bad_pointer       = 0ULL; //!< a pointer at or beyond the zone
        std::uint64_t orphan_octets     = 0ULL; //!< residue with no fragment to complete
        std::uint64_t oversize_dropped  = 0ULL; //!< a packet longer than `max_packet_length`

        [[nodiscard]] bool operator==(const Counters&) const noexcept = default;
    };

    PacketExtractor() : PacketExtractor(Config{}) {}

    explicit PacketExtractor(Config config) : config_{config.max_packet_length == 0UZ ? kMaxPacketOctets : std::min(config.max_packet_length, kMaxPacketOctets), config.count_modulus} { partial_.reserve(config_.max_packet_length); }

    /**
     * @brief Take one zone and the pointer that came with it.
     *
     * `zone` is the TM data field, or the AOS M_PDU packet zone with its header already removed; `fhp`
     * is the first header pointer that arrived with it; `count` is the virtual channel frame count,
     * already widened where AOS's cycle field applies. Every completed packet is passed whole to
     * `emit`, primary header included, so a consumer that wants the raw packet has it.
     */
    template<typename Emit>
    void feed(std::span<const std::uint8_t> zone, std::uint16_t fhp, std::uint32_t count, Emit&& emit) {
        // 1. Continuity. The modular subtraction is the whole of the wrap handling.
        if (have_count_) {
            const SequenceGap gap = frameGap(count, last_count_, config_.count_modulus);
            if (gap.duplicate) {
                ++counters_.duplicate_frames;
                return; // re-feeding the zone would emit every packet in it a second time
            }
            if (!gap.continuous) {
                counters_.frames_lost += gap.lost;
                dropFragment(); // the octets that would have completed it were in the frames that did not arrive
            }
        }
        last_count_ = count;
        have_count_ = true;

        // 2. Only idle data. 4.1.4.6.3 NOTE 3 permits an OID frame in the middle of a split packet, so
        //    the fill is not the packet's continuation and appending it would corrupt the packet silently.
        if (fhp == kFhpOnlyIdleData) {
            ++counters_.idle_frames;
            dropFragment();
            return;
        }

        // 3. No packet starts here: the whole zone continues a packet begun earlier (4.1.2.7.6.4 NOTE).
        if (fhp == kFhpNoPacketStart) {
            if (partial_.empty()) {
                counters_.orphan_octets += zone.size();
                return;
            }
            append(zone); // the length that trims this is in the fragment's own header, so the trim is below
            std::size_t resume = 0UZ;
            if (settleFragment(zone.size(), resume, emit)) {
                walk(zone, resume, emit);
            }
            return;
        }

        // 4a. Out of range. The one field that says where anything is has been contradicted.
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
                // The completed header declared a packet past `max_packet_length`; already counted.
            } else if (!resolved || partial_.size() != expected_) {
                // The fragment and the pointer disagree, and the pointer is correct. The same drop
                // whether the fragment came out short (a packet was lost between) or long (its own
                // length field was corrupt).
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
        SpacePacketHeader header{};
        static_cast<void>(parseSpacePacketHeader(partial_, header));
        const std::size_t total = totalPacketOctets(header);
        if (total > config_.max_packet_length) {
            ++counters_.oversize_dropped;
            clearFragment();
            return false;
        }
        expected_ = total;
        return true;
    }

    /**
     * @brief Step 6: complete a fragment that has just been appended to.
     *
     * Returns true when the fragment completed and left a surplus, in which case `resume` is the zone
     * offset the forward walk continues at — the surplus belongs to the next packet in the same zone.
     * The surplus case arises only from a whole-zone continuation, because a residue is bounded by the
     * pointer that follows it.
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

    /// @brief Step 4c: read whole packets forward from `offset` until the zone ends or one spills.
    template<typename Emit>
    void walk(std::span<const std::uint8_t> zone, std::size_t offset, Emit& emit) {
        std::size_t o = offset;
        while (o < zone.size()) {
            const std::size_t remaining = zone.size() - o;
            if (remaining < kSpacePacketHeaderSize) {
                // The primary header itself is split across the zone boundary; its length is unknown
                // until the sixth octet arrives, which bounds this fragment at five octets.
                append(zone.subspan(o));
                expected_ = 0UZ;
                return;
            }
            SpacePacketHeader header{};
            static_cast<void>(parseSpacePacketHeader(zone.subspan(o, kSpacePacketHeaderSize), header));
            const std::size_t total = totalPacketOctets(header);
            if (total > config_.max_packet_length) {
                // The length is not trustworthy and there is no second pointer to recover from.
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

    /// @brief Step 5: an idle packet is counted and discarded; every other packet is emitted whole.
    template<typename Emit>
    void emitPacket(std::span<const std::uint8_t> packet, Emit& emit) {
        SpacePacketHeader header{};
        static_cast<void>(parseSpacePacketHeader(packet, header));
        if (isIdlePacket(header)) {
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
