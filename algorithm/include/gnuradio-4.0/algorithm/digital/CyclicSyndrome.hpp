#ifndef GNURADIO_CYCLIC_SYNDROME_HPP
#define GNURADIO_CYCLIC_SYNDROME_HPP

#include <cstdint>
#include <format>
#include <stdexcept>

/**
 * A systematic shortened cyclic block code with coset offset words, the arithmetic behind
 * offset-word framing (EN 50067 / RBDS Annex C is the driving instantiation).
 *
 * A block is `data * x^C | check`, the check being the remainder of `data * x^C` modulo the
 * generator, zero register preset, no inversion — where `Crc` next door carries reflection
 * and inversion vocabulary, an offset-word code carries none, which is why this is its own
 * kernel rather than a Crc configuration. A position label rides the block as an offset word
 * xored onto the checkword, and the load-bearing identity is that the syndrome is the offset:
 * a valid un-offset block's remainder is zero, so a valid offset block's remainder is exactly
 * its offset word, and framing recovery is one reduction and a lookup rather than a trial per
 * offset.
 */
namespace gr::digital {

struct CyclicSyndrome {
    std::uint32_t polynomial; //!< the generator, its degree-checkBits term present
    unsigned      checkBits;
    unsigned      dataBits;

    constexpr CyclicSyndrome(std::uint32_t poly, unsigned nCheck, unsigned nData) : polynomial(poly), checkBits(nCheck), dataBits(nData) {
        if (nCheck == 0U || nCheck > 16U || nData == 0U || nData > 32U) {
            throw std::invalid_argument(std::format("CyclicSyndrome: {} data and {} check bits; the offset scheme's registers are one word (data <= 32, check in 1..16)", nData, nCheck));
        }
        if (((poly >> nCheck) & 1U) == 0U || (poly >> nCheck) > 1U) {
            throw std::invalid_argument(std::format("CyclicSyndrome: the generator {:#x} must carry its degree-{} term and nothing above it", poly, nCheck));
        }
    }

    //! The remainder of a word of `bits` coefficients modulo the generator.
    [[nodiscard]] constexpr std::uint16_t remainder(std::uint64_t word, unsigned bits) const noexcept {
        std::uint64_t r = word;
        for (unsigned bit = bits; bit-- > checkBits;) {
            if ((r >> bit) & 1U) {
                r ^= static_cast<std::uint64_t>(polynomial) << (bit - checkBits);
            }
        }
        return static_cast<std::uint16_t>(r & ((1ULL << checkBits) - 1ULL));
    }

    //! The checkword of a data word: the remainder of `data * x^C`.
    [[nodiscard]] constexpr std::uint16_t check(std::uint32_t data) const noexcept { return remainder(static_cast<std::uint64_t>(data) << checkBits, dataBits + checkBits); }

    //! The syndrome of a whole received block. Zero for a valid un-offset block; exactly the
    //! offset word for a valid block carrying one.
    [[nodiscard]] constexpr std::uint16_t syndrome(std::uint64_t block) const noexcept { return remainder(block, dataBits + checkBits); }

    //! The transmitted block for a data word under an offset: `data * x^C | (check ^ offset)`.
    [[nodiscard]] constexpr std::uint64_t block(std::uint32_t data, std::uint16_t offset) const noexcept { return (static_cast<std::uint64_t>(data) << checkBits) | static_cast<std::uint64_t>(check(data) ^ offset); }
};

} // namespace gr::digital

#endif // GNURADIO_CYCLIC_SYNDROME_HPP
