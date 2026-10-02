#ifndef GNURADIO_CYCLIC_SYNDROME_HPP
#define GNURADIO_CYCLIC_SYNDROME_HPP

#include <cstdint>
#include <format>

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/fec/Gf2Polynomial.hpp>

/**
 * A systematic shortened cyclic block code with coset offset words. It is the arithmetic of
 * offset-word framing. EN 50067 and RBDS Annex C are its main instance.
 *
 * A block is `data * x^C | check`. The check is the remainder of `data * x^C` modulo the generator,
 * with a zero register preset and no inversion. The code has no reflection and no inversion, and it
 * is not a CRC configuration. Each block's checkword is xored with an offset word that labels the
 * block's position. The syndrome of a valid block equals its offset word. A valid block with no
 * offset has a remainder of zero. A valid block with an offset therefore has its offset word as the
 * remainder. Framing recovery is one reduction and one lookup, with no trial per offset. The constructor
 * throws `gr::exception` for a shape the registers cannot hold.
 */
namespace gr::digital {

struct CyclicSyndrome {
    std::uint32_t polynomial; //!< the generator, its degree-checkBits term present
    unsigned      checkBits;
    unsigned      dataBits;

    constexpr CyclicSyndrome(std::uint32_t poly, unsigned nCheck, unsigned nData) : polynomial(poly), checkBits(nCheck), dataBits(nData) {
        if (nCheck == 0U || nCheck > 16U || nData == 0U || nData > 32U) {
            throw gr::exception(std::format("CyclicSyndrome: {} data and {} check bits; the offset scheme's registers are one word (data <= 32, check in 1..16)", nData, nCheck));
        }
        if (((poly >> nCheck) & 1U) == 0U || (poly >> nCheck) > 1U) {
            throw gr::exception(std::format("CyclicSyndrome: the generator {:#x} must carry its degree-{} term and nothing above it", poly, nCheck));
        }
    }

    //! The remainder of a word of `bits` coefficients modulo the generator.
    [[nodiscard]] constexpr std::uint16_t remainder(std::uint64_t word, unsigned bits) const noexcept { return static_cast<std::uint16_t>(gr::fec::gf2Remainder(word, bits, polynomial, checkBits)); }

    //! The checkword of a data word, the remainder of `data * x^C`.
    [[nodiscard]] constexpr std::uint16_t check(std::uint32_t data) const noexcept { return remainder(static_cast<std::uint64_t>(data) << checkBits, dataBits + checkBits); }

    //! The syndrome of a whole received block. It is zero for a valid block with no offset.
    //! It is exactly the offset word for a valid block with one.
    [[nodiscard]] constexpr std::uint16_t syndrome(std::uint64_t block) const noexcept { return remainder(block, dataBits + checkBits); }

    //! The transmitted block for a data word under an offset, `data * x^C | (check ^ offset)`.
    [[nodiscard]] constexpr std::uint64_t block(std::uint32_t data, std::uint16_t offset) const noexcept { return (static_cast<std::uint64_t>(data) << checkBits) | static_cast<std::uint64_t>(check(data) ^ offset); }
};

} // namespace gr::digital

#endif // GNURADIO_CYCLIC_SYNDROME_HPP
