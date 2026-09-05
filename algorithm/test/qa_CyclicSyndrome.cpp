#include <boost/ut.hpp>

#include <array>
#include <cstdint>
#include <format>

#include <gnuradio-4.0/algorithm/digital/CyclicSyndrome.hpp>

namespace {

// EN 50067's code, the driving instantiation: g = x^10+x^8+x^7+x^5+x^4+x^3+1 over 16 data bits.
constexpr std::uint32_t kRdsPoly  = 0x5B9U;
constexpr std::uint16_t kOffsetA  = 0x0FCU;
constexpr std::uint16_t kOffsetB  = 0x198U;
constexpr std::uint16_t kOffsetC  = 0x168U;
constexpr std::uint16_t kOffsetCp = 0x350U;
constexpr std::uint16_t kOffsetD  = 0x1B4U;

} // namespace

const boost::ut::suite<"CyclicSyndrome"> cyclicSyndromeTests = [] {
    using namespace boost::ut;
    using gr::digital::CyclicSyndrome;

    "the checkword vectors, as a standard's appendix would give them"_test = [] {
        const CyclicSyndrome code(kRdsPoly, 10U, 16U);
        struct Row {
            std::uint32_t data;
            std::uint16_t check;
        };
        for (const Row row : std::array<Row, 6>{{{0x0000U, 0x000U}, {0x0001U, 0x1B9U}, {0x8000U, 0x077U}, {0x1234U, 0x096U}, {0xB147U, 0x1AEU}, {0xFFFFU, 0x0CDU}}}) {
            expect(eq(code.check(row.data), row.check)) << std::format("check of {:#06x}", row.data);
        }
        expect(eq(code.block(0x1234U, kOffsetA), std::uint64_t{0x048D06AULL})) << "the offset-applied block, held against a value this tree did not compute";
        expect(eq(code.block(0xB147U, kOffsetCp), std::uint64_t{0x2C51EFEULL}));
    };

    "the syndrome IS the offset word"_test = [] {
        const CyclicSyndrome code(kRdsPoly, 10U, 16U);
        for (const std::uint16_t offset : {kOffsetA, kOffsetB, kOffsetC, kOffsetCp, kOffsetD}) {
            for (const std::uint32_t data : {0x0000U, 0x1234U, 0xB147U, 0xFFFFU, 0x5BABU}) {
                expect(eq(code.syndrome(code.block(data, offset)), offset)) << std::format("offset {:#05x} at data {:#06x}", offset, data);
            }
        }
        expect(eq(code.syndrome(code.block(0x5BABU, 0U)), std::uint16_t{0U})) << "an un-offset valid block's syndrome is zero";
    };

    "offsets C and D sit one bit apart, the edge lock must not key on"_test = [] {
        const CyclicSyndrome code(kRdsPoly, 10U, 16U);
        // The syndrome of the single-bit error pattern at bit 18 equals C xor D, so one bit
        // error relabels a C block as D or back.
        expect(eq(code.syndrome(1ULL << 18U), std::uint16_t(kOffsetC ^ kOffsetD))) << "bit 18's syndrome is C^D";
        const std::uint64_t cBlock = code.block(0x5BABU, kOffsetC);
        expect(eq(code.syndrome(cBlock ^ (1ULL << 18U)), kOffsetD)) << "a C block with bit 18 flipped validates as D";
    };

    "refusals fire by name"_test = [] {
        expect(throws([] { CyclicSyndrome code(0x1B9U, 10U, 16U); })) << "a generator without its degree term";
        expect(throws([] { CyclicSyndrome code(0x5B9U, 10U, 40U); })) << "data past one word";
        expect(throws([] { CyclicSyndrome code(0x5B9U, 0U, 16U); })) << "no checkword";
    };
};

int main() { /* not needed for UT */ }
