#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/Ppm.hpp>

namespace {

using gr::digital::PpmConfig;
using gr::digital::PpmScanner;

/// One hexadecimal frame as octets, first bit received in the first octet's most significant bit.
[[nodiscard]] std::vector<std::uint8_t> octetsOf(std::string_view hex) {
    std::vector<std::uint8_t> octets;
    octets.reserve(hex.size() / 2UZ);
    const auto nibble = [](char c) { return static_cast<std::uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10); };
    for (std::size_t i = 0UZ; i + 1UZ < hex.size(); i += 2UZ) {
        octets.push_back(static_cast<std::uint8_t>((nibble(hex[i]) << 4U) | nibble(hex[i + 1UZ])));
    }
    return octets;
}

/// A scanner carrying `modeS()` at one sample a slot, which is all `remainderOf` needs.
[[nodiscard]] PpmScanner modeSScanner() {
    PpmScanner scanner;
    scanner.prepare(gr::digital::modeS(), 1UZ);
    return scanner;
}

/// The five published extended squitters, whose parity fields are the bare CRC.
constexpr std::array<std::string_view, 5> kPublished{"8D4840D6202CC371C32CE0576098", "8D40621D58C382D690C8AC2863A7", "8D40621D58C386435CC412692AD6", "8D485020994409940838175B284F", "8D40621D58C3862590C412D77427"};

} // namespace

const boost::ut::suite<"ppm"> ppmTests = [] {
    using namespace boost::ut;

    "modeS states Annex 10's geometry, length rule and parity generator"_test = [] {
        const PpmConfig cfg = gr::digital::modeS();
        expect(that % (cfg.pulseSlots == std::vector<std::size_t>{0UZ, 2UZ, 7UZ, 9UZ})) << "pulses at 0.0, 1.0, 3.5 and 4.5 microseconds, in half-microsecond slots";
        expect(eq(cfg.preambleSlots, 16UZ)) << "eight microseconds of preamble";
        expect(eq(cfg.formatBits, 5UZ));
        expect(eq(cfg.longBits, 112UZ));
        expect(eq(cfg.shortBits, 56UZ));
        expect(eq(static_cast<unsigned>(cfg.crcWidth), 24U));
        expect(eq(cfg.crcPoly, 0xFFF409ULL));
        expect(eq(cfg.crcInit, 0ULL));
        expect(eq(cfg.crcFinalXor, 0ULL));
        expect(!cfg.crcInputReflected);
        expect(!cfg.crcResultReflected);
    };

    "the published frames reduce to zero, and one flipped bit does not"_test = [] {
        const PpmScanner scanner = modeSScanner();
        for (const std::string_view hex : kPublished) {
            const std::vector<std::uint8_t> frame = octetsOf(hex);
            expect(eq(frame.size(), 14UZ)) << hex;
            expect(eq(scanner.remainderOf(std::span<const std::uint8_t>(frame)), 0ULL)) << hex << " is a bare-CRC frame";
        }

        std::vector<std::uint8_t> damaged = octetsOf(kPublished[0UZ]);
        damaged[0UZ] ^= 0x01U;
        expect(neq(scanner.remainderOf(std::span<const std::uint8_t>(damaged)), 0ULL)) << "a single flipped bit leaves the remainder";
    };

    "a parity field XORed with an address leaves that address as the remainder"_test = [] {
        const PpmScanner          scanner = modeSScanner();
        std::vector<std::uint8_t> message{0x20U, 0x00U, 0x11U, 0x22U}; // downlink format 4, a surveillance altitude reply
        message.insert(message.end(), 3UZ, std::uint8_t{0});
        const std::uint64_t parity = scanner.remainderOf(std::span<const std::uint8_t>(message));

        for (const std::uint64_t address : {0x4840D6ULL, 0xABCDEFULL}) {
            const std::uint64_t       field = parity ^ address;
            std::vector<std::uint8_t> frame(message.begin(), message.begin() + 4);
            frame.push_back(static_cast<std::uint8_t>(field >> 16U));
            frame.push_back(static_cast<std::uint8_t>(field >> 8U));
            frame.push_back(static_cast<std::uint8_t>(field));
            expect(eq(scanner.remainderOf(std::span<const std::uint8_t>(frame)), address)) << "the reduction is linear, so the XORed address survives it";
        }
    };

    "configure refuses every geometry the scanner cannot read"_test = [] {
        const auto refused = [](auto&& edit) {
            PpmConfig cfg = gr::digital::modeS();
            edit(cfg);
            return throws([&cfg] { gr::digital::configure(cfg); });
        };

        expect(nothrow([] { gr::digital::configure(gr::digital::modeS()); }));
        expect(refused([](PpmConfig& c) { c.pulseSlots.clear(); })) << "no pulse to measure";
        expect(refused([](PpmConfig& c) { c.pulseSlots.push_back(16UZ); })) << "a pulse slot outside the preamble";
        expect(refused([](PpmConfig& c) { c.pulseSlots.push_back(2UZ); })) << "a pulse slot named twice";
        expect(refused([](PpmConfig& c) { c.preambleSlots = 4UZ; })) << "every slot a pulse leaves no gap";
        expect(refused([](PpmConfig& c) { c.preambleSlots = 0UZ; }));
        expect(refused([](PpmConfig& c) { c.formatBits = 0UZ; }));
        expect(refused([](PpmConfig& c) { c.formatBits = 33UZ; })) << "a format field wider than the word that reads it, whose top bit could not be shifted out";
        expect(nothrow([] {
            PpmConfig cfg  = gr::digital::modeS();
            cfg.formatBits = 32UZ;
            gr::digital::configure(cfg);
        })) << "the widest field the word still reads";
        expect(refused([](PpmConfig& c) { c.shortBits = 120UZ; })) << "the short frame is longer than the long one";
        expect(refused([](PpmConfig& c) { c.shortBits = 52UZ; })) << "a frame length that is not whole octets";
        expect(refused([](PpmConfig& c) { c.longBits = 116UZ; }));
        expect(refused([](PpmConfig& c) { c.crcWidth = 2U; })) << "below the register width the CRC kernel accepts";
        expect(refused([](PpmConfig& c) { c.crcWidth = 20U; })) << "a parity field that is not whole octets";
        expect(refused([](PpmConfig& c) {
            c.shortBits = 24UZ;
            c.longBits  = 24UZ;
        })) << "a parity field that leaves no message";

        PpmScanner scanner;
        expect(throws([&scanner] { scanner.prepare(gr::digital::modeS(), 0UZ); })) << "a slot spans at least one sample";
        expect(!scanner.configured()) << "a refused preparation leaves the scanner inert";
    };
};

int main() { /* not needed for UT */ }
