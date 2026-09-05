#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <print>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/ofdm/CarrierMap.hpp>

using namespace boost::ut;
using gr::ofdm::CarrierMap;
using gr::ofdm::Occupancy;
using gr::ofdm::schmidlCoxPreamble;

namespace {

using Cplx = std::complex<float>;

/// The QA numerology the OFDM spec names: 64 carriers, 52 occupied, 4 pilots — the 802.11a shape, used here as
/// a test numerology and explicitly not as an interoperability claim.
struct Numerology {
    std::vector<int> data;
    std::vector<int> pilot{-21, -7, 7, 21};
};

[[nodiscard]] Numerology qaNumerology() {
    Numerology numerology;
    for (int carrier = -26; carrier <= 26; ++carrier) {
        if (carrier == 0 || carrier == -21 || carrier == -7 || carrier == 7 || carrier == 21) {
            continue;
        }
        numerology.data.push_back(carrier);
    }
    return numerology;
}

} // namespace

const boost::ut::suite<"CarrierMap"> carrierMapTests = [] {
    // The one conversion point, checked at both ends of the range and at the asymmetry in the middle of it.
    // `-fft_len/2` is the Nyquist bin and a real carrier; `+fft_len/2` is not a carrier at all, and folding it
    // rather than refusing it would move a carrier to the other side of the band without saying so.
    "a signed carrier and its bin are the same thing said twice"_test = [] {
        expect(eq(CarrierMap::binOf(64UZ, 0), 0UZ)) << "DC is bin zero";
        expect(eq(CarrierMap::binOf(64UZ, 1), 1UZ));
        expect(eq(CarrierMap::binOf(64UZ, 31), 31UZ)) << "the highest positive carrier";
        expect(eq(CarrierMap::binOf(64UZ, -1), 63UZ)) << "the negative carriers are the upper half";
        expect(eq(CarrierMap::binOf(64UZ, -31), 33UZ));
        expect(eq(CarrierMap::binOf(64UZ, -32), 32UZ)) << "-fft_len/2 is the Nyquist bin";

        expect(eq(CarrierMap::carrierOf(64UZ, 0UZ), 0));
        expect(eq(CarrierMap::carrierOf(64UZ, 31UZ), 31));
        expect(eq(CarrierMap::carrierOf(64UZ, 32UZ), -32));
        expect(eq(CarrierMap::carrierOf(64UZ, 63UZ), -1));

        for (const std::size_t fftLength : {4UZ, 8UZ, 64UZ, 128UZ, 1'024UZ}) {
            const int half = static_cast<int>(fftLength / 2UZ);
            for (int carrier = -half; carrier < half; ++carrier) {
                const std::size_t bin = CarrierMap::binOf(fftLength, carrier);
                expect(lt(bin, fftLength));
                expect(eq(CarrierMap::carrierOf(fftLength, bin), carrier)) << std::format("fft_len {} carrier {} did not come back", fftLength, carrier);
            }
            expect(throws<std::invalid_argument>([fftLength, half] { (void)CarrierMap::binOf(fftLength, half); })) << "+fft_len/2 is not a carrier";
            expect(throws<std::invalid_argument>([fftLength, half] { (void)CarrierMap::binOf(fftLength, -half - 1); })) << "one below the lowest carrier";
            expect(throws<std::invalid_argument>([fftLength] { (void)CarrierMap::carrierOf(fftLength, fftLength); })) << "a bin past the transform";
        }
    };

    // The spec's QA numerology, and the property that makes it one: the three sets partition the transform.
    "the QA numerology's three sets are disjoint and complete"_test = [] {
        const Numerology numerology = qaNumerology();
        const CarrierMap map(64UZ, numerology.data, numerology.pilot);

        expect(eq(map.fftLength(), 64UZ));
        expect(eq(map.nData(), 48UZ));
        expect(eq(map.nPilots(), 4UZ));
        expect(eq(map.occupiedCount(), 52UZ)) << "64 carriers, 52 occupied, 4 pilots";
        expect(eq(map.guardCarriers().size(), 12UZ)) << "DC, -32..-27 and 27..31";
        expect(that % !map.dcOccupied()) << "this numerology guards DC";

        std::vector<int> claimed(64UZ, 0);
        for (const std::size_t bin : map.dataBins()) {
            claimed[bin] += 1;
        }
        for (const std::size_t bin : map.pilotBins()) {
            claimed[bin] += 1;
        }
        for (const std::size_t bin : map.guardBins()) {
            claimed[bin] += 1;
        }
        expect(that % std::ranges::all_of(claimed, [](int n) { return n == 1; })) << "every bin is claimed exactly once by exactly one set";

        // The bins are in the order the carriers were named, because that order is what a symbol assembler
        // walks: the n-th data symbol goes to the n-th data carrier.
        for (std::size_t i = 0UZ; i < map.nData(); ++i) {
            expect(eq(map.dataBins()[i], CarrierMap::binOf(64UZ, numerology.data[i])));
        }
        expect(that % std::ranges::equal(map.pilotBins(), std::vector<std::size_t>{43UZ, 57UZ, 7UZ, 21UZ})) << "pilots at -21, -7, +7, +21";

        expect(that % (map.occupancyOfBin(0UZ) == Occupancy::Guard)) << "DC";
        expect(that % (map.occupancyOf(7) == Occupancy::Pilot));
        expect(that % (map.occupancyOf(1) == Occupancy::Data));
        expect(that % (map.occupancyOf(-27) == Occupancy::Guard));
        expect(that % (map.occupancyOf(-32) == Occupancy::Guard)) << "the Nyquist carrier is guard here";

        // A numerology may use DC, and the map says so rather than leaving a consumer to infer it.
        std::vector<int> withDc = numerology.data;
        withDc.push_back(0);
        const CarrierMap dcMap(64UZ, withDc, numerology.pilot);
        expect(that % dcMap.dcOccupied());
        expect(eq(dcMap.occupiedCount(), 53UZ));
        expect(eq(dcMap.guardCarriers().size(), 11UZ));
    };

    // Two levels of one cycle: the slots of a symbol take consecutive entries and the next symbol carries on
    // where the last stopped. A cycle whose length is a multiple of n_pilots repeats per symbol; one that is
    // not rotates, which is the behavior the rule exists to state.
    "the pilot cycle is read at two levels"_test = [] {
        for (const std::size_t nPilots : {1UZ, 4UZ, 7UZ}) {
            for (const std::size_t cycleLength : {1UZ, 4UZ, 6UZ, 13UZ}) {
                for (std::size_t symbol = 0UZ; symbol < 40UZ; ++symbol) {
                    for (std::size_t slot = 0UZ; slot < nPilots; ++slot) {
                        expect(eq(CarrierMap::pilotSymbolIndex(symbol, slot, nPilots, cycleLength), (symbol * nPilots + slot) % cycleLength)) << std::format("n_pilots {} len {} symbol {} slot {}", nPilots, cycleLength, symbol, slot);
                    }
                }
            }
        }

        // A cycle of 4 over 4 pilots gives every symbol the same four; a cycle of 6 rotates by two each symbol.
        expect(eq(CarrierMap::pilotSymbolIndex(0UZ, 2UZ, 4UZ, 4UZ), CarrierMap::pilotSymbolIndex(99UZ, 2UZ, 4UZ, 4UZ)));
        expect(eq(CarrierMap::pilotSymbolIndex(1UZ, 0UZ, 4UZ, 6UZ), 4UZ));
        expect(eq(CarrierMap::pilotSymbolIndex(2UZ, 0UZ, 4UZ, 6UZ), 2UZ));

        // The reduction before the multiply is what keeps a stream-absolute symbol index from overflowing, and
        // it is congruent to the unreduced form rather than merely close to it.
        constexpr std::size_t kHuge = std::size_t{1} << 62;
        for (const std::size_t cycleLength : {3UZ, 6UZ, 13UZ}) {
            for (std::size_t slot = 0UZ; slot < 4UZ; ++slot) {
                expect(eq(CarrierMap::pilotSymbolIndex(kHuge, slot, 4UZ, cycleLength), CarrierMap::pilotSymbolIndex(kHuge % cycleLength, slot, 4UZ, cycleLength)));
            }
        }

        const Numerology numerology = qaNumerology();
        const CarrierMap map(64UZ, numerology.data, numerology.pilot);
        expect(eq(map.pilotSymbolIndex(3UZ, 1UZ, 8UZ), (3UZ * 4UZ + 1UZ) % 8UZ)) << "the member form reads n_pilots off the numerology";
        expect(nothrow([&] { map.validatePilotCycle(4UZ); }));
        expect(throws<std::invalid_argument>([&] { map.validatePilotCycle(0UZ); })) << "pilot carriers with no symbols to put on them";

        const CarrierMap noPilots(64UZ, numerology.data, std::span<const int>{});
        expect(nothrow([&] { noPilots.validatePilotCycle(0UZ); })) << "no pilots, nothing to cycle";
    };

    "a numerology that does not describe a transform is refused"_test = [] {
        const std::vector<int> data{1, 2, 3};

        expect(throws<std::invalid_argument>([&] { (void)CarrierMap(2UZ, data, std::span<const int>{}); })) << "an fft_len below four";
        expect(throws<std::invalid_argument>([&] { (void)CarrierMap(65UZ, data, std::span<const int>{}); })) << "an odd fft_len";
        expect(throws<std::invalid_argument>([&] { (void)CarrierMap(CarrierMap::kMaxFftLength + 2UZ, data, std::span<const int>{}); })) << "an fft_len past the ceiling";
        expect(throws<std::invalid_argument>([] { (void)CarrierMap(64UZ, std::span<const int>{}, std::span<const int>{}); })) << "no data carriers";

        expect(throws<std::invalid_argument>([] {
            const std::vector<int> outside{1, 32};
            (void)CarrierMap(64UZ, outside, std::span<const int>{});
        })) << "a carrier at +fft_len/2";
        expect(throws<std::invalid_argument>([] {
            const std::vector<int> outside{-33, 1};
            (void)CarrierMap(64UZ, outside, std::span<const int>{});
        })) << "a carrier below -fft_len/2";
        expect(throws<std::invalid_argument>([] {
            const std::vector<int> repeated{1, 2, 1};
            (void)CarrierMap(64UZ, repeated, std::span<const int>{});
        })) << "a data carrier named twice";
        expect(throws<std::invalid_argument>([] {
            const std::vector<int> data2{1, 2};
            const std::vector<int> pilots{3, 3};
            (void)CarrierMap(64UZ, data2, pilots);
        })) << "a pilot carrier named twice";
        expect(throws<std::invalid_argument>([] {
            const std::vector<int> data2{1, 2};
            const std::vector<int> pilots{2};
            (void)CarrierMap(64UZ, data2, pilots);
        })) << "a carrier that is both data and pilot";

        const Numerology numerology = qaNumerology();
        const CarrierMap map(64UZ, numerology.data, numerology.pilot);
        expect(nothrow([&] { map.validateSyncWord(std::vector<Cplx>(64UZ)); }));
        expect(throws<std::invalid_argument>([&] { map.validateSyncWord(std::vector<Cplx>(63UZ)); })) << "a sync word is a whole symbol";
        expect(throws<std::invalid_argument>([&] { (void)map.occupancyOfBin(64UZ); }));
        expect(throws<std::invalid_argument>([&] { (void)map.occupancyOf(32); }));
    };

    // The construction's claim is exact and has nothing to do with the values: `exp(j*pi*k)` is +1 on every even
    // bin, so a symbol whose odd bins are all zero has x[n + N/2] = x[n] for every n. The transform is in the
    // test only — a kernel that states a preamble needs no FFT to state it.
    "the preamble's odd bins are empty and its time-domain halves match"_test = [] {
        constexpr std::size_t kFftLength = 64UZ;
        constexpr std::size_t kOccupied  = 52UZ;

        const std::vector<Cplx> preamble = schmidlCoxPreamble(kFftLength, kOccupied, 0xc0ffeeULL);
        expect(eq(preamble.size(), kFftLength));
        expect(eq(preamble[0UZ], Cplx{0.f, 0.f})) << "DC carries nothing";

        std::size_t occupiedBins = 0UZ;
        double      energy       = 0.;
        for (std::size_t bin = 0UZ; bin < kFftLength; ++bin) {
            const int carrier = CarrierMap::carrierOf(kFftLength, bin);
            energy += static_cast<double>(std::norm(preamble[bin]));
            if (bin % 2UZ != 0UZ) {
                expect(eq(preamble[bin], Cplx{0.f, 0.f})) << std::format("odd bin {} must be empty", bin);
                continue;
            }
            if (preamble[bin] != Cplx{0.f, 0.f}) {
                ++occupiedBins;
                expect(lt(std::abs(static_cast<double>(std::norm(preamble[bin])) - 2.), 1e-6)) << "a QPSK point scaled by sqrt(2)";
                expect(le(std::abs(carrier), 26)) << std::format("carrier {} is outside the occupied band", carrier);
                expect(neq(carrier, 0)) << "DC is not occupied";
            }
        }
        expect(eq(occupiedBins, 26UZ)) << "the even members of -26..-1 and 1..26";
        std::println("qa_CarrierMap: preamble energy {:.1f} over {} even carriers, against an occupancy of {}", energy, occupiedBins, kOccupied);
        expect(lt(std::abs(energy - static_cast<double>(kOccupied)), 1e-5)) << "the sqrt(2) scaling puts a fully occupied symbol's energy on half the carriers";

        gr::algorithm::FFT<Cplx, Cplx, gr::algorithm::Direction::Backward> inverse{};
        const std::vector<Cplx>                                            time = inverse.compute(preamble);
        expect(eq(time.size(), kFftLength));

        double worst = 0.;
        double scale = 0.;
        for (std::size_t n = 0UZ; n < kFftLength / 2UZ; ++n) {
            worst = std::max(worst, static_cast<double>(std::abs(time[n] - time[n + kFftLength / 2UZ])));
            scale = std::max(scale, static_cast<double>(std::abs(time[n])));
        }
        std::println("qa_CarrierMap: the two time-domain halves differ by at most {:.3e} against a peak of {:.3f}", worst, scale);
        expect(lt(worst, 1e-5 * scale)) << "the upper half must repeat the lower half";

        // The property is the empty odd bins and nothing else, so putting anything on one must break it.
        std::vector<Cplx> spoiled            = preamble;
        spoiled[3UZ]                         = Cplx{1.f, 0.f};
        const std::vector<Cplx> spoiledTime  = inverse.compute(spoiled);
        double                  spoiledWorst = 0.;
        for (std::size_t n = 0UZ; n < kFftLength / 2UZ; ++n) {
            spoiledWorst = std::max(spoiledWorst, static_cast<double>(std::abs(spoiledTime[n] - spoiledTime[n + kFftLength / 2UZ])));
        }
        expect(gt(spoiledWorst, 1.)) << "one occupied odd bin is enough to destroy the repeat";

        // A different seed is a different preamble, and the repeat holds regardless — it is the addressing that
        // guarantees it, not the values.
        for (const std::size_t fftLength : {32UZ, 64UZ, 256UZ}) {
            const std::vector<Cplx> other      = schmidlCoxPreamble(fftLength, fftLength - 12UZ, 7ULL);
            const std::vector<Cplx> otherTime  = inverse.compute(other);
            double                  otherWorst = 0.;
            double                  otherScale = 0.;
            for (std::size_t n = 0UZ; n < fftLength / 2UZ; ++n) {
                otherWorst = std::max(otherWorst, static_cast<double>(std::abs(otherTime[n] - otherTime[n + fftLength / 2UZ])));
                otherScale = std::max(otherScale, static_cast<double>(std::abs(otherTime[n])));
            }
            expect(lt(otherWorst, 1e-5 * otherScale)) << std::format("fft_len {} did not repeat", fftLength);
        }

        expect(that % !std::ranges::equal(preamble, schmidlCoxPreamble(kFftLength, kOccupied, 1ULL))) << "the seed is the sequence";
        expect(that % std::ranges::equal(preamble, schmidlCoxPreamble(kFftLength, kOccupied, 0xc0ffeeULL))) << "and the same seed is the same sequence";
    };

    "a preamble that cannot be built is refused"_test = [] {
        expect(throws<std::invalid_argument>([] { (void)schmidlCoxPreamble(63UZ, 52UZ, 0ULL); })) << "an odd fft_len";
        expect(throws<std::invalid_argument>([] { (void)schmidlCoxPreamble(2UZ, 2UZ, 0ULL); })) << "an fft_len below four";
        expect(throws<std::invalid_argument>([] { (void)schmidlCoxPreamble(64UZ, 51UZ, 0ULL); })) << "an odd occupancy";
        expect(throws<std::invalid_argument>([] { (void)schmidlCoxPreamble(64UZ, 2UZ, 0ULL); })) << "too few to hold an even carrier each side";
        expect(throws<std::invalid_argument>([] { (void)schmidlCoxPreamble(64UZ, 64UZ, 0ULL); })) << "an occupancy that leaves no room for DC";
        expect(nothrow([] { (void)schmidlCoxPreamble(64UZ, 62UZ, 0ULL); })) << "the widest occupancy that does";
    };
};

int main() { /* not needed for UT */ }
