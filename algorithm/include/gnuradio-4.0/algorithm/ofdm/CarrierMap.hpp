#ifndef GNURADIO_ALGORITHM_CARRIER_MAP_HPP
#define GNURADIO_ALGORITHM_CARRIER_MAP_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

namespace gr::ofdm {

/// What a carrier is used for. Guards are everything the numerology did not claim, so the three are disjoint
/// and complete over the transform by construction rather than by a check that could be forgotten.
enum class Occupancy : std::uint8_t { Guard, Data, Pilot };

/**
 * @brief One OFDM numerology: the signed carrier indices a human writes, resolved into FFT bin order once.
 *
 * Two conventions meet in every OFDM implementation and are the source of most of its off-by-one bugs. A
 * published numerology names carriers by **signed logical index**: DC is 0, the carrier one spacing above it is
 * +1, the one below is -1, and the range is `-fft_len/2 … fft_len/2-1`. A transform holds the same symbol in
 * **bin order**: DC at index 0, the positive carriers ascending from index 1, and the negative ones in the upper
 * half. The map between them is `bin = carrier` for a non-negative carrier and `bin = carrier + fft_len` for a
 * negative one, and this class is the only place it is written. Everything downstream reads bins.
 *
 * `fft_len/2` is the asymmetry the range makes explicit: `-fft_len/2` is a real carrier and lands on the Nyquist
 * bin `fft_len/2`, while `+fft_len/2` does not exist. A numerology that names it is refused rather than folded,
 * because folding it would silently move a carrier to the other side of the band.
 *
 * A numerology is the data set and the pilot set. Everything else is guard, which is what makes the three
 * disjoint and complete over `fft_len` without a completeness check: guards are computed as the complement, so
 * there is nothing for the caller to get wrong and nothing for the class to have to reject. What is checked is
 * that no carrier falls outside the range, that neither set repeats a carrier, and that the two sets do not
 * share one. DC is a carrier like any other here — a numerology may use it or leave it guarded — and
 * `dcOccupied()` says which, so a consumer never has to infer it from a set it did not build.
 *
 * The class holds no symbol and no transform. It is the addressing, and it is const once constructed.
 */
class CarrierMap {
public:
    /// Nothing in the tiers asks for a longer transform, and the ceiling keeps the per-bin tables a numerology
    /// builds bounded by something stated rather than by available memory.
    static constexpr std::size_t kMaxFftLength = std::size_t{1} << 16;

    /// The signed logical index of a carrier, as an FFT bin. Static because the mapping is a property of the
    /// transform length alone and a caller often has one before it has a numerology.
    [[nodiscard]] static std::size_t binOf(std::size_t fftLength, int carrier) {
        requireLength(fftLength);
        requireCarrier(fftLength, carrier);
        return (carrier < 0) ? static_cast<std::size_t>(carrier + static_cast<int>(fftLength)) : static_cast<std::size_t>(carrier);
    }

    /// The inverse: which signed logical carrier a bin holds.
    [[nodiscard]] static int carrierOf(std::size_t fftLength, std::size_t bin) {
        requireLength(fftLength);
        if (bin >= fftLength) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: bin {} is outside a transform of {} bins", bin, fftLength));
        }
        const int half  = static_cast<int>(fftLength / 2UZ);
        const int index = static_cast<int>(bin);
        return (index >= half) ? index - static_cast<int>(fftLength) : index;
    }

    CarrierMap(std::size_t fftLength, std::span<const int> dataCarriers, std::span<const int> pilotCarriers) : _fftLength(fftLength), _data(dataCarriers.begin(), dataCarriers.end()), _pilot(pilotCarriers.begin(), pilotCarriers.end()) {
        requireLength(fftLength);
        if (_data.empty()) {
            throw std::invalid_argument("gr::ofdm::CarrierMap: data_carriers is empty — a numerology that carries no data is not one");
        }

        _occupancy.assign(fftLength, Occupancy::Guard);
        _dataBins.reserve(_data.size());
        _pilotBins.reserve(_pilot.size());

        claim(_data, _dataBins, Occupancy::Data, "data_carriers");
        claim(_pilot, _pilotBins, Occupancy::Pilot, "pilot_carriers");

        for (std::size_t bin = 0UZ; bin < fftLength; ++bin) {
            if (_occupancy[bin] == Occupancy::Guard) {
                _guard.push_back(carrierOf(fftLength, bin));
                _guardBins.push_back(bin);
            }
        }
        _dcOccupied = _occupancy[0UZ] != Occupancy::Guard;
    }

    [[nodiscard]] std::size_t fftLength() const noexcept { return _fftLength; }
    [[nodiscard]] std::size_t nData() const noexcept { return _data.size(); }
    [[nodiscard]] std::size_t nPilots() const noexcept { return _pilot.size(); }
    [[nodiscard]] std::size_t occupiedCount() const noexcept { return _data.size() + _pilot.size(); }
    [[nodiscard]] bool        dcOccupied() const noexcept { return _dcOccupied; }

    [[nodiscard]] std::span<const int> dataCarriers() const noexcept { return _data; }
    [[nodiscard]] std::span<const int> pilotCarriers() const noexcept { return _pilot; }

    /// The guard carriers, ascending in signed index. They are the complement of the two occupied sets, which is
    /// what makes the three complete over `fftLength()`.
    [[nodiscard]] std::span<const int> guardCarriers() const noexcept { return _guard; }

    /// The bin each data carrier occupies, in `dataCarriers()` order — the gather a symbol assembler applies.
    [[nodiscard]] std::span<const std::size_t> dataBins() const noexcept { return _dataBins; }
    [[nodiscard]] std::span<const std::size_t> pilotBins() const noexcept { return _pilotBins; }
    [[nodiscard]] std::span<const std::size_t> guardBins() const noexcept { return _guardBins; }

    /// What a bin carries. Bounds-checked, because a caller indexing this with a bin it computed itself is
    /// exactly the mistake the class exists to catch.
    [[nodiscard]] Occupancy occupancyOfBin(std::size_t bin) const {
        if (bin >= _fftLength) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: bin {} is outside a transform of {} bins", bin, _fftLength));
        }
        return _occupancy[bin];
    }

    [[nodiscard]] Occupancy occupancyOf(int carrier) const { return _occupancy[binOf(_fftLength, carrier)]; }

    /// A sync word is a whole symbol, emitted verbatim, so the only thing to check about it is its length —
    /// and that is worth checking loudly, because a short one would be read as a symbol with silent carriers.
    void validateSyncWord(std::span<const std::complex<float>> word, std::size_t index = 0UZ) const {
        if (word.size() != _fftLength) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: sync word {} holds {} carriers, not the {} a symbol has", index, word.size(), _fftLength));
        }
    }

    /// Refuses a pilot-symbol cycle that cannot be indexed. Called once at configure, so that
    /// `pilotSymbolIndex` can be arithmetic with no check in it.
    void validatePilotCycle(std::size_t cycleLength) const {
        if (_pilot.empty()) {
            return;
        }
        if (cycleLength == 0UZ) {
            throw std::invalid_argument("gr::ofdm::CarrierMap: pilot_symbols is empty while the numerology has pilot carriers");
        }
    }

    /**
     * @brief Which entry of the pilot-symbol vector symbol `symbolIndex` reads for its pilot slot `pilotSlot`.
     *
     * The rule is one cycle read at two levels: the slots of a symbol take consecutive entries, and the next
     * symbol carries on where the last one stopped — `pilot_symbols[(s * n_pilots + p) % len]`. A cycle whose
     * length is a multiple of `n_pilots` therefore gives every symbol the same pilots, and one that is not
     * rotates them from symbol to symbol, which is the whole point of stating the rule rather than leaving each
     * block to invent it.
     *
     * `symbolIndex` is reduced before the multiply. A symbol index is stream-absolute and a long run makes it
     * large, and `s * n_pilots` would overflow long before `s` itself did; `(s mod len) * n_pilots` is congruent
     * to it and cannot.
     */
    [[nodiscard]] static constexpr std::size_t pilotSymbolIndex(std::size_t symbolIndex, std::size_t pilotSlot, std::size_t nPilots, std::size_t cycleLength) noexcept { return ((symbolIndex % cycleLength) * nPilots + pilotSlot) % cycleLength; }

    [[nodiscard]] constexpr std::size_t pilotSymbolIndex(std::size_t symbolIndex, std::size_t pilotSlot, std::size_t cycleLength) const noexcept { return pilotSymbolIndex(symbolIndex, pilotSlot, _pilot.size(), cycleLength); }

private:
    std::size_t              _fftLength{0UZ};
    std::vector<int>         _data{};
    std::vector<int>         _pilot{};
    std::vector<int>         _guard{};
    std::vector<std::size_t> _dataBins{};
    std::vector<std::size_t> _pilotBins{};
    std::vector<std::size_t> _guardBins{};
    std::vector<Occupancy>   _occupancy{};
    bool                     _dcOccupied{false};

    static void requireLength(std::size_t fftLength) {
        if (fftLength < 4UZ || fftLength > kMaxFftLength) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: fft_len is {} — a transform holds between 4 and {} bins", fftLength, kMaxFftLength));
        }
        if (fftLength % 2UZ != 0UZ) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: fft_len is {} — an odd transform has no symmetric carrier range about DC", fftLength));
        }
    }

    /// The range is asymmetric because bin order is: `-fft_len/2` is the Nyquist bin and `+fft_len/2` is not a
    /// carrier at all.
    static void requireCarrier(std::size_t fftLength, int carrier) {
        const int half = static_cast<int>(fftLength / 2UZ);
        if (carrier < -half || carrier >= half) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: carrier {} is outside the range {} to {} an fft_len of {} spans", carrier, -half, half - 1, fftLength));
        }
    }

    void claim(std::span<const int> carriers, std::vector<std::size_t>& bins, Occupancy what, const char* parameter) {
        for (const int carrier : carriers) {
            requireCarrier(_fftLength, carrier);
            const std::size_t bin = binOf(_fftLength, carrier);
            if (_occupancy[bin] != Occupancy::Guard) {
                throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: {} names carrier {} twice, or after the other set already claimed it", parameter, carrier));
            }
            _occupancy[bin] = what;
            bins.push_back(bin);
        }
    }
};

/**
 * @brief A Schmidl-Cox preamble: one full-length frequency-domain symbol whose time-domain form repeats its
 * lower half in its upper half.
 *
 * The repeat is not a property of the values, it is a property of which carriers hold them. Writing the inverse
 * transform out,
 *
 *     x[n + N/2] = (1/N) * sum_k X[k] * exp(j*2*pi*k*n/N) * exp(j*pi*k)
 *
 * and `exp(j*pi*k)` is `+1` on every even `k` and `-1` on every odd one. So a symbol whose odd bins are all
 * zero has `x[n + N/2] = x[n]` exactly, for every `n`, with no approximation and no dependence on what the even
 * carriers hold. That is the whole construction: occupy the even carriers, leave the odd ones empty, and the
 * time-domain half-repeat that `SchmidlCoxSync`'s correlator looks for follows. `fftLength` is even, so a
 * carrier's parity and its bin's parity are the same and "even carrier" and "even bin" name the same set.
 *
 * The occupied band is the `occupied` carriers closest to DC, DC itself excluded: `-occupied/2 … -1` and
 * `1 … occupied/2`. DC carries nothing, because a preamble is what a receiver measures its own frequency error
 * against and a DC term is the one component a residual offset cannot be seen against. The even members of that
 * band each take a QPSK point scaled by `sqrt(2)`, so each holds twice the power a fully occupied symbol would
 * put on a carrier and the preamble's total energy comes to `2 * evenCarrierCount`, which is `occupied` itself
 * whenever `occupied/2` is even — the case every published numerology is in.
 *
 * The points come from a seeded PRNG rather than a published sequence: what the correlator needs is that the
 * lower half looks like noise to a timing metric, not that it matches any standard's table. A standard's
 * preamble is a sync word the caller supplies verbatim; this is the helper for when there is no standard.
 */
[[nodiscard]] inline std::vector<std::complex<float>> schmidlCoxPreamble(std::size_t fftLength, std::size_t occupied, std::uint64_t seed) {
    if (fftLength < 4UZ || fftLength > CarrierMap::kMaxFftLength || fftLength % 2UZ != 0UZ) {
        throw std::invalid_argument(std::format("gr::ofdm::schmidlCoxPreamble: fft_len is {} — an even transform of between 4 and {} bins", fftLength, CarrierMap::kMaxFftLength));
    }
    if (occupied < 4UZ || occupied % 2UZ != 0UZ) {
        throw std::invalid_argument(std::format("gr::ofdm::schmidlCoxPreamble: occupied is {} — an even count of at least 4, so that each side of DC holds an even carrier", occupied));
    }
    if (occupied > fftLength - 2UZ) {
        throw std::invalid_argument(std::format("gr::ofdm::schmidlCoxPreamble: occupied is {}, which with DC excluded does not fit in an fft_len of {}", occupied, fftLength));
    }

    gr::rng::Xoshiro256pp            rng(seed);
    std::vector<std::complex<float>> symbol(fftLength, std::complex<float>{0.f, 0.f});
    const int                        half = static_cast<int>(occupied / 2UZ);
    for (int carrier = -half; carrier <= half; ++carrier) {
        if (carrier == 0 || (carrier % 2) != 0) {
            continue;
        }
        const std::uint64_t bits                      = rng();
        const float         real                      = ((bits & 1ULL) != 0ULL) ? 1.f : -1.f;
        const float         imag                      = ((bits & 2ULL) != 0ULL) ? 1.f : -1.f;
        symbol[CarrierMap::binOf(fftLength, carrier)] = std::complex<float>{real, imag};
    }
    return symbol;
}

} // namespace gr::ofdm

#endif // GNURADIO_ALGORITHM_CARRIER_MAP_HPP
