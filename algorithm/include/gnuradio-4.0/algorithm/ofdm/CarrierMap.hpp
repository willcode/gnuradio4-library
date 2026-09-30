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

/// The role of a carrier. A guard is every carrier the numerology does not name. The three roles are therefore
/// disjoint and cover the transform by construction, with no separate check.
enum class Occupancy : std::uint8_t { Guard, Data, Pilot };

/**
 * @brief One OFDM numerology, with signed carrier indices resolved once into FFT bin order.
 *
 * Two conventions meet in every OFDM implementation. They cause most of its off-by-one errors. A published
 * numerology names carriers by **signed logical index**. DC is 0, the carrier one spacing above it is +1, and
 * the one below is -1. The range is `-fft_len/2 … fft_len/2-1`. A transform holds the same symbol in **bin
 * order**. DC is at index 0, the positive carriers ascend from index 1, and the negative carriers fill the upper
 * half. The map is `bin = carrier` for a non-negative carrier and `bin = carrier + fft_len` for a negative one.
 * This class applies the map.
 *
 * The range is asymmetric at `fft_len/2`. `-fft_len/2` is a real carrier and falls on the Nyquist bin
 * `fft_len/2`. `+fft_len/2` does not exist. A numerology that names it is refused and not folded. Folding would
 * move a carrier to the other side of the band without any sign.
 *
 * A numerology is the data set and the pilot set. Every other carrier is a guard. The guards are computed as the
 * complement. The three sets are therefore disjoint and cover `fft_len` without a completeness check. The caller
 * cannot get the guards wrong, and the class has nothing to reject for them. The constructor checks that no
 * carrier falls outside the range. It checks that neither set repeats a carrier and that the two sets share
 * none. DC is an ordinary carrier here. A numerology may use it or leave it as a guard. `dcOccupied()` reports
 * which, and a caller need not infer it from the sets.
 *
 * The class holds no symbol and no transform. It holds the addressing and is const once constructed.
 */
class CarrierMap {
public:
    /// The longest transform accepted. The ceiling bounds a numerology's per-bin tables by a stated size and not
    /// by available memory.
    static constexpr std::size_t kMaxFftLength = std::size_t{1} << 16;

    /// The signed logical index of a carrier, as an FFT bin. The function is static. The mapping depends on the
    /// transform length alone, and a caller often has the length before it has a numerology.
    [[nodiscard]] static std::size_t binOf(std::size_t fftLength, int carrier) {
        requireLength(fftLength);
        requireCarrier(fftLength, carrier);
        return (carrier < 0) ? static_cast<std::size_t>(carrier + static_cast<int>(fftLength)) : static_cast<std::size_t>(carrier);
    }

    /// The inverse, the signed logical carrier a bin holds.
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

    /// The guard carriers, ascending in signed index. They are the complement of the two occupied sets. The three
    /// sets therefore cover `fftLength()`.
    [[nodiscard]] std::span<const int> guardCarriers() const noexcept { return _guard; }

    /// The bin each data carrier occupies, in `dataCarriers()` order. A symbol assembler gathers by this list.
    [[nodiscard]] std::span<const std::size_t> dataBins() const noexcept { return _dataBins; }
    [[nodiscard]] std::span<const std::size_t> pilotBins() const noexcept { return _pilotBins; }
    [[nodiscard]] std::span<const std::size_t> guardBins() const noexcept { return _guardBins; }

    /// The role of a bin. The index is bounds-checked. A caller that computes its own bin index makes the
    /// mistake this class exists to catch.
    [[nodiscard]] Occupancy occupancyOfBin(std::size_t bin) const {
        if (bin >= _fftLength) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: bin {} is outside a transform of {} bins", bin, _fftLength));
        }
        return _occupancy[bin];
    }

    [[nodiscard]] Occupancy occupancyOf(int carrier) const { return _occupancy[binOf(_fftLength, carrier)]; }

    /// A sync word is a whole symbol, emitted verbatim. Its length is therefore the one property to check. The
    /// check throws, because a short word would read as a symbol with silent carriers.
    void validateSyncWord(std::span<const std::complex<float>> word, std::size_t index = 0UZ) const {
        if (word.size() != _fftLength) {
            throw std::invalid_argument(std::format("gr::ofdm::CarrierMap: sync word {} holds {} carriers, not the {} a symbol has", index, word.size(), _fftLength));
        }
    }

    /// Refuses a pilot-symbol cycle that cannot be indexed. It runs once at configuration.
    /// `pilotSymbolIndex` is then plain arithmetic with no check.
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
     * The rule reads one cycle at two levels. The slots of a symbol take consecutive entries. The next symbol
     * continues where the previous one stopped, `pilot_symbols[(s * n_pilots + p) % len]`. A cycle whose length
     * is a multiple of `n_pilots` gives every symbol the same pilots. Any other cycle rotates them from symbol to
     * symbol. This function is the single definition of the rule.
     *
     * `symbolIndex` is reduced before the multiply. A symbol index counts from the stream start and grows large
     * over a long run. `s * n_pilots` would overflow long before `s` itself. `(s mod len) * n_pilots` is
     * congruent to it and cannot overflow.
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

    /// The range is asymmetric, as bin order is. `-fft_len/2` is the Nyquist bin, and `+fft_len/2` is not a
    /// carrier.
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
 * @brief A Schmidl-Cox preamble, one frequency-domain symbol whose time-domain upper half repeats its lower half.
 *
 * The repeat depends on which carriers hold values and not on the values. Write out the inverse transform.
 *
 *     x[n + N/2] = (1/N) * sum_k X[k] * exp(j*2*pi*k*n/N) * exp(j*pi*k)
 *
 * `exp(j*pi*k)` is `+1` on every even `k` and `-1` on every odd `k`. A symbol whose odd bins are all zero
 * therefore has `x[n + N/2] = x[n]` exactly, for every `n`. The result is exact and does not depend on the
 * values of the even carriers. The construction follows. Occupy the even carriers and leave the odd ones empty.
 * The time-domain half-repeat that a Schmidl-Cox correlator looks for then follows. `fftLength` is even. A
 * carrier's parity and its bin's parity are then the same, and "even carrier" and "even bin" name one set.
 *
 * The occupied band is the `occupied` carriers closest to DC, with DC excluded. They are `-occupied/2 … -1`
 * and `1 … occupied/2`. DC carries nothing. A receiver measures its frequency error against the preamble, and a
 * residual offset cannot be seen against a DC term. Each even member of the band takes a QPSK point scaled by
 * `sqrt(2)`. Each then holds twice the power a fully occupied symbol puts on a carrier. The total energy of the
 * preamble is `2 * evenCarrierCount`. That equals `occupied` whenever `occupied/2` is even, as it is in every
 * published numerology.
 *
 * The points come from a seeded PRNG and not from a published sequence. The correlator needs the lower half to
 * look like noise to a timing metric. It does not need a match to any standard's table. A standard's preamble is
 * a sync word the caller supplies verbatim. This helper serves when there is no standard.
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
