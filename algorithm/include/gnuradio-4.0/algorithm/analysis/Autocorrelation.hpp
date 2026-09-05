#ifndef GNURADIO_ALGORITHM_ANALYSIS_AUTOCORRELATION_HPP
#define GNURADIO_ALGORITHM_ANALYSIS_AUTOCORRELATION_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

/**
 * @brief What a signal repeats at: the windowed, averaged autocorrelation of a stream, with the scatter and the
 * detection threshold that decide whether a peak in it is a peak.
 *
 * Two different instruments share the machinery here and differ only in the sequence handed to it.
 *
 * - The **complex** kind estimates `R(tau) = E[x(t) x*(t - tau)]` on the samples as they arrive. It is coherent, so
 *   a frequency offset `df` multiplies `R(tau)` by `exp(j 2 pi df tau)` and leaves `|R(tau)|` untouched; the phase at
 *   a known lag is therefore a frequency estimate, `df = arg R(tau) / (2 pi tau)`, unambiguous while `|df| < 1/(2 tau)`.
 *   Its usable lag is bounded by the receiver's own coherence and not by this estimator.
 * - The **envelope** kind estimates the autocorrelation of `|x|^2` with its mean removed. It is immune to local
 *   oscillator phase noise, because `|s(t) exp(j phi(t))| = |s(t)|` exactly.
 *
 * **The mean must go, and the reason is arithmetic.** Write `y = |x|^2`, `P = E[y]`. Without the removal the
 * published ratio is `[P^2 + R_y~(tau)] / [P^2 + Var(y)]`: for circular complex Gaussian `x`, `Var(y) = P^2` and
 * every lag reads `0.5` plus half the real answer; a feature that is a fraction `f` of the fluctuation reads `f/2`;
 * and for a constant-modulus signal `Var(y) = 0` and the ratio is exactly `1` at every lag, so the instrument reads
 * full scale on the one class it can say nothing about. Removing the mean makes `R(0) = Var(y)` and every lag a pure
 * fluctuation correlation. The same argument applies to the complex kind whenever the input carries a DC offset,
 * which local-oscillator leakage puts in every direct-conversion capture, so the removal defaults on for both kinds.
 *
 * The estimate is Wiener-Khinchin (N. Wiener, *Generalized Harmonic Analysis*, 1930; A. Khinchin, 1934): the
 * correlation over lags `0..L` is the inverse transform of the magnitude-squared forward transform of one window,
 * zero-padded so that no circular wraparound reaches a published lag. Windows are averaged in the manner of
 * P. D. Welch, *IEEE Trans. Audio Electroacoust.* AU-15 (1967), with one difference that is stated here because it
 * is easy to assume otherwise: **overlap does not reduce the scatter of a linear estimator.** The variance of the
 * average is `sigma^4 / N_distinct` with `N_distinct = N + (K-1)H` whatever the overlap is, so two half-overlapped
 * windows are the same estimate as one window of twice the length at twice the transform cost. Overlap buys a record
 * every `K*H` samples instead of every `K*N`, and it buys capturing a feature shorter than a window whole instead of
 * split across a boundary.
 */
namespace gr::analysis {

/// @brief Which sequence the estimator is handed: the samples themselves, or their mean-removed squared magnitude.
enum class AcfKind : std::uint8_t { Complex, Envelope };

/// @brief Which divisor turns the accumulated products into an estimate of `R(tau)`.
enum class AcfNormalization : std::uint8_t { Biased, Unbiased };

[[nodiscard]] inline constexpr std::string_view acfKindName(AcfKind kind) noexcept { return kind == AcfKind::Complex ? "complex" : "envelope"; }

/// @brief The two spellings a setting may carry, and nothing else.
[[nodiscard]] inline constexpr std::optional<AcfKind> acfKindFrom(std::string_view name) noexcept {
    if (name == "complex") {
        return AcfKind::Complex;
    }
    if (name == "envelope") {
        return AcfKind::Envelope;
    }
    return std::nullopt;
}

[[nodiscard]] inline constexpr std::string_view acfNormalizationName(AcfNormalization normalization) noexcept { return normalization == AcfNormalization::Biased ? "biased" : "unbiased"; }

[[nodiscard]] inline constexpr std::optional<AcfNormalization> acfNormalizationFrom(std::string_view name) noexcept {
    if (name == "biased") {
        return AcfNormalization::Biased;
    }
    if (name == "unbiased") {
        return AcfNormalization::Unbiased;
    }
    return std::nullopt;
}

/// @brief The geometry and the estimator choices one accumulation is made under.
struct AcfConfig {
    std::size_t                 windowLength{};                                            ///< `N`, samples per window
    std::size_t                 maxLag{};                                                  ///< `L`, the longest published lag, in samples
    AcfKind                     kind           = AcfKind::Envelope;                        //
    AcfNormalization            normalization  = AcfNormalization::Unbiased;               //
    double                      overlap        = 0.5;                                      ///< fraction a window shares with the next, in [0, 1)
    std::size_t                 nAverages      = 16UZ;                                     ///< windows per published record
    bool                        removeMean     = true;                                     //
    gr::algorithm::window::Type window         = gr::algorithm::window::Type::Rectangular; //
    double                      falseAlarmRate = 1e-3;                                     ///< per published record, which is what sets the threshold
};

/**
 * @brief The transform length one window of `n` samples needs to publish lags `0..maxLag` without wraparound.
 *
 * The circular correlation of a length-`M` zero-padded sequence at lag `tau` is the linear correlation at `tau` plus
 * the linear correlation at `tau - M`. The linear correlation of an `n`-sample sequence vanishes outside
 * `|lag| <= n - 1`, so no published lag is contaminated exactly when `maxLag - M < -(n - 1)`, that is `M >= n + maxLag`.
 * The next power of two at or above that is taken, because it is the length the radix-2 path and the ordered
 * real-to-complex layout already agree on; the nearest length factoring into {2, 3, 4, 5} would be 1.33 to 1.78 times
 * shorter at the sizes this estimator is used at.
 */
[[nodiscard]] inline std::size_t acfTransformLength(std::size_t n, std::size_t maxLag) noexcept { return std::bit_ceil(n + maxLag); }

/**
 * @brief The standard deviation of a normalized autocorrelation at a lag standing on `nPairs` independent products.
 *
 * For `N` samples of circular complex white noise the biased estimator at `tau != 0` is a sum of `N - tau`
 * uncorrelated terms each of variance `sigma^4`, so `E|R^(tau)|^2 = (N - tau) sigma^4 / N^2` against `R^(0) = sigma^2`.
 * Dividing instead by the pair count gives `1/sqrt(N - tau)` for the unbiased estimator; averaging `K` windows
 * replaces `N - tau` by the count of distinct products behind the estimate. `|R^|/R^(0)` is Rayleigh with
 * `E|.|^2 = 1/nPairs`, so `P(|R^|/R^(0) > u) = exp(-u^2 nPairs)`.
 */
[[nodiscard]] inline double acfScatter(std::size_t nPairs) noexcept { return nPairs == 0UZ ? std::numeric_limits<double>::infinity() : 1. / std::sqrt(static_cast<double>(nPairs)); }

/**
 * @brief The normalized height a peak must reach for a record of `nLags` searched lags to false-alarm at rate `pFa`.
 *
 * Per lag `P(|R^|/R^(0) > gamma) = exp(-gamma^2 nPairs)`; over `nLags` searched lags the union bound gives a
 * per-record rate `nLags exp(-gamma^2 nPairs)`, so `gamma = sqrt(ln(nLags/pFa) / nPairs)` — the scatter times
 * `sqrt(ln(nLags/pFa))`, which depends on the searched-lag count and the stated rate and on nothing else.
 */
[[nodiscard]] inline double acfThreshold(std::size_t nPairs, std::size_t nLags, double pFa) noexcept {
    if (nPairs == 0UZ || nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(std::log(static_cast<double>(nLags) / pFa) / static_cast<double>(nPairs));
}

/**
 * @brief The same threshold in decibels above the record's own median, which is the unit a peak detector reads.
 *
 * On noise the record's values are Rayleigh with `E|.|^2 = 1/nPairs`, so the median is `sqrt(ln2 / nPairs)` and
 * `10 log10(gamma / median) = 5 log10(ln(nLags/pFa) / ln 2)` — the pair count cancels, leaving a figure that depends
 * on the searched-lag count and the stated rate alone. A median reference and not an absolute one is what makes this
 * usable on a real capture, where a receiver's own envelope wander sits as a broad pedestal under every lag.
 */
[[nodiscard]] inline double acfPeakDetectThresholdDb(std::size_t nLags, double pFa) noexcept {
    if (nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return 5. * std::log10(std::log(static_cast<double>(nLags) / pFa) / std::numbers::ln2);
}

/// @brief The standard normal deviate `t` with `P(|z| > t) = tail`, by bisection on the complementary error function.
/// One evaluation per published record, against a transform pair, so a closed rational approximation would buy
/// nothing that its own error did not cost back.
[[nodiscard]] inline double acfNormalDeviate(double tail) noexcept {
    if (!(tail > 0.) || !(tail < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    double low  = 0.;
    double high = 40.; // erfc(40/sqrt2) underflows every representable tail
    for (std::size_t step = 0UZ; step < 100UZ; ++step) {
        const double middle = 0.5 * (low + high);
        if (std::erfc(middle / std::numbers::sqrt2) > tail) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return 0.5 * (low + high);
}

/**
 * @brief The same two thresholds for an estimate that is real rather than circular complex.
 *
 * The envelope kind correlates `|x|^2`, and the complex kind of a real input correlates real samples; in both cases
 * the transform's magnitude squared is real and even, so `R(tau)` is real and `|R(tau)|/R(0)` is a folded normal
 * rather than a Rayleigh. Its tail is `erfc(u sqrt(nPairs/2))` and not `exp(-u^2 nPairs)`, which is far heavier at
 * the same height: at `nLags = 1024` and `pFa = 1e-3` the Rayleigh threshold realizes a per-lag rate of `2e-4` on a
 * real estimate against the `9.8e-7` it was designed for, a per-record rate of about 0.2. A real estimate therefore
 * takes `u = normalDeviate(pFa/nLags) / sqrt(nPairs)`, and its median is `0.6745/sqrt(nPairs)` rather than
 * `sqrt(ln2/nPairs)`, which is what the decibel form divides by.
 */
[[nodiscard]] inline double acfRealThreshold(std::size_t nPairs, std::size_t nLags, double pFa) noexcept {
    if (nPairs == 0UZ || nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return acfNormalDeviate(pFa / static_cast<double>(nLags)) / std::sqrt(static_cast<double>(nPairs));
}

/// The median of `|z|` for a standard normal `z`, which is the level `above_median` reads a real estimate against.
inline constexpr double kNormalAbsoluteMedian = 0.674489750196081743;

[[nodiscard]] inline double acfRealPeakDetectThresholdDb(std::size_t nLags, double pFa) noexcept {
    if (nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return 10. * std::log10(acfNormalDeviate(pFa / static_cast<double>(nLags)) / kNormalAbsoluteMedian);
}

/**
 * @brief The window grid every analysis estimator in this family segments a stream on.
 *
 * It holds the geometry (`windowLength`, `overlap` and the hop they imply), the taper and the taper's own
 * autocorrelation `r_w`, and the running count of windows and distinct samples that is every such estimator's
 * scatter denominator. It carries no sample buffer: a caller presents a span, is told how many leading samples the
 * grid is done with, and presents the remainder again with more behind it. The grid is anchored at the stream start,
 * so the same input split differently yields the same windows.
 */
template<typename T>
struct WindowedSegmenter {
    std::size_t                 windowLength = 0UZ;
    std::size_t                 maxLag       = 0UZ;
    double                      overlap      = 0.;
    std::size_t                 hop          = 1UZ;
    gr::algorithm::window::Type windowType   = gr::algorithm::window::Type::Rectangular;

    std::vector<float>  taper{};    ///< `w[0..windowLength-1]`
    std::vector<double> taperAcf{}; ///< `r_w(tau) = sum_n w[n] w[n+tau]` for `tau = 0..maxLag`

    std::uint64_t streamAt     = 0ULL; ///< absolute index of the next presented span's first sample
    std::size_t   windows      = 0UZ;  ///< windows in the accumulation in progress
    std::uint64_t groupStartAt = 0ULL; ///< absolute index of that accumulation's first window's first sample

    /// @brief The hop an overlap implies, never less than one sample.
    [[nodiscard]] static std::size_t hopFor(std::size_t length, double fraction) noexcept { return std::max(1UZ, static_cast<std::size_t>(std::llround(static_cast<double>(length) * (1. - fraction)))); }

    /// @brief Adopt a geometry and a taper. `taperAcf` is left empty for the caller to fill, since the only cheap way
    /// to compute a correlation of the taper is the transform the estimator already holds.
    void prepare(std::size_t length, std::size_t lag, double fraction, gr::algorithm::window::Type type) {
        windowLength = length;
        maxLag       = lag;
        overlap      = fraction;
        windowType   = type;
        hop          = hopFor(length, fraction);
        taper        = gr::algorithm::window::create<float>(type, length);
        taperAcf.assign(lag + 1UZ, 0.);
        reset();
    }

    /// @brief Drop the accumulation in progress, keeping the stream position.
    void restart() noexcept { windows = 0UZ; }

    /// @brief Drop everything, the stream position included.
    void reset() noexcept {
        streamAt     = 0ULL;
        groupStartAt = 0ULL;
        restart();
    }

    /// @brief State where the next presented span's first sample sits in the stream.
    void seek(std::uint64_t absolute) noexcept {
        streamAt     = absolute;
        groupStartAt = absolute;
    }

    /// @brief Take one window into the accumulation, remembering where the group started.
    void noteWindow(std::uint64_t absoluteStart) noexcept {
        if (windows == 0UZ) {
            groupStartAt = absoluteStart;
        }
        ++windows;
    }

    /// @brief Distinct input samples behind the accumulation in progress, `N + (K-1)H`.
    [[nodiscard]] std::size_t distinctSamples() const noexcept { return windows == 0UZ ? 0UZ : windowLength + (windows - 1UZ) * hop; }

    /**
     * @brief Independent sample products standing behind lag @p lag, which is the scatter's denominator.
     *
     * Each window contributes `windowLength - lag` products, so `K` windows contribute `K (windowLength - lag)` when
     * their product sets are disjoint. When the hop is short enough that consecutive windows' product sets overlap
     * (`hop <= windowLength - lag`) those products repeat, and the count is the union's size `N + (K-1)H - lag`
     * instead. The smaller of the two is the count in either regime, and the two are equal at `hop = windowLength - lag`.
     */
    [[nodiscard]] std::size_t independentPairs(std::size_t lag) const noexcept {
        if (windows == 0UZ || lag >= windowLength) {
            return 0UZ;
        }
        return std::min(windows * (windowLength - lag), distinctSamples() - lag);
    }

    /**
     * @brief Present @p in to the grid, calling @p onWindow with each whole window that lies inside it.
     * @return how many leading samples the grid is done with; the remainder is undecided and is presented again.
     */
    template<typename Fn>
    std::size_t forEachWindow(std::span<const T> in, Fn&& onWindow) {
        std::size_t consumed = 0UZ;
        if (windowLength == 0UZ) {
            return 0UZ;
        }
        while (consumed + windowLength <= in.size()) {
            onWindow(in.subspan(consumed, windowLength), streamAt + static_cast<std::uint64_t>(consumed));
            consumed += hop;
        }
        streamAt += static_cast<std::uint64_t>(consumed);
        return consumed;
    }
};

/// @brief One published record: the normalized correlation over lags `0..L` with everything needed to read it.
struct AcfResult {
    std::span<const float> magnitude;          ///< `|R(tau)|/R(0)`, lags `0..L`; `magnitude[0]` is exactly 1
    std::span<const float> phase;              ///< `arg R(tau)` in radians; empty for the envelope kind
    double                 power{};            ///< `R(0)` before normalization, so the scale the ratio throws away survives
    std::uint64_t          sampleStart{};      ///< absolute index of the first averaged window's first sample
    std::size_t            nAveraged{};        ///< windows behind this record
    std::size_t            nDistinctSamples{}; ///< `N + (nAveraged-1) H`
    std::size_t            nPairs{};           ///< independent products behind the worst lag `L`
    double                 scatter{};          ///< `acfScatter(nPairs)`
    double                 threshold{};        ///< the height a peak must clear at the worst lag, for this estimate's own tail
    double                 peakThresholdDb{};  ///< the same threshold in decibels over the record's median
    bool                   realValued{};       ///< the estimate is real, so its tail is folded normal rather than Rayleigh
};

/**
 * @brief The windowed, averaged autocorrelation of a stream, scheduler-free.
 *
 * `process()` carries no sample buffer: it consumes whole hops whose window lies inside the span it was given and
 * returns how many leading samples it is done with, so the caller owns the undecided tail and can consume every span
 * it is handed. Averaging is coherent over the complex estimate with the magnitude taken once at the end: at a fixed
 * lag `R(tau)` has the same value and phase in every window of a stationary signal, while averaging `|R^|` would
 * carry each window's noise magnitude into the mean as a positive bias that no mean removal can undo.
 *
 * The transform runs in double precision. The published ratio at a long lag under the unbiased normalization divides
 * by as few as one sample product, and a single-precision transform's rounding is then a relative error of about
 * `1e-5` on that lag — the same size as the agreement a direct sum is asked for. Double leaves twelve orders of
 * margin for twice the memory traffic.
 *
 * A window whose sequence has no fluctuation left after the mean is removed produces no estimate at all: a
 * constant-modulus signal has no envelope autocorrelation, and the published curve would be whatever the estimator
 * did with `0/0`. Such a window is counted in `nDegenerate` and contributes to no record, so a stream that is
 * constant-modulus throughout publishes nothing rather than publishing a flat unity curve.
 */
template<typename T>
requires(std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>)
struct Autocorrelation {
    using Scalar    = double;
    using Transform = std::complex<Scalar>;
    template<typename U>
    using AlignedVector = std::vector<U, gr::allocator::Aligned<U>>;

    /// The fluctuation power, relative to the sequence's own power, below which a window states nothing. A
    /// constant-modulus waveform stored in single precision fluctuates by a few units in the last place, which is a
    /// relative power of about `1e-14`; the smallest envelope feature any of this estimator's users can resolve is
    /// larger than this bound by ten orders of magnitude.
    static constexpr double kDegenerateFluctuation = 1e-12;

    std::size_t nWindows{};        ///< windows the grid produced, degenerate ones included
    std::size_t nRecords{};        ///< records of a full `nAverages` windows
    std::size_t nSamples{};        ///< input samples the estimator is done with
    std::size_t nDegenerate{};     ///< windows with no fluctuation to correlate
    std::size_t nPartialFlushes{}; ///< records published with fewer than `nAverages` windows
    std::size_t nPartialWindows{}; ///< windows those partial records were made of

    AcfConfig            _config{};
    WindowedSegmenter<T> _segmenter{};
    std::size_t          _transformLength = 0UZ;

    gr::algorithm::FFT<Transform, Transform, gr::algorithm::Direction::Forward>  _forward{};
    gr::algorithm::FFT<Transform, Transform, gr::algorithm::Direction::Backward> _backward{};

    AlignedVector<Transform> _work{};     ///< one window, tapered and zero-padded, then the transform's output
    AlignedVector<Transform> _spectrum{}; ///< the forward transform's output, then its magnitude squared
    std::vector<Transform>   _accumulator{};
    std::vector<float>       _magnitude{};
    std::vector<float>       _phase{};
    std::vector<double>      _noiseCurve{}; ///< the modeled median of each published lag, which sets the median-referred threshold

    /**
     * @brief Adopt a configuration, sizing every buffer and building the taper and its own autocorrelation.
     * @throws std::invalid_argument naming the setting and the bound it broke.
     */
    void prepare(const AcfConfig& config) {
        if (config.windowLength < 16UZ || config.windowLength > 1048576UZ) {
            throw std::invalid_argument(std::format("gr::analysis::Autocorrelation: windowLength must lie in [16, 1048576], got {}", config.windowLength));
        }
        if (config.maxLag == 0UZ) {
            throw std::invalid_argument("gr::analysis::Autocorrelation: maxLag is the longest published lag and must be at least 1");
        }
        if (config.maxLag >= config.windowLength) {
            throw std::invalid_argument(std::format("gr::analysis::Autocorrelation: maxLag ({}) must be below windowLength ({}): at a lag of a whole window there is not one sample pair separated by it inside a window, so the taper's own autocorrelation is zero and the estimate has nothing to average", config.maxLag, config.windowLength));
        }
        if (config.nAverages == 0UZ) {
            throw std::invalid_argument("gr::analysis::Autocorrelation: nAverages counts the windows a record is made of and must be at least 1");
        }
        if (!(config.overlap >= 0.) || !(config.overlap < 1.)) {
            throw std::invalid_argument(std::format("gr::analysis::Autocorrelation: overlap is the fraction a window shares with the next and must lie in [0, 1), got {}", config.overlap));
        }
        if (!(config.falseAlarmRate > 0.) || !(config.falseAlarmRate < 1.)) {
            throw std::invalid_argument(std::format("gr::analysis::Autocorrelation: falseAlarmRate is a probability per record and must lie in (0, 1), got {}", config.falseAlarmRate));
        }

        _config          = config;
        _transformLength = acfTransformLength(config.windowLength, config.maxLag);
        _segmenter.prepare(config.windowLength, config.maxLag, config.overlap, config.window);

        _forward  = gr::algorithm::FFT<Transform, Transform, gr::algorithm::Direction::Forward>{};
        _backward = gr::algorithm::FFT<Transform, Transform, gr::algorithm::Direction::Backward>{};
        _work.assign(_transformLength, Transform{});
        _spectrum.assign(_transformLength, Transform{});
        _accumulator.assign(config.maxLag + 1UZ, Transform{});
        _magnitude.assign(config.maxLag + 1UZ, 0.f);
        _phase.assign(config.maxLag + 1UZ, 0.f);
        _noiseCurve.assign(config.maxLag, 0.);

        buildTaperAutocorrelation();
        reset();
    }

    /// @brief Return to the state a fresh stream starts from, counters included.
    void reset() {
        _segmenter.reset();
        std::ranges::fill(_accumulator, Transform{});
        nWindows = nRecords = nSamples = nDegenerate = nPartialFlushes = nPartialWindows = 0UZ;
    }

    /// @brief Drop the accumulation in progress, keeping the stream position and the counters.
    void restart() {
        _segmenter.restart();
        std::ranges::fill(_accumulator, Transform{});
    }

    /// @brief Forward a stream position to the grid, so that a record's own start index is absolute.
    void seek(std::uint64_t absolute) noexcept { _segmenter.seek(absolute); }

    /// @brief Move the windows per record. It changes when a record is published and nothing about the estimate, so
    /// the accumulation in progress and the window grid both survive it.
    void setAveraging(std::size_t nAverages) {
        if (nAverages == 0UZ) {
            throw std::invalid_argument("gr::analysis::Autocorrelation: nAverages counts the windows a record is made of and must be at least 1");
        }
        _config.nAverages = nAverages;
    }

    /// @brief Move the per-record false-alarm rate the published threshold is computed for.
    void setFalseAlarmRate(double pFa) {
        if (!(pFa > 0.) || !(pFa < 1.)) {
            throw std::invalid_argument(std::format("gr::analysis::Autocorrelation: falseAlarmRate is a probability per record and must lie in (0, 1), got {}", pFa));
        }
        _config.falseAlarmRate = pFa;
    }

    /// @brief Whether the correlated sequence is real, which decides the tail a threshold is computed from: the
    /// envelope kind squares magnitudes, and the complex kind of a real input correlates real samples.
    [[nodiscard]] bool                    realValued() const noexcept { return _config.kind == AcfKind::Envelope || std::is_same_v<T, float>; }
    [[nodiscard]] std::uint64_t           streamAt() const noexcept { return _segmenter.streamAt; }
    [[nodiscard]] bool                    configured() const noexcept { return _segmenter.windowLength != 0UZ; }
    [[nodiscard]] std::size_t             hop() const noexcept { return _segmenter.hop; }
    [[nodiscard]] std::size_t             transformLength() const noexcept { return _transformLength; }
    [[nodiscard]] const AcfConfig&        config() const noexcept { return _config; }
    [[nodiscard]] std::size_t             windowsPending() const noexcept { return _segmenter.windows; }
    [[nodiscard]] std::span<const double> taperAutocorrelation() const noexcept { return std::span<const double>(_segmenter.taperAcf); }

    /**
     * @brief Estimate over every whole window inside @p in, reporting each completed record to @p onResult.
     * @return how many leading samples the estimator is done with; the remainder is presented again.
     */
    template<typename Sink>
    std::size_t process(std::span<const T> in, Sink&& onResult) {
        if (!configured()) {
            return 0UZ;
        }
        const std::size_t consumed = _segmenter.forEachWindow(in, [&](std::span<const T> window, std::uint64_t at) {
            ++nWindows;
            if (!accumulate(window)) {
                ++nDegenerate;
                return;
            }
            _segmenter.noteWindow(at);
            if (_segmenter.windows >= _config.nAverages) {
                publish(onResult);
                ++nRecords;
                restart();
            }
        });
        nSamples += consumed;
        return consumed;
    }

    /**
     * @brief Publish the accumulation in progress as a short record marked with the windows it actually holds.
     * @return whether a record was published; an accumulation holding no complete window publishes nothing.
     */
    template<typename Sink>
    bool flush(Sink&& onResult) {
        if (_segmenter.windows == 0UZ) {
            return false;
        }
        nPartialWindows += _segmenter.windows;
        ++nPartialFlushes;
        publish(onResult);
        restart();
        return true;
    }

private:
    /// @brief The taper's own autocorrelation, through the same transform pair the estimate runs on.
    void buildTaperAutocorrelation() {
        std::ranges::fill(_work, Transform{});
        for (std::size_t n = 0UZ; n < _segmenter.windowLength; ++n) {
            _work[n] = Transform(static_cast<Scalar>(_segmenter.taper[n]), Scalar{0});
        }
        correlate();
        const Scalar inverse = Scalar{1} / static_cast<Scalar>(_transformLength);
        for (std::size_t lag = 0UZ; lag <= _config.maxLag; ++lag) {
            _segmenter.taperAcf[lag] = _work[lag].real() * inverse;
        }
    }

    /// @brief `_work` holds a zero-padded sequence on entry and `transformLength` times its circular autocorrelation
    /// on exit, the transform pair being unnormalized.
    void correlate() {
        _forward.compute(_work, _spectrum);
        for (Transform& bin : _spectrum) {
            bin = Transform(bin.real() * bin.real() + bin.imag() * bin.imag(), Scalar{0});
        }
        _backward.compute(_spectrum, _work);
    }

    /**
     * @brief Fold one window into the accumulator.
     * @return whether the window carried a fluctuation to correlate; a window that did not is degenerate.
     */
    [[nodiscard]] bool accumulate(std::span<const T> window) {
        const std::size_t n = _segmenter.windowLength;
        std::ranges::fill(_work, Transform{});

        Transform mean{};
        Scalar    scale = Scalar{0};
        for (std::size_t k = 0UZ; k < n; ++k) {
            const Transform value = sequence(window[k]);
            _work[k]              = value;
            mean += value;
            scale += std::norm(value);
        }
        mean /= static_cast<Scalar>(n);
        if (!_config.removeMean) {
            mean = Transform{};
        }

        Scalar energy = Scalar{0};
        for (std::size_t k = 0UZ; k < n; ++k) {
            const Transform centered = _work[k] - mean;
            energy += std::norm(centered);
            _work[k] = centered * static_cast<Scalar>(_segmenter.taper[k]);
        }
        if (!(scale > Scalar{0}) || !(energy > kDegenerateFluctuation * scale)) {
            return false;
        }

        correlate();
        const Scalar inverse = Scalar{1} / static_cast<Scalar>(_transformLength);
        for (std::size_t lag = 0UZ; lag <= _config.maxLag; ++lag) {
            _accumulator[lag] += _work[lag] * inverse;
        }
        return true;
    }

    /// @brief The sequence element one input sample contributes, promoted to the transform's own type.
    [[nodiscard]] Transform sequence(const T& sample) const noexcept {
        if constexpr (std::is_same_v<T, float>) {
            const Scalar value = static_cast<Scalar>(sample);
            return _config.kind == AcfKind::Envelope ? Transform(value * value, Scalar{0}) : Transform(value, Scalar{0});
        } else {
            const Scalar re = static_cast<Scalar>(sample.real());
            const Scalar im = static_cast<Scalar>(sample.imag());
            return _config.kind == AcfKind::Envelope ? Transform(re * re + im * im, Scalar{0}) : Transform(re, im);
        }
    }

    /**
     * @brief The worst-lag threshold expressed as decibels over the record's own median, which is the unit a
     * median-referring peak detector reads.
     *
     * The scatter falls with the lag, because a longer lag stands on fewer products, so the median of a record's
     * noise curve is not the scale of its worst lag: at `L = N/4` and one window it is the scale of lag `L/2`, seven
     * per cent smaller, and a threshold that assumed otherwise would sit seven per cent low and fire six times its
     * design rate. This models each published lag's own noise scale — `1/sqrt(nPairs(tau))` under the unbiased
     * normalization, times `r_w(tau)/r_w(0)` under the biased one — takes the median of those, and states the
     * threshold against it. Lag zero is excluded: it is exactly one by construction and describes no noise.
     */
    [[nodiscard]] Scalar medianReferredThresholdDb(Scalar worst, bool real) {
        const std::size_t lags = _config.maxLag;
        if (lags == 0UZ || !(worst > Scalar{0}) || !std::isfinite(worst)) {
            return std::numeric_limits<Scalar>::infinity();
        }
        const Scalar shape = real ? kNormalAbsoluteMedian : std::sqrt(std::numbers::ln2);
        for (std::size_t lag = 1UZ; lag <= lags; ++lag) {
            const std::size_t pairs = _segmenter.independentPairs(lag);
            Scalar            scale = pairs == 0UZ ? std::numeric_limits<Scalar>::infinity() : Scalar{1} / std::sqrt(static_cast<Scalar>(pairs));
            if (_config.normalization == AcfNormalization::Biased && _segmenter.taperAcf[0UZ] > Scalar{0}) {
                scale *= _segmenter.taperAcf[lag] / _segmenter.taperAcf[0UZ];
            }
            _noiseCurve[lag - 1UZ] = shape * scale;
        }
        const auto middle = _noiseCurve.begin() + static_cast<std::ptrdiff_t>(_noiseCurve.size() / 2UZ);
        std::nth_element(_noiseCurve.begin(), middle, _noiseCurve.end());
        const Scalar median = *middle;
        return median > Scalar{0} && std::isfinite(median) ? 10. * std::log10(worst / median) : std::numeric_limits<Scalar>::infinity();
    }

    /// @brief Normalize the accumulation and hand it to the sink.
    template<typename Sink>
    void publish(Sink&& onResult) {
        const std::size_t averaged = _segmenter.windows;
        const std::size_t lags     = _config.maxLag;
        const Scalar      zeroLag  = std::abs(_accumulator[0UZ]);
        const Scalar      taperAt0 = _segmenter.taperAcf[0UZ];

        for (std::size_t lag = 0UZ; lag <= lags; ++lag) {
            const Scalar taperAtLag = _segmenter.taperAcf[lag];
            Scalar       ratio      = Scalar{0};
            if (zeroLag > Scalar{0} && (_config.normalization == AcfNormalization::Biased || taperAtLag > Scalar{0})) {
                ratio = std::abs(_accumulator[lag]) / zeroLag;
                if (_config.normalization == AcfNormalization::Unbiased) {
                    ratio *= taperAt0 / taperAtLag;
                }
            }
            _magnitude[lag] = static_cast<float>(ratio);
            _phase[lag]     = static_cast<float>(std::arg(_accumulator[lag]));
        }

        const std::size_t pairs = _segmenter.independentPairs(lags);
        const bool        real  = realValued();
        // The published threshold is on the scale of the published values: the biased normalization scales every
        // lag by the taper's own lag window, and its noise with it, so the height a peak must reach at the worst
        // lag scales the same way. Comparing a biased estimate against the unbiased threshold would demand of it a
        // height its own normalization has already removed.
        Scalar worst      = real ? acfRealThreshold(pairs, lags, _config.falseAlarmRate) : acfThreshold(pairs, lags, _config.falseAlarmRate);
        Scalar dispersion = acfScatter(pairs);
        if (_config.normalization == AcfNormalization::Biased && taperAt0 > Scalar{0}) {
            const Scalar lagWindow = _segmenter.taperAcf[lags] / taperAt0;
            worst *= lagWindow;
            dispersion *= lagWindow;
        }
        const AcfResult result{
            .magnitude        = std::span<const float>(_magnitude),
            .phase            = _config.kind == AcfKind::Complex ? std::span<const float>(_phase) : std::span<const float>{},
            .power            = zeroLag / (static_cast<Scalar>(averaged) * taperAt0),
            .sampleStart      = _segmenter.groupStartAt,
            .nAveraged        = averaged,
            .nDistinctSamples = _segmenter.distinctSamples(),
            .nPairs           = pairs,
            .scatter          = dispersion,
            .threshold        = worst,
            .peakThresholdDb  = medianReferredThresholdDb(worst, real),
            .realValued       = real,
        };
        onResult(result);
    }
};

} // namespace gr::analysis

#endif // GNURADIO_ALGORITHM_ANALYSIS_AUTOCORRELATION_HPP
