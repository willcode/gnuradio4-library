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
#include <string_view>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

/**
 * @brief The windowed, averaged autocorrelation of a stream, with its scatter and its detection threshold.
 *
 * The threshold decides whether a peak in the autocorrelation is a real feature. The estimator has two kinds. They
 * share the code and differ only in the sequence they correlate.
 *
 * - The **complex** kind estimates `R(tau) = E[x(t) x*(t - tau)]` on the input samples. The estimate is coherent.
 *   A frequency offset `df` multiplies `R(tau)` by `exp(j 2 pi df tau)` and leaves `|R(tau)|` unchanged. The phase
 *   at a known lag is therefore a frequency estimate, `df = arg R(tau) / (2 pi tau)`. The estimate is unambiguous
 *   while `|df| < 1/(2 tau)`. The receiver's coherence bounds the usable lag. This estimator does not.
 * - The **envelope** kind estimates the autocorrelation of `|x|^2` with its mean removed. Local oscillator phase
 *   noise does not change it, because `|s(t) exp(j phi(t))| = |s(t)|` exactly.
 *
 * **The mean is removed, for an arithmetic reason.** Write `y = |x|^2` and `P = E[y]`. Without the removal the
 * published ratio is `[P^2 + R_y~(tau)] / [P^2 + Var(y)]`. For circular complex Gaussian `x`, `Var(y) = P^2`. Every
 * lag then reads `0.5` plus half the true value. A feature that is a fraction `f` of the fluctuation reads `f/2`. For
 * a constant-modulus signal `Var(y) = 0`, and the ratio is exactly `1` at every lag. The estimate would read full
 * scale on the one signal class it can say nothing about. With the mean removed, `R(0) = Var(y)` and every lag is a
 * pure fluctuation correlation. The same holds for the complex kind when the input carries a DC offset.
 * Local-oscillator leakage puts a DC offset in every direct-conversion capture. The removal is on by default for
 * both kinds.
 *
 * The estimate uses the Wiener-Khinchin theorem (N. Wiener, *Generalized Harmonic Analysis*, 1930; A. Khinchin,
 * 1934). The correlation over lags `0..L` is the inverse transform of the magnitude-squared forward transform of one
 * window. The window is zero-padded. The padding keeps circular wraparound out of every published lag.
 *
 * Windows are averaged after P. D. Welch, *IEEE Trans. Audio Electroacoust.* AU-15 (1967). One point differs from
 * Welch's spectral use. **Overlap does not reduce the scatter of a linear estimator.** The variance of the average is
 * `sigma^4 / N_distinct` with `N_distinct = N + (K-1)H`, whatever the overlap. Two half-overlapped windows give the
 * same estimate as one window of twice the length, at twice the transform cost. Overlap gives a record every `K*H`
 * samples instead of every `K*N`. Overlap also captures a feature shorter than a window whole instead of split
 * across a window boundary.
 */
namespace gr::analysis {

/// @brief The sequence the estimator correlates, the samples themselves or their mean-removed squared magnitude.
enum class AcfKind : std::uint8_t { Complex, Envelope };

/// @brief Which divisor turns the accumulated products into an estimate of `R(tau)`.
enum class AcfNormalization : std::uint8_t { Biased, Unbiased };

[[nodiscard]] inline constexpr std::string_view acfKindName(AcfKind kind) noexcept { return kind == AcfKind::Complex ? "complex" : "envelope"; }

/// @brief The kind named by a setting's text, `complex` or `envelope`, or no value for any other text.
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
    double                      falseAlarmRate = 1e-3;                                     ///< per published record, the rate the threshold is set for
};

/**
 * @brief The transform length one window of `n` samples needs to publish lags `0..maxLag` without wraparound.
 *
 * The circular correlation of a length-`M` zero-padded sequence at lag `tau` is the linear correlation at `tau` plus
 * the linear correlation at `tau - M`. The linear correlation of an `n`-sample sequence vanishes outside
 * `|lag| <= n - 1`. No published lag is contaminated exactly when `maxLag - M < -(n - 1)`, that is when
 * `M >= n + maxLag`. The function returns the next power of two at or above that length. The radix-2 path and the
 * ordered real-to-complex layout both accept that length. The nearest length that factors into {2, 3, 4, 5} would be
 * 1.33 to 1.78 times shorter at the sizes this estimator runs at.
 */
[[nodiscard]] inline std::size_t acfTransformLength(std::size_t n, std::size_t maxLag) noexcept { return std::bit_ceil(n + maxLag); }

/**
 * @brief The standard deviation of a normalized autocorrelation at a lag standing on `nPairs` independent products.
 *
 * For `N` samples of circular complex white noise, the biased estimator at `tau != 0` is a sum of `N - tau`
 * uncorrelated terms. Each term has variance `sigma^4`. This gives `E|R^(tau)|^2 = (N - tau) sigma^4 / N^2` against
 * `R^(0) = sigma^2`. The unbiased estimator divides by the pair count instead and gives `1/sqrt(N - tau)`. Averaging
 * `K` windows replaces `N - tau` by the count of distinct products behind the estimate. `|R^|/R^(0)` is Rayleigh
 * with `E|.|^2 = 1/nPairs`. Therefore `P(|R^|/R^(0) > u) = exp(-u^2 nPairs)`.
 */
[[nodiscard]] inline double acfScatter(std::size_t nPairs) noexcept { return nPairs == 0UZ ? std::numeric_limits<double>::infinity() : 1. / std::sqrt(static_cast<double>(nPairs)); }

/**
 * @brief The normalized height a peak must reach for a record of `nLags` searched lags to false-alarm at rate `pFa`.
 *
 * Per lag, `P(|R^|/R^(0) > gamma) = exp(-gamma^2 nPairs)`. Over `nLags` searched lags the union bound gives a
 * per-record rate of `nLags exp(-gamma^2 nPairs)`. Solving for the threshold gives
 * `gamma = sqrt(ln(nLags/pFa) / nPairs)`. That is the scatter times `sqrt(ln(nLags/pFa))`. The factor depends only on
 * the searched-lag count and the stated rate.
 */
[[nodiscard]] inline double acfThreshold(std::size_t nPairs, std::size_t nLags, double pFa) noexcept {
    if (nPairs == 0UZ || nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(std::log(static_cast<double>(nLags) / pFa) / static_cast<double>(nPairs));
}

/**
 * @brief The same threshold in decibels above the record's own median, the unit a peak detector reads.
 *
 * On noise the record's values are Rayleigh with `E|.|^2 = 1/nPairs`. The median is `sqrt(ln2 / nPairs)`, and
 * `10 log10(gamma / median) = 5 log10(ln(nLags/pFa) / ln 2)`. The pair count cancels. The figure depends only on the
 * searched-lag count and the stated rate. The reference is the median and not an absolute level. A real capture has
 * the receiver's own envelope wander as a broad pedestal under every lag. The median reference keeps the threshold
 * usable on such a capture.
 */
[[nodiscard]] inline double acfPeakDetectThresholdDb(std::size_t nLags, double pFa) noexcept {
    if (nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return 5. * std::log10(std::log(static_cast<double>(nLags) / pFa) / std::numbers::ln2);
}

/// @brief The standard normal deviate `t` with `P(|z| > t) = tail`, by bisection on the complementary error function.
/// The function runs once per published record, next to a transform pair. A closed rational approximation would save
/// little time and would add its own error.
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
 * @brief The same two thresholds for an estimate that is real instead of circular complex.
 *
 * The envelope kind correlates `|x|^2`. The complex kind of a real input correlates real samples. In both cases the
 * transform's magnitude squared is real and even. `R(tau)` is then real, and `|R(tau)|/R(0)` is a folded normal and
 * not a Rayleigh. Its tail is `erfc(u sqrt(nPairs/2))` and not `exp(-u^2 nPairs)`. That tail is far heavier at the
 * same height. At `nLags = 1024` and `pFa = 1e-3` the Rayleigh threshold gives a per-lag rate of `2e-4` on a real
 * estimate. It was designed for `9.8e-7`. The per-record rate is then about 0.2. A real estimate therefore takes
 * `u = normalDeviate(pFa/nLags) / sqrt(nPairs)`. Its median is `0.6745/sqrt(nPairs)` in place of
 * `sqrt(ln2/nPairs)`. The decibel form divides by that median.
 */
[[nodiscard]] inline double acfRealThreshold(std::size_t nPairs, std::size_t nLags, double pFa) noexcept {
    if (nPairs == 0UZ || nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return acfNormalDeviate(pFa / static_cast<double>(nLags)) / std::sqrt(static_cast<double>(nPairs));
}

/// The median of `|z|` for a standard normal `z`. The decibel threshold of a real estimate is referred to it.
inline constexpr double kNormalAbsoluteMedian = 0.674489750196081743;

[[nodiscard]] inline double acfRealPeakDetectThresholdDb(std::size_t nLags, double pFa) noexcept {
    if (nLags == 0UZ || !(pFa > 0.) || !(pFa < 1.)) {
        return std::numeric_limits<double>::infinity();
    }
    return 10. * std::log10(acfNormalDeviate(pFa / static_cast<double>(nLags)) / kNormalAbsoluteMedian);
}

/**
 * @brief The window grid an analysis estimator segments a stream on.
 *
 * The grid holds the geometry, which is `windowLength`, `overlap` and the hop they imply. It holds the taper and the
 * taper's own autocorrelation `r_w`. It counts the windows and the distinct samples. That count is the estimator's
 * scatter denominator. The grid holds no sample buffer. The caller presents a span and gets back how many leading
 * samples the grid has finished with. The caller presents the remainder again, with more samples after it. The grid is
 * anchored at the stream start. The same input split at different points gives the same windows.
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

    /// @brief The hop an overlap implies, at least one sample.
    [[nodiscard]] static std::size_t hopFor(std::size_t length, double fraction) noexcept { return std::max(1UZ, static_cast<std::size_t>(std::llround(static_cast<double>(length) * (1. - fraction)))); }

    /// @brief Adopt a geometry and a taper. `taperAcf` is left zeroed for the caller to fill. The estimator fills it
    /// with the transform it already holds, which is the least costly way to correlate the taper.
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
     * @brief Independent sample products behind lag @p lag, the scatter's denominator.
     *
     * Each window contributes `windowLength - lag` products. `K` windows with disjoint product sets contribute
     * `K (windowLength - lag)`. When `hop <= windowLength - lag`, the product sets of consecutive windows overlap and
     * those products repeat. The count is then the size of the union, `N + (K-1)H - lag`. The smaller of the two
     * values is the count in both cases. The two are equal at `hop = windowLength - lag`.
     */
    [[nodiscard]] std::size_t independentPairs(std::size_t lag) const noexcept {
        if (windows == 0UZ || lag >= windowLength) {
            return 0UZ;
        }
        return std::min(windows * (windowLength - lag), distinctSamples() - lag);
    }

    /**
     * @brief Present @p in to the grid, calling @p onWindow with each whole window that lies inside it.
     * @return how many leading samples the grid has finished with. The caller presents the remainder again.
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

/// @brief One published record, the normalized correlation over lags `0..L` with everything needed to read it.
struct AcfResult {
    std::span<const float> magnitude;          ///< `|R(tau)|/R(0)` for lags `0..L`, with `magnitude[0]` exactly 1
    std::span<const float> phase;              ///< `arg R(tau)` in radians, empty for the envelope kind
    double                 power{};            ///< `R(0)` before normalization, the scale the ratio removes
    std::uint64_t          sampleStart{};      ///< absolute index of the first averaged window's first sample
    std::size_t            nAveraged{};        ///< windows behind this record
    std::size_t            nDistinctSamples{}; ///< `N + (nAveraged-1) H`
    std::size_t            nPairs{};           ///< independent products behind the worst lag `L`
    double                 scatter{};          ///< `acfScatter(nPairs)`
    double                 threshold{};        ///< the height a peak must clear at the worst lag, for this estimate's own tail
    double                 peakThresholdDb{};  ///< the same threshold in decibels over the record's median
    bool                   realValued{};       ///< the estimate is real, with a folded normal tail in place of a Rayleigh tail
};

/**
 * @brief The windowed, averaged autocorrelation of a stream, without a scheduler.
 *
 * `process()` holds no sample buffer. It consumes each whole hop whose window lies inside the given span. It returns
 * how many leading samples it has finished with. The caller keeps the unused tail and can consume every span it
 * receives. Averaging is coherent over the complex estimate, and the magnitude is taken once at the end. At a fixed
 * lag, `R(tau)` has the same value and phase in every window of a stationary signal. Averaging `|R^|` instead would
 * add each window's noise magnitude to the mean as a positive bias. Mean removal cannot undo that bias.
 *
 * The transform runs in double precision. Under the unbiased normalization, the published ratio at a long lag
 * divides by as few as one sample product. A single-precision transform then has a rounding error of about `1e-5`
 * relative on that lag. That is the same size as the agreement required against a direct sum. Double precision
 * leaves twelve orders of margin for twice the memory traffic.
 *
 * A window with no fluctuation left after mean removal produces no estimate. A constant-modulus signal has no
 * envelope autocorrelation, and its curve would be the result of `0/0`. The estimator counts such a window in
 * `nDegenerate` and adds it to no record. A stream that is constant-modulus throughout publishes no record. It does
 * not publish a flat unity curve.
 */
template<typename T>
requires(std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>)
struct Autocorrelation {
    using Scalar    = double;
    using Transform = std::complex<Scalar>;
    template<typename U>
    using AlignedVector = std::vector<U, gr::allocator::Aligned<U>>;

    /// The fluctuation power, relative to the sequence's own power, below which a window is degenerate. A
    /// constant-modulus waveform stored in single precision fluctuates by a few units in the last place. That is a
    /// relative power of about `1e-14`. The smallest resolvable envelope feature is ten orders of magnitude above
    /// this bound.
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
     * @throws gr::exception naming the setting and the bound it broke.
     */
    void prepare(const AcfConfig& config) {
        if (config.windowLength < 16UZ || config.windowLength > 1048576UZ) {
            throw gr::exception(std::format("gr::analysis::Autocorrelation: windowLength must lie in [16, 1048576], got {}", config.windowLength));
        }
        if (config.maxLag == 0UZ) {
            throw gr::exception("gr::analysis::Autocorrelation: maxLag is the longest published lag and must be at least 1");
        }
        if (config.maxLag >= config.windowLength) {
            throw gr::exception(std::format("gr::analysis::Autocorrelation: maxLag ({}) must be below windowLength ({}): at a lag of a whole window there is not one sample pair separated by it inside a window, so the taper's own autocorrelation is zero and the estimate has nothing to average", config.maxLag, config.windowLength));
        }
        if (config.nAverages == 0UZ) {
            throw gr::exception("gr::analysis::Autocorrelation: nAverages counts the windows a record is made of and must be at least 1");
        }
        if (!(config.overlap >= 0.) || !(config.overlap < 1.)) {
            throw gr::exception(std::format("gr::analysis::Autocorrelation: overlap is the fraction a window shares with the next and must lie in [0, 1), got {}", config.overlap));
        }
        if (!(config.falseAlarmRate > 0.) || !(config.falseAlarmRate < 1.)) {
            throw gr::exception(std::format("gr::analysis::Autocorrelation: falseAlarmRate is a probability per record and must lie in (0, 1), got {}", config.falseAlarmRate));
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

    /// @brief Forward a stream position to the grid. A record's start index is then absolute.
    void seek(std::uint64_t absolute) noexcept { _segmenter.seek(absolute); }

    /// @brief Set the windows per record. The setting changes how many windows make a record and nothing in the
    /// estimate.
    /// The accumulation in progress and the window grid are kept.
    void setAveraging(std::size_t nAverages) {
        if (nAverages == 0UZ) {
            throw gr::exception("gr::analysis::Autocorrelation: nAverages counts the windows a record is made of and must be at least 1");
        }
        _config.nAverages = nAverages;
    }

    /// @brief Set the per-record false-alarm rate the published threshold is computed for.
    void setFalseAlarmRate(double pFa) {
        if (!(pFa > 0.) || !(pFa < 1.)) {
            throw gr::exception(std::format("gr::analysis::Autocorrelation: falseAlarmRate is a probability per record and must lie in (0, 1), got {}", pFa));
        }
        _config.falseAlarmRate = pFa;
    }

    /// @brief True when the correlated sequence is real. The threshold uses the real tail then. The envelope kind
    /// squares magnitudes. The complex kind of a real input correlates real samples.
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
     * @return how many leading samples the estimator has finished with. The caller presents the remainder again.
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
     * @return whether a record was published. An accumulation with no complete window publishes nothing.
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
     * @return whether the window had a fluctuation to correlate. A window without one is degenerate.
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
     * @brief The worst-lag threshold in decibels over the record's own median.
     *
     * A median-referred peak detector reads this unit. A longer lag stands on fewer products. Its scatter is larger.
     * The median of a record's noise curve is therefore not the scale of its worst lag. At `L = N/4` and one window the
     * median is the scale of lag `L/2`. That is seven percent smaller. A threshold that ignored this would sit seven
     * percent low and fire at six times its design rate. The function models the noise scale of each published lag.
     * Under the unbiased normalization that scale is `1/sqrt(nPairs(tau))`. The biased normalization multiplies it by
     * `r_w(tau)/r_w(0)`. The function takes the median of those scales and states the threshold against it. Lag zero is
     * excluded. It is exactly one by construction and carries no noise.
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
        // The published threshold is on the scale of the published values. The biased normalization scales every lag
        // and its noise by the taper's own lag window. The threshold at the worst lag scales the same way. A biased
        // estimate held to the unbiased threshold would need a height that its normalization has already removed.
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
