#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numbers>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/analysis/Autocorrelation.hpp>
#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

using namespace boost::ut;
using gr::analysis::AcfConfig;
using gr::analysis::AcfKind;
using gr::analysis::acfNormalDeviate;
using gr::analysis::AcfNormalization;
using gr::analysis::acfPeakDetectThresholdDb;
using gr::analysis::acfRealPeakDetectThresholdDb;
using gr::analysis::acfRealThreshold;
using gr::analysis::AcfResult;
using gr::analysis::acfScatter;
using gr::analysis::acfThreshold;
using gr::analysis::acfTransformLength;

namespace {

using Complex           = std::complex<float>;
constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

/// One published record, copied out of the spans the kernel lends its sink.
struct Record {
    std::vector<float> magnitude{};
    std::vector<float> phase{};
    double             power{};
    double             scatter{};
    double             threshold{};
    double             peakThresholdDb{};
    std::uint64_t      sampleStart{};
    std::size_t        nAveraged{};
    std::size_t        nDistinct{};
    std::size_t        nPairs{};
    bool               realValued{};
};

[[nodiscard]] Record copyOf(const AcfResult& result) {
    return Record{
        .magnitude       = std::vector<float>(result.magnitude.begin(), result.magnitude.end()),
        .phase           = std::vector<float>(result.phase.begin(), result.phase.end()),
        .power           = result.power,
        .scatter         = result.scatter,
        .threshold       = result.threshold,
        .peakThresholdDb = result.peakThresholdDb,
        .sampleStart     = result.sampleStart,
        .nAveraged       = result.nAveraged,
        .nDistinct       = result.nDistinctSamples,
        .nPairs          = result.nPairs,
        .realValued      = result.realValued,
    };
}

/// @brief Drive a prepared kernel over one span, presenting the undecided remainder again as a caller must.
template<typename T>
[[nodiscard]] std::vector<Record> drive(gr::analysis::Autocorrelation<T>& kernel, std::span<const T> in, bool flushTail = true) {
    std::vector<Record> records;
    const auto          sink = [&records](const AcfResult& result) { records.push_back(copyOf(result)); };
    std::size_t         at   = 0UZ;
    while (at < in.size()) {
        const std::size_t used = kernel.process(in.subspan(at), sink);
        if (used == 0UZ) {
            break;
        }
        at += used;
    }
    if (flushTail) {
        std::ignore = kernel.flush(sink);
    }
    return records;
}

/// @brief The whole estimate of one window by the direct `O(N L)` sum, in double, against which the transform path
/// is checked. It states the estimator's definition once more in the plainest possible form: taper the mean-removed
/// sequence, sum the products at each lag, divide by the taper's own autocorrelation or by its value at zero.
template<typename T>
[[nodiscard]] std::vector<double> directRatios(const AcfConfig& config, std::span<const T> window) {
    const std::size_t        n     = config.windowLength;
    const std::size_t        lags  = config.maxLag;
    const std::vector<float> taper = gr::algorithm::window::create<float>(config.window, n);

    std::vector<std::complex<double>> u(n);
    std::complex<double>              mean{};
    for (std::size_t k = 0UZ; k < n; ++k) {
        std::complex<double> value{};
        if constexpr (std::is_same_v<T, float>) {
            const double sample = static_cast<double>(window[k]);
            value               = config.kind == AcfKind::Envelope ? std::complex<double>(sample * sample, 0.) : std::complex<double>(sample, 0.);
        } else {
            const double re = static_cast<double>(window[k].real());
            const double im = static_cast<double>(window[k].imag());
            value           = config.kind == AcfKind::Envelope ? std::complex<double>(re * re + im * im, 0.) : std::complex<double>(re, im);
        }
        u[k] = value;
        mean += value;
    }
    mean /= static_cast<double>(n);
    if (!config.removeMean) {
        mean = std::complex<double>{};
    }
    for (std::size_t k = 0UZ; k < n; ++k) {
        u[k] = (u[k] - mean) * static_cast<double>(taper[k]);
    }

    std::vector<double> ratios(lags + 1UZ, 0.);
    std::vector<double> taperAcf(lags + 1UZ, 0.);
    std::vector<double> magnitudes(lags + 1UZ, 0.);
    for (std::size_t lag = 0UZ; lag <= lags; ++lag) {
        double               rw{};
        std::complex<double> sum{};
        for (std::size_t k = 0UZ; k + lag < n; ++k) {
            rw += static_cast<double>(taper[k]) * static_cast<double>(taper[k + lag]);
            sum += u[k + lag] * std::conj(u[k]);
        }
        taperAcf[lag]   = rw;
        magnitudes[lag] = std::abs(sum);
    }
    for (std::size_t lag = 0UZ; lag <= lags; ++lag) {
        double ratio = magnitudes[0UZ] > 0. ? magnitudes[lag] / magnitudes[0UZ] : 0.;
        if (config.normalization == AcfNormalization::Unbiased) {
            ratio = taperAcf[lag] > 0. ? ratio * taperAcf[0UZ] / taperAcf[lag] : 0.;
        }
        ratios[lag] = ratio;
    }
    return ratios;
}

[[nodiscard]] std::vector<Complex> complexNoise(std::size_t n, std::uint64_t seed) {
    gr::rng::Xoshiro256pp         rng(seed);
    gr::rng::GaussianNoise<float> gauss(rng);
    std::vector<Complex>          out(n);
    for (Complex& sample : out) {
        sample = gauss.complexSample();
    }
    return out;
}

[[nodiscard]] std::vector<float> realNoise(std::size_t n, std::uint64_t seed) {
    gr::rng::Xoshiro256pp         rng(seed);
    gr::rng::GaussianNoise<float> gauss(rng);
    std::vector<float>            out(n);
    for (float& sample : out) {
        sample = gauss();
    }
    return out;
}

/// @brief The largest published ratio over lags `[first, last]`, and where it sits.
[[nodiscard]] std::pair<std::size_t, double> peakOver(std::span<const float> magnitude, std::size_t first, std::size_t last) {
    std::size_t at   = first;
    double      best = -1.;
    for (std::size_t lag = first; lag <= last && lag < magnitude.size(); ++lag) {
        if (static_cast<double>(magnitude[lag]) > best) {
            best = static_cast<double>(magnitude[lag]);
            at   = lag;
        }
    }
    return {at, best};
}

[[nodiscard]] AcfConfig baseConfig(std::size_t n, std::size_t lags, AcfKind kind, AcfNormalization normalization) {
    AcfConfig config{};
    config.windowLength  = n;
    config.maxLag        = lags;
    config.kind          = kind;
    config.normalization = normalization;
    config.overlap       = 0.;
    config.nAverages     = 1UZ;
    return config;
}

} // namespace

const suite<"autocorrelation free functions"> _acfFunctions = [] {
    "the transform length is the next power of two at or above N + L"_test = [] {
        expect(eq(acfTransformLength(8192UZ, 2048UZ), 16384UZ));
        expect(eq(acfTransformLength(8192UZ, 4096UZ), 16384UZ));
        expect(eq(acfTransformLength(4096UZ, 1024UZ), 8192UZ));
        expect(eq(acfTransformLength(16384UZ, 2048UZ), 32768UZ));
        expect(eq(acfTransformLength(240UZ, 16UZ), 256UZ)) << "N + L exactly a power of two needs no further rounding";
        expect(eq(acfTransformLength(1024UZ, 1023UZ), 2048UZ));
        expect(eq(acfTransformLength(64UZ, 8UZ), 128UZ));
    };

    "the scatter is one over the root of the products behind the lag"_test = [] {
        expect(std::abs(acfScatter(4096UZ) - 1. / 64.) < 1e-15);
        expect(std::isinf(acfScatter(0UZ))) << "no products behind a lag is no estimate";
    };

    "the detection multiple depends on the searched lags and the rate alone"_test = [] {
        // the multiple is the scatter times sqrt(ln(L/p)), and depends on the searched lags and the rate alone
        constexpr std::array<std::pair<std::size_t, double>, 4> kCases{{{256UZ, 3.186}, {1024UZ, 3.397}, {2048UZ, 3.497}, {4096UZ, 3.595}}};
        for (const auto& [lags, expected] : kCases) {
            const double multiple = acfThreshold(1UZ, lags, 1e-2);
            expect(std::abs(multiple - expected) < 1e-3) << std::format("L={} p=1e-2: multiple {:.4f} against {:.4f}", lags, multiple, expected);
        }
        constexpr std::array<std::pair<std::size_t, double>, 4> kTighter{{{256UZ, 3.529}, {1024UZ, 3.720}, {2048UZ, 3.812}, {4096UZ, 3.902}}};
        for (const auto& [lags, expected] : kTighter) {
            const double multiple = acfThreshold(1UZ, lags, 1e-3);
            expect(std::abs(multiple - expected) < 1e-3) << std::format("L={} p=1e-3: multiple {:.4f} against {:.4f}", lags, multiple, expected);
        }
        // and it scales as the scatter does
        expect(std::abs(acfThreshold(4096UZ, 1024UZ, 1e-3) - 3.720 / 64.) < 1e-4);
    };

    "the peak-detector threshold in decibels over the median"_test = [] {
        expect(std::abs(acfPeakDetectThresholdDb(4096UZ, 1e-3) - 6.708) < 5e-3) << std::format("{:.4f}", acfPeakDetectThresholdDb(4096UZ, 1e-3));
        expect(std::abs(acfPeakDetectThresholdDb(1024UZ, 1e-3) - 6.501) < 5e-3) << std::format("{:.4f}", acfPeakDetectThresholdDb(1024UZ, 1e-3));
    };

    "a real estimate has a heavier tail and needs a higher threshold"_test = [] {
        // erfc(t/sqrt2) = tail: the two-sided normal deviate, checked against values that are common knowledge
        expect(std::abs(acfNormalDeviate(0.05) - 1.959964) < 1e-5) << std::format("{:.6f}", acfNormalDeviate(0.05));
        expect(std::abs(acfNormalDeviate(0.01) - 2.575829) < 1e-5) << std::format("{:.6f}", acfNormalDeviate(0.01));
        expect(std::abs(acfNormalDeviate(0.002699796) - 3.) < 1e-5) << std::format("{:.6f}", acfNormalDeviate(0.002699796));

        const double rayleigh = acfThreshold(3072UZ, 1024UZ, 1e-3);
        const double normal   = acfRealThreshold(3072UZ, 1024UZ, 1e-3);
        expect(normal > rayleigh) << std::format("real {:.5f} against complex {:.5f}", normal, rayleigh);
        std::println("thresholds at nPairs=3072, L=1024, p=1e-3: complex {:.5f} ({:.3f} dB over median), real {:.5f} ({:.3f} dB)", rayleigh, acfPeakDetectThresholdDb(1024UZ, 1e-3), normal, acfRealPeakDetectThresholdDb(1024UZ, 1e-3));
    };
};

const suite<"autocorrelation kernel"> _acfKernel = [] {
    "criterion 1: Wiener-Khinchin equals the direct sum"_test = [] {
        constexpr std::array<std::pair<std::size_t, std::size_t>, 4> kGeometries{{{64UZ, 8UZ}, {256UZ, 100UZ}, {4096UZ, 1024UZ}, {1024UZ, 1023UZ}}};
        double                                                       worst = 0.;
        for (const auto& [n, lags] : kGeometries) {
            const std::vector<Complex> complexInput = complexNoise(n, 0x9E37ULL + n);
            const std::vector<float>   realInput    = realNoise(n, 0xC0FFEEULL + n);
            for (const AcfKind kind : {AcfKind::Complex, AcfKind::Envelope}) {
                for (const AcfNormalization normalization : {AcfNormalization::Unbiased, AcfNormalization::Biased}) {
                    const AcfConfig config = baseConfig(n, lags, kind, normalization);

                    gr::analysis::Autocorrelation<Complex> complexKernel;
                    complexKernel.prepare(config);
                    expect(eq(complexKernel.transformLength(), acfTransformLength(n, lags))) << "the transform length is the padding rule's own";
                    const std::vector<Record> complexRecords = drive(complexKernel, std::span<const Complex>(complexInput));
                    expect(eq(complexRecords.size(), 1UZ)) << "one window, one record";
                    const std::vector<double> complexReference = directRatios<Complex>(config, std::span<const Complex>(complexInput));

                    gr::analysis::Autocorrelation<float> realKernel;
                    realKernel.prepare(config);
                    const std::vector<Record> realRecords = drive(realKernel, std::span<const float>(realInput));
                    expect(eq(realRecords.size(), 1UZ));
                    const std::vector<double> realReference = directRatios<float>(config, std::span<const float>(realInput));

                    for (const auto& [records, reference, label] : {std::tuple{&complexRecords, &complexReference, "complex input"}, std::tuple{&realRecords, &realReference, "real input"}}) {
                        const std::vector<float>& magnitude = (*records)[0UZ].magnitude;
                        expect(eq(magnitude.size(), lags + 1UZ));
                        expect(magnitude[0UZ] == 1.f) << std::format("N={} L={} {}: lag zero is exactly one by construction", n, lags, label);
                        for (std::size_t lag = 0UZ; lag <= lags; ++lag) {
                            const double want     = (*reference)[lag];
                            const double relative = std::abs(static_cast<double>(magnitude[lag]) - want) / std::max(want, 1e-6);
                            worst                 = std::max(worst, relative);
                            expect(relative < 1e-5) << std::format("N={} L={} kind={} norm={} {} lag {}: {:.9f} against {:.9f}", n, lags, gr::analysis::acfKindName(kind), gr::analysis::acfNormalizationName(normalization), label, lag, magnitude[lag], want);
                        }
                    }
                }
            }
        }
        std::println("criterion 1: worst relative disagreement with the direct sum over 4 geometries x 2 kinds x 2 normalizations x 2 input types: {:.3e}", worst);
    };

    "criterion 2: the scatter, per lag and per normalization"_test = [] {
        constexpr std::size_t                n            = 4096UZ;
        constexpr std::size_t                lags         = 3072UZ;
        constexpr std::size_t                realizations = 400UZ;
        constexpr std::array<std::size_t, 5> kLags{1UZ, 512UZ, 1024UZ, 2048UZ, 3072UZ};

        for (const AcfNormalization normalization : {AcfNormalization::Biased, AcfNormalization::Unbiased}) {
            const AcfConfig                        config = baseConfig(n, lags, AcfKind::Complex, normalization);
            gr::analysis::Autocorrelation<Complex> kernel;
            kernel.prepare(config);

            std::array<double, kLags.size()> sumSquares{};
            for (std::size_t trial = 0UZ; trial < realizations; ++trial) {
                const std::vector<Complex> noise   = complexNoise(n, 0x51ED0000ULL + trial);
                const std::vector<Record>  records = drive(kernel, std::span<const Complex>(noise));
                expect(eq(records.size(), 1UZ));
                for (std::size_t k = 0UZ; k < kLags.size(); ++k) {
                    const double value = static_cast<double>(records[0UZ].magnitude[kLags[k]]);
                    sumSquares[k] += value * value;
                }
            }
            for (std::size_t k = 0UZ; k < kLags.size(); ++k) {
                const std::size_t lag      = kLags[k];
                const double      measured = std::sqrt(sumSquares[k] / static_cast<double>(realizations));
                const double      expected = normalization == AcfNormalization::Biased ? std::sqrt(static_cast<double>(n - lag)) / static_cast<double>(n) : 1. / std::sqrt(static_cast<double>(n - lag));
                std::println("criterion 2: {} lag {:>4}: rms {:.5f} against {:.5f} ({:+.1f}%)", gr::analysis::acfNormalizationName(normalization), lag, measured, expected, 100. * (measured / expected - 1.));
                expect(std::abs(measured / expected - 1.) < 0.10) << std::format("{} lag {}: {:.5f} against {:.5f}", gr::analysis::acfNormalizationName(normalization), lag, measured, expected);
            }
        }
    };

    "criterion 3: the detection multiple holds its rate and the naive one does not"_test = [] {
        constexpr std::size_t n       = 4096UZ;
        constexpr std::size_t lags    = 1024UZ;
        constexpr std::size_t windows = 2000UZ;
        constexpr double      pFa     = 1e-3;

        const AcfConfig                        config = baseConfig(n, lags, AcfKind::Complex, AcfNormalization::Unbiased);
        gr::analysis::Autocorrelation<Complex> kernel;
        kernel.prepare(config);

        const double derived      = acfThreshold(n - lags, lags, pFa);
        const double naive        = acfThreshold(n, lags, pFa); // the same multiple against 1/sqrt(N) rather than 1/sqrt(N-L)
        std::size_t  firedDerived = 0UZ;
        std::size_t  firedNaive   = 0UZ;
        for (std::size_t trial = 0UZ; trial < windows; ++trial) {
            const std::vector<Complex> noise   = complexNoise(n, 0xFA00000ULL + trial);
            const std::vector<Record>  records = drive(kernel, std::span<const Complex>(noise));
            const auto [at, best]              = peakOver(records[0UZ].magnitude, 1UZ, lags);
            std::ignore                        = at;
            firedDerived += best > derived ? 1UZ : 0UZ;
            firedNaive += best > naive ? 1UZ : 0UZ;
            expect(std::abs(records[0UZ].threshold - derived) < 1e-12) << "the record publishes the threshold this criterion computes by hand";
        }
        const double rateDerived = static_cast<double>(firedDerived) / static_cast<double>(windows);
        const double rateNaive   = static_cast<double>(firedNaive) / static_cast<double>(windows);
        const double spread      = 3. * std::sqrt(pFa * (1. - pFa) / static_cast<double>(windows));
        std::println("criterion 3: derived threshold {:.5f} fires on {}/{} = {:.4f}; the 1/sqrt(N) threshold {:.5f} fires on {}/{} = {:.4f} ({:.1f} times)", derived, firedDerived, windows, rateDerived, naive, firedNaive, windows, rateNaive, rateNaive / std::max(rateDerived, 1. / static_cast<double>(windows)));
        expect(rateDerived <= pFa + spread) << std::format("derived rate {:.4f} against design {:.4f} + spread {:.4f}", rateDerived, pFa, spread);
        expect(rateNaive > rateDerived) << "the lag dependence is not a refinement";

        // a negative result needs the instrument verified positive: the same geometry with a sinusoidal envelope
        // modulation at a known period, which the envelope kind must find
        constexpr std::size_t period    = 64UZ;
        constexpr double      depth     = 0.9;
        std::vector<Complex>  modulated = complexNoise(n, 0x9005ULL);
        for (std::size_t k = 0UZ; k < n; ++k) {
            const auto scale = static_cast<float>(std::sqrt(1. + depth * std::cos(kTwoPi * static_cast<double>(k) / static_cast<double>(period))));
            modulated[k] *= scale;
        }
        AcfConfig                              envelopeConfig = baseConfig(n, lags, AcfKind::Envelope, AcfNormalization::Unbiased);
        gr::analysis::Autocorrelation<Complex> envelopeKernel;
        envelopeKernel.prepare(envelopeConfig);
        const std::vector<Record> found = drive(envelopeKernel, std::span<const Complex>(modulated));
        // The autocorrelation of a cosine is a cosine and the record is its magnitude, so the anti-nodes at the odd
        // half-periods stand exactly as high as the period itself; the band searched is therefore the one period
        // centered on the expected lag, inside which the magnitude has a single maximum. That maximum is quadratically
        // flat: one lag either side of it the cosine falls by 5e-3 of its height, far under the scatter, so the
        // located lag is asserted to a fraction of a period and no amount of averaging would make it exact.
        const std::size_t tolerance = period / 8UZ;
        const auto [at, best]       = peakOver(found[0UZ].magnitude, period - period / 4UZ, period + period / 4UZ);
        std::println("criterion 3 positive arm: envelope modulation at {} samples found at lag {} reading {:.4f} against the record's threshold {:.4f}", period, at, best, found[0UZ].threshold);
        expect(at >= period - tolerance && at <= period + tolerance) << std::format("the injected period is where the maximum sits: lag {} against {} +/- {}", at, period, tolerance);
        expect(best > found[0UZ].threshold) << "and it clears the record's own threshold";
        expect(found[0UZ].realValued) << "an envelope estimate is real and takes the folded-normal threshold";
    };

    "criterion 4: the two normalizations differ by exactly the lag window"_test = [] {
        constexpr std::size_t n    = 1024UZ;
        constexpr std::size_t lags = 256UZ;
        for (const auto windowType : {gr::algorithm::window::Type::Rectangular, gr::algorithm::window::Type::Hann}) {
            const std::vector<Complex> input = complexNoise(n, 0xBEEFULL);

            AcfConfig unbiasedConfig   = baseConfig(n, lags, AcfKind::Complex, AcfNormalization::Unbiased);
            unbiasedConfig.window      = windowType;
            AcfConfig biasedConfig     = unbiasedConfig;
            biasedConfig.normalization = AcfNormalization::Biased;

            gr::analysis::Autocorrelation<Complex> unbiasedKernel;
            unbiasedKernel.prepare(unbiasedConfig);
            const std::vector<Record> unbiased = drive(unbiasedKernel, std::span<const Complex>(input));

            gr::analysis::Autocorrelation<Complex> biasedKernel;
            biasedKernel.prepare(biasedConfig);
            const std::vector<Record> biased = drive(biasedKernel, std::span<const Complex>(input));

            const std::span<const double> taperAcf = unbiasedKernel.taperAutocorrelation();
            for (std::size_t lag = 0UZ; lag <= lags; ++lag) {
                const double window = taperAcf[lag] / taperAcf[0UZ];
                const double want   = static_cast<double>(unbiased[0UZ].magnitude[lag]) * window;
                const double got    = static_cast<double>(biased[0UZ].magnitude[lag]);
                expect(std::abs(got - want) <= 1e-5 * std::max(want, 1e-6)) << std::format("{} lag {}: biased {:.9f} against unbiased x r_w(tau)/r_w(0) {:.9f}", magic_enum::enum_name(windowType), lag, got, want);
            }

            // and the taper divides out of the unbiased estimate, so the direct sum still agrees
            const std::vector<double> reference = directRatios<Complex>(unbiasedConfig, std::span<const Complex>(input));
            for (std::size_t lag = 0UZ; lag <= lags; ++lag) {
                const double relative = std::abs(static_cast<double>(unbiased[0UZ].magnitude[lag]) - reference[lag]) / std::max(reference[lag], 1e-6);
                expect(relative < 1e-5) << std::format("{} unbiased lag {}: {:.9f} against {:.9f}", magic_enum::enum_name(windowType), lag, unbiased[0UZ].magnitude[lag], reference[lag]);
            }
        }
    };

    "criterion 5: averaging, and the overlap correction"_test = [] {
        constexpr std::size_t                n            = 1024UZ;
        constexpr std::size_t                lags         = 256UZ;
        constexpr std::size_t                realizations = 200UZ;
        constexpr std::array<std::size_t, 4> kAverages{1UZ, 4UZ, 16UZ, 64UZ};

        for (const std::size_t averages : kAverages) {
            AcfConfig disjoint   = baseConfig(n, lags, AcfKind::Complex, AcfNormalization::Unbiased);
            disjoint.nAverages   = averages;
            AcfConfig overlapped = disjoint;
            overlapped.overlap   = 0.5;
            overlapped.nAverages = 2UZ * averages - 1UZ; // the same distinct samples: N + (K'-1)H == K N at H = N/2

            gr::analysis::Autocorrelation<Complex> disjointKernel;
            disjointKernel.prepare(disjoint);
            gr::analysis::Autocorrelation<Complex> overlappedKernel;
            overlappedKernel.prepare(overlapped);

            double      disjointSum   = 0.;
            double      overlappedSum = 0.;
            std::size_t disjointDistinct{};
            std::size_t overlappedDistinct{};
            for (std::size_t trial = 0UZ; trial < realizations; ++trial) {
                const std::vector<Complex> noise = complexNoise(n * averages, 0xA5A50000ULL + trial);
                const std::vector<Record>  a     = drive(disjointKernel, std::span<const Complex>(noise), false);
                const std::vector<Record>  b     = drive(overlappedKernel, std::span<const Complex>(noise), false);
                expect(ge(a.size(), 1UZ));
                expect(ge(b.size(), 1UZ));
                const double av = static_cast<double>(a[0UZ].magnitude[1UZ]);
                const double bv = static_cast<double>(b[0UZ].magnitude[1UZ]);
                disjointSum += av * av;
                overlappedSum += bv * bv;
                disjointDistinct   = a[0UZ].nDistinct;
                overlappedDistinct = b[0UZ].nDistinct;
                disjointKernel.reset();
                overlappedKernel.reset();
            }
            const double measuredDisjoint   = std::sqrt(disjointSum / static_cast<double>(realizations));
            const double measuredOverlapped = std::sqrt(overlappedSum / static_cast<double>(realizations));
            const double expected           = 1. / std::sqrt(static_cast<double>(averages * (n - 1UZ)));

            std::println("criterion 5: K={:>2} disjoint rms {:.5f} against 1/sqrt(K(N-1)) {:.5f}; overlapped K'={} rms {:.5f}; n_samples {} and {}", averages, measuredDisjoint, expected, overlapped.nAverages, measuredOverlapped, disjointDistinct, overlappedDistinct);
            expect(std::abs(measuredDisjoint / expected - 1.) < 0.10) << std::format("K={}: disjoint rms {:.5f} against {:.5f}", averages, measuredDisjoint, expected);
            expect(std::abs(measuredOverlapped / measuredDisjoint - 1.) < 0.10) << std::format("K={}: overlap does not reduce the scatter, {:.5f} against {:.5f}", averages, measuredOverlapped, measuredDisjoint);
            expect(eq(disjointDistinct, averages * n)) << "disjoint windows: n_samples is K N";
            expect(eq(overlappedDistinct, averages * n)) << "half-overlapped windows over the same input: the same distinct samples";
        }
    };

    "criterion 6: the mean, its removal, and a sequence with nothing left to correlate"_test = [] {
        constexpr std::size_t n    = 4096UZ;
        constexpr std::size_t lags = 512UZ;

        std::vector<Complex> offset = complexNoise(n, 0xD1CEULL);
        for (Complex& sample : offset) {
            sample += Complex{1.f, 0.f}; // a DC offset equal to the rms
        }

        AcfConfig kept  = baseConfig(n, lags, AcfKind::Complex, AcfNormalization::Unbiased);
        kept.removeMean = false;
        gr::analysis::Autocorrelation<Complex> keptKernel;
        keptKernel.prepare(kept);
        const std::vector<Record> withPedestal = drive(keptKernel, std::span<const Complex>(offset));
        double                    lowest       = 1.;
        for (std::size_t lag = 1UZ; lag <= lags; ++lag) {
            lowest = std::min(lowest, static_cast<double>(withPedestal[0UZ].magnitude[lag]));
        }
        std::println("criterion 6: an unremoved offset puts a pedestal of at least {:.4f} under every lag", lowest);
        expect(lowest >= 0.4) << std::format("every lag reads at least 0.4, lowest {:.4f}", lowest);

        AcfConfig removed  = kept;
        removed.removeMean = true;
        gr::analysis::Autocorrelation<Complex> removedKernel;
        removedKernel.prepare(removed);
        const std::vector<Record> withoutPedestal = drive(removedKernel, std::span<const Complex>(offset));
        const auto [at, best]                     = peakOver(withoutPedestal[0UZ].magnitude, 1UZ, lags);
        std::ignore                               = at;
        std::println("criterion 6: with the mean removed the largest lag reads {:.4f} against a scatter of {:.4f}", best, withoutPedestal[0UZ].scatter);
        expect(best < 6. * withoutPedestal[0UZ].scatter) << "the pedestal is gone and what is left is scatter";

        // a constant-modulus input has no envelope autocorrelation at all, and the estimator says so rather than
        // publishing whatever it does with 0/0
        std::vector<Complex>  constant(n);
        gr::rng::Xoshiro256pp rng(0x1234ULL);
        for (std::size_t k = 0UZ; k < n; ++k) {
            const double phase = kTwoPi * rng.uniform01<double>();
            constant[k]        = Complex{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
        }
        AcfConfig                              envelope = baseConfig(n, lags, AcfKind::Envelope, AcfNormalization::Unbiased);
        gr::analysis::Autocorrelation<Complex> envelopeKernel;
        envelopeKernel.prepare(envelope);
        const std::vector<Record> none = drive(envelopeKernel, std::span<const Complex>(constant));
        expect(none.empty()) << "a constant-modulus envelope publishes no record";
        expect(eq(envelopeKernel.nDegenerate, 1UZ)) << "and the window is counted degenerate";
        expect(eq(envelopeKernel.nWindows, 1UZ));
        expect(eq(envelopeKernel.nRecords, 0UZ));
    };

    "the segmenter's counters close, and the tail is the caller's"_test = [] {
        constexpr std::size_t n      = 256UZ;
        constexpr std::size_t lags   = 64UZ;
        AcfConfig             config = baseConfig(n, lags, AcfKind::Complex, AcfNormalization::Unbiased);
        config.overlap               = 0.5;
        config.nAverages             = 4UZ;

        gr::analysis::Autocorrelation<Complex> kernel;
        kernel.prepare(config);
        expect(eq(kernel.hop(), 128UZ));

        const std::vector<Complex> input = complexNoise(2000UZ, 0x777ULL);
        std::vector<Record>        records;
        const auto                 sink = [&records](const AcfResult& result) { records.push_back(copyOf(result)); };
        std::size_t                at   = 0UZ;
        while (at < input.size()) {
            const std::size_t used = kernel.process(std::span<const Complex>(input).subspan(at), sink);
            if (used == 0UZ) {
                break;
            }
            at += used;
        }
        const std::size_t tail = input.size() - at;
        expect(lt(tail, n)) << "the undecided remainder is shorter than a window";
        expect(eq(kernel.nSamples, at));
        expect(eq(kernel.nWindows, config.nAverages * kernel.nRecords + kernel.windowsPending() + kernel.nDegenerate));
        expect(eq(records.size(), kernel.nRecords));
        for (std::size_t k = 0UZ; k < records.size(); ++k) {
            expect(eq(records[k].nAveraged, config.nAverages));
            expect(eq(records[k].sampleStart, k * config.nAverages * kernel.hop()));
            expect(eq(records[k].nDistinct, n + (config.nAverages - 1UZ) * kernel.hop()));
        }
        std::ignore = kernel.flush(sink);
        expect(eq(kernel.nPartialFlushes, records.size() > 0UZ ? 1UZ : 0UZ));
    };

    "prepare refuses by name"_test = [] {
        gr::analysis::Autocorrelation<Complex> kernel;
        AcfConfig                              config = baseConfig(1024UZ, 256UZ, AcfKind::Complex, AcfNormalization::Unbiased);

        AcfConfig tooShort    = config;
        tooShort.windowLength = 8UZ;
        expect(throws<std::invalid_argument>([&] { kernel.prepare(tooShort); }));
        AcfConfig zeroLag = config;
        zeroLag.maxLag    = 0UZ;
        expect(throws<std::invalid_argument>([&] { kernel.prepare(zeroLag); }));
        AcfConfig lagAtWindow = config;
        lagAtWindow.maxLag    = lagAtWindow.windowLength;
        expect(throws<std::invalid_argument>([&] { kernel.prepare(lagAtWindow); }));
        AcfConfig noAverages = config;
        noAverages.nAverages = 0UZ;
        expect(throws<std::invalid_argument>([&] { kernel.prepare(noAverages); }));
        AcfConfig fullOverlap = config;
        fullOverlap.overlap   = 1.;
        expect(throws<std::invalid_argument>([&] { kernel.prepare(fullOverlap); }));
        AcfConfig zeroRate      = config;
        zeroRate.falseAlarmRate = 0.;
        expect(throws<std::invalid_argument>([&] { kernel.prepare(zeroRate); }));
        expect(nothrow([&] { kernel.prepare(config); }));
        expect(throws<std::invalid_argument>([&] { kernel.setAveraging(0UZ); }));
        expect(throws<std::invalid_argument>([&] { kernel.setFalseAlarmRate(1.); }));
    };
};

int main() { /* not needed for UT */ }
