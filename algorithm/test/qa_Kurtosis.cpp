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
#include <vector>

#include <gnuradio-4.0/algorithm/measurement/Kurtosis.hpp>
#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

using namespace boost::ut;
using gr::measurement::gaussianFourthMoment;
using gr::measurement::KurtosisAccumulator;
using gr::measurement::kurtosisBias;
using gr::measurement::kurtosisSpread;
using gr::measurement::kurtosisVariance;
using gr::measurement::spectralKurtosis;
using gr::measurement::SpectralKurtosisAccumulator;
using gr::measurement::spectralKurtosisCw;
using gr::measurement::spectralKurtosisExpectation;
using gr::measurement::spectralKurtosisSpread;
using gr::measurement::spectralKurtosisVariance;
using gr::measurement::SpectrumResponse;

namespace {

constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;
using Complex           = std::complex<float>;

/// The mean and the sample standard deviation of a run of readings, which is what every band below is applied
/// to: the criteria assert a distribution, not a value.
struct Spread {
    double mean{0.};
    double sigma{0.};

    [[nodiscard]] double standardError(std::size_t n) const { return sigma / std::sqrt(static_cast<double>(n)); }
};

[[nodiscard]] Spread spreadOf(std::span<const double> values) {
    Spread out;
    for (const double value : values) {
        out.mean += value;
    }
    out.mean /= static_cast<double>(values.size());
    for (const double value : values) {
        out.sigma += (value - out.mean) * (value - out.mean);
    }
    out.sigma = std::sqrt(out.sigma / static_cast<double>(values.size() - 1UZ));
    return out;
}

/// The excess kurtosis of a real stream of unit Gaussian plus a Bernoulli impulse of amplitude `A`, in closed
/// form: `E[x^2] = 1 + p A^2` and `E[x^4] = 3 + 6 p A^2 + p A^4`, the cross terms vanishing with `E[g]` and
/// `E[g^3]`. This is the discriminant's positive side and it is written here so the test asserts a derivation.
[[nodiscard]] constexpr double impulsiveExcess(double p, double amplitude) noexcept {
    const double a2 = amplitude * amplitude;
    const double m2 = 1. + p * a2;
    return (3. + 6. * p * a2 + p * a2 * a2) / (m2 * m2) - 3.;
}

/// One bin's power under `d` independent periodograms, which is Gamma of shape `d`. The statistic is
/// homogeneous of degree zero, so the `1/d` an averaging producer would apply divides straight out and is not
/// applied here.
[[nodiscard]] float gammaPower(gr::rng::GaussianNoise<float>& noise, std::size_t shape) {
    float sum = 0.f;
    for (std::size_t i = 0UZ; i < shape; ++i) {
        const Complex z = noise.complexSample();
        sum += z.real() * z.real() + z.imag() * z.imag();
    }
    return sum;
}

} // namespace

const boost::ut::suite<"Kurtosis"> _kurtosis = [] {
    "the closed forms the criteria are written against"_test = [] {
        expect(gaussianFourthMoment(false) == 3.);
        expect(gaussianFourthMoment(true) == 2.);
        expect(kurtosisBias(4'096UZ, true) == -2. / 4'096.);
        expect(kurtosisBias(4'096UZ, false) == -6. / 4'096.);
        expect(kurtosisVariance(4'096UZ, true) == 4. / 4'096.);
        expect(kurtosisVariance(4'096UZ, false) == 24. / 4'096.);
        expect(approx(kurtosisSpread(4'096UZ, true), 0.03125, 1e-12)) << "2/sqrt(N) at N = 4096";
        expect(approx(kurtosisSpread(4'096UZ, false), 0.0765466, 1e-6)) << "4.899/sqrt(N) at N = 4096";
        expect(kurtosisBias(0UZ, true) == 0.) << "an empty window has no bias to report";
    };

    "seeded Gaussian sits inside the bias and the spread its own closed forms give"_test = [] {
        constexpr std::size_t kWindows = 400UZ;
        constexpr std::size_t kWindow  = 4'096UZ;

        gr::rng::Xoshiro256pp         rng(0x5EEDU);
        gr::rng::GaussianNoise<float> noise(rng);
        std::vector<Complex>          complexWindow(kWindow);
        std::vector<float>            realWindow(kWindow);
        std::vector<double>           complexReadings(kWindows);
        std::vector<double>           realReadings(kWindows);

        for (std::size_t w = 0UZ; w < kWindows; ++w) {
            noise.fillComplex(complexWindow);
            KurtosisAccumulator<Complex> accumulator;
            accumulator.add(std::span<const Complex>(complexWindow));
            complexReadings[w] = accumulator.excess();

            noise.fill(realWindow);
            KurtosisAccumulator<float> real;
            real.add(std::span<const float>(realWindow));
            realReadings[w] = real.excess();
        }

        for (const bool complexDomain : {true, false}) {
            const Spread measured = spreadOf(complexDomain ? complexReadings : realReadings);
            const double bias     = kurtosisBias(kWindow, complexDomain);
            const double spread   = kurtosisSpread(kWindow, complexDomain);
            const double band     = 4. * spread / std::sqrt(static_cast<double>(kWindows));

            std::println("{} Gaussian at N = {}: mean excess {:+.6f} against the bias {:+.6f} (band +/-{:.6f}); spread {:.6f} against {:.6f}", complexDomain ? "complex" : "real   ", kWindow, measured.mean, bias, band, measured.sigma, spread);
            expect(std::abs(measured.mean - bias) < band) << "the mean sits on the closed-form bias";
            expect(std::abs(measured.sigma - spread) < 0.1 * spread) << "and the per-window spread is the closed-form one, so a right mean over a wrong shape fails";
        }
    };

    "the four closed values, and the sign that separates them"_test = [] {
        constexpr std::size_t kSamples = 262'144UZ;
        // An increment that is not a simple fraction of a turn, so the window averages the phase rather than
        // sampling a short cycle of it.
        constexpr double kIncrement = 0.30901699437494745;

        // Constant modulus: `E|z|^4 = A^4` and `(E|z|^2)^2 = A^4`, so `K = 1`. This is the `ka = 1` the tree's
        // m2m4 signal-to-noise estimator already leans on, reached from the same line.
        KurtosisAccumulator<Complex> constantModulus;
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            const double phase = kTwoPi * kIncrement * static_cast<double>(k);
            constantModulus.add(Complex{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))});
        }
        expect(approx(constantModulus.normalized(), 1., 1e-6)) << "a constant-modulus stream reads K = 1 to float rounding, got" << constantModulus.normalized();
        expect(approx(constantModulus.excess(), -1., 1e-6));

        // Two equal complex tones: `|z|^2 = 2 + 2 cos(D)` with `D` uniform, so `E|z|^2 = 2`, `E|z|^4 = 6` and
        // `K = 3/2` exactly.
        KurtosisAccumulator<Complex> twoTones;
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            const double a = kTwoPi * 0.1 * static_cast<double>(k);
            const double b = kTwoPi * (0.1 + kIncrement) * static_cast<double>(k);
            twoTones.add(Complex{static_cast<float>(std::cos(a) + std::cos(b)), static_cast<float>(std::sin(a) + std::sin(b))});
        }
        expect(approx(twoTones.normalized(), 1.5, 1e-4)) << "two equal complex tones read K = 3/2, got" << twoTones.normalized();
        expect(approx(twoTones.excess(), -0.5, 1e-4)) << "an excess of exactly -0.5";

        // A real sinusoid: `m4 = 3A^4/8` and `m2 = A^2/2`, so `K = 3/2` and the excess is -1.5.
        KurtosisAccumulator<float> sinusoid;
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            sinusoid.add(static_cast<float>(std::cos(kTwoPi * kIncrement * static_cast<double>(k))));
        }
        expect(approx(sinusoid.normalized(), 1.5, 1e-4)) << "a real sinusoid reads K = 3/2, got" << sinusoid.normalized();
        expect(approx(sinusoid.excess(), -1.5, 1e-4));

        expect(lt(constantModulus.excess(), 0.)) << "bounded structure drives the fourth moment down";
        expect(lt(twoTones.excess(), 0.));
        expect(lt(sinusoid.excess(), 0.));
    };

    "the impulsive sign, against the closed form for the scene"_test = [] {
        constexpr std::size_t kWindows = 16UZ;
        constexpr std::size_t kWindow  = 1UZ << 18;

        struct Point {
            double p;
            double amplitude;
        };
        constexpr std::array<Point, 3> points{{{0.001, 10.}, {0.01, 5.}, {0.001, 20.}}};

        for (const Point& point : points) {
            gr::rng::Xoshiro256pp         rng(0xC0FFEEU);
            gr::rng::GaussianNoise<float> noise(rng);
            std::vector<float>            window(kWindow);
            std::vector<double>           readings(kWindows);

            for (std::size_t w = 0UZ; w < kWindows; ++w) {
                noise.fill(window);
                for (float& sample : window) {
                    if (rng.uniform01<float>() < static_cast<float>(point.p)) {
                        sample += static_cast<float>(point.amplitude);
                    }
                }
                KurtosisAccumulator<float> accumulator;
                accumulator.add(std::span<const float>(window));
                readings[w] = accumulator.excess();
            }

            const Spread measured = spreadOf(readings);
            const double closed   = impulsiveExcess(point.p, point.amplitude);
            const double band     = 4. * measured.standardError(kWindows);
            std::println("impulsive p = {:.3f}, A = {:4.1f}: mean excess {:.4f} against the closed form {:.4f}, band +/-{:.4f}", point.p, point.amplitude, measured.mean, closed, band);
            expect(std::abs(measured.mean - closed) < band) << "the closed value, inside the run's own standard error";
            expect(gt(measured.mean, 0.)) << "and impulsive structure drives the fourth moment up, where bounded structure drove it down";
        }
    };

    "a degenerate window is reported, not guessed"_test = [] {
        KurtosisAccumulator<Complex> accumulator;
        expect(accumulator.degenerate()) << "an empty accumulator has nothing to divide by";
        expect(accumulator.normalized() == 0.) << "and reports zero rather than a NaN a consumer would have to test for";
        expect(accumulator.excess() == 0.);
        expect(accumulator.meanPower() == 0.);

        const std::vector<Complex> zeros(1'024UZ);
        accumulator.add(std::span<const Complex>(zeros));
        expect(accumulator.count() == 1'024UZ);
        expect(accumulator.degenerate()) << "an all-zero window is degenerate on exact equality with zero, not on an epsilon";
        expect(accumulator.normalized() == 0.);

        accumulator.reset();
        expect(accumulator.count() == 0UZ);
        expect(accumulator.sumPower() == 0.);
    };

    "one sample at a time is one span at a time"_test = [] {
        gr::rng::Xoshiro256pp         rng(7ULL);
        gr::rng::GaussianNoise<float> noise(rng);
        std::vector<Complex>          window(4'096UZ);
        noise.fillComplex(window);

        KurtosisAccumulator<Complex> bulk;
        bulk.add(std::span<const Complex>(window));

        KurtosisAccumulator<Complex> chunked;
        for (const std::size_t chunk : {1UZ, 7UZ, 1'000UZ}) {
            chunked.reset();
            for (std::size_t at = 0UZ; at < window.size(); at += chunk) {
                chunked.add(std::span<const Complex>(window).subspan(at, std::min(chunk, window.size() - at)));
            }
            expect(chunked.sumPower() == bulk.sumPower()) << "the sums are the same bits however the stream is split";
            expect(chunked.sumPowerSquared() == bulk.sumPowerSquared());
        }

        KurtosisAccumulator<Complex> scalar;
        for (const Complex sample : window) {
            scalar.add(sample);
        }
        expect(scalar.sumPower() == bulk.sumPower()) << "and the scalar form is the span form";
    };

    "spectral kurtosis reads exactly 1 on noise, at every shape"_test = [] {
        constexpr std::size_t kBins    = 2'000UZ;
        constexpr std::size_t kSpectra = 64UZ;

        for (const std::size_t shape : {1UZ, 4UZ, 16UZ}) {
            gr::rng::Xoshiro256pp         rng(0xA11CEULL + shape);
            gr::rng::GaussianNoise<float> noise(rng);

            SpectralKurtosisAccumulator accumulator(kBins);
            std::vector<float>          spectrum(kBins);
            for (std::size_t m = 0UZ; m < kSpectra; ++m) {
                for (float& bin : spectrum) {
                    bin = gammaPower(noise, shape);
                }
                expect(accumulator.accumulate(spectrum) == SpectrumResponse::accepted);
            }

            std::vector<double> sk(kBins);
            expect(accumulator.evaluate(static_cast<double>(shape), sk) == 0UZ) << "no bin is degenerate on noise";

            const Spread measured = spreadOf(sk);
            const double closed   = spectralKurtosisSpread(kSpectra, static_cast<double>(shape));
            const double band     = 4. * closed / std::sqrt(static_cast<double>(kBins));
            std::println("SK at M = {}, d = {:2}: mean {:.5f} against the exact 1 (band +/-{:.5f}); spread {:.5f} against {:.5f}", kSpectra, shape, measured.mean, band, measured.sigma, closed);
            expect(std::abs(measured.mean - spectralKurtosisExpectation()) < band) << "E[SK] is 1 exactly, at every M and every d, and not asymptotically";
            expect(std::abs(measured.sigma - closed) < 0.1 * closed) << "and the spread is the Dirichlet variance the header derives";
        }

        // The three figures the header states, so a change to the variance is caught here and not in a band.
        expect(approx(spectralKurtosisSpread(64UZ, 1.), 0.24251, 1e-4));
        expect(approx(spectralKurtosisSpread(64UZ, 4.), 0.19728, 1e-4));
        expect(approx(spectralKurtosisSpread(64UZ, 16.), 0.18321, 1e-4));
        expect(approx(spectralKurtosisSpread(16UZ, 1.), 0.44678, 1e-4));
    };

    "spectral kurtosis reads exactly 0 on a noiseless tone"_test = [] {
        constexpr std::size_t kBins = 4UZ;

        for (const std::size_t spectra : {8UZ, 64UZ, 1'024UZ}) {
            SpectralKurtosisAccumulator accumulator(kBins);
            const std::vector<float>    spectrum{1.f, 3.f, 0.25f, 128.f};
            for (std::size_t m = 0UZ; m < spectra; ++m) {
                expect(accumulator.accumulate(spectrum) == SpectrumResponse::accepted);
            }

            std::vector<double> sk(kBins);
            (void)accumulator.evaluate(1., sk);
            for (const double value : sk) {
                // Identical bin powers give `M*S2/S1^2 = 1` algebraically, so this is an exact comparison and
                // not a tolerance: a tolerance here would hide the identity it exists to check.
                expect(value == 0.) << "a noiseless tone reads SK = 0 identically at M =" << spectra;
            }
        }
    };

    "the two closed shapes in between, and the two-sided discriminant"_test = [] {
        constexpr std::size_t kBins    = 200UZ;
        constexpr std::size_t kSpectra = 1'024UZ;

        for (const double rho : {1., 2., 10., 100.}) {
            gr::rng::Xoshiro256pp         rng(0xBEEFULL + static_cast<std::uint64_t>(rho));
            gr::rng::GaussianNoise<float> noise(rng);

            SpectralKurtosisAccumulator accumulator(kBins);
            std::vector<float>          spectrum(kBins);
            const float                 amplitude = static_cast<float>(std::sqrt(rho));
            for (std::size_t m = 0UZ; m < kSpectra; ++m) {
                for (float& bin : spectrum) {
                    const Complex z = Complex{amplitude, 0.f} + noise.complexSample();
                    bin             = z.real() * z.real() + z.imag() * z.imag();
                }
                (void)accumulator.accumulate(spectrum);
            }

            std::vector<double> sk(kBins);
            (void)accumulator.evaluate(1., sk);
            const Spread measured = spreadOf(sk);
            const double closed   = spectralKurtosisCw(rho);
            std::println("SK on CW at rho = {:5.1f}: mean {:.5f} against (2*rho+1)/(rho+1)^2 = {:.5f}", rho, measured.mean, closed);
            expect(std::abs(measured.mean - closed) < 0.01) << "the carrier-in-noise closed form";
            expect(lt(measured.mean, 1.)) << "steady structure reads below 1";
        }

        // A pulsed interferer present a fraction `p` of the time at bin power `S` over noise `N0`:
        // `E[u] = pS + N0` and `E[u^2] = p S^2 + 4 p S N0 + 2 N0^2`, so the **large-M limit** is their ratio
        // less one. At S/N0 = 100 that is 7.6116, 2.8491 and 1.0000 at p = 0.1, 0.25 and 0.5.
        //
        // Unlike the carrier arm this one has a finite-M correction that is not derived anywhere, and it is
        // largest where the interferer is rarest: at p = 0.1 only about a tenth of the M estimates carry it. So
        // the criterion is convergence rather than a pinned tolerance -- the deviation is measured at two values
        // of M and asserted to fall, which is what says the closed form is the limit and the gap is the
        // correction.
        for (const double p : {0.1, 0.25, 0.5}) {
            constexpr double kSignal = 100.;
            const double     mean    = p * kSignal + 1.;
            const double     closed  = (p * kSignal * kSignal + 4. * p * kSignal + 2.) / (mean * mean) - 1.;

            double previousDeviation = 0.;
            for (const std::size_t spectra : {1'024UZ, 8'192UZ}) {
                gr::rng::Xoshiro256pp         rng(0xFEEDULL + static_cast<std::uint64_t>(100. * p));
                gr::rng::GaussianNoise<float> noise(rng);

                SpectralKurtosisAccumulator accumulator(kBins);
                std::vector<float>          spectrum(kBins);
                for (std::size_t m = 0UZ; m < spectra; ++m) {
                    for (float& bin : spectrum) {
                        const bool    present = rng.uniform01<double>() < p;
                        const Complex z       = Complex{present ? static_cast<float>(std::sqrt(kSignal)) : 0.f, 0.f} + noise.complexSample();
                        bin                   = z.real() * z.real() + z.imag() * z.imag();
                    }
                    (void)accumulator.accumulate(spectrum);
                }

                std::vector<double> sk(kBins);
                (void)accumulator.evaluate(1., sk);
                const Spread measured  = spreadOf(sk);
                const double deviation = measured.mean - closed;
                std::println("SK on a pulsed interferer at p = {:.2f}, M = {:5}: mean {:.4f} against the closed form {:.4f}, deviation {:+.4f}", p, spectra, measured.mean, closed, deviation);

                expect(measured.mean >= 1. - 0.05) << "intermittent structure reads at or above 1, the other side of the discriminant";
                if (spectra == 1'024UZ) {
                    previousDeviation = std::abs(deviation);
                } else {
                    expect(std::abs(deviation) < previousDeviation) << "the gap to the closed form falls with M, so it is the finite-M correction and not a wrong limit";
                    expect(std::abs(deviation) < 0.05) << "and at M = 8192 it is inside 0.05 of the closed form";
                }
            }
        }
    };

    "what the spectral accumulator refuses, and the bin it cannot divide by"_test = [] {
        SpectralKurtosisAccumulator accumulator(4UZ);
        expect(accumulator.bins() == 4UZ);
        expect(accumulator.count() == 0UZ);

        const std::vector<float> wrongSize(5UZ, 1.f);
        expect(accumulator.accumulate(wrongSize) == SpectrumResponse::wrongBinCount);
        expect(accumulator.count() == 0UZ) << "a refused record folds nothing in";

        const std::vector<float> negative{1.f, -1.f, 1.f, 1.f};
        expect(accumulator.accumulate(negative) == SpectrumResponse::negativeBin);
        expect(accumulator.count() == 0UZ);
        expect(accumulator.nRefused() == 2ULL);

        const std::vector<float> dead{1.f, 0.f, 1.f, 1.f};
        for (std::size_t m = 0UZ; m < 16UZ; ++m) {
            expect(accumulator.accumulate(dead) == SpectrumResponse::accepted);
        }
        std::vector<double> sk(4UZ);
        expect(accumulator.evaluate(1., sk) == 1UZ) << "the dead bin is counted";
        expect(sk[1] == 0.) << "and written as zero, because one dead bin is a property of the signal";

        expect(throws<std::invalid_argument>([&] {
            std::vector<double> tooFew(3UZ);
            (void)accumulator.evaluate(1., tooFew);
        })) << "the output span takes one entry per bin";

        expect(spectralKurtosis(0., 0., 64UZ, 1.) == 0.) << "a bin with no power has no defined statistic";
        expect(spectralKurtosisVariance(1UZ, 1.) == 0.) << "and one spectrum is not a run of them";
    };
};

int main() { /* not needed for UT */ }
