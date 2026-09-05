#include <boost/ut.hpp>

#include <algorithm>
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

#include <gnuradio-4.0/algorithm/measurement/PolarizationCombiner.hpp>
#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

using namespace boost::ut;
using gr::measurement::BranchCovariance;
using gr::measurement::kPolarizationSaturation;
using gr::measurement::polarizationCombine;
using gr::measurement::PolarizationEstimate;
using gr::measurement::PolarizationMode;
using gr::measurement::polarizationModeFrom;
using gr::measurement::PolarizationNormalization;
using gr::measurement::polarizationNormalizationFrom;
using gr::measurement::polarizationPassthrough;

namespace {

using Complex = std::complex<float>;

/// A two-branch scene: one signal through two channels, plus independent branch noise.
struct Scene {
    std::vector<Complex> signal0; ///< the signal component on branch 0, alone
    std::vector<Complex> signal1;
    std::vector<Complex> noise0; ///< the noise component on branch 0, alone
    std::vector<Complex> noise1;
    std::vector<Complex> branch0; ///< their sum, which is what the estimator sees
    std::vector<Complex> branch1;
};

[[nodiscard]] Scene makeScene(gr::rng::GaussianNoise<float>& noise, std::complex<double> h0, std::complex<double> h1, double noisePower0, double noisePower1, std::size_t n) {
    Scene scene;
    scene.signal0.resize(n);
    scene.signal1.resize(n);
    scene.noise0.resize(n);
    scene.noise1.resize(n);
    scene.branch0.resize(n);
    scene.branch1.resize(n);

    const Complex g0 = static_cast<Complex>(h0);
    const Complex g1 = static_cast<Complex>(h1);
    const float   a0 = static_cast<float>(std::sqrt(noisePower0));
    const float   a1 = static_cast<float>(std::sqrt(noisePower1));

    noise.fillComplex(scene.noise0, a0);
    noise.fillComplex(scene.noise1, a1);
    for (std::size_t k = 0UZ; k < n; ++k) {
        const Complex s  = noise.complexSample();
        scene.signal0[k] = g0 * s;
        scene.signal1[k] = g1 * s;
        scene.branch0[k] = scene.signal0[k] + scene.noise0[k];
        scene.branch1[k] = scene.signal1[k] + scene.noise1[k];
    }
    return scene;
}

[[nodiscard]] double meanPower(std::span<const Complex> in) {
    double sum = 0.;
    for (const Complex value : in) {
        sum += static_cast<double>(value.real()) * static_cast<double>(value.real()) + static_cast<double>(value.imag()) * static_cast<double>(value.imag());
    }
    return sum / static_cast<double>(in.size());
}

/// The realized output ratio: the signal-only and noise-only streams put through the same weights, so the
/// measurement is of the combination and not of one particular draw of the sum.
[[nodiscard]] double realizedSnr(const Scene& scene, const PolarizationEstimate& estimate) {
    std::vector<Complex> signal(scene.signal0.size());
    std::vector<Complex> noise(scene.noise0.size());
    polarizationCombine(scene.signal0, scene.signal1, estimate, signal);
    polarizationCombine(scene.noise0, scene.noise1, estimate, noise);
    return meanPower(signal) / meanPower(noise);
}

[[nodiscard]] double toDb(double linear) { return 10. * std::log10(linear); }

} // namespace

const boost::ut::suite<"PolarizationCombiner"> _polarization = [] {
    "the spellings, and nothing else"_test = [] {
        expect(polarizationModeFrom("mrc").value() == PolarizationMode::mrc);
        expect(polarizationModeFrom("selection").value() == PolarizationMode::selection);
        expect(!polarizationModeFrom("mrc ").has_value());
        expect(polarizationNormalizationFrom("unit_noise").value() == PolarizationNormalization::unit_noise);
        expect(polarizationNormalizationFrom("unit_signal").value() == PolarizationNormalization::unit_signal);
        expect(!polarizationNormalizationFrom("unity").has_value());
    };

    "the estimator's own identity holds whatever the data is"_test = [] {
        // No assumed model: arbitrary seeded pairs. `a - lambda_-` and `b - lambda_-` sum to
        // `lambda_+ - lambda_-` by construction, so the two branch ratios sum to the combined one as algebra
        // rather than as an approximation. This is the criterion that catches a sign or an eigenvalue swap.
        gr::rng::Xoshiro256pp         rng(0xDECAFULL);
        gr::rng::GaussianNoise<float> noise(rng);

        double worst = 0.;
        for (std::size_t trial = 0UZ; trial < 20UZ; ++trial) {
            BranchCovariance covariance;
            for (std::size_t k = 0UZ; k < 4'096UZ; ++k) {
                const Complex a = noise.complexSample() + Complex{0.3f * static_cast<float>(trial), 0.1f};
                const Complex b = 0.7f * a + noise.complexSample();
                covariance.add(a, b);
            }
            const PolarizationEstimate estimate = covariance.solve();
            const double               sum      = estimate.branchSnr0 + estimate.branchSnr1;
            worst                               = std::max(worst, std::abs(sum - estimate.combinedSnr) / estimate.combinedSnr);
        }
        std::println("the additive identity on arbitrary data: worst relative departure {:.3e}", worst);
        expect(lt(worst, 1e-9)) << "SNR_out = SNR_0 + SNR_1 holds on the estimates themselves";
    };

    "the estimates recover the channel, free of noise bias"_test = [] {
        constexpr std::size_t kWindow = 262'144UZ;
        constexpr double      kNoise  = 0.02;

        gr::rng::Xoshiro256pp         rng(0x1234ULL);
        gr::rng::GaussianNoise<float> noise(rng);
        const std::complex<double>    h1    = 0.5 * std::exp(std::complex<double>{0., 2.});
        const Scene                   scene = makeScene(noise, {1., 0.}, h1, kNoise, kNoise, kWindow);

        BranchCovariance covariance;
        covariance.add(scene.branch0, scene.branch1);
        const PolarizationEstimate estimate = covariance.solve();

        const double tolerance = 4. / std::sqrt(static_cast<double>(kWindow));
        std::println("channel recovery at M = {}: relative phase {:.5f} against -2, amplitude ratio {:.5f} against 2, tolerance {:.5f}", kWindow, estimate.relativePhase, estimate.amplitudeRatio, tolerance);
        expect(std::abs(estimate.relativePhase + 2.) < tolerance) << "arg(h0) - arg(h1) is -2 rad";
        expect(std::abs(estimate.amplitudeRatio - 2.) < tolerance) << "|h0|/|h1| is 2";

        // The naive ratio is biased toward 1 by the noise, which is the whole reason the estimator does not use
        // it. Measured here so the difference is a number rather than a claim.
        const double naive = std::sqrt(covariance.a() / covariance.b());
        std::println("the naive sqrt(a/b) reads {:.5f}, biased toward 1 by the branch noise", naive);
        expect(lt(naive, 2.)) << "and it really is biased low";
    };

    "maximal-ratio combining adds the two ratios, to within an envelope of two named terms"_test = [] {
        constexpr std::size_t kWindow = 262'144UZ;
        constexpr double      kNoise  = 0.1;
        // 4.343/sqrt(N) dB is the spread of one power measurement; the weight-estimation loss is the other term
        // and is far smaller at this window. Four of the first is 0.034 dB, so the envelope is 0.05.
        constexpr double kEnvelopeDb = 0.05;

        struct Arm {
            std::complex<double> h1;
            double               expectedGainDb;
            std::string_view     label;
        };
        const Arm arms[] = {{{std::cos(0.7), std::sin(0.7)}, 3.0103, "equal branches"}, {0.5 * std::exp(std::complex<double>{0., 2.}), 0.9691, "branch 1 at -6 dB"}};

        for (const Arm& arm : arms) {
            gr::rng::Xoshiro256pp         rng(0xF00DULL);
            gr::rng::GaussianNoise<float> noise(rng);

            // The weights are estimated on one window and applied to an independent one, so the figure carries
            // the estimation loss rather than hiding it by fitting the same samples it is measured on.
            const Scene training = makeScene(noise, {1., 0.}, arm.h1, kNoise, kNoise, kWindow);
            const Scene applied  = makeScene(noise, {1., 0.}, arm.h1, kNoise, kNoise, kWindow);

            BranchCovariance covariance;
            covariance.add(training.branch0, training.branch1);
            const PolarizationEstimate estimate = covariance.solve();

            const double combined = realizedSnr(applied, estimate);
            const double branch0  = meanPower(applied.signal0) / meanPower(applied.noise0);
            const double branch1  = meanPower(applied.signal1) / meanPower(applied.noise1);
            const double gainDb   = toDb(combined) - toDb(std::max(branch0, branch1));

            std::println("{}: branch ratios {:.3f} and {:.3f} dB, combined {:.3f} dB, gain {:.4f} dB against the predicted {:.4f}", arm.label, toDb(branch0), toDb(branch1), toDb(combined), gainDb, arm.expectedGainDb);
            expect(std::abs(gainDb - arm.expectedGainDb) < kEnvelopeDb) << arm.label << "misses 10*log10(1 + SNR_min/SNR_max)";
            expect(approx(toDb(combined), toDb(branch0 + branch1), kEnvelopeDb)) << "and the realized output ratio is the sum of the two branch ratios";
        }
    };

    "the gauge is pinned, so no window boundary puts a phase step in the output"_test = [] {
        constexpr std::size_t kWindow = 8'192UZ;

        gr::rng::Xoshiro256pp         rng(0xABCDULL);
        gr::rng::GaussianNoise<float> noise(rng);
        const std::complex<double>    h1 = std::exp(std::complex<double>{0., 1.3});

        double worstStep = 0.;
        double previous  = 0.;
        for (std::size_t window = 0UZ; window < 8UZ; ++window) {
            const Scene      scene = makeScene(noise, {1., 0.}, h1, 0.05, 0.05, kWindow);
            BranchCovariance covariance;
            covariance.add(scene.branch0, scene.branch1);
            const PolarizationEstimate estimate = covariance.solve();

            // An eigenvector is defined up to a unit-modulus scalar; a solver that did not pin one would put a
            // random phase on the combined output at every one of these boundaries.
            expect(estimate.weight0.imag() == 0.) << "arg(weight 0) is zero exactly, window" << window;
            expect(estimate.weight0.real() > 0.) << "and it is the positive real axis, not the negative one";

            const double phase = std::arg(estimate.weight1);
            if (window > 0UZ) {
                worstStep = std::max(worstStep, std::abs(phase - previous));
            }
            previous = phase;
        }
        std::println("across eight windows the second weight's phase moves at most {:.6f} rad", worstStep);
        expect(lt(worstStep, 0.05)) << "the weights move with the channel and not with the solver";
    };

    "the orthogonal output nulls the signal"_test = [] {
        constexpr std::size_t kWindow = 65'536UZ;

        gr::rng::Xoshiro256pp         rng(0x5150ULL);
        gr::rng::GaussianNoise<float> noise(rng);
        const std::complex<double>    h1    = 0.8 * std::exp(std::complex<double>{0., -0.4});
        const Scene                   scene = makeScene(noise, {1., 0.}, h1, 0.01, 0.01, kWindow);

        BranchCovariance covariance;
        covariance.add(scene.branch0, scene.branch1);
        const PolarizationEstimate estimate = covariance.solve();

        // The algebraic identity first: `u = [-conj(w_1), conj(w_0)]` is orthogonal to `w` by construction.
        const std::complex<double> inner = std::conj(estimate.weight0) * estimate.ortho0 + std::conj(estimate.weight1) * estimate.ortho1;
        expect(lt(std::abs(inner), 1e-15)) << "w^H v_perp is zero to rounding, got" << std::abs(inner);

        std::vector<Complex> combined(kWindow);
        std::vector<Complex> ortho(kWindow);
        polarizationCombine(scene.signal0, scene.signal1, estimate, combined, ortho);
        const double leak = toDb(meanPower(ortho) / meanPower(combined));
        std::println("the orthogonal port carries the signal {:.1f} dB below the combined port", leak);
        expect(lt(leak, -40.)) << "the null is a null";
    };

    "selection follows the stronger branch and counts the crossing"_test = [] {
        constexpr std::size_t kWindow = 16'384UZ;

        gr::rng::Xoshiro256pp         rng(0x2468ULL);
        gr::rng::GaussianNoise<float> noise(rng);

        int         selected  = -1;
        std::size_t switches  = 0UZ;
        std::size_t mrcBetter = 0UZ;
        for (std::size_t step = 0UZ; step < 6UZ; ++step) {
            // Branch 1's amplitude crosses branch 0's halfway through.
            const double               magnitude = 0.5 + 0.25 * static_cast<double>(step);
            const std::complex<double> h1        = magnitude * std::exp(std::complex<double>{0., 0.9});
            const Scene                scene     = makeScene(noise, {1., 0.}, h1, 0.05, 0.05, kWindow);

            BranchCovariance covariance;
            covariance.add(scene.branch0, scene.branch1);
            const PolarizationEstimate chosen = covariance.solve(PolarizationMode::selection);
            const PolarizationEstimate mrc    = covariance.solve(PolarizationMode::mrc);

            if (selected >= 0 && chosen.selectedBranch != selected) {
                ++switches;
            }
            selected = chosen.selectedBranch;

            const double bySelection = realizedSnr(scene, chosen);
            const double byMrc       = realizedSnr(scene, mrc);
            if (byMrc > bySelection) {
                ++mrcBetter;
            }
            expect(mrc.selectedBranch == -1) << "mrc names no branch";
        }
        expect(switches == 1UZ) << "the selection changes exactly once as the branches cross, got" << switches;
        expect(mrcBetter == 6UZ) << "and the maximal-ratio combination beats the stronger branch at every step";
    };

    "saturation is counted, never silent"_test = [] {
        constexpr std::size_t kWindow = 4'096UZ;

        // No noise in either branch: the two branches carry the same deterministic sequence, so `a`, `b` and
        // `|c|` are the same sums and `lambda_-` is exactly zero rather than a rounding of it.
        BranchCovariance noiseless;
        for (std::size_t k = 0UZ; k < kWindow; ++k) {
            const Complex sample{static_cast<float>(std::cos(0.31 * static_cast<double>(k))), static_cast<float>(std::sin(0.31 * static_cast<double>(k)))};
            noiseless.add(sample, sample);
        }
        const PolarizationEstimate hot = noiseless.solve();
        expect(hot.saturatedSnr) << "an unbounded ratio is flagged";
        expect(hot.branchSnr0 == kPolarizationSaturation) << "and reported at the stated 60 dB cap";
        expect(hot.combinedSnr == kPolarizationSaturation);
        expect(approx(hot.combinedSnrDb(), 60., 1e-9));

        // Branch 1 carries nothing correlated at all: `lambda_+ - a` is exactly zero, the amplitude ratio is
        // unbounded, and the weights degenerate to branch 0 alone -- which is the right answer, not an error.
        gr::rng::Xoshiro256pp         rng(0x9999ULL);
        gr::rng::GaussianNoise<float> noise(rng);
        std::vector<Complex>          only0(kWindow);
        const std::vector<Complex>    empty(kWindow);
        noise.fillComplex(only0);

        BranchCovariance lonely;
        lonely.add(only0, empty);
        const PolarizationEstimate degenerate = lonely.solve();
        expect(degenerate.saturatedRatio) << "the unbounded amplitude ratio is flagged";
        expect(degenerate.amplitudeRatio == kPolarizationSaturation);
        expect(degenerate.weight0 == std::complex<double>{1., 0.});
        expect(degenerate.weight1 == std::complex<double>{0., 0.});

        std::vector<Complex> out(kWindow);
        polarizationCombine(only0, empty, degenerate, out);
        expect(that % std::ranges::equal(out, only0)) << "and the output is branch 0, bit for bit";
    };

    "unequal branch noise reduces to the equal case by whitening"_test = [] {
        constexpr std::size_t kWindow = 262'144UZ;

        gr::rng::Xoshiro256pp         rng(0x77ULL);
        gr::rng::GaussianNoise<float> noise(rng);
        const std::complex<double>    h1     = std::exp(std::complex<double>{0., 0.4});
        const Scene                   scene  = makeScene(noise, {1., 0.}, h1, 0.02, 0.2, kWindow);
        const std::vector<double>     powers = {0.02, 0.2};

        BranchCovariance covariance;
        covariance.add(scene.branch0, scene.branch1);

        const PolarizationEstimate whitened = covariance.solve(PolarizationMode::mrc, PolarizationNormalization::unit_noise, powers);
        const PolarizationEstimate assumed  = covariance.solve();

        const double branch0 = meanPower(scene.signal0) / meanPower(scene.noise0);
        const double branch1 = meanPower(scene.signal1) / meanPower(scene.noise1);
        const double best    = realizedSnr(scene, whitened);
        const double naive   = realizedSnr(scene, assumed);

        std::println("branch ratios {:.3f} and {:.3f} dB; whitened combination {:.3f} dB, the equal-noise assumption {:.3f} dB, the optimum {:.3f}", toDb(branch0), toDb(branch1), toDb(best), toDb(naive), toDb(branch0 + branch1));
        expect(approx(toDb(best), toDb(branch0 + branch1), 0.05)) << "whitening reaches the optimum with a ten-to-one noise imbalance";
        expect(gt(best, naive)) << "and the equal-noise assumption does not";

        expect(throws<std::invalid_argument>([&] { (void)covariance.solve(PolarizationMode::mrc, PolarizationNormalization::unit_noise, std::vector<double>{1.}); })) << "two branches take two noise powers";
        expect(throws<std::invalid_argument>([&] { (void)covariance.solve(PolarizationMode::mrc, PolarizationNormalization::unit_noise, std::vector<double>{1., 0.}); })) << "and both must be positive";
    };

    "the covariance does not depend on how the stream is chunked"_test = [] {
        constexpr std::size_t kWindow = 10'000UZ;

        gr::rng::Xoshiro256pp         rng(0x3333ULL);
        gr::rng::GaussianNoise<float> noise(rng);
        const Scene                   scene = makeScene(noise, {1., 0.}, {0.6, 0.3}, 0.05, 0.05, kWindow);

        BranchCovariance whole;
        whole.add(scene.branch0, scene.branch1);

        for (const std::size_t chunk : {1UZ, 7UZ, 1'000UZ, 12'345UZ}) {
            BranchCovariance pieced;
            for (std::size_t at = 0UZ; at < kWindow; at += chunk) {
                const std::size_t n = std::min(chunk, kWindow - at);
                pieced.add(std::span<const Complex>(scene.branch0).subspan(at, n), std::span<const Complex>(scene.branch1).subspan(at, n));
            }
            expect(pieced.a() == whole.a()) << std::format("chunk {} changed the covariance", chunk);
            expect(pieced.b() == whole.b());
            expect(pieced.c() == whole.c());
            expect(pieced.count() == whole.count());
        }
    };

    "the startup estimate is branch 0 unchanged, and the refusals are paired"_test = [] {
        const PolarizationEstimate first = polarizationPassthrough(0UZ);
        expect(first.weight0 == std::complex<double>{1., 0.});
        expect(first.weight1 == std::complex<double>{0., 0.});
        expect(first.selectedBranch == 0);

        const std::vector<Complex> in0{{1.f, 2.f}, {3.f, 4.f}};
        const std::vector<Complex> in1{{5.f, 6.f}, {7.f, 8.f}};
        std::vector<Complex>       out(2UZ);
        std::vector<Complex>       ortho(2UZ);
        polarizationCombine(in0, in1, first, out, ortho);
        expect(that % std::ranges::equal(out, in0)) << "branch 0 passes through bit for bit before anything is measured";
        expect(that % std::ranges::equal(ortho, in1)) << "and branch 1 goes to the orthogonal port";

        const PolarizationEstimate second = polarizationPassthrough(1UZ);
        polarizationCombine(in0, in1, second, out, ortho);
        expect(that % std::ranges::equal(out, in1));

        expect(throws<std::invalid_argument>([&] { polarizationCombine(in0, std::span<const Complex>(in1).first(1UZ), first, out); })) << "the two branches are consumed in lockstep";
        expect(throws<std::invalid_argument>([&] { polarizationCombine(in0, in1, first, std::span<Complex>(out).first(1UZ)); })) << "one output short";
        expect(throws<std::invalid_argument>([&] { polarizationCombine(in0, in1, first, out, std::span<Complex>(ortho).first(1UZ)); })) << "one orthogonal output short";

        BranchCovariance covariance;
        expect(throws<std::invalid_argument>([&] { covariance.add(in0, std::span<const Complex>(in1).first(1UZ)); }));
        expect(covariance.count() == 0UZ);
        expect(covariance.a() == 0.) << "an empty covariance divides by nothing";
    };

    "unit_signal states its own scale"_test = [] {
        constexpr std::size_t kWindow = 65'536UZ;

        gr::rng::Xoshiro256pp         rng(0x8080ULL);
        gr::rng::GaussianNoise<float> noise(rng);
        const Scene                   scene = makeScene(noise, {1., 0.}, {0.7, 0.2}, 0.05, 0.05, kWindow);

        BranchCovariance covariance;
        covariance.add(scene.branch0, scene.branch1);

        const PolarizationEstimate unitNoise  = covariance.solve(PolarizationMode::mrc, PolarizationNormalization::unit_noise);
        const PolarizationEstimate unitSignal = covariance.solve(PolarizationMode::mrc, PolarizationNormalization::unit_signal);

        expect(unitNoise.scale == 1.);
        expect(approx(std::norm(unitNoise.weight0) + std::norm(unitNoise.weight1), 1., 1e-12)) << "unit_noise keeps a unit weight vector, so the output noise power is one branch's";

        // The signal's own amplitude is not identifiable from the covariance -- only `P|h|^2 = lambda_+ -
        // lambda_-` is -- so unit_signal is the scale that makes the combined output's signal power one, and
        // that is what it reports.
        std::vector<Complex> signal(kWindow);
        polarizationCombine(scene.signal0, scene.signal1, unitSignal, signal);
        std::println("unit_signal scale {:.6f}, realized output signal power {:.6f}", unitSignal.scale, meanPower(signal));
        expect(approx(meanPower(signal), 1., 0.02)) << "the combined output's signal power is 1";
        expect(approx(unitSignal.scale, 1. / std::sqrt(unitSignal.lambdaPlus - unitSignal.lambdaMinus), 1e-12));

        // The ratios do not depend on the scale, which is what makes them comparable across normalizations.
        expect(approx(unitNoise.combinedSnr, unitSignal.combinedSnr, 1e-12));
    };
};

int main() { /* not needed for UT */ }
