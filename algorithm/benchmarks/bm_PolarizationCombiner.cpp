#include <benchmark.hpp>

#include <gnuradio-4.0/algorithm/measurement/PolarizationCombiner.hpp>
#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

#include <complex>
#include <cstddef>
#include <format>
#include <string_view>
#include <vector>

namespace {

using Complex = std::complex<float>;

constexpr std::size_t kSamples = 65'536UZ;
constexpr std::size_t kRepeats = 200UZ;

} // namespace

/// Per output sample the combination is two complex multiply-accumulates, the orthogonal output two more, and
/// the covariance four real multiply-adds. The eigen solve is once per window -- one square root, one `atan2`
/// and a handful of multiplies -- so it does not appear per sample at all and is not timed here.
void benchPolarizationCombiner() {
    using namespace benchmark;
    using boost::ut::operator""_test;

    "PolarizationCombiner: ns/sample"_test = [] {
        gr::rng::Xoshiro256pp         rng(3ULL);
        gr::rng::GaussianNoise<float> noise(rng);

        std::vector<Complex> branch0(kSamples);
        std::vector<Complex> branch1(kSamples);
        noise.fillComplex(branch0);
        noise.fillComplex(branch1);

        gr::measurement::BranchCovariance covariance;
        covariance.add(branch0, branch1);
        const gr::measurement::PolarizationEstimate estimate = covariance.solve();

        std::vector<Complex> out(kSamples);
        std::vector<Complex> ortho(kSamples);

        const auto combined                                                    = std::format("polarizationCombine, no orthogonal port, N={}", kSamples);
        ::benchmark::benchmark<kRepeats>(std::string_view(combined), kSamples) = [&] { gr::measurement::polarizationCombine(branch0, branch1, estimate, out); };
        const auto both                                                        = std::format("polarizationCombine, with it,           N={}", kSamples);
        ::benchmark::benchmark<kRepeats>(std::string_view(both), kSamples)     = [&] { gr::measurement::polarizationCombine(branch0, branch1, estimate, out, ortho); };

        const auto accumulate                                                    = std::format("BranchCovariance::add,                  N={}", kSamples);
        ::benchmark::benchmark<kRepeats>(std::string_view(accumulate), kSamples) = [&] {
            gr::measurement::BranchCovariance window;
            window.add(branch0, branch1);
            ::benchmark::force_to_memory(window);
        };
        ::benchmark::results::add_separator();
    };
}

inline const boost::ut::suite<"polarization benchmarks"> _polarization_bm = [] { benchPolarizationCombiner(); };

int main() { /* not needed by the UT framework */ }
