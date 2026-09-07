#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <print>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

// The three backends of gr::algorithm::FFT over the receiver's own shape: one long single-precision complex
// transform of a power-of-two length, repeated at a fixed size, and the short real-input audio transform.
//
// Two columns beyond the elapsed time. The cost of one butterfly unit -- the time divided by N log2(N) -- is what
// compares across lengths and across engines. The error is the relative L2 norm of the whole spectrum against a
// double-precision reference of the same input, which says how much of a backend's speed is bought from accuracy.

using namespace std::chrono;

template<typename T>
using Aligned = std::vector<T, gr::allocator::Aligned<T>>;

template<typename T>
Aligned<T> broadbandSignal(std::size_t N) {
    Aligned<T> signal(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double x = std::sin(0.7 * static_cast<double>(i) + 0.3) + 0.4 * std::cos(2.9 * static_cast<double>(i));
        const double y = std::cos(1.3 * static_cast<double>(i) - 0.9) - 0.2 * std::sin(0.11 * static_cast<double>(i));
        if constexpr (gr::meta::complex_like<T>) {
            signal[i] = T(static_cast<typename T::value_type>(x), static_cast<typename T::value_type>(y));
        } else {
            signal[i] = static_cast<T>(x);
        }
    }
    return signal;
}

/// the same transform in double precision, through PocketFFT, which qa_algorithm_fourier pins against a direct DFT
template<typename TIn>
[[nodiscard]] std::vector<std::complex<double>> doubleReference(const Aligned<TIn>& signal) {
    using WideIn = std::conditional_t<gr::meta::complex_like<TIn>, std::complex<double>, double>;

    Aligned<WideIn> wide(signal.size());
    for (std::size_t i = 0; i < signal.size(); ++i) {
        wide[i] = static_cast<WideIn>(signal[i]);
    }

    gr::algorithm::FFT<WideIn, std::complex<double>> fft{};
    fft.backend = gr::algorithm::FftBackend::PocketFFT;
    Aligned<std::complex<double>> out(signal.size());
    fft.compute(wide, out);
    return {out.begin(), out.end()};
}

[[nodiscard]] double relativeL2Error(const Aligned<std::complex<float>>& actual, const std::vector<std::complex<double>>& reference) {
    double numerator   = 0.;
    double denominator = 0.;
    for (std::size_t k = 0; k < reference.size(); ++k) {
        const std::complex<double> difference(static_cast<double>(actual[k].real()) - reference[k].real(), static_cast<double>(actual[k].imag()) - reference[k].imag());
        numerator += std::norm(difference);
        denominator += std::norm(reference[k]);
    }
    return denominator > 0. ? std::sqrt(numerator / denominator) : std::sqrt(numerator);
}

// each measurement is given about the same amount of work, so a short length is not timed against the clock's
// own resolution and a long one does not dominate the run
[[nodiscard]] std::size_t repetitionsFor(std::size_t N) {
    constexpr double  kUnitBudget = 3.e8; // butterfly units per measurement
    const double      units       = static_cast<double>(N) * std::log2(static_cast<double>(N));
    const std::size_t requested   = static_cast<std::size_t>(kUnitBudget / units);
    return std::clamp(requested, 5UZ, 2000UZ);
}

template<typename TIn>
void benchBackend(std::string_view label, const Aligned<TIn>& signal, const std::vector<std::complex<double>>& reference, gr::algorithm::FftBackend backend) {
    const std::size_t                            N = signal.size();
    Aligned<std::complex<float>>                 out(N);
    gr::algorithm::FFT<TIn, std::complex<float>> fft{};
    fft.backend = backend;
    if (fft.resolveBackend(N) != backend) {
        std::println("  {:>10} N = {:8}           -- the backend does not take this length", label, N);
        return;
    }
    fft.compute(signal, out); // the plan and the tables are built here, not in the timed loop

    const std::size_t repetitions = repetitionsFor(N);
    const auto        start       = steady_clock::now();
    for (std::size_t i = 0; i < repetitions; ++i) {
        fft.compute(signal, out);
    }
    const auto elapsed = duration_cast<nanoseconds>(steady_clock::now() - start);

    const double perTransform = static_cast<double>(elapsed.count()) / static_cast<double>(repetitions);
    const double units        = static_cast<double>(N) * std::log2(static_cast<double>(N));
    std::println("  {:>10} N = {:8} {:12.1f} us {:9.3f} ns/unit  error {:9.2e}  ({} repetitions, {} plan build(s))", //
        label, N, perTransform * 1.e-3, perTransform / units, relativeL2Error(out, reference), repetitions, fft.pocketPlanBuilds);
}

template<typename TIn>
void sweep(std::size_t N) {
    using gr::algorithm::FftBackend;
    const auto signal    = broadbandSignal<TIn>(N);
    const auto reference = doubleReference(signal);
    benchBackend("native", signal, reference, FftBackend::Native);
    benchBackend("simd", signal, reference, FftBackend::Simd);
    benchBackend("pocketfft", signal, reference, FftBackend::PocketFFT);
}

int main() {
    std::println("complex<float> forward, one instance per backend and length");
    for (std::size_t exponent = 9UZ; exponent <= 22UZ; ++exponent) {
        sweep<std::complex<float>>(1UZ << exponent);
    }

    std::println("");
    std::println("composite lengths, complex<float> forward");
    for (const std::size_t N : {1250UZ, 4000UZ, 105UZ, 1009UZ}) {
        sweep<std::complex<float>>(N);
    }

    std::println("");
    std::println("float input, forward only (the audio transform)");
    for (const std::size_t N : {2048UZ, 8192UZ}) {
        sweep<float>(N);
    }
    return 0;
}
