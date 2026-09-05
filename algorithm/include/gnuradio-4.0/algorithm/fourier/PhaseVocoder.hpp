#ifndef GNURADIO_PHASE_VOCODER_HPP
#define GNURADIO_PHASE_VOCODER_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

namespace gr::algorithm {

/**
 * @brief Time-scale modification of a complex stream that keeps every partial coherent.
 *
 * Frames of `frameSize` samples are taken every `analysisHop` samples under a Hann window and
 * resynthesized every `synthesisHop` samples under the same window, so the stream's spectrum is
 * replayed at `analysisHop / synthesisHop` times its original pace with the spectral content
 * untouched. Each bin's synthesis phase advances by the bin's MEASURED frequency — its center plus
 * the principal value of the analysis phase advance's deviation from the center's, divided by the
 * analysis hop — rather than by the center alone, which is what keeps a partial that sits between
 * bins coherent from frame to frame. The first frame adopts its analysis phase outright, so a
 * stream that begins mid-tone begins coherent instead of at an arbitrary offset.
 *
 * The spectrum is never modified, so no two bins collide in the output and there is no amplitude
 * ambiguity to resolve: the overlap-add is normalized by the summed squared window actually laid
 * down, which preserves level at any hop pair. Equal hops replay the stream exactly; the identity
 * case is short-circuited to a copy with no latency.
 *
 * State carries across `process` calls and the call boundaries are not observable in the output:
 * a stream fed whole and the same stream fed one sample at a time produce identical results.
 */
struct PhaseVocoder {
    std::size_t frameSize    = 0UZ;
    std::size_t analysisHop  = 0UZ;
    std::size_t synthesisHop = 0UZ;

    /**
     * @brief Size the engine. Throws on a shape the algorithm cannot run.
     *
     * The frame must be a power of two (the transform's shape) and both hops whole, positive and
     * no longer than the frame; a hop above the frame would leave gaps no window overlaps.
     */
    void configure(std::size_t frame, std::size_t hopIn, std::size_t hopOut) {
        if (frame == 0UZ || (frame & (frame - 1UZ)) != 0UZ) {
            throw std::invalid_argument("PhaseVocoder: frame must be a power of two");
        }
        if (hopIn == 0UZ || hopOut == 0UZ || hopIn > frame || hopOut > frame) {
            throw std::invalid_argument("PhaseVocoder: hops must be positive and no longer than the frame");
        }
        frameSize    = frame;
        analysisHop  = hopIn;
        synthesisHop = hopOut;

        _window.resize(frame);
        for (std::size_t i = 0UZ; i < frame; ++i) {
            _window[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(frame)));
        }
        _scratch.resize(frame);
        _spectrum.resize(frame);
        _synth.resize(frame);
        _frameOut.resize(frame);
        reset();
    }

    /// @brief Forget all stream state; the next call primes from nothing, as at construction.
    void reset() {
        _pending.clear();
        _lastPhase.assign(frameSize, 0.0);
        _sumPhase.assign(frameSize, 0.0);
        _acc.assign(frameSize, std::complex<float>{0.f, 0.f});
        _accWindow.assign(frameSize, 0.f);
        _primed = false;
    }

    /**
     * @brief Consume a block of input, appending however much output it completes.
     *
     * Output arrives at `synthesisHop / analysisHop` of the input count in the steady state; a
     * final partial frame stays pending, since a window that cannot fill resolves no frequency.
     */
    void process(std::span<const std::complex<float>> in, std::vector<std::complex<float>>& out) {
        if (analysisHop == synthesisHop) { // identity: replaying at the same pace is the input
            out.insert(out.end(), in.begin(), in.end());
            return;
        }
        _pending.insert(_pending.end(), in.begin(), in.end());

        std::size_t consumed = 0UZ;
        while (_pending.size() - consumed >= frameSize) {
            analyzeResynthesize(_pending.data() + consumed, out);
            consumed += analysisHop;
        }
        _pending.erase(_pending.begin(), _pending.begin() + static_cast<std::ptrdiff_t>(consumed));
    }

private:
    void analyzeResynthesize(const std::complex<float>* src, std::vector<std::complex<float>>& out) {
        for (std::size_t i = 0UZ; i < frameSize; ++i) {
            _scratch[i] = src[i] * _window[i];
        }
        _fft.compute(_scratch, _spectrum);

        const double expectedPerBin = 2.0 * std::numbers::pi * static_cast<double>(analysisHop) / static_cast<double>(frameSize);
        for (std::size_t k = 0UZ; k < frameSize; ++k) {
            const double magnitude = std::abs(_spectrum[k]);
            const double phase     = std::arg(_spectrum[k]);
            if (!_primed) {
                _lastPhase[k] = phase;
                _sumPhase[k]  = phase;
            } else {
                const double deviation = principal(phase - _lastPhase[k] - expectedPerBin * static_cast<double>(k));
                const double omega     = 2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(frameSize) + deviation / static_cast<double>(analysisHop);
                _lastPhase[k]          = phase;
                _sumPhase[k] += omega * static_cast<double>(synthesisHop);
            }
            _synth[k] = std::polar(static_cast<float>(magnitude), static_cast<float>(_sumPhase[k]));
        }
        _primed = true;

        inverse(_synth, _frameOut);
        for (std::size_t i = 0UZ; i < frameSize; ++i) {
            _acc[i] += _frameOut[i] * _window[i];
            _accWindow[i] += _window[i] * _window[i];
        }

        // The leading synthesis hop can receive no further contribution, so it is final;
        // normalizing by the window energy actually laid down keeps the level independent of
        // the overlap.
        for (std::size_t i = 0UZ; i < synthesisHop; ++i) {
            out.push_back(_accWindow[i] > 1e-12f ? _acc[i] / _accWindow[i] : std::complex<float>{0.f, 0.f});
        }
        std::rotate(_acc.begin(), _acc.begin() + static_cast<std::ptrdiff_t>(synthesisHop), _acc.end());
        std::rotate(_accWindow.begin(), _accWindow.begin() + static_cast<std::ptrdiff_t>(synthesisHop), _accWindow.end());
        std::fill(_acc.end() - static_cast<std::ptrdiff_t>(synthesisHop), _acc.end(), std::complex<float>{0.f, 0.f});
        std::fill(_accWindow.end() - static_cast<std::ptrdiff_t>(synthesisHop), _accWindow.end(), 0.f);
    }

    /// @brief Wrap to (-pi, pi].
    [[nodiscard]] static double principal(double angle) noexcept { return angle - 2.0 * std::numbers::pi * std::round(angle / (2.0 * std::numbers::pi)); }

    /// @brief Inverse transform through the forward one: conjugate, transform, conjugate, scale.
    void inverse(const std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out) {
        for (std::size_t i = 0UZ; i < frameSize; ++i) {
            _scratch[i] = std::conj(in[i]);
        }
        _fft.compute(_scratch, _spectrum);
        const float scale = 1.f / static_cast<float>(frameSize);
        for (std::size_t i = 0UZ; i < frameSize; ++i) {
            out[i] = std::conj(_spectrum[i]) * scale;
        }
    }

    std::vector<float>               _window;
    std::vector<std::complex<float>> _pending;
    std::vector<double>              _lastPhase;
    std::vector<double>              _sumPhase;
    std::vector<std::complex<float>> _acc;
    std::vector<float>               _accWindow;
    bool                             _primed = false;

    FFT<std::complex<float>>         _fft;
    std::vector<std::complex<float>> _scratch, _spectrum, _synth, _frameOut;
};

} // namespace gr::algorithm

#endif // GNURADIO_PHASE_VOCODER_HPP
