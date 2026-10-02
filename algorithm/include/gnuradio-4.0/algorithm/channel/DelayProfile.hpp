#ifndef GNURADIO_ALGORITHM_CHANNEL_DELAY_PROFILE_HPP
#define GNURADIO_ALGORITHM_CHANNEL_DELAY_PROFILE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <expected>
#include <source_location>
#include <span>
#include <vector>

#include <gnuradio-4.0/Message.hpp>

namespace gr::channel {

/**
 * @brief Complex FIR taps for a published delay/power profile at a sample rate.
 *
 * A profile states delays in seconds and average powers in dB. Rounding each delay to the nearest sample
 * gives the tap positions; powers convert to amplitudes and, when `normalize` is set, are scaled so the
 * profile carries unit mean power — which keeps a channel from quietly changing the level a receiver sees.
 * Paths that round onto the same sample are summed in power, as they are in every published tabulation.
 *
 * Static multipath is this function plus an FIR filter. A time-varying channel needs tap gains that are
 * processes rather than constants, which is what a fading-channel block covers.
 *
 * Tables of different lengths, an empty profile, a sample rate that is not positive or a negative delay is an
 * error.
 */
[[nodiscard]] inline std::expected<std::vector<std::complex<float>>, gr::Error> tapsFromProfile(std::span<const double> delaysSeconds, std::span<const double> powersDb, double sampleRate, bool normalize = true, std::source_location location = std::source_location::current()) {
    if (delaysSeconds.size() != powersDb.size()) {
        return std::unexpected(gr::Error("tapsFromProfile: the delay and power tables must be the same length", location));
    }
    if (delaysSeconds.empty()) {
        return std::unexpected(gr::Error("tapsFromProfile: the profile is empty", location));
    }
    if (!(sampleRate > 0.)) {
        return std::unexpected(gr::Error("tapsFromProfile: the sample rate must be positive", location));
    }

    std::size_t span = 0UZ;
    for (const double delay : delaysSeconds) {
        if (delay < 0.) {
            return std::unexpected(gr::Error("tapsFromProfile: delays must not be negative", location));
        }
        span = std::max(span, static_cast<std::size_t>(std::llround(delay * sampleRate)));
    }

    std::vector<double> power(span + 1UZ, 0.);
    for (std::size_t p = 0UZ; p < delaysSeconds.size(); ++p) {
        power[static_cast<std::size_t>(std::llround(delaysSeconds[p] * sampleRate))] += std::pow(10., powersDb[p] / 10.);
    }

    double total = 0.;
    for (const double value : power) {
        total += value;
    }
    const double scale = (normalize && total > 0.) ? 1. / total : 1.;

    std::vector<std::complex<float>> taps(power.size());
    for (std::size_t k = 0UZ; k < power.size(); ++k) {
        taps[k] = std::complex<float>(static_cast<float>(std::sqrt(power[k] * scale)), 0.f);
    }
    return taps;
}

} // namespace gr::channel

#endif // GNURADIO_ALGORITHM_CHANNEL_DELAY_PROFILE_HPP
