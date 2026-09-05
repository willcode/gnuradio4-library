#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/sigmf/Json.hpp>
#include <gnuradio-4.0/algorithm/sigmf/SampleCodec.hpp>
#include <gnuradio-4.0/algorithm/sigmf/SigMfMetadata.hpp>

// What the SigMF codec costs per component, against a plain memcpy of the same byte count as the floor.
//
// The specification asserts no timing number because none was measured for it. The reasoning it records is that the
// identity pairs should reach the file system's own throughput because the codec does no arithmetic; that the scaled
// integer paths should be memory-bound rather than arithmetic-bound, because one multiply per component against four
// to eight bytes moved is well under one operation per byte; and that the `_be` paths should cost nothing measurable
// on a target with a byte-swap instruction. A run that contradicts any of the three means the per-component loop is
// not vectorizing and the no-branch-in-the-loop requirement is not being met.
//
// Three buffer sizes so the L2 boundary is visible, and the metadata parser and writer in isolation at three capture
// counts, so the per-recording cost is measured rather than assumed.

namespace {

using namespace std::chrono;
using namespace std::string_view_literals;

constexpr std::array<std::size_t, 3> kBufferSizes{1024UZ, 65536UZ, 1048576UZ};

/// Nanoseconds per component, taking the best of a few passes so a scheduling hiccup does not become the figure.
template<typename TWork>
[[nodiscard]] double nanosecondsPerComponent(std::size_t components, TWork&& work) {
    constexpr std::size_t kPasses = 5UZ;
    double                best    = std::numeric_limits<double>::max();
    for (std::size_t pass = 0UZ; pass < kPasses; ++pass) {
        const auto start = steady_clock::now();
        work();
        const auto elapsed = duration_cast<nanoseconds>(steady_clock::now() - start).count();
        best               = std::min(best, static_cast<double>(elapsed) / static_cast<double>(components));
    }
    return best;
}

void reportDecode(std::string_view spelling, gr::sigmf::Scaling scaling, std::size_t bufferBytes) {
    const auto datatype = gr::sigmf::parseDatatype(spelling);
    if (!datatype) {
        std::println("{:>10}  unsupported", spelling);
        return;
    }
    const auto decode = gr::sigmf::selectDecoder<float>(*datatype, scaling);
    if (!decode) {
        std::println("{:>10}  {}", spelling, decode.error());
        return;
    }

    const std::size_t      components = bufferBytes / datatype->bytesPerComponent();
    std::vector<std::byte> source(components * datatype->bytesPerComponent(), std::byte{0x11});
    std::vector<float>     destination(components);
    std::vector<std::byte> floor(source.size());

    const double codec = nanosecondsPerComponent(components, [&] { std::ignore = (*decode)(source.data(), destination.data(), components); });
    const double copy  = nanosecondsPerComponent(components, [&] { std::memcpy(floor.data(), source.data(), source.size()); });
    std::println("  {:>10} {:>5} {:>10} B  {:8.3f} ns/component   memcpy floor {:8.3f}", spelling, scaling == gr::sigmf::Scaling::Unit ? "unit"sv : "raw"sv, bufferBytes, codec, copy);
}

void reportEncode(std::string_view spelling, gr::sigmf::Scaling scaling, std::size_t bufferBytes) {
    const auto datatype = gr::sigmf::parseDatatype(spelling);
    if (!datatype) {
        return;
    }
    const auto encode = gr::sigmf::selectEncoder<float>(*datatype, scaling);
    if (!encode) {
        return;
    }
    const std::size_t      components = bufferBytes / datatype->bytesPerComponent();
    std::vector<float>     source(components, 0.25f);
    std::vector<std::byte> destination(components * datatype->bytesPerComponent());

    const double codec = nanosecondsPerComponent(components, [&] { std::ignore = (*encode)(source.data(), destination.data(), components); });
    std::println("  {:>10} {:>5} {:>10} B  {:8.3f} ns/component   (write)", spelling, scaling == gr::sigmf::Scaling::Unit ? "unit"sv : "raw"sv, bufferBytes, codec);
}

[[nodiscard]] gr::sigmf::Metadata metadataWith(std::size_t captureCount) {
    gr::sigmf::Metadata metadata;
    metadata.global.datatype   = "cf32_le";
    metadata.global.version    = std::string(gr::sigmf::kWrittenVersion);
    metadata.global.sampleRate = 61440000.0;
    metadata.global.recorder   = "gnuradio4";
    metadata.captures.reserve(captureCount);
    for (std::size_t i = 0UZ; i < captureCount; ++i) {
        gr::sigmf::Capture capture;
        capture.sampleStart = static_cast<std::uint64_t>(i) * 4096U;
        capture.frequency   = 433921337.0;
        capture.datetime    = "2026-08-26T12:00:00.000000Z";
        metadata.captures.push_back(std::move(capture));
    }
    return metadata;
}

void reportMetadata(std::size_t captureCount) {
    const gr::sigmf::Metadata metadata = metadataWith(captureCount);
    const auto                text     = gr::sigmf::write(metadata);
    if (!text) {
        return;
    }

    constexpr std::size_t kRepeats = 200UZ;
    const auto            writeNs  = nanosecondsPerComponent(kRepeats, [&] {
        for (std::size_t i = 0UZ; i < kRepeats; ++i) {
            std::ignore = gr::sigmf::write(metadata);
        }
    });
    const auto            parseNs  = nanosecondsPerComponent(kRepeats, [&] {
        for (std::size_t i = 0UZ; i < kRepeats; ++i) {
            gr::sigmf::ParseCounters counters;
            std::ignore = gr::sigmf::parse(*text, gr::sigmf::Limits{}, counters);
        }
    });
    std::println("  {:>7} captures  {:>9} B   write {:10.0f} ns   parse {:10.0f} ns", captureCount, text->size(), writeNs, parseNs);
}

} // namespace

int main() {
    std::println("SigMF sample codec — read path (dataset into a float stream)");
    for (const std::size_t bufferBytes : kBufferSizes) {
        reportDecode("cf32_le", gr::sigmf::Scaling::Unit, bufferBytes);
        reportDecode("ci16_le", gr::sigmf::Scaling::Unit, bufferBytes);
        reportDecode("ci16_be", gr::sigmf::Scaling::Unit, bufferBytes);
        reportDecode("ci16_le", gr::sigmf::Scaling::Raw, bufferBytes);
        reportDecode("cu8", gr::sigmf::Scaling::Unit, bufferBytes);
        reportDecode("cf64_le", gr::sigmf::Scaling::Unit, bufferBytes);
    }

    std::println("");
    std::println("SigMF sample codec — write path (a float stream into a dataset)");
    for (const std::size_t bufferBytes : kBufferSizes) {
        reportEncode("cf32_le", gr::sigmf::Scaling::Unit, bufferBytes);
        reportEncode("ci16_le", gr::sigmf::Scaling::Unit, bufferBytes);
        reportEncode("cu8", gr::sigmf::Scaling::Unit, bufferBytes);
    }

    std::println("");
    std::println("SigMF metadata — the parser and the writer in isolation");
    for (const std::size_t captureCount : {1UZ, 100UZ, 10000UZ}) {
        reportMetadata(captureCount);
    }
    return 0;
}
