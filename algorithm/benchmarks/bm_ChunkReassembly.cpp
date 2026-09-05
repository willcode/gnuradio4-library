#include <benchmark.hpp>

#include <gnuradio-4.0/algorithm/packet/ChunkReassembly.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kChunkSize = 224UZ;
constexpr std::size_t kChunks    = 512UZ;
constexpr std::size_t kFileBytes = kChunkSize * kChunks;
constexpr std::size_t kRepeats   = 50UZ;

[[nodiscard]] std::vector<std::uint8_t> seededBytes(std::size_t n) {
    gr::rng::Xoshiro256pp     rng(0xb0'0bULL);
    std::vector<std::uint8_t> data(n);
    for (std::uint8_t& value : data) {
        value = static_cast<std::uint8_t>(rng() & 0xffULL);
    }
    return data;
}

void putBigEndian(std::vector<std::uint8_t>& record, std::size_t at, std::size_t width, std::uint64_t value) {
    for (std::size_t i = 0UZ; i < width; ++i) {
        record[at + width - 1UZ - i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFULL);
    }
}

/// identifier (2) | index (2) | total size (4) | payload
[[nodiscard]] std::vector<std::vector<std::uint8_t>> indexedRecords(const std::vector<std::uint8_t>& file) {
    std::vector<std::vector<std::uint8_t>> records;
    for (std::size_t base = 0UZ; base < file.size(); base += kChunkSize) {
        std::vector<std::uint8_t> record(8UZ + kChunkSize, 0U);
        putBigEndian(record, 0UZ, 2UZ, 1ULL);
        putBigEndian(record, 2UZ, 2UZ, base / kChunkSize);
        putBigEndian(record, 4UZ, 4UZ, file.size());
        std::copy_n(file.begin() + static_cast<std::ptrdiff_t>(base), kChunkSize, record.begin() + 8);
        records.push_back(std::move(record));
    }
    return records;
}

/// identifier (2) | offset (4) | total size (4) | payload
[[nodiscard]] std::vector<std::vector<std::uint8_t>> offsetRecords(const std::vector<std::uint8_t>& file) {
    std::vector<std::vector<std::uint8_t>> records;
    for (std::size_t base = 0UZ; base < file.size(); base += kChunkSize) {
        std::vector<std::uint8_t> record(10UZ + kChunkSize, 0U);
        putBigEndian(record, 0UZ, 2UZ, 1ULL);
        putBigEndian(record, 2UZ, 4UZ, base);
        putBigEndian(record, 6UZ, 4UZ, file.size());
        std::copy_n(file.begin() + static_cast<std::ptrdiff_t>(base), kChunkSize, record.begin() + 10);
        records.push_back(std::move(record));
    }
    return records;
}

[[nodiscard]] gr::packet::ChunkReassembler::Config bitmapConfig() {
    gr::packet::IndexedChunkFormat format{};
    format.identifier     = gr::packet::FieldSpec{0UZ, 2UZ};
    format.index          = gr::packet::FieldSpec{2UZ, 2UZ};
    format.total_size     = gr::packet::FieldSpec{4UZ, 4UZ};
    format.payload_offset = 8UZ;
    format.chunk_size     = kChunkSize;

    gr::packet::ChunkReassembler::Config config{};
    config.format         = format;
    config.max_open_files = 2UZ;
    config.max_file_bytes = 1UZ << 20U;
    return config;
}

[[nodiscard]] gr::packet::ChunkReassembler::Config intervalConfig(std::size_t maxGaps) {
    gr::packet::OffsetChunkFormat format{};
    format.identifier     = gr::packet::FieldSpec{0UZ, 2UZ};
    format.offset         = gr::packet::FieldSpec{2UZ, 4UZ};
    format.total_size     = gr::packet::FieldSpec{6UZ, 4UZ};
    format.payload_offset = 10UZ;

    gr::packet::ChunkReassembler::Config config{};
    config.format         = format;
    config.max_open_files = 2UZ;
    config.max_file_bytes = 1UZ << 20U;
    config.max_gaps       = maxGaps;
    return config;
}

void runOnce(const gr::packet::ChunkReassembler::Config& config, const std::vector<std::vector<std::uint8_t>>& records, bool alternating) {
    gr::packet::ChunkReassembler engine{config};
    if (alternating) {
        // Every other chunk first, so the interval set grows to one interval per pair before the
        // second pass merges them away. This is the worst fragmentation a chunked delivery produces.
        for (std::size_t i = 0UZ; i < records.size(); i += 2UZ) {
            engine.push(records[i]);
        }
        for (std::size_t i = 1UZ; i < records.size(); i += 2UZ) {
            engine.push(records[i]);
        }
        return;
    }
    for (const std::vector<std::uint8_t>& record : records) {
        engine.push(record);
    }
}

} // namespace

/// Every row is one chunk's worth of work: one format parse of a handful of shift-and-mask reads, one hash
/// lookup, one coverage update, and one memcpy of the payload. The memcpy is the floor, and the rows differ
/// only in what the coverage update costs. The bitmap path sets one bit and compares one running count; the
/// interval path binary-searches a sorted vector and merges, which is free while the file arrives in order —
/// every chunk extends the single interval — and is paid for only when the file arrives fragmented, which is
/// what the third row measures. The file is copied exactly once end to end in every row, because the storage
/// is reserved from the declared size and never reallocated.
void benchChunkReassembly() {
    using namespace benchmark;
    using boost::ut::operator""_test;

    "chunk reassembly: one chunk's work, by coverage representation"_test = [] {
        const std::vector<std::uint8_t>              file    = seededBytes(kFileBytes);
        const std::vector<std::vector<std::uint8_t>> indexed = indexedRecords(file);
        const std::vector<std::vector<std::uint8_t>> offsets = offsetRecords(file);

        {
            const auto config                                                 = bitmapConfig();
            const auto name                                                   = std::format("bitmap    in order,    {} chunks of {} B", kChunks, kChunkSize);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kChunks) = [&] { runOnce(config, indexed, false); };
        }
        {
            const auto config                                                 = intervalConfig(1024UZ);
            const auto name                                                   = std::format("interval  in order,    {} chunks of {} B", kChunks, kChunkSize);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kChunks) = [&] { runOnce(config, offsets, false); };
        }
        {
            const auto config                                                 = intervalConfig(1024UZ);
            const auto name                                                   = std::format("interval  alternating, {} chunks of {} B", kChunks, kChunkSize);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kChunks) = [&] { runOnce(config, offsets, true); };
        }
        {
            const auto config                                                 = bitmapConfig();
            const auto name                                                   = std::format("bitmap    alternating, {} chunks of {} B", kChunks, kChunkSize);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kChunks) = [&] { runOnce(config, indexed, true); };
        }
        ::benchmark::results::add_separator();
    };
}

inline const boost::ut::suite<"chunk reassembly benchmarks"> _chunk_reassembly_bm = [] { benchChunkReassembly(); };

int main() { /* not needed by the UT framework */ }
