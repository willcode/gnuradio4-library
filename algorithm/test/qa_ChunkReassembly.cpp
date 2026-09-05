#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/packet/ChunkReassembly.hpp>

/*
 * The headline scene is one known file and twelve arrival orders. 8192 bytes at a chunk size of 224
 * is 36 full chunks and one final chunk of 128, thirty-seven in all -- 8192 = 36 * 224 + 128 -- and
 * those three numbers are asserted, because a scene whose last chunk happens to be full never
 * exercises the short-last-chunk arithmetic at all. Pushed in order, in reverse, and in ten seeded
 * permutations, the twelve completed files are byte-identical.
 *
 * The criterion that separates this engine from the one the survey found is criterion 9: a file whose
 * declared size is reached by its last chunk while an earlier chunk is missing publishes nothing.
 * Completion is coverage, and a write pointer is not coverage.
 *
 * Two numbers the engine exposes as arithmetic rather than as documentation: the coverage bitmap for
 * a 1 MiB file at 224-byte chunks is 586 bytes, and the peak for eight files of 1 MiB at 1024 gaps is
 * 8 520 832 bytes, exactly 8.125 MiB. The per-chunk cost of both coverage paths is bm_ChunkReassembly's.
 */
namespace {

using namespace gr::packet;

std::uint64_t rng = 0xD1B54A32D192ED03ULL;

std::uint64_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng >> 11U;
}

constexpr std::size_t kChunkSize  = 224UZ;
constexpr std::size_t kFileBytes  = 8192UZ;
constexpr std::size_t kHeaderSize = 9UZ;

/// identifier (2) | index (2) | total size (4) | flags (1) | payload
IndexedChunkFormat indexedFormat() {
    IndexedChunkFormat format{};
    format.identifier     = FieldSpec{0UZ, 2UZ};
    format.index          = FieldSpec{2UZ, 2UZ};
    format.total_size     = FieldSpec{4UZ, 4UZ};
    format.last_flag      = FlagSpec{8UZ, 0U};
    format.payload_offset = kHeaderSize;
    format.chunk_size     = kChunkSize;
    return format;
}

/// identifier (2) | offset (4) | total size (4) | flags (1) | payload
OffsetChunkFormat offsetFormat() {
    OffsetChunkFormat format{};
    format.identifier     = FieldSpec{0UZ, 2UZ};
    format.offset         = FieldSpec{2UZ, 4UZ};
    format.total_size     = FieldSpec{6UZ, 4UZ};
    format.last_flag      = FlagSpec{10UZ, 0U};
    format.payload_offset = 11UZ;
    return format;
}

void putBigEndian(std::vector<std::uint8_t>& record, std::size_t at, std::size_t width, std::uint64_t value) {
    for (std::size_t i = 0UZ; i < width; ++i) {
        record[at + width - 1UZ - i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFULL);
    }
}

std::vector<std::uint8_t> indexedChunk(std::uint64_t id, std::uint64_t index, std::uint64_t totalSize, bool last, std::span<const std::uint8_t> payload) {
    std::vector<std::uint8_t> record(kHeaderSize + payload.size(), 0U);
    putBigEndian(record, 0UZ, 2UZ, id);
    putBigEndian(record, 2UZ, 2UZ, index);
    putBigEndian(record, 4UZ, 4UZ, totalSize);
    record[8] = last ? 1U : 0U;
    std::copy(payload.begin(), payload.end(), record.begin() + static_cast<std::ptrdiff_t>(kHeaderSize));
    return record;
}

std::vector<std::uint8_t> offsetChunk(std::uint64_t id, std::uint64_t offset, std::uint64_t totalSize, bool last, std::span<const std::uint8_t> payload) {
    std::vector<std::uint8_t> record(11UZ + payload.size(), 0U);
    putBigEndian(record, 0UZ, 2UZ, id);
    putBigEndian(record, 2UZ, 4UZ, offset);
    putBigEndian(record, 6UZ, 4UZ, totalSize);
    record[10] = last ? 1U : 0U;
    std::copy(payload.begin(), payload.end(), record.begin() + 11);
    return record;
}

/// A known file: byte i is a cheap mixing of i, so a chunk written at the wrong offset is visible.
std::vector<std::uint8_t> knownFile(std::size_t bytes, std::uint8_t seed) {
    std::vector<std::uint8_t> file(bytes);
    for (std::size_t i = 0UZ; i < bytes; ++i) {
        file[i] = static_cast<std::uint8_t>((i * 131UZ + seed * 17UZ + (i >> 5U)) & 0xFFUZ);
    }
    return file;
}

ChunkReassembler::Config indexedConfig(std::size_t maxOpenFiles = 4UZ, std::uint64_t maxFileBytes = 65536ULL) {
    ChunkReassembler::Config config{};
    config.format         = indexedFormat();
    config.max_open_files = maxOpenFiles;
    config.max_file_bytes = maxFileBytes;
    return config;
}

/// The records of one file, one per chunk, in index order.
std::vector<std::vector<std::uint8_t>> indexedChunksOf(std::uint64_t id, const std::vector<std::uint8_t>& file) {
    std::vector<std::vector<std::uint8_t>> records;
    for (std::size_t base = 0UZ; base < file.size(); base += kChunkSize) {
        const std::size_t length = std::min(kChunkSize, file.size() - base);
        records.push_back(indexedChunk(id, base / kChunkSize, file.size(), base + length == file.size(), std::span{file}.subspan(base, length)));
    }
    return records;
}

bool sameBytes(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }

} // namespace

const boost::ut::suite<"chunk reassembly"> chunkReassemblyTests = [] {
    using namespace boost::ut;

    "the file's own arithmetic, asserted before anything is reassembled"_test = [] {
        expect(eq(kFileBytes, 36UZ * kChunkSize + 128UZ)) << "8192 = 36 * 224 + 128";
        expect(eq((kFileBytes + kChunkSize - 1UZ) / kChunkSize, 37UZ)) << "thirty-seven chunks, the last one short";
        expect(eq(kFileBytes % kChunkSize, 128UZ));

        // The two figures the memory model states, as the expressions that produce them.
        expect(eq(bitmapBytes(1048576ULL, 224ULL), 586ULL)) << "ceil(ceil(1 048 576 / 224) / 8)";
        expect(eq((1048576ULL + 223ULL) / 224ULL, 4682ULL));
        expect(eq(peakBytes(8ULL, 1048576ULL, 1024ULL), 8520832ULL)) << "8 * (1 048 576 + 16 400 + 128), which is 8.125 MiB";
        expect(eq(8520832ULL, 8ULL * 1065104ULL));
        expect(eq(kMaxFileBytes, 2147483647ULL)) << "so every payload fits a record extent by construction";
    };

    "1. out of order reassembles identically to in order"_test = [] {
        const std::vector<std::uint8_t>              file    = knownFile(kFileBytes, 3U);
        const std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(7ULL, file);
        expect(eq(records.size(), 37UZ));
        expect(eq(records.back().size(), kHeaderSize + 128UZ)) << "the final chunk is 128 bytes";
        expect(eq(records.front().size(), kHeaderSize + kChunkSize));

        std::vector<std::vector<std::size_t>> orders;
        std::vector<std::size_t>              identity(records.size());
        std::iota(identity.begin(), identity.end(), 0UZ);
        orders.push_back(identity);
        std::vector<std::size_t> reversed = identity;
        std::reverse(reversed.begin(), reversed.end());
        orders.push_back(reversed);
        for (std::size_t trial = 0UZ; trial < 10UZ; ++trial) {
            std::vector<std::size_t> shuffled = identity;
            for (std::size_t i = shuffled.size(); i > 1UZ; --i) {
                std::swap(shuffled[i - 1UZ], shuffled[next() % i]);
            }
            orders.push_back(shuffled);
        }
        expect(eq(orders.size(), 12UZ));

        for (const std::vector<std::size_t>& order : orders) {
            ChunkReassembler engine{indexedConfig()};
            expect(engine.validate() == FormatError::ok);
            std::size_t completed = 0UZ;
            for (const std::size_t at : order) {
                const ChunkReassembler::ChunkResult result = engine.push(records[at]);
                expect(result.status != ChunkReassembler::Status::refused);
                if (result.status == ChunkReassembler::Status::completed) {
                    ++completed;
                    expect(eq(*result.completed_id, 7ULL));
                    expect(sameBytes(engine.file(7ULL), file)) << "byte for byte, in every arrival order";
                    const FileFacts facts = engine.facts(7ULL);
                    expect(eq(facts.chunks, 37ULL));
                    expect(eq(facts.covered_bytes, kFileBytes));
                    engine.release(7ULL);
                }
            }
            expect(eq(completed, 1UZ));
            expect(eq(engine.counters().duplicate_chunks, 0ULL));
            expect(eq(engine.counters().conflicting_chunks, 0ULL));
        }
    };

    "2. duplicates with equal content are counted and change nothing"_test = [] {
        const std::vector<std::uint8_t>              file    = knownFile(kFileBytes, 5U);
        const std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(1ULL, file);

        ChunkReassembler engine{indexedConfig()};
        for (std::size_t i = 0UZ; i + 1UZ < records.size(); ++i) {
            expect(engine.push(records[i]).status == ChunkReassembler::Status::accepted);
            expect(engine.push(records[i]).status == ChunkReassembler::Status::accepted) << "a retransmitting downlink is not an error";
        }
        const ChunkReassembler::ChunkResult completed = engine.push(records.back());
        expect(completed.status == ChunkReassembler::Status::completed);
        expect(sameBytes(engine.file(1ULL), file)) << "identical to the file criterion 1 assembles";
        expect(eq(engine.counters().duplicate_chunks, 36ULL));
        expect(eq(engine.counters().conflicting_chunks, 0ULL));

        // A retransmission that arrives after the file has been published has nothing to attach to,
        // because the file left the engine when it completed. It opens a new one, and says so.
        engine.release(1ULL);
        expect(engine.push(records.back()).status == ChunkReassembler::Status::accepted);
        expect(eq(engine.counters().files_opened, 2ULL));
    };

    "3. a conflicting duplicate does not overwrite, and the next chunk is processed"_test = [] {
        const std::vector<std::uint8_t>        file    = knownFile(kFileBytes, 9U);
        std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(2ULL, file);

        ChunkReassembler engine{indexedConfig()};
        engine.push(records[0]);

        std::vector<std::uint8_t> altered            = records[0];
        altered[kHeaderSize + 100UZ]                 = static_cast<std::uint8_t>(altered[kHeaderSize + 100UZ] ^ 0xFFU);
        const ChunkReassembler::ChunkResult conflict = engine.push(altered);
        expect(conflict.status == ChunkReassembler::Status::refused);
        expect(conflict.reason == ChunkReassembler::RefusalReason::content_conflict);
        expect(eq(engine.counters().conflicting_chunks, 1ULL));

        for (std::size_t i = 1UZ; i < records.size(); ++i) {
            const ChunkReassembler::ChunkResult result = engine.push(records[i]);
            expect(result.status != ChunkReassembler::Status::refused) << "the following chunk is processed";
            if (result.status == ChunkReassembler::Status::completed) {
                expect(sameBytes(engine.file(2ULL), file)) << "the first write stands";
                expect(eq(engine.facts(2ULL).conflicts, 1ULL));
            }
        }
    };

    "4. a missing chunk never completes and is evicted after N records, with a count"_test = [] {
        const std::vector<std::uint8_t>              file         = knownFile(kFileBytes, 11U);
        const std::vector<std::vector<std::uint8_t>> records      = indexedChunksOf(3ULL, file);
        const std::vector<std::uint8_t>              other        = knownFile(kFileBytes, 12U);
        const std::vector<std::vector<std::uint8_t>> otherRecords = indexedChunksOf(4ULL, other);

        ChunkReassembler::Config config = indexedConfig();
        config.evict_after_records      = 50ULL;
        ChunkReassembler engine{config};

        for (std::size_t i = 0UZ; i < records.size(); ++i) {
            if (i == 17UZ) {
                continue; // chunk 17 withheld
            }
            expect(engine.push(records[i]).status == ChunkReassembler::Status::accepted);
        }
        expect(eq(engine.counters().files_completed, 0ULL)) << "coverage is not complete, so nothing is published";

        bool evicted = false;
        for (std::size_t i = 0UZ; i < 51UZ; ++i) {
            engine.push(otherRecords[i % otherRecords.size()]);
            for (const std::uint64_t id : engine.departed()) {
                if (id == 3ULL) {
                    evicted               = true;
                    const FileFacts facts = engine.facts(3ULL);
                    expect(facts.reason == EvictReason::stale);
                    expect(eq(facts.covered_bytes, kFileBytes - kChunkSize));
                    expect(eq(facts.covered_bytes, 7968ULL)) << "8192 - 224, asserted as a number";
                    engine.release(3ULL);
                }
            }
        }
        expect(evicted);
        expect(eq(engine.counters().evicted_stale, 1ULL));
    };

    "5. two interleaved files complete independently"_test = [] {
        const std::vector<std::uint8_t>              first  = knownFile(2240UZ, 21U);
        const std::vector<std::uint8_t>              second = knownFile(2240UZ, 22U);
        const std::vector<std::vector<std::uint8_t>> a      = indexedChunksOf(10ULL, first);
        const std::vector<std::vector<std::uint8_t>> b      = indexedChunksOf(11ULL, second);

        ChunkReassembler           engine{indexedConfig()};
        std::vector<std::uint64_t> order;
        for (std::size_t i = 0UZ; i < a.size(); ++i) {
            for (const ChunkReassembler::ChunkResult result : {engine.push(a[i]), engine.push(b[i])}) {
                if (result.status == ChunkReassembler::Status::completed) {
                    order.push_back(*result.completed_id);
                }
            }
        }
        expect(eq(order.size(), 2UZ));
        expect(eq(order[0], 10ULL)) << "in completion order";
        expect(eq(order[1], 11ULL));
        expect(sameBytes(engine.file(10ULL), first));
        expect(sameBytes(engine.file(11ULL), second));
        expect(eq(engine.counters().files_opened, 2ULL));
        expect(eq(engine.counters().evicted_for_cap, 0ULL));
    };

    "6. max_open_files evicts the least recently touched and admits the new file"_test = [] {
        const std::vector<std::uint8_t>              third   = knownFile(2240UZ, 33U);
        const std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(30ULL, third);

        ChunkReassembler engine{indexedConfig(2UZ)};
        engine.push(indexedChunksOf(20ULL, knownFile(2240UZ, 31U))[0]);
        engine.push(indexedChunksOf(21ULL, knownFile(2240UZ, 32U))[0]);
        engine.push(indexedChunksOf(21ULL, knownFile(2240UZ, 32U))[1]); // 21 is touched more recently

        bool evictedFirst = false;
        for (const std::vector<std::uint8_t>& record : records) {
            engine.push(record);
            for (const std::uint64_t id : engine.departed()) {
                if (id == 20ULL) {
                    evictedFirst = true;
                    expect(engine.facts(20ULL).reason == EvictReason::cap);
                    engine.release(20ULL);
                }
            }
        }
        expect(evictedFirst) << "the least recently touched file leaves, not the arriving one";
        expect(eq(engine.counters().evicted_for_cap, 1ULL));
        expect(sameBytes(engine.file(30ULL), third)) << "and the third file completes";
    };

    "7. max_file_bytes refuses by range, before any multiplication"_test = [] {
        // An eight-byte index field, so an index near 2^64 / chunk_size can be expressed at all.
        IndexedChunkFormat format{};
        format.identifier     = FieldSpec{0UZ, 2UZ};
        format.index          = FieldSpec{2UZ, 8UZ};
        format.total_size     = FieldSpec{10UZ, 4UZ};
        format.payload_offset = 14UZ;
        format.chunk_size     = kChunkSize;

        ChunkReassembler::Config config{};
        config.format         = format;
        config.max_open_files = 4UZ;
        config.max_file_bytes = 65536ULL;
        ChunkReassembler engine{config};
        expect(engine.validate() == FormatError::ok);

        const std::vector<std::uint8_t> payload = knownFile(kChunkSize, 44U);
        const auto                      wide    = [&payload](std::uint64_t index, std::uint64_t totalSize) {
            std::vector<std::uint8_t> record(14UZ + payload.size(), 0U);
            putBigEndian(record, 0UZ, 2UZ, 1ULL);
            putBigEndian(record, 2UZ, 8UZ, index);
            putBigEndian(record, 10UZ, 4UZ, totalSize);
            std::copy(payload.begin(), payload.end(), record.begin() + 14);
            return record;
        };

        // An index whose product with the chunk size really wraps a 64-bit register: 224 * ((2^64 - 1)/224 + 1)
        // is 96 modulo 2^64, and 96 is inside both the declared size and the cap, so an engine that multiplied
        // first would place this chunk at byte 96 and accept it. The guard divides instead.
        constexpr std::uint64_t kWrappingIndex = (0xFFFFFFFFFFFFFFFFULL / kChunkSize) + 1ULL;
        static_assert(kWrappingIndex * kChunkSize == 96ULL, "the scene only tests the guard if the product wraps");
        static_assert(kWrappingIndex > 65536ULL / kChunkSize);

        const ChunkReassembler::ChunkResult huge = engine.push(wide(kWrappingIndex, 65536ULL));
        expect(huge.status == ChunkReassembler::Status::refused);
        expect(huge.reason == ChunkReassembler::RefusalReason::over_max_bytes) << "the comparison is made before the multiply, so no product wraps";
        expect(eq(engine.counters().refused_over_max, 1ULL));
        expect(eq(engine.facts(1ULL).covered_bytes, 0ULL)) << "nothing landed at the wrapped product's byte 96";

        const ChunkReassembler::ChunkResult inRange = engine.push(wide(0ULL, 65536ULL));
        expect(inRange.status == ChunkReassembler::Status::accepted) << "the file is not marked broken by one bad header";
        expect(eq(engine.facts(1ULL).covered_bytes, kChunkSize));

        // And the configuration bounds themselves, each refused where the configuration is offered.
        const auto refused = [](const ChunkReassembler::Config& bad) {
            ChunkReassembler engine2{bad};
            std::ignore = engine2.openFiles();
        };
        ChunkReassembler::Config oversize = config;
        oversize.max_file_bytes           = kMaxFileBytes + 1ULL;
        expect(ChunkReassembler::validate(oversize) == FormatError::file_bytes_too_large);
        expect(throws([&refused, &oversize] { refused(oversize); }));
        ChunkReassembler::Config unset = config;
        unset.max_open_files           = 0UZ;
        expect(ChunkReassembler::validate(unset) == FormatError::open_files_required);
        expect(throws([&refused, &unset] { refused(unset); })) << "a cap of zero admits no file, so nothing could ever be evicted to make room";
        ChunkReassembler::Config noBytes = config;
        noBytes.max_file_bytes           = 0ULL;
        expect(ChunkReassembler::validate(noBytes) == FormatError::file_bytes_required);
        expect(throws([&refused, &noBytes] { refused(noBytes); }));
        ChunkReassembler::Config noGaps = config;
        noGaps.max_gaps                 = 0UZ;
        expect(ChunkReassembler::validate(noGaps) == FormatError::gaps_required);
        expect(throws([&refused, &noGaps] { refused(noGaps); }));
    };

    "8. max_gaps refuses only chunks that would add a gap"_test = [] {
        ChunkReassembler::Config config{};
        config.format         = offsetFormat();
        config.max_open_files = 2UZ;
        config.max_file_bytes = 65536ULL;
        config.max_gaps       = 4UZ;
        ChunkReassembler engine{config};
        expect(engine.validate() == FormatError::ok);

        const std::vector<std::uint8_t> file = knownFile(2000UZ, 55U);
        const auto                      at   = [&file](std::uint64_t offset, std::size_t length) { return offsetChunk(5ULL, offset, file.size(), false, std::span{file}.subspan(static_cast<std::size_t>(offset), length)); };

        // Alternating 100-byte chunks: each one opens a new interval until the cap.
        for (std::uint64_t i = 0ULL; i < 5ULL; ++i) {
            const ChunkReassembler::ChunkResult result = engine.push(at(i * 200ULL, 100UZ));
            expect(result.status == ChunkReassembler::Status::accepted) << "interval " << i;
        }
        const ChunkReassembler::ChunkResult tooMany = engine.push(at(1000ULL, 100UZ));
        expect(tooMany.status == ChunkReassembler::Status::refused);
        expect(tooMany.reason == ChunkReassembler::RefusalReason::too_fragmented);
        expect(eq(engine.counters().refused_too_fragmented, 1ULL));

        // A chunk that fills an existing hole is accepted in the same state, because merging is
        // attempted before the cap is tested.
        const ChunkReassembler::ChunkResult fills = engine.push(at(100ULL, 100UZ));
        expect(fills.status == ChunkReassembler::Status::accepted) << "a chunk that fills a hole is never refused for making one";
    };

    "9. completion is coverage, not a write pointer"_test = [] {
        const std::vector<std::uint8_t>              file    = knownFile(kFileBytes, 66U);
        const std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(6ULL, file);

        ChunkReassembler engine{indexedConfig()};
        for (std::size_t i = 0UZ; i < records.size(); ++i) {
            if (i == 3UZ) {
                continue;
            }
            const ChunkReassembler::ChunkResult result = engine.push(records[i]);
            expect(result.status != ChunkReassembler::Status::completed) << "chunk " << i << " must not complete a file with a hole in it";
        }
        expect(eq(engine.counters().files_completed, 0ULL)) << "the last chunk reached the declared size and the file is still not complete";
        expect(eq(engine.facts(6ULL).covered_bytes, kFileBytes - kChunkSize));

        // And it completes the moment the hole is filled, in the same engine.
        const ChunkReassembler::ChunkResult filled = engine.push(records[3]);
        expect(filled.status == ChunkReassembler::Status::completed);
        expect(sameBytes(engine.file(6ULL), file));
    };

    "10. the two declarations, the last-chunk flag, and the first-wins conflict rule"_test = [] {
        const std::vector<std::uint8_t> file = knownFile(672UZ, 77U); // exactly three chunks

        // (a) a declared total size alone completes.
        {
            ChunkReassembler              engine{indexedConfig()};
            ChunkReassembler::ChunkResult last{};
            for (std::size_t base = 0UZ; base < file.size(); base += kChunkSize) {
                last = engine.push(indexedChunk(1ULL, base / kChunkSize, file.size(), false, std::span{file}.subspan(base, kChunkSize)));
            }
            expect(last.status == ChunkReassembler::Status::completed);
            expect(sameBytes(engine.file(1ULL), file));
        }

        // (b) a declared chunk count alone completes.
        {
            IndexedChunkFormat format       = indexedFormat();
            format.total_size               = FieldSpec{};
            format.chunk_count              = FieldSpec{4UZ, 4UZ};
            ChunkReassembler::Config config = indexedConfig();
            config.format                   = format;
            ChunkReassembler              engine{config};
            ChunkReassembler::ChunkResult last{};
            for (std::size_t base = 0UZ; base < file.size(); base += kChunkSize) {
                last = engine.push(indexedChunk(1ULL, base / kChunkSize, 3ULL, false, std::span{file}.subspan(base, kChunkSize)));
            }
            expect(last.status == ChunkReassembler::Status::completed);
            expect(sameBytes(engine.file(1ULL), file));
        }

        // (c) a last-chunk flag alone completes, and is exactly a declaration of total_chunks = index + 1.
        {
            IndexedChunkFormat format       = indexedFormat();
            format.total_size               = FieldSpec{};
            ChunkReassembler::Config config = indexedConfig();
            config.format                   = format;
            ChunkReassembler              engine{config};
            ChunkReassembler::ChunkResult last{};
            for (std::size_t base = 0UZ; base < file.size(); base += kChunkSize) {
                const bool isLast = base + kChunkSize == file.size();
                last              = engine.push(indexedChunk(1ULL, base / kChunkSize, 0ULL, isLast, std::span{file}.subspan(base, kChunkSize)));
            }
            expect(last.status == ChunkReassembler::Status::completed);
            expect(eq(*engine.facts(1ULL).declared_chunks, 3ULL)) << "the flag on chunk 2 declared three chunks";
        }

        // (d) a second, differing declaration is counted and the first stands -- in both orders.
        for (const bool bigFirst : {true, false}) {
            ChunkReassembler    engine{indexedConfig()};
            const std::uint64_t honest = file.size();
            const std::uint64_t wrong  = file.size() + kChunkSize;
            engine.push(indexedChunk(1ULL, 0ULL, bigFirst ? wrong : honest, false, std::span{file}.subspan(0UZ, kChunkSize)));
            engine.push(indexedChunk(1ULL, 1ULL, bigFirst ? honest : wrong, false, std::span{file}.subspan(kChunkSize, kChunkSize)));
            expect(eq(engine.counters().declaration_conflicts, 1ULL)) << "counted once, whichever arrived first";
            expect(eq(*engine.facts(1ULL).declared_size, bigFirst ? wrong : honest)) << "the first declaration stands";
        }
    };

    "11. a new file on index regression, both ways"_test = [] {
        // No identifier: the format carries none, so the engine keeps one current file. The declared
        // size names three chunks and only two are sent, so the file never completes and the current
        // file is still current when the index regresses.
        IndexedChunkFormat format       = indexedFormat();
        format.identifier               = FieldSpec{};
        ChunkReassembler::Config config = indexedConfig();
        config.format                   = format;

        const std::vector<std::uint8_t> file  = knownFile(448UZ, 88U);
        const auto                      chunk = [&file](std::uint64_t index) { return indexedChunk(0ULL, index, 3ULL * kChunkSize, false, std::span{file}.subspan(static_cast<std::size_t>(index) * kChunkSize, kChunkSize)); };

        {
            ChunkReassembler engine{config};
            engine.push(chunk(0ULL));
            engine.push(chunk(1ULL));
            engine.push(chunk(0ULL)); // a descending index
            expect(eq(engine.counters().files_opened, 2ULL)) << "a transmitter that restarts its numbering has said so";
        }
        {
            ChunkReassembler::Config held     = config;
            held.new_file_on_index_regression = false;
            ChunkReassembler engine{held};
            engine.push(chunk(0ULL));
            engine.push(chunk(1ULL));
            engine.push(chunk(0ULL));
            expect(eq(engine.counters().files_opened, 1ULL));
            expect(eq(engine.counters().duplicate_chunks, 1ULL)) << "placed by index in the current file, where it is a duplicate";
        }
    };

    "12. every completed payload is inside the record carriers' own extent bound"_test = [] {
        const std::vector<std::uint8_t>              file    = knownFile(kFileBytes, 99U);
        const std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(8ULL, file);

        ChunkReassembler engine{indexedConfig()};
        for (const std::vector<std::uint8_t>& record : records) {
            engine.push(record);
        }
        const std::span<const std::uint8_t> payload = engine.file(8ULL);
        expect(ge(payload.size(), 1UZ)) << "a zero-item payload is not admissible at either carrier boundary";
        expect(le(payload.size(), kMaxFileBytes));

        // The bound holds by construction from the setting, which is asserted as the setting's own
        // refusal rather than by building a two-gigabyte file.
        ChunkReassembler::Config oversize = indexedConfig();
        oversize.max_file_bytes           = kMaxFileBytes + 1ULL;
        expect(ChunkReassembler::validate(oversize) == FormatError::file_bytes_too_large);
        expect(throws([&oversize] {
            ChunkReassembler engine2{oversize};
            std::ignore = engine2.openFiles();
        }));

        // A declared size of zero is refused where it is read.
        ChunkReassembler                    fresh{indexedConfig()};
        const std::vector<std::uint8_t>     zero   = indexedChunk(9ULL, 0ULL, 0ULL, false, std::span{file}.first(16UZ));
        const ChunkReassembler::ChunkResult result = fresh.push(zero);
        expect(result.status == ChunkReassembler::Status::refused);
        expect(result.reason == ChunkReassembler::RefusalReason::zero_size);
    };

    "13. the degenerate descriptor's strict continuity, and what it costs"_test = [] {
        // Neither an index nor an offset: a running write pointer, which is meaningful only when
        // nothing has been lost. The failure is made visible rather than set as a broken flag.
        ChunkReassembler::Config config{};
        config.format         = offsetFormat();
        config.max_open_files = 2UZ;
        config.max_file_bytes = 65536ULL;
        ChunkReassembler engine{config};

        const std::vector<std::uint8_t> file = knownFile(600UZ, 111U);
        const auto                      feed = [&engine, &file](std::size_t begin, std::size_t length, bool declare) {
            ChunkDescriptor descriptor{};
            descriptor.file_id       = 1ULL;
            descriptor.payload_begin = 0UZ;
            descriptor.payload_end   = length;
            if (declare) {
                descriptor.total_size = file.size();
            }
            return engine.pushDescriptor(descriptor, std::span{file}.subspan(begin, length));
        };

        expect(feed(0UZ, 200UZ, true).status == ChunkReassembler::Status::accepted);
        expect(feed(200UZ, 200UZ, true).status == ChunkReassembler::Status::accepted);
        expect(eq(engine.facts(1ULL).covered_bytes, 400ULL)) << "contiguous chunks assemble exactly";

        // Now withhold one and offer the next: the pointer can no longer say where anything goes.
        ChunkReassembler::Config other = config;
        ChunkReassembler         second{other};
        ChunkDescriptor          descriptor{};
        descriptor.file_id       = 2ULL;
        descriptor.payload_begin = 0UZ;
        descriptor.payload_end   = 100UZ;
        descriptor.total_size    = file.size();
        expect(second.pushDescriptor(descriptor, std::span{file}.first(100UZ)).status == ChunkReassembler::Status::accepted);

        ChunkDescriptor placed{};
        placed.file_id       = 2ULL;
        placed.offset        = 300ULL;
        placed.payload_begin = 0UZ;
        placed.payload_end   = 100UZ;
        placed.total_size    = file.size();
        expect(second.pushDescriptor(placed, std::span{file}.subspan(300UZ, 100UZ)).status == ChunkReassembler::Status::accepted);

        const ChunkReassembler::ChunkResult lost = second.pushDescriptor(descriptor, std::span{file}.subspan(400UZ, 100UZ));
        expect(lost.status == ChunkReassembler::Status::refused);
        expect(lost.reason == ChunkReassembler::RefusalReason::no_position);
        expect(eq(second.counters().refused_no_position, 1ULL));
        expect(second.facts(2ULL).reason == EvictReason::no_position) << "and the file is evicted in that same call, with its coverage reported";
        expect(eq(second.facts(2ULL).covered_bytes, 200ULL));
    };

    "14. the integrity accounting, and the strict mode"_test = [] {
        const std::vector<std::uint8_t>              file    = knownFile(kFileBytes, 123U);
        const std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(12ULL, file);

        {
            ChunkReassembler engine{indexedConfig()};
            for (std::size_t i = 0UZ; i < records.size(); ++i) {
                const bool bad = i == 4UZ || i == 9UZ || i == 20UZ;
                engine.push(records[i], !bad, true, bad ? 3ULL : 1ULL, bad ? 1ULL : 0ULL);
            }
            const FileFacts facts = engine.facts(12ULL);
            expect(facts.crc_stated);
            expect(!facts.all_chunks_crc_ok) << "the AND over every contributing record that stated one";
            expect(eq(engine.counters().chunks_crc_failed, 3ULL));
            expect(eq(facts.corrected_errors, 3ULL * 3ULL + 34ULL * 1ULL)) << "summed, which is the key's own declared accumulation";
            expect(eq(facts.uncorrectable_errors, 3ULL));
            expect(sameBytes(engine.file(12ULL), file)) << "an imperfect chunk still contributes its bytes";
        }
        {
            ChunkReassembler::Config strict = indexedConfig();
            strict.require_crc_ok           = true;
            ChunkReassembler engine{strict};
            for (std::size_t i = 0UZ; i < records.size(); ++i) {
                const bool                          bad    = i == 4UZ || i == 9UZ || i == 20UZ;
                const ChunkReassembler::ChunkResult result = engine.push(records[i], !bad, true, 0ULL, 0ULL);
                expect((result.status == ChunkReassembler::Status::refused) == bad);
            }
            expect(eq(engine.counters().files_completed, 0ULL));
            expect(eq(engine.counters().refused_crc_failed, 3ULL));
        }
    };

    "a bit says a whole cell is covered, so a short chunk is only the last one"_test = [] {
        // identifier (2) | index (2) | chunk count (2) | payload, at a chunk size of 16.
        constexpr std::size_t kCell = 16UZ;
        IndexedChunkFormat    counted{};
        counted.identifier     = FieldSpec{0UZ, 2UZ};
        counted.index          = FieldSpec{2UZ, 2UZ};
        counted.chunk_count    = FieldSpec{4UZ, 2UZ};
        counted.payload_offset = 6UZ;
        counted.chunk_size     = kCell;

        const auto record = [](std::uint64_t index, std::uint64_t count, std::span<const std::uint8_t> payload) {
            std::vector<std::uint8_t> bytes(6UZ + payload.size(), 0U);
            putBigEndian(bytes, 0UZ, 2UZ, 1ULL);
            putBigEndian(bytes, 2UZ, 2UZ, index);
            putBigEndian(bytes, 4UZ, 2UZ, count);
            std::copy(payload.begin(), payload.end(), bytes.begin() + 6);
            return bytes;
        };

        ChunkReassembler::Config config{};
        config.format         = counted;
        config.max_open_files = 2UZ;
        config.max_file_bytes = 65536ULL;

        const std::vector<std::uint8_t> file = knownFile(3UZ * kCell, 131U);
        const auto                      cell = [&file](std::size_t which, std::size_t length) { return std::span{file}.subspan(which * kCell, length); };

        {
            ChunkReassembler engine{config};
            expect(engine.push(record(0ULL, 3ULL, cell(0UZ, kCell))).status == ChunkReassembler::Status::accepted);

            // Half a cell at index 1, which the count on this very record says is not the last index.
            const ChunkReassembler::ChunkResult shortMiddle = engine.push(record(1ULL, 3ULL, cell(1UZ, kCell / 2UZ)));
            expect(shortMiddle.status == ChunkReassembler::Status::refused);
            expect(shortMiddle.reason == ChunkReassembler::RefusalReason::bad_payload_span);
            expect(eq(engine.counters().refused_payload_span, 1ULL));

            expect(engine.push(record(2ULL, 3ULL, cell(2UZ, kCell))).status == ChunkReassembler::Status::accepted);
            expect(eq(engine.counters().files_completed, 0ULL)) << "two of three cells covered is not a complete file";

            const ChunkReassembler::ChunkResult filled = engine.push(record(1ULL, 3ULL, cell(1UZ, kCell)));
            expect(filled.status == ChunkReassembler::Status::completed);
            expect(sameBytes(engine.file(1ULL), file)) << "and the file is the one that was sent, with no zero-filled hole";
        }

        // The same short chunk where nothing has yet said where the file ends: it is admitted, and the
        // declaration that arrives later decides that this file can no longer complete.
        {
            IndexedChunkFormat flagged{};
            flagged.identifier     = FieldSpec{0UZ, 2UZ};
            flagged.index          = FieldSpec{2UZ, 2UZ};
            flagged.last_flag      = FlagSpec{4UZ, 0U};
            flagged.payload_offset = 5UZ;
            flagged.chunk_size     = kCell;

            const auto flaggedRecord = [](std::uint64_t index, bool last, std::span<const std::uint8_t> payload) {
                std::vector<std::uint8_t> bytes(5UZ + payload.size(), 0U);
                putBigEndian(bytes, 0UZ, 2UZ, 1ULL);
                putBigEndian(bytes, 2UZ, 2UZ, index);
                bytes[4] = last ? 1U : 0U;
                std::copy(payload.begin(), payload.end(), bytes.begin() + 5);
                return bytes;
            };

            ChunkReassembler::Config held = config;
            held.format                   = flagged;
            ChunkReassembler engine{held};

            expect(engine.push(flaggedRecord(1ULL, false, cell(1UZ, kCell / 2UZ))).status == ChunkReassembler::Status::accepted) << "nothing declares an extent yet";
            expect(engine.push(flaggedRecord(0ULL, false, cell(0UZ, kCell))).status == ChunkReassembler::Status::accepted);
            expect(engine.push(flaggedRecord(2ULL, true, cell(2UZ, kCell))).status == ChunkReassembler::Status::accepted);
            expect(eq(engine.counters().files_completed, 0ULL)) << "the flag names index 2 as the last, so the short index 1 is a hole";

            // The cell cannot be repaired either: its first write stands and the full chunk disagrees with it.
            const ChunkReassembler::ChunkResult repair = engine.push(flaggedRecord(1ULL, false, cell(1UZ, kCell)));
            expect(repair.status == ChunkReassembler::Status::refused);
            expect(repair.reason == ChunkReassembler::RefusalReason::content_conflict);
            expect(eq(engine.counters().files_completed, 0ULL)) << "a file that took a short chunk at a non-final index stays open until it is evicted";

            // A second short chunk at another index, while nothing has yet declared an extent, is refused:
            // only one index can turn out to be the last one.
            ChunkReassembler second{held};
            expect(second.push(flaggedRecord(1ULL, false, cell(1UZ, kCell / 2UZ))).status == ChunkReassembler::Status::accepted);
            const ChunkReassembler::ChunkResult twoShort = second.push(flaggedRecord(2ULL, false, cell(2UZ, kCell / 2UZ)));
            expect(twoShort.status == ChunkReassembler::Status::refused);
            expect(twoShort.reason == ChunkReassembler::RefusalReason::bad_payload_span);
        }

        // A chunk carrying no bytes covers nothing, on either coverage path.
        {
            ChunkReassembler                    engine{config};
            const ChunkReassembler::ChunkResult empty = engine.push(record(0ULL, 3ULL, std::span<const std::uint8_t>{}));
            expect(empty.status == ChunkReassembler::Status::refused);
            expect(empty.reason == ChunkReassembler::RefusalReason::bad_payload_span);

            ChunkReassembler::Config intervals{};
            intervals.format         = offsetFormat();
            intervals.max_open_files = 2UZ;
            intervals.max_file_bytes = 65536ULL;
            ChunkReassembler                    other{intervals};
            const ChunkReassembler::ChunkResult none = other.push(offsetChunk(1ULL, 0ULL, 64ULL, false, std::span<const std::uint8_t>{}));
            expect(none.status == ChunkReassembler::Status::refused);
            expect(none.reason == ChunkReassembler::RefusalReason::bad_payload_span);
            expect(eq(other.counters().duplicate_chunks, 0ULL)) << "and it is not a duplicate of the nothing that is there";
        }
    };

    "an offset addresses bytes, so a configured chunk size never puts two of its chunks in one cell"_test = [] {
        OffsetChunkFormat format = offsetFormat();
        format.chunk_size        = 8UZ; // beside an offset this only makes the two declarations checkable

        ChunkReassembler::Config config{};
        config.format         = format;
        config.max_open_files = 2UZ;
        config.max_file_bytes = 65536ULL;
        ChunkReassembler engine{config};

        const std::vector<std::uint8_t> file = knownFile(16UZ, 141U);
        expect(engine.push(offsetChunk(1ULL, 0ULL, 16ULL, false, std::span{file}.first(8UZ))).status == ChunkReassembler::Status::accepted);
        const ChunkReassembler::ChunkResult second = engine.push(offsetChunk(1ULL, 8ULL, 16ULL, false, std::span{file}.subspan(8UZ, 8UZ)));
        expect(second.status == ChunkReassembler::Status::completed) << "both halves land where their offsets say";
        expect(eq(engine.counters().conflicting_chunks, 0ULL));
        expect(sameBytes(engine.file(1ULL), file));
    };

    "a chunk that contradicts part of what is held is refused entire"_test = [] {
        ChunkReassembler::Config config{};
        config.format         = offsetFormat();
        config.max_open_files = 2UZ;
        config.max_file_bytes = 65536ULL;
        ChunkReassembler engine{config};

        const std::vector<std::uint8_t> file = knownFile(300UZ, 151U);
        expect(engine.push(offsetChunk(1ULL, 0ULL, 300ULL, false, std::span{file}.first(100UZ))).status == ChunkReassembler::Status::accepted);

        std::vector<std::uint8_t> overlapping(file.begin() + 50, file.begin() + 150);
        overlapping[10] = static_cast<std::uint8_t>(overlapping[10] ^ 0xFFU); // inside the covered half

        const ChunkReassembler::ChunkResult conflict = engine.push(offsetChunk(1ULL, 50ULL, 300ULL, false, overlapping));
        expect(conflict.status == ChunkReassembler::Status::refused);
        expect(conflict.reason == ChunkReassembler::RefusalReason::content_conflict) << "so there is a record to publish on reject and a reason to name";
        expect(eq(engine.counters().conflicting_chunks, 1ULL));
        expect(eq(engine.facts(1ULL).conflicts, 1ULL));
        expect(eq(engine.facts(1ULL).covered_bytes, 100ULL)) << "the uncovered half of a contradicting chunk is not written either";
        expect(sameBytes(engine.file(1ULL).first(100UZ), std::span{file}.first(100UZ))) << "the first write stands";

        // The same range with the content that is already held is a duplicate, not a conflict.
        const ChunkReassembler::ChunkResult agreeing = engine.push(offsetChunk(1ULL, 50ULL, 300ULL, false, std::span{file}.subspan(50UZ, 100UZ)));
        expect(agreeing.status == ChunkReassembler::Status::accepted);
        expect(eq(engine.facts(1ULL).covered_bytes, 150ULL)) << "and its uncovered half is written";
    };

    "the format's own refusals, each naming what is wrong"_test = [] {
        IndexedChunkFormat noIndex = indexedFormat();
        noIndex.index              = FieldSpec{};
        expect(validateChunkFormat(noIndex) == FormatError::index_required);

        IndexedChunkFormat noChunkSize = indexedFormat();
        noChunkSize.chunk_size         = 0UZ;
        expect(validateChunkFormat(noChunkSize) == FormatError::chunk_size_required);

        IndexedChunkFormat wide = indexedFormat();
        wide.identifier         = FieldSpec{0UZ, 9UZ};
        expect(validateChunkFormat(wide) == FormatError::field_too_wide);

        IndexedChunkFormat overlapping = indexedFormat();
        overlapping.total_size         = FieldSpec{1UZ, 4UZ};
        expect(validateChunkFormat(overlapping) == FormatError::fields_overlap);

        IndexedChunkFormat intoPayload = indexedFormat();
        intoPayload.payload_offset     = 4UZ;
        expect(validateChunkFormat(intoPayload) == FormatError::field_reaches_payload);

        IndexedChunkFormat cannotComplete = indexedFormat();
        cannotComplete.total_size         = FieldSpec{};
        cannotComplete.last_flag          = FlagSpec{};
        expect(validateChunkFormat(cannotComplete) == FormatError::cannot_complete) << "refusing it here beats discovering it as a stalled graph";

        OffsetChunkFormat noOffset = offsetFormat();
        noOffset.offset            = FieldSpec{};
        expect(validateChunkFormat(noOffset) == FormatError::offset_required);

        OffsetChunkFormat countOnly = offsetFormat();
        countOnly.total_size        = FieldSpec{};
        countOnly.last_flag         = FlagSpec{};
        countOnly.chunk_count       = FieldSpec{6UZ, 4UZ};
        countOnly.chunk_size        = kChunkSize;
        expect(validateChunkFormat(countOnly) == FormatError::cannot_complete) << "a count names cells an offset-addressed file does not have, so only a declared size completes one";

        // A record shorter than the header cannot be read, and says so rather than throwing.
        const std::vector<std::uint8_t>     tiny(4UZ, 0U);
        ChunkReassembler                    engine{indexedConfig()};
        const ChunkReassembler::ChunkResult result = engine.push(tiny);
        expect(result.status == ChunkReassembler::Status::refused);
        expect(result.reason == ChunkReassembler::RefusalReason::unparsable);
        expect(eq(engine.counters().refused_unparsable, 1ULL));

        // Byte order is read where it is configured and nowhere else.
        IndexedChunkFormat little                   = indexedFormat();
        little.byte_order                           = ByteOrder::little;
        const std::vector<std::uint8_t>      record = indexedChunk(1ULL, 0x0100ULL, 224ULL, false, std::vector<std::uint8_t>(224UZ, 7U));
        const std::optional<ChunkDescriptor> big    = parseChunk(indexedFormat(), record);
        const std::optional<ChunkDescriptor> small  = parseChunk(little, record);
        expect(big.has_value() && small.has_value());
        expect(eq(*big->index, 0x0100ULL));
        expect(eq(*small->index, 1ULL)) << "an index of 0x0100 read the other way is 1, and nothing else says so";
    };

    "end of stream retires what is still open, and reports it"_test = [] {
        const std::vector<std::uint8_t>              file    = knownFile(kFileBytes, 200U);
        const std::vector<std::vector<std::uint8_t>> records = indexedChunksOf(50ULL, file);

        ChunkReassembler engine{indexedConfig()};
        for (std::size_t i = 0UZ; i < 10UZ; ++i) {
            engine.push(records[i]);
        }
        const std::span<const std::uint64_t> open = engine.finish();
        expect(eq(open.size(), 1UZ));
        expect(eq(open[0], 50ULL));
        expect(eq(engine.counters().incomplete_at_stop, 1ULL));
        expect(engine.facts(50ULL).reason == EvictReason::at_stop);
        expect(eq(engine.facts(50ULL).covered_bytes, 10ULL * kChunkSize)) << "the operator learns what the pass lost";
    };
};

int main() { /* tests are automatically registered and run */ }
