#ifndef GNURADIO_ALGORITHM_PACKET_CHUNK_REASSEMBLY_HPP
#define GNURADIO_ALGORITHM_PACKET_CHUNK_REASSEMBLY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

/**
 * @brief Reassembles a file delivered as chunks in ordinary frames, in bounded memory.
 *
 * Many mission protocols deliver an image, a spectrum sweep or a log as a sequence of chunks. The chunks
 * travel in the frames the link already has. One engine serves all such protocols, each with its own
 * header reader. This file holds that engine. It is general and independent of the protocol. The
 * per-protocol part is a pure function from a record to a `ChunkDescriptor`. Two general descriptor
 * formats are provided. One is a fixed header with a chunk index over a fixed chunk size. The other
 * carries a byte offset. A protocol with its own header format supplies a descriptor directly.
 *
 * **Completion is decided on coverage and not on a write pointer.** Suppose an engine completes a file
 * when its write pointer reaches the declared size. It then finishes when the *last* chunk arrives,
 * however many earlier chunks are missing. It publishes a file with holes as complete. Here a file is
 * complete when every byte of it has been written. The cost is one bitmap.
 *
 * **Coverage is a bit per cell for an index and a byte interval for an offset.** One bit marks a whole
 * cell as covered. An index-addressed chunk with fewer bytes than the chunk size is therefore admitted
 * only at the file's last index. A chunk with no bytes is refused on both paths. A cell marked covered
 * by a payload that did not fill it would make a file with a hole look complete. An offset-addressed
 * file is placed by intervals whatever its chunk size, because two of its chunks may share a cell.
 *
 * **Two declarations, not three.** A declared total size and a declared chunk count each complete a
 * file. A last-chunk flag is not a third criterion. It is a *declaration* of `total_chunks = index + 1`
 * and takes the same path with the same conflict rule. When two declarations disagree, or a second
 * declaration differs from the first, the **first stands**. A later header is no more trustworthy than
 * an earlier one. If the later one won, completion would depend on arrival order.
 *
 * **The first write wins on conflicting content.** A chunk whose range is already covered with
 * different bytes is refused and counted. The first write stands. Overwriting would discard a copy that
 * may have been the good one, and nothing would remain to compare. Keeping the first copy leaves the
 * conflict counted, and the file's own integrity figures decide. The other policy needs two engines and
 * a comparison. That works because this engine does not overwrite.
 *
 * **Two required settings and one expression bound the memory.** `max_open_files` and `max_file_bytes`
 * have no defaults, and zero is refused for both. The bounded quantity is the sum over all open files
 * and not one record. The values that size it are an offset, a declared size and a chunk count. All of
 * them are received data. `peakBytes` states the worst case from the settings alone. `max_file_bytes`
 * is refused above `2^31 - 1`. Every payload this engine produces then fits a record extent by
 * construction.
 *
 * **Eviction counts records and reads no clock.** With a wall-clock timeout the output would depend on
 * the speed of the machine. The same recording replayed on a slower host would evict different files.
 * Both rules here count records observed. One evicts the least recently touched file under cap
 * pressure. The other evicts a file untouched for more than `evict_after_records`.
 *
 * **The engine computes no check value.** Per-chunk integrity arrives with the record from an earlier
 * stage. The file's own value is the AND over every contributing record that stated one. It also holds
 * the sums of their corrected and uncorrectable error counts. A whole-file check is a CRC applied after
 * the engine, with the protocol's own parameters.
 */
namespace gr::packet {

/// @brief The byte order of a multi-byte header field. It is stated and not assumed. An index of
/// `0x0100` read in the other order is 1. A receiver that reads it wrongly reassembles a scrambled
/// file that decodes to nothing, and no counter shows the error.
enum class ByteOrder { big, little };

/// @brief One header field, its start and its width. A width of zero means the field is absent.
struct FieldSpec {
    std::size_t offset = 0UZ;
    std::size_t width  = 0UZ; //!< 1 to 8 bytes, or 0 when the format does not carry this field

    [[nodiscard]] constexpr bool        present() const noexcept { return width != 0UZ; }
    [[nodiscard]] constexpr std::size_t end() const noexcept { return present() ? offset + width : 0UZ; }
    [[nodiscard]] bool                  operator==(const FieldSpec&) const noexcept = default;
};

/// @brief The position of a last-chunk flag, a byte and a bit within it. Bit 8 disables the flag.
struct FlagSpec {
    std::size_t  offset = 0UZ;
    std::uint8_t bit    = 8U;

    [[nodiscard]] constexpr bool present() const noexcept { return bit < 8U; }
    [[nodiscard]] bool           operator==(const FlagSpec&) const noexcept = default;
};

/**
 * @brief The result of a protocol's format function, with no protocol knowledge in it.
 *
 * The position is an index, an offset or neither. With both present the **offset wins**. The index is
 * then kept only to detect a regression. An offset describes the file, and an index describes the
 * transmission.
 *
 * The payload is a span of the record and not a copy. `payload_begin` and `payload_end` are item
 * indices into the record given to the format. A format that strips a link header and a trailer copies
 * nothing. The engine checks both indices against the record's length before it reads. A format cannot
 * make the engine read out of bounds.
 *
 * `declared_check` is carried and not interpreted.
 */
struct ChunkDescriptor {
    std::optional<std::uint64_t> file_id{};
    std::optional<std::uint64_t> index{};
    std::optional<std::uint64_t> offset{};
    std::size_t                  payload_begin = 0UZ;
    std::size_t                  payload_end   = 0UZ;
    std::optional<std::uint64_t> total_size{};
    std::optional<std::uint64_t> total_chunks{};
    bool                         last_chunk = false;
    std::optional<std::uint64_t> declared_check{};
};

/// @brief A fixed header carrying an identifier and a 0-based chunk index over a fixed chunk size.
struct IndexedChunkFormat {
    FieldSpec   identifier{};
    FieldSpec   index{}; //!< required, the format's defining field
    FieldSpec   chunk_count{};
    FieldSpec   total_size{};
    FieldSpec   check_value{};
    FlagSpec    last_flag{};
    std::size_t payload_offset = 0UZ;
    std::size_t payload_trim   = 0UZ;
    std::size_t chunk_size     = 0UZ; //!< required, since an index means nothing without it
    ByteOrder   byte_order     = ByteOrder::big;
};

/// @brief A header stating where in the file this chunk's bytes go, so chunk sizes need not be uniform.
struct OffsetChunkFormat {
    FieldSpec   identifier{};
    FieldSpec   offset{}; //!< required
    FieldSpec   chunk_count{};
    FieldSpec   total_size{};
    FieldSpec   check_value{};
    FlagSpec    last_flag{};
    std::size_t payload_offset = 0UZ;
    std::size_t payload_trim   = 0UZ;
    std::size_t chunk_size     = 0UZ; //!< optional, and used only to check the two declarations against each other
    ByteOrder   byte_order     = ByteOrder::big;
};

using ChunkFormat = std::variant<IndexedChunkFormat, OffsetChunkFormat>;

/// @brief The reason a configuration was refused. Each value names the fault.
enum class FormatError { ok, field_too_wide, field_reaches_payload, fields_overlap, index_required, offset_required, chunk_size_required, cannot_complete, open_files_required, file_bytes_required, file_bytes_too_large, gaps_required };

[[nodiscard]] inline constexpr std::string_view formatErrorName(FormatError error) noexcept {
    switch (error) {
    case FormatError::ok: return "ok";
    case FormatError::field_too_wide: return "a field wider than eight bytes does not fit a 64-bit value";
    case FormatError::field_reaches_payload: return "a header field reaches past payload_offset into the payload";
    case FormatError::fields_overlap: return "two header fields overlap";
    case FormatError::index_required: return "the indexed format's index field is required";
    case FormatError::offset_required: return "the offset format's offset field is required";
    case FormatError::chunk_size_required: return "an index means nothing without a chunk size";
    case FormatError::open_files_required: return "max_open_files is required and zero is the unset state, not a spelling of infinite";
    case FormatError::file_bytes_required: return "max_file_bytes is required and zero is the unset state, not a spelling of infinite";
    case FormatError::file_bytes_too_large: return "max_file_bytes exceeds the extent a record carrier can express";
    case FormatError::gaps_required: return "the interval path needs room for at least one gap";
    default: return "no total size, no chunk count and no last-chunk flag: this format can never complete a file";
    }
}

namespace detail {

[[nodiscard]] inline constexpr std::uint64_t readField(std::span<const std::uint8_t> record, const FieldSpec& field, ByteOrder order) noexcept {
    std::uint64_t value = 0ULL;
    for (std::size_t i = 0UZ; i < field.width; ++i) {
        const std::uint8_t byte = record[field.offset + i];
        if (order == ByteOrder::big) {
            value = (value << 8U) | byte;
        } else {
            value |= static_cast<std::uint64_t>(byte) << (8U * i);
        }
    }
    return value;
}

/// @brief Every configured field of a format, so the overlap and payload checks read one list.
[[nodiscard]] inline std::vector<FieldSpec> fieldsOf(const ChunkFormat& format) {
    std::vector<FieldSpec> fields;
    const auto             collect = [&fields](std::initializer_list<FieldSpec> list) {
        for (const FieldSpec& field : list) {
            if (field.present()) {
                fields.push_back(field);
            }
        }
    };
    if (const IndexedChunkFormat* indexed = std::get_if<IndexedChunkFormat>(&format); indexed != nullptr) {
        collect({indexed->identifier, indexed->index, indexed->chunk_count, indexed->total_size, indexed->check_value});
    } else {
        const OffsetChunkFormat& offsetFormat = std::get<OffsetChunkFormat>(format);
        collect({offsetFormat.identifier, offsetFormat.offset, offsetFormat.chunk_count, offsetFormat.total_size, offsetFormat.check_value});
    }
    return fields;
}

} // namespace detail

/// @brief The header bytes and payload extent a format needs before a record can be read at all.
///
/// It is computed without heap allocation. It runs once per record on the accept path.
[[nodiscard]] inline constexpr std::size_t minimumRecordItems(const ChunkFormat& format) noexcept {
    const auto fold = [](std::initializer_list<FieldSpec> fields, const FlagSpec& flag, std::size_t payloadOffset, std::size_t payloadTrim) {
        std::size_t minimum = payloadOffset;
        for (const FieldSpec& field : fields) {
            minimum = std::max(minimum, field.end());
        }
        if (flag.present()) {
            minimum = std::max(minimum, flag.offset + 1UZ);
        }
        return minimum + payloadTrim;
    };
    if (const IndexedChunkFormat* indexed = std::get_if<IndexedChunkFormat>(&format); indexed != nullptr) {
        return fold({indexed->identifier, indexed->index, indexed->chunk_count, indexed->total_size, indexed->check_value}, indexed->last_flag, indexed->payload_offset, indexed->payload_trim);
    }
    const OffsetChunkFormat& offsetFormat = std::get<OffsetChunkFormat>(format);
    return fold({offsetFormat.identifier, offsetFormat.offset, offsetFormat.chunk_count, offsetFormat.total_size, offsetFormat.check_value}, offsetFormat.last_flag, offsetFormat.payload_offset, offsetFormat.payload_trim);
}

/**
 * @brief Validate a format's configuration and name the fault. It returns more than a bool.
 *
 * A header field that reaches into the payload is a configuration error and not a runtime error. It is
 * caught here. A format that cannot complete a file is caught here too. Refusing it at configuration is
 * better than a run that stalls with no file completed.
 */
[[nodiscard]] inline FormatError validateChunkFormat(const ChunkFormat& format) {
    const std::vector<FieldSpec> fields = detail::fieldsOf(format);
    for (const FieldSpec& field : fields) {
        if (field.width > 8UZ) {
            return FormatError::field_too_wide;
        }
    }
    for (std::size_t i = 0UZ; i < fields.size(); ++i) {
        for (std::size_t j = i + 1UZ; j < fields.size(); ++j) {
            if (fields[i].offset < fields[j].end() && fields[j].offset < fields[i].end()) {
                return FormatError::fields_overlap;
            }
        }
    }

    const auto common = [&fields](std::size_t payloadOffset, const FlagSpec& flag, bool canComplete) -> FormatError {
        for (const FieldSpec& field : fields) {
            if (field.end() > payloadOffset) {
                return FormatError::field_reaches_payload;
            }
        }
        if (flag.present() && flag.offset >= payloadOffset) {
            return FormatError::field_reaches_payload;
        }
        if (!canComplete) {
            return FormatError::cannot_complete;
        }
        return FormatError::ok;
    };

    if (const IndexedChunkFormat* indexed = std::get_if<IndexedChunkFormat>(&format); indexed != nullptr) {
        if (!indexed->index.present()) {
            return FormatError::index_required;
        }
        if (indexed->chunk_size == 0UZ) {
            return FormatError::chunk_size_required;
        }
        // A declared size, a declared count and a last-chunk flag each complete an index-addressed file. The
        // flag declares the count from the index it arrives with. Any one of the three suffices.
        return common(indexed->payload_offset, indexed->last_flag, indexed->chunk_count.present() || indexed->total_size.present() || indexed->last_flag.present());
    }
    const OffsetChunkFormat& offsetFormat = std::get<OffsetChunkFormat>(format);
    if (!offsetFormat.offset.present()) {
        return FormatError::offset_required;
    }
    // An offset-addressed file is placed by byte intervals and completes on a declared size alone. A chunk
    // count names cells this format does not address. A last-chunk flag declares a count in the same way.
    // Neither can decide completion here. A format with no size field cannot complete a file.
    return common(offsetFormat.payload_offset, offsetFormat.last_flag, offsetFormat.total_size.present());
}

/**
 * @brief Read one record under a format. `std::nullopt` is a refusal for the caller to count.
 *
 * A received record cannot stop the caller. A record the format cannot read yields no value and no
 * exception. Such a record is shorter than `minimumRecordItems` or lacks the stated payload extent.
 */
[[nodiscard]] inline std::optional<ChunkDescriptor> parseChunk(const ChunkFormat& format, std::span<const std::uint8_t> record) {
    if (record.size() < minimumRecordItems(format)) {
        return std::nullopt;
    }
    ChunkDescriptor descriptor{};
    const auto      readOptional = [&record](const FieldSpec& field, ByteOrder order) -> std::optional<std::uint64_t> {
        if (!field.present()) {
            return std::nullopt;
        }
        return detail::readField(record, field, order);
    };

    const auto fill = [&](const FieldSpec& identifier, const FieldSpec& count, const FieldSpec& size, const FieldSpec& check, const FlagSpec& flag, std::size_t payloadOffset, std::size_t payloadTrim, ByteOrder order) {
        descriptor.file_id        = readOptional(identifier, order);
        descriptor.total_chunks   = readOptional(count, order);
        descriptor.total_size     = readOptional(size, order);
        descriptor.declared_check = readOptional(check, order);
        descriptor.last_chunk     = flag.present() && ((record[flag.offset] >> flag.bit) & 1U) != 0U;
        descriptor.payload_begin  = payloadOffset;
        descriptor.payload_end    = record.size() - payloadTrim;
    };

    if (const IndexedChunkFormat* indexed = std::get_if<IndexedChunkFormat>(&format); indexed != nullptr) {
        fill(indexed->identifier, indexed->chunk_count, indexed->total_size, indexed->check_value, indexed->last_flag, indexed->payload_offset, indexed->payload_trim, indexed->byte_order);
        descriptor.index = detail::readField(record, indexed->index, indexed->byte_order);
    } else {
        const OffsetChunkFormat& offsetFormat = std::get<OffsetChunkFormat>(format);
        fill(offsetFormat.identifier, offsetFormat.chunk_count, offsetFormat.total_size, offsetFormat.check_value, offsetFormat.last_flag, offsetFormat.payload_offset, offsetFormat.payload_trim, offsetFormat.byte_order);
        descriptor.offset = detail::readField(record, offsetFormat.offset, offsetFormat.byte_order);
    }
    if (descriptor.payload_end < descriptor.payload_begin || descriptor.payload_end > record.size()) {
        return std::nullopt;
    }
    return descriptor;
}

/// @brief The chunk-index bitmap's size in bytes: `ceil(ceil(bytes / chunk) / 8)`.
///
/// For example, a 1 MiB file in 224-byte chunks is `ceil(1 048 576 / 224) = 4682` chunks. The bitmap is
/// then `ceil(4682 / 8) = 586` bytes, 0.056 % of the file's own storage. The bitmap path exists for that
/// reason. The common case avoids the insert cost of the interval path.
[[nodiscard]] inline constexpr std::uint64_t bitmapBytes(std::uint64_t fileBytes, std::uint64_t chunkBytes) noexcept {
    if (chunkBytes == 0ULL) {
        return 0ULL;
    }
    const std::uint64_t chunks = (fileBytes + chunkBytes - 1ULL) / chunkBytes;
    return (chunks + 7ULL) / 8ULL;
}

/// @brief The engine's worst case from its own settings, on the interval path (the bitmap is smaller).
///
/// `max_open_files * (max_file_bytes + 16 * (max_gaps + 1) + 128)`. The 128 bounds the bookkeeping and the
/// identifier. At 8 files, 1 MiB and 1024 gaps that is 8 520 832 bytes, exactly 8.125 MiB.
[[nodiscard]] inline constexpr std::uint64_t peakBytes(std::uint64_t maxOpenFiles, std::uint64_t maxFileBytes, std::uint64_t maxGaps) noexcept { return maxOpenFiles * (maxFileBytes + 16ULL * (maxGaps + 1ULL) + 128ULL); }

/// @brief The record extent bound, `2^31 - 1`. `max_file_bytes` may not exceed it.
inline constexpr std::uint64_t kMaxFileBytes = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());

/// @brief The reason a file left the engine before it was released.
enum class EvictReason { none, cap, stale, superseded, no_position, reconfigure, at_stop, completed };

/// @brief What a completed or evicted file reports. Every value is a count and not a verdict.
struct FileFacts {
    std::uint64_t                id                   = 0ULL;
    bool                         anonymous            = false; //!< the format carried no identifier
    std::uint64_t                size                 = 0ULL;  //!< bytes held
    std::uint64_t                covered_bytes        = 0ULL;
    std::uint64_t                chunks               = 0ULL;
    bool                         crc_stated           = false; //!< at least one record stated `crc_ok`
    bool                         all_chunks_crc_ok    = true;  //!< the AND over the records that stated one
    std::uint64_t                corrected_errors     = 0ULL;  //!< the sum over contributing records
    std::uint64_t                uncorrectable_errors = 0ULL;  //!< likewise
    std::uint64_t                conflicts            = 0ULL;
    std::optional<std::uint64_t> declared_size{};
    std::optional<std::uint64_t> declared_chunks{};
    std::optional<std::uint64_t> declared_check{};
    EvictReason                  reason = EvictReason::none;
};

class ChunkReassembler {
public:
    struct Config {
        ChunkFormat   format                       = IndexedChunkFormat{};
        std::size_t   max_open_files               = 0UZ;  //!< required, zero refused
        std::uint64_t max_file_bytes               = 0ULL; //!< required, zero and values above `kMaxFileBytes` refused
        std::size_t   max_gaps                     = 1024UZ;
        std::uint64_t evict_after_records          = 0ULL; //!< 0 leaves cap pressure as the sole rule
        bool          require_crc_ok               = false;
        bool          new_file_on_index_regression = true;
    };

    enum class Status { accepted, completed, refused };

    enum class RefusalReason { none, unparsable, bad_payload_span, zero_size, offset_beyond_size, over_max_bytes, too_fragmented, no_position, content_conflict, crc_failed };

    struct ChunkResult {
        Status                       status = Status::accepted;
        std::optional<std::uint64_t> completed_id{};
        RefusalReason                reason = RefusalReason::none;
    };

    struct Counters {
        std::uint64_t records                = 0ULL;
        std::uint64_t chunks_written         = 0ULL;
        std::uint64_t payload_bytes          = 0ULL;
        std::uint64_t files_opened           = 0ULL;
        std::uint64_t files_completed        = 0ULL;
        std::uint64_t duplicate_chunks       = 0ULL;
        std::uint64_t conflicting_chunks     = 0ULL;
        std::uint64_t declaration_conflicts  = 0ULL;
        std::uint64_t evicted_for_cap        = 0ULL;
        std::uint64_t evicted_stale          = 0ULL;
        std::uint64_t incomplete_at_stop     = 0ULL;
        std::uint64_t chunks_crc_failed      = 0ULL;
        std::uint64_t refused_unparsable     = 0ULL;
        std::uint64_t refused_payload_span   = 0ULL;
        std::uint64_t refused_zero_size      = 0ULL;
        std::uint64_t refused_beyond_size    = 0ULL;
        std::uint64_t refused_over_max       = 0ULL;
        std::uint64_t refused_too_fragmented = 0ULL;
        std::uint64_t refused_no_position    = 0ULL;
        std::uint64_t refused_crc_failed     = 0ULL;

        [[nodiscard]] bool operator==(const Counters&) const noexcept = default;
    };

    /// @brief Refuses an inadmissible configuration at construction. No later operation then works with a
    /// bound that does not exist. For example, a cap of zero open files admits no file. Cap-pressure eviction
    /// would then have nothing to evict and no way to make room.
    explicit ChunkReassembler(Config config) : config_{std::move(config)} {
        if (const FormatError error = validate(config_); error != FormatError::ok) {
            throw std::invalid_argument("gr::packet::ChunkReassembler: " + std::string(formatErrorName(error)));
        }
    }

    /// @brief Checks a configuration and returns `ok` or the error that names the fault.
    [[nodiscard]] static FormatError validate(const Config& config) {
        if (config.max_open_files == 0UZ) {
            return FormatError::open_files_required;
        }
        if (config.max_file_bytes == 0ULL) {
            return FormatError::file_bytes_required;
        }
        if (config.max_file_bytes > kMaxFileBytes) {
            return FormatError::file_bytes_too_large;
        }
        if (config.max_gaps == 0UZ) {
            return FormatError::gaps_required;
        }
        return validateChunkFormat(config.format);
    }

    [[nodiscard]] FormatError validate() const { return validate(config_); }

    /// @brief Read a record under the configured format and fold it in.
    ChunkResult push(std::span<const std::uint8_t> record, bool crcOk = true, bool crcStated = false, std::uint64_t correctedErrors = 0ULL, std::uint64_t uncorrectableErrors = 0ULL) {
        const std::optional<ChunkDescriptor> descriptor = parseChunk(config_.format, record);
        if (!descriptor.has_value()) {
            beginRecord();
            ++counters_.refused_unparsable;
            return ChunkResult{Status::refused, std::nullopt, RefusalReason::unparsable};
        }
        return pushDescriptor(*descriptor, record, crcOk, crcStated, correctedErrors, uncorrectableErrors);
    }

    /**
     * @brief Fold in a descriptor a protocol's own parser produced, with the record it came from.
     *
     * This is the hook for a protocol. The protocol supplies a pure function from a record to a descriptor
     * and calls this function. The engine does not know which mission it serves.
     */
    ChunkResult pushDescriptor(const ChunkDescriptor& descriptor, std::span<const std::uint8_t> record, bool crcOk = true, bool crcStated = false, std::uint64_t correctedErrors = 0ULL, std::uint64_t uncorrectableErrors = 0ULL) {
        beginRecord();

        if (descriptor.payload_end < descriptor.payload_begin || descriptor.payload_end > record.size()) {
            ++counters_.refused_payload_span;
            return ChunkResult{Status::refused, std::nullopt, RefusalReason::bad_payload_span};
        }
        if (crcStated && !crcOk) {
            ++counters_.chunks_crc_failed;
            if (config_.require_crc_ok) {
                ++counters_.refused_crc_failed;
                return ChunkResult{Status::refused, std::nullopt, RefusalReason::crc_failed};
            }
        }
        if (descriptor.total_size.has_value() && *descriptor.total_size == 0ULL) {
            ++counters_.refused_zero_size;
            return ChunkResult{Status::refused, std::nullopt, RefusalReason::zero_size};
        }

        const std::span<const std::uint8_t> payload = record.subspan(descriptor.payload_begin, descriptor.payload_end - descriptor.payload_begin);

        File& file = resolveFile(descriptor);
        foldDeclarations(file, descriptor);

        std::uint64_t       begin  = 0ULL;
        const RefusalReason placed = resolvePosition(file, descriptor, payload.size(), begin);
        if (placed != RefusalReason::none) {
            countRefusal(placed);
            if (placed == RefusalReason::no_position) {
                // No later chunk can be placed either. Holding the file open until the cap or the age rule
                // acts would only waste the bound. This refusal is the one that also evicts.
                retire(file.id, EvictReason::no_position);
            }
            return ChunkResult{Status::refused, std::nullopt, placed};
        }

        const RefusalReason written = writeChunk(file, begin, payload);
        if (written != RefusalReason::none) {
            countRefusal(written);
            return ChunkResult{Status::refused, std::nullopt, written};
        }

        file.last_touch = counters_.records;
        if (crcStated) {
            file.facts.crc_stated        = true;
            file.facts.all_chunks_crc_ok = file.facts.all_chunks_crc_ok && crcOk;
        }
        file.facts.corrected_errors += correctedErrors;
        file.facts.uncorrectable_errors += uncorrectableErrors;
        if (descriptor.index.has_value()) {
            file.next_index = *descriptor.index + 1ULL;
        }

        if (complete(file)) {
            const std::uint64_t id = file.id;
            ++counters_.files_completed;
            retire(id, EvictReason::completed);
            return ChunkResult{Status::completed, id, RefusalReason::none};
        }
        return ChunkResult{Status::accepted, std::nullopt, RefusalReason::none};
    }

    /// @brief The assembled bytes of a file that has been completed or evicted and not yet released.
    [[nodiscard]] std::span<const std::uint8_t> file(std::uint64_t id) const {
        const auto retired = retired_.find(id);
        if (retired != retired_.end()) {
            return retired->second.bytes;
        }
        const auto open = files_.find(id);
        return open != files_.end() ? std::span<const std::uint8_t>{open->second.bytes} : std::span<const std::uint8_t>{};
    }

    [[nodiscard]] FileFacts facts(std::uint64_t id) const {
        const auto retired = retired_.find(id);
        if (retired != retired_.end()) {
            return retired->second.facts;
        }
        const auto open = files_.find(id);
        return open != files_.end() ? open->second.facts : FileFacts{};
    }

    /// @brief The files that left the engine during the last `push`, in the order they left it.
    [[nodiscard]] std::span<const std::uint64_t> departed() const noexcept { return departed_; }

    /// @brief Drop a retired file's storage once the caller has published it.
    void release(std::uint64_t id) { retired_.erase(id); }

    /// @brief End of stream. Every file still open is incomplete by definition and is retired.
    std::span<const std::uint64_t> finish() {
        departed_.clear();
        std::vector<std::uint64_t> open;
        open.reserve(files_.size());
        for (const auto& [id, unused] : files_) {
            open.push_back(id);
        }
        std::sort(open.begin(), open.end());
        for (const std::uint64_t id : open) {
            ++counters_.incomplete_at_stop;
            retire(id, EvictReason::at_stop);
        }
        return departed_;
    }

    /// @brief Reset for a reconfiguration. Every open file is evicted and the state is cleared.
    ///
    /// A chunk format and a memory bound are properties of a link. A reconfigured engine must not complete
    /// a file from bytes received under the previous configuration.
    std::span<const std::uint64_t> reconfigure(Config config) {
        if (const FormatError error = validate(config); error != FormatError::ok) {
            // Refused before any file is retired. A rejected configuration leaves the engine unchanged.
            throw std::invalid_argument("gr::packet::ChunkReassembler: " + std::string(formatErrorName(error)));
        }
        departed_.clear();
        std::vector<std::uint64_t> open;
        open.reserve(files_.size());
        for (const auto& [id, unused] : files_) {
            open.push_back(id);
        }
        std::sort(open.begin(), open.end());
        for (const std::uint64_t id : open) {
            retire(id, EvictReason::reconfigure);
        }
        config_         = std::move(config);
        anonymous_next_ = 0ULL;
        current_        = std::nullopt;
        return departed_;
    }

    void reset() {
        files_.clear();
        retired_.clear();
        departed_.clear();
        counters_       = Counters{};
        anonymous_next_ = 0ULL;
        current_        = std::nullopt;
    }

    [[nodiscard]] const Counters& counters() const noexcept { return counters_; }
    [[nodiscard]] const Config&   config() const noexcept { return config_; }
    [[nodiscard]] std::size_t     openFiles() const noexcept { return files_.size(); }

private:
    struct Interval {
        std::uint64_t begin = 0ULL;
        std::uint64_t end   = 0ULL;
    };

    struct File {
        std::uint64_t             id = 0ULL;
        FileFacts                 facts{};
        std::vector<std::uint8_t> bytes{};
        std::vector<std::uint8_t> bitmap{};    //!< one bit per chunk index, where a chunk size is configured
        std::vector<Interval>     intervals{}; //!< sorted, disjoint and half-open, for the offset path
        std::uint64_t             highest_index = 0ULL;
        std::uint64_t             bits_set      = 0ULL;
        std::uint64_t             write_pointer = 0ULL;
        std::uint64_t             last_touch    = 0ULL;
        std::uint64_t             next_index    = 0ULL;
        /// the index whose payload was shorter than a chunk, taken while no declaration stated the file's end
        std::optional<std::uint64_t> short_index{};
    };

    struct RetiredFile {
        FileFacts                 facts{};
        std::vector<std::uint8_t> bytes{};
    };

    void beginRecord() {
        ++counters_.records;
        departed_.clear();
        evictStale();
    }

    void countRefusal(RefusalReason reason) {
        switch (reason) {
        case RefusalReason::zero_size: ++counters_.refused_zero_size; break;
        case RefusalReason::offset_beyond_size: ++counters_.refused_beyond_size; break;
        case RefusalReason::over_max_bytes: ++counters_.refused_over_max; break;
        case RefusalReason::too_fragmented: ++counters_.refused_too_fragmented; break;
        case RefusalReason::no_position: ++counters_.refused_no_position; break;
        case RefusalReason::content_conflict: ++counters_.conflicting_chunks; break;
        case RefusalReason::bad_payload_span: ++counters_.refused_payload_span; break;
        default: break;
        }
    }

    /// @brief True for the indexed format, which alone uses the bitmap. An index addresses a fixed grid, one
    /// cell per bit. An offset addresses bytes. Two chunks of an offset-addressed protocol may share a cell,
    /// whatever the chunk size. That format is placed by intervals throughout. Its chunk size serves only to
    /// check the two declarations against each other.
    [[nodiscard]] bool bitmapPath() const noexcept { return std::holds_alternative<IndexedChunkFormat>(config_.format); }

    [[nodiscard]] std::size_t chunkSize() const noexcept {
        if (const IndexedChunkFormat* indexed = std::get_if<IndexedChunkFormat>(&config_.format); indexed != nullptr) {
            return indexed->chunk_size;
        }
        return std::get<OffsetChunkFormat>(config_.format).chunk_size;
    }

    /// @brief Move a file out of the open set, keeping its bytes for the caller to publish.
    void retire(std::uint64_t id, EvictReason reason) {
        const auto found = files_.find(id);
        if (found == files_.end()) {
            return;
        }
        File& file        = found->second;
        file.facts.reason = reason;
        file.facts.size   = file.bytes.size();
        RetiredFile retired{file.facts, std::move(file.bytes)};
        files_.erase(found);
        retired_.insert_or_assign(id, std::move(retired));
        departed_.push_back(id);
        if (current_.has_value() && *current_ == id) {
            current_ = std::nullopt;
        }
    }

    void evictStale() {
        if (config_.evict_after_records == 0ULL) {
            return;
        }
        std::vector<std::uint64_t> stale;
        for (const auto& [id, file] : files_) {
            if (counters_.records > file.last_touch && counters_.records - file.last_touch > config_.evict_after_records) {
                stale.push_back(id);
            }
        }
        std::sort(stale.begin(), stale.end());
        for (const std::uint64_t id : stale) {
            ++counters_.evicted_stale;
            retire(id, EvictReason::stale);
        }
    }

    /// @brief Evicts the least recently touched file first. A transmitter that has moved on loses the old
    /// file and not the new one. Refusing the new file would turn a bounded loss into a permanent one.
    void evictForCap() {
        while (files_.size() >= config_.max_open_files) {
            std::uint64_t oldest      = 0ULL;
            std::uint64_t oldestTouch = std::numeric_limits<std::uint64_t>::max();
            for (const auto& [id, file] : files_) {
                if (file.last_touch < oldestTouch || (file.last_touch == oldestTouch && id < oldest)) {
                    oldest      = id;
                    oldestTouch = file.last_touch;
                }
            }
            ++counters_.evicted_for_cap;
            retire(oldest, EvictReason::cap);
        }
    }

    File& openFile(std::uint64_t id, bool anonymous, const ChunkDescriptor& descriptor) {
        evictForCap();
        File file{};
        file.id              = id;
        file.facts.id        = id;
        file.facts.anonymous = anonymous;
        file.last_touch      = counters_.records;
        if (descriptor.total_size.has_value() && *descriptor.total_size <= config_.max_file_bytes) {
            file.bytes.assign(static_cast<std::size_t>(*descriptor.total_size), 0U);
        }
        ++counters_.files_opened;
        return files_.insert_or_assign(id, std::move(file)).first->second;
    }

    File& resolveFile(const ChunkDescriptor& descriptor) {
        if (descriptor.file_id.has_value()) {
            const auto found = files_.find(*descriptor.file_id);
            if (found != files_.end()) {
                return found->second;
            }
            return openFile(*descriptor.file_id, false, descriptor);
        }

        // No identifier, so there is one current file. A regressing index is the general signal that a
        // transmitter has started a new file.
        if (current_.has_value()) {
            const auto found = files_.find(*current_);
            if (found != files_.end()) {
                const bool regressed = config_.new_file_on_index_regression && descriptor.index.has_value() && *descriptor.index < found->second.next_index;
                if (!regressed) {
                    return found->second;
                }
                // A transmitter that restarts its numbering has ended the previous file. That file is
                // retired and not counted as an eviction. No data was lost to a bound.
                retire(*current_, EvictReason::superseded);
            }
        }
        const std::uint64_t id = anonymousId();
        current_               = id;
        return openFile(id, true, descriptor);
    }

    [[nodiscard]] std::uint64_t anonymousId() noexcept { return kAnonymousBase + anonymous_next_++; }

    /// @brief Fold a chunk's declarations in. The first wins. A disagreement is counted and not applied.
    void foldDeclarations(File& file, const ChunkDescriptor& descriptor) {
        const auto declare = [this](std::optional<std::uint64_t>& held, std::optional<std::uint64_t> offered) {
            if (!offered.has_value()) {
                return;
            }
            if (!held.has_value()) {
                held = offered;
                return;
            }
            if (*held != *offered) {
                ++counters_.declaration_conflicts;
            }
        };
        declare(file.facts.declared_size, descriptor.total_size);

        // A last-chunk flag declares `total_chunks = index + 1`. It takes the same path under the same
        // conflict rule and is not a third criterion.
        std::optional<std::uint64_t> chunks = descriptor.total_chunks;
        if (descriptor.last_chunk && descriptor.index.has_value()) {
            const std::uint64_t implied = *descriptor.index + 1ULL;
            if (chunks.has_value() && *chunks != implied) {
                ++counters_.declaration_conflicts;
            } else {
                chunks = implied;
            }
        }
        declare(file.facts.declared_chunks, chunks);
        if (!file.facts.declared_check.has_value()) {
            file.facts.declared_check = descriptor.declared_check;
        }

        // With both declared and a chunk size configured, the two agree exactly when the last
        // chunk is non-empty and at most one chunk long.
        if (file.facts.declared_size.has_value() && file.facts.declared_chunks.has_value() && chunkSize() != 0UZ) {
            const std::uint64_t chunk = chunkSize();
            const std::uint64_t count = *file.facts.declared_chunks;
            const std::uint64_t size  = *file.facts.declared_size;
            if (count == 0ULL || size <= (count - 1ULL) * chunk || size > count * chunk) {
                ++counters_.declaration_conflicts;
            }
        }

        if (file.facts.declared_size.has_value() && file.bytes.size() < *file.facts.declared_size && *file.facts.declared_size <= config_.max_file_bytes) {
            file.bytes.resize(static_cast<std::size_t>(*file.facts.declared_size), 0U);
        }
    }

    /// @brief The position of this chunk's bytes. The multiplication is checked before it is performed.
    [[nodiscard]] RefusalReason resolvePosition(const File& file, const ChunkDescriptor& descriptor, std::size_t length, std::uint64_t& begin) const {
        const std::uint64_t chunk = chunkSize();
        if (descriptor.offset.has_value()) {
            begin = *descriptor.offset;
        } else if (descriptor.index.has_value()) {
            if (chunk == 0ULL) {
                return RefusalReason::no_position;
            }
            if (*descriptor.index > config_.max_file_bytes / chunk) {
                return RefusalReason::over_max_bytes; // tested before the multiply, and no product wraps
            }
            begin = *descriptor.index * chunk;
        } else {
            // The degenerate descriptor. It is placed at the write pointer, and only when the file's
            // coverage is exactly [0, write_pointer). A running pointer is meaningful only when no data
            // has been lost. This check tests that condition.
            if (!contiguous(file)) {
                return RefusalReason::no_position;
            }
            begin = file.write_pointer;
        }

        const std::uint64_t end = begin + length;
        if (end > config_.max_file_bytes) {
            return RefusalReason::over_max_bytes;
        }
        if (file.facts.declared_size.has_value() && end > *file.facts.declared_size) {
            return RefusalReason::offset_beyond_size;
        }
        return RefusalReason::none;
    }

    [[nodiscard]] bool contiguous(const File& file) const noexcept {
        if (bitmapPath()) {
            return file.bits_set == 0ULL || (file.bits_set == file.highest_index + 1ULL);
        }
        return file.intervals.empty() || (file.intervals.size() == 1UZ && file.intervals[0].begin == 0ULL);
    }

    [[nodiscard]] static bool bitSet(const std::vector<std::uint8_t>& bitmap, std::uint64_t index) noexcept {
        const std::size_t byte = static_cast<std::size_t>(index / 8ULL);
        return byte < bitmap.size() && ((bitmap[byte] >> (index % 8ULL)) & 1U) != 0U;
    }

    /// @brief The index of the file's last chunk, when a declaration already states where the file ends.
    ///
    /// A chunk count names it directly. A declared size names it through the chunk size. A last-chunk flag
    /// has already been folded in as a chunk count before this runs.
    [[nodiscard]] std::optional<std::uint64_t> lastIndex(const File& file) const noexcept {
        const std::uint64_t chunk = chunkSize();
        if (chunk == 0ULL) {
            return std::nullopt;
        }
        if (file.facts.declared_chunks.has_value() && *file.facts.declared_chunks != 0ULL) {
            return *file.facts.declared_chunks - 1ULL;
        }
        if (file.facts.declared_size.has_value() && *file.facts.declared_size != 0ULL) {
            return (*file.facts.declared_size + chunk - 1ULL) / chunk - 1ULL;
        }
        return std::nullopt;
    }

    RefusalReason writeChunk(File& file, std::uint64_t begin, std::span<const std::uint8_t> payload) {
        const std::uint64_t end = begin + payload.size();

        // A chunk with no bytes covers nothing. Admitting it would mark a cell or an interval as covered
        // over an empty range. A file with a hole would then look complete.
        if (payload.empty()) {
            return RefusalReason::bad_payload_span;
        }

        std::uint64_t index = 0ULL;
        if (bitmapPath()) {
            index = begin / chunkSize();
            if (payload.size() > chunkSize()) {
                // The index addresses a fixed grid. A chunk longer than one cell would overlap the next
                // cell's bytes, and the bitmap could not record what is covered.
                return RefusalReason::bad_payload_span;
            }
            if (payload.size() < chunkSize()) {
                // One bit marks one cell as covered. A cell filled only in part would report coverage the
                // file does not have. Only the last chunk of a file may be short. When a declaration already
                // names that index, a short payload at any other index is a malformed chunk. Until a
                // declaration states where the file ends, the chunk is admitted and its index is kept. Only
                // one index can turn out to be the last.
                const std::optional<std::uint64_t> last = lastIndex(file);
                if (last.has_value()) {
                    if (index != *last) {
                        return RefusalReason::bad_payload_span;
                    }
                } else if (file.short_index.has_value() && *file.short_index != index) {
                    return RefusalReason::bad_payload_span;
                }
            }
        }

        if (file.bytes.size() < end) {
            file.bytes.resize(static_cast<std::size_t>(end), 0U);
        }

        if (bitmapPath()) {
            const std::size_t byte = static_cast<std::size_t>(index / 8ULL);
            if (file.bitmap.size() <= byte) {
                file.bitmap.resize(byte + 1UZ, 0U);
            }
            if (bitSet(file.bitmap, index)) {
                return compareCovered(file, begin, payload);
            }
            if (payload.size() < chunkSize()) {
                file.short_index = index;
            }
            file.bitmap[byte] = static_cast<std::uint8_t>(file.bitmap[byte] | (1U << (index % 8ULL)));
            ++file.bits_set;
            file.highest_index = std::max(file.highest_index, index);
            std::copy(payload.begin(), payload.end(), file.bytes.begin() + static_cast<std::ptrdiff_t>(begin));
            file.facts.covered_bytes += payload.size();
            file.write_pointer = std::max(file.write_pointer, end);
            ++file.facts.chunks;
            ++counters_.chunks_written;
            counters_.payload_bytes += payload.size();
            return RefusalReason::none;
        }

        // The interval path. The intervals are sorted and disjoint. Two binary searches find the run this
        // chunk touches. The update is one erase-and-insert, a memmove of at most `16 * (max_gaps + 1)`
        // bytes, with no allocation once the vector has grown. The merge is tried before the cap is
        // tested. A chunk that fills a hole is therefore not refused as if it made one.
        const auto lower = std::lower_bound(file.intervals.begin(), file.intervals.end(), begin, [](const Interval& interval, std::uint64_t value) { return interval.end < value; });
        const auto upper = std::upper_bound(lower, file.intervals.end(), end, [](std::uint64_t value, const Interval& interval) { return value < interval.begin; });

        std::uint64_t alreadyIn = 0ULL;
        for (auto it = lower; it != upper; ++it) {
            alreadyIn += overlap(*it, begin, end);
        }
        if (alreadyIn == payload.size()) {
            return compareCovered(file, begin, payload);
        }

        // Part of this chunk is already held with different bytes. The first write stands, whether it
        // covered the whole range or only part of it. The whole chunk is refused and not written around
        // the disagreement. Half a chunk placed beside a copy that contradicts the other half would make
        // the file inconsistent. The refusal reports the whole chunk instead.
        if (alreadyIn != 0ULL && coveredDiffers(file, begin, payload)) {
            ++file.facts.conflicts;
            return RefusalReason::content_conflict;
        }

        const std::size_t touched = static_cast<std::size_t>(std::distance(lower, upper));
        if (file.intervals.size() - touched + 1UZ > config_.max_gaps + 1UZ) {
            return RefusalReason::too_fragmented;
        }

        // Write only the uncovered parts. A first write then stands.
        std::uint64_t position = begin;
        for (auto it = lower; it != upper; ++it) {
            if (it->end <= position || it->begin >= end) {
                continue;
            }
            if (it->begin > position) {
                copyRange(file, position, std::min(it->begin, end), begin, payload);
            }
            position = std::max(position, it->end);
        }
        if (position < end) {
            copyRange(file, position, end, begin, payload);
        }

        Interval merged{begin, end};
        if (touched != 0UZ) {
            merged.begin = std::min(merged.begin, lower->begin);
            merged.end   = std::max(merged.end, std::prev(upper)->end);
        }
        const auto at = file.intervals.erase(lower, upper);
        file.intervals.insert(at, merged);

        file.write_pointer = std::max(file.write_pointer, end);
        ++file.facts.chunks;
        ++counters_.chunks_written;
        counters_.payload_bytes += payload.size();
        return RefusalReason::none;
    }

    void copyRange(File& file, std::uint64_t from, std::uint64_t to, std::uint64_t begin, std::span<const std::uint8_t> payload) {
        const std::size_t at    = from - begin;
        const std::size_t count = to - from;
        std::copy_n(payload.begin() + static_cast<std::ptrdiff_t>(at), count, file.bytes.begin() + static_cast<std::ptrdiff_t>(from));
        file.facts.covered_bytes += count;
    }

    [[nodiscard]] static std::uint64_t overlap(const Interval& interval, std::uint64_t begin, std::uint64_t end) noexcept {
        const std::uint64_t low  = std::max(interval.begin, begin);
        const std::uint64_t high = std::min(interval.end, end);
        return high > low ? high - low : 0ULL;
    }

    /// @brief True when the already covered parts of this chunk disagree with the bytes held.
    ///
    /// Only the covered sub-ranges are compared. The uncovered ones hold no data yet. Comparing them would
    /// report a conflict against zeros for every partly overlapping chunk.
    [[nodiscard]] bool coveredDiffers(const File& file, std::uint64_t begin, std::span<const std::uint8_t> payload) const {
        const std::uint64_t end     = begin + payload.size();
        const auto          differs = [&](std::uint64_t from, std::uint64_t to) {
            const std::size_t at    = from - begin;
            const std::size_t count = to - from;
            return !std::equal(payload.begin() + static_cast<std::ptrdiff_t>(at), payload.begin() + static_cast<std::ptrdiff_t>(at + count), file.bytes.begin() + static_cast<std::ptrdiff_t>(from));
        };
        if (bitmapPath()) {
            return differs(begin, end);
        }
        for (const Interval& interval : file.intervals) {
            const std::uint64_t low  = std::max(interval.begin, begin);
            const std::uint64_t high = std::min(interval.end, end);
            if (high > low && differs(low, high)) {
                return true;
            }
        }
        return false;
    }

    /// @brief Compares a fully covered range. Equal content is a duplicate. Different content is a conflict.
    RefusalReason compareCovered(File& file, std::uint64_t begin, std::span<const std::uint8_t> payload) {
        if (!coveredDiffers(file, begin, payload)) {
            ++counters_.duplicate_chunks;
            return RefusalReason::none;
        }
        ++file.facts.conflicts;
        return RefusalReason::content_conflict;
    }

    [[nodiscard]] bool complete(const File& file) const noexcept {
        if (file.facts.declared_size.has_value() && file.facts.covered_bytes == *file.facts.declared_size) {
            return true;
        }
        if (file.facts.declared_chunks.has_value() && bitmapPath()) {
            const std::uint64_t count = *file.facts.declared_chunks;
            // A short chunk covers its cell only in part. The file may hold a short chunk at an index that
            // the count now shows is not the last. The file then has a hole at that index. It cannot complete
            // on the count alone.
            if (file.short_index.has_value() && count != 0ULL && *file.short_index != count - 1ULL) {
                return false;
            }
            return count != 0ULL && file.bits_set == count && file.highest_index + 1ULL == count;
        }
        return false;
    }

    static constexpr std::uint64_t kAnonymousBase = 1ULL << 63U;

    Config                                         config_;
    Counters                                       counters_{};
    std::unordered_map<std::uint64_t, File>        files_{};
    std::unordered_map<std::uint64_t, RetiredFile> retired_{};
    std::vector<std::uint64_t>                     departed_{};
    std::uint64_t                                  anonymous_next_ = 0ULL;
    std::optional<std::uint64_t>                   current_{};
};

} // namespace gr::packet

#endif // GNURADIO_ALGORITHM_PACKET_CHUNK_REASSEMBLY_HPP
