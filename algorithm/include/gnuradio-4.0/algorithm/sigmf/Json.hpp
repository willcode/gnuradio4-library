#ifndef GNURADIO_ALGORITHM_SIGMF_JSON_HPP
#define GNURADIO_ALGORITHM_SIGMF_JSON_HPP

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

/**
 * @brief A strict, non-throwing RFC 8259 reader and writer for SigMF metadata documents.
 *
 * The document model preserves object member order, because the emitted text has to be a pure
 * function of the value: two equal documents serialize to identical bytes. Numbers record the shape
 * of the token they came from, so an integral token is recovered from its own digits and never
 * routed through a `double`. No exception escapes this header's API; every failure is reported as a
 * `ParseError` carrying the one-based line and column of the offending byte.
 */
namespace gr::sigmf::json {

/// Maximum nesting depth the parser accepts. SigMF's schema is three levels deep, so the only thing
/// a deeper document can be is an attempt to exhaust the stack. The limit is fixed rather than
/// settable: sixty-four is far past any legitimate extension and far short of a thread stack.
inline constexpr std::size_t kMaxDepth = 64UZ;

struct ParseError {
    std::size_t line{1UZ};
    std::size_t column{1UZ};
    std::string message{}; ///< stable refusal name a test can assert on
    std::string detail{};  ///< the offending key or token, empty when the refusal has none
};

/**
 * @brief A JSON number together with the shape of the token that produced it.
 *
 * `magnitude` holds the digits before the decimal point, parsed by `std::from_chars` on the text.
 * A consumer reading an unsigned schema field uses `magnitude` and never `real`, so a token such as
 * `9007199254740993` survives exactly.
 */
struct Number {
    bool          isInteger{false};         ///< the token carried neither a fractional part nor an exponent
    bool          isIntegralFloat{false};   ///< the token carried a fractional part of all zeros and no exponent
    bool          negative{false};          ///< the token carried a leading minus sign
    bool          magnitudeOverflow{false}; ///< the integer digits do not fit `std::uint64_t`
    std::uint64_t magnitude{0U};            ///< the token's integer digits, valid when integral or integral-float
    double        real{0.0};                ///< the token as a `double`, for fields with a fractional domain

    [[nodiscard]] bool operator==(const Number&) const noexcept = default;
};

/**
 * @brief A JSON value of one of six kinds, with object member order preserved.
 *
 * Objects are held as two parallel vectors rather than a map, so that member order is a property of
 * the document rather than of a container's hashing, and lookup is linear over the few members a
 * SigMF object carries.
 */
class Value {
public:
    enum class Kind : std::uint8_t { Null, Bool, Number, String, Array, Object };

private:
    Kind                     _kind{Kind::Null};
    bool                     _bool{false};
    Number                   _number{};
    std::string              _string{};
    std::vector<std::string> _keys{};   ///< object member keys, in document order
    std::vector<Value>       _values{}; ///< array elements, or object member values parallel to `_keys`

public:
    Value() = default;

    [[nodiscard]] static Value fromBool(bool state) {
        Value result;
        result._kind = Kind::Bool;
        result._bool = state;
        return result;
    }

    [[nodiscard]] static Value fromUnsigned(std::uint64_t magnitude) {
        Value result;
        result._kind             = Kind::Number;
        result._number.isInteger = true;
        result._number.magnitude = magnitude;
        result._number.real      = static_cast<double>(magnitude);
        return result;
    }

    [[nodiscard]] static Value fromSigned(std::int64_t signedValue) {
        Value result;
        result._kind             = Kind::Number;
        result._number.isInteger = true;
        result._number.negative  = signedValue < 0;
        result._number.magnitude = signedValue < 0 ? (static_cast<std::uint64_t>(-(signedValue + 1)) + 1U) : static_cast<std::uint64_t>(signedValue);
        result._number.real      = static_cast<double>(signedValue);
        return result;
    }

    [[nodiscard]] static Value fromDouble(double realValue) {
        Value result;
        result._kind        = Kind::Number;
        result._number.real = realValue;
        return result;
    }

    [[nodiscard]] static Value fromNumber(Number number) {
        Value result;
        result._kind   = Kind::Number;
        result._number = number;
        return result;
    }

    [[nodiscard]] static Value fromString(std::string text) {
        Value result;
        result._kind   = Kind::String;
        result._string = std::move(text);
        return result;
    }

    [[nodiscard]] static Value makeArray() {
        Value result;
        result._kind = Kind::Array;
        return result;
    }

    [[nodiscard]] static Value makeObject() {
        Value result;
        result._kind = Kind::Object;
        return result;
    }

    [[nodiscard]] Kind kind() const noexcept { return _kind; }
    [[nodiscard]] bool isNull() const noexcept { return _kind == Kind::Null; }
    [[nodiscard]] bool isBool() const noexcept { return _kind == Kind::Bool; }
    [[nodiscard]] bool isNumber() const noexcept { return _kind == Kind::Number; }
    [[nodiscard]] bool isString() const noexcept { return _kind == Kind::String; }
    [[nodiscard]] bool isArray() const noexcept { return _kind == Kind::Array; }
    [[nodiscard]] bool isObject() const noexcept { return _kind == Kind::Object; }

    [[nodiscard]] bool               boolValue() const noexcept { return _bool; }
    [[nodiscard]] const Number&      number() const noexcept { return _number; }
    [[nodiscard]] const std::string& str() const noexcept { return _string; }

    /// Number of array elements, or of object members.
    [[nodiscard]] std::size_t size() const noexcept { return _values.size(); }

    [[nodiscard]] const Value&       at(std::size_t index) const noexcept { return _values[index]; }
    [[nodiscard]] const std::string& keyAt(std::size_t index) const noexcept { return _keys[index]; }
    [[nodiscard]] const Value&       valueAt(std::size_t index) const noexcept { return _values[index]; }

    void push(Value element) { _values.push_back(std::move(element)); }

    /// Append a member without checking for an existing one; the parser rejects duplicates first.
    void append(std::string key, Value member) {
        _keys.push_back(std::move(key));
        _values.push_back(std::move(member));
    }

    /// Append the member, or replace the value of an existing one in place, keeping its position.
    void set(std::string_view key, Value member) {
        for (std::size_t i = 0UZ; i < _keys.size(); ++i) {
            if (_keys[i] == key) {
                _values[i] = std::move(member);
                return;
            }
        }
        _keys.emplace_back(key);
        _values.push_back(std::move(member));
    }

    void erase(std::string_view key) {
        for (std::size_t i = 0UZ; i < _keys.size(); ++i) {
            if (_keys[i] == key) {
                _keys.erase(_keys.begin() + static_cast<std::ptrdiff_t>(i));
                _values.erase(_values.begin() + static_cast<std::ptrdiff_t>(i));
                return;
            }
        }
    }

    [[nodiscard]] const Value* find(std::string_view key) const noexcept {
        for (std::size_t i = 0UZ; i < _keys.size(); ++i) {
            if (_keys[i] == key) {
                return &_values[i];
            }
        }
        return nullptr;
    }

    [[nodiscard]] bool contains(std::string_view key) const noexcept { return find(key) != nullptr; }

    [[nodiscard]] bool operator==(const Value& other) const {
        if (_kind != other._kind) {
            return false;
        }
        switch (_kind) {
        case Kind::Null: return true;
        case Kind::Bool: return _bool == other._bool;
        case Kind::Number: return _number == other._number;
        case Kind::String: return _string == other._string;
        case Kind::Array: return _values == other._values;
        case Kind::Object: return _keys == other._keys && _values == other._values;
        }
        return false;
    }
};

namespace detail {

[[nodiscard]] inline bool isDigit(char character) noexcept { return character >= '0' && character <= '9'; }

[[nodiscard]] inline bool isAsciiLetter(char character) noexcept { return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z'); }

/// Length in bytes of the UTF-8 sequence introduced by `lead`, or 0 when `lead` cannot introduce one.
[[nodiscard]] inline std::size_t utf8SequenceLength(unsigned char lead) noexcept {
    if (lead < 0x80U) {
        return 1UZ;
    }
    if (lead >= 0xC2U && lead <= 0xDFU) {
        return 2UZ;
    }
    if (lead >= 0xE0U && lead <= 0xEFU) {
        return 3UZ;
    }
    if (lead >= 0xF0U && lead <= 0xF4U) {
        return 4UZ;
    }
    return 0UZ;
}

/// Append `codePoint` to `out` as UTF-8. The caller has already excluded surrogates.
inline void appendUtf8(std::string& out, std::uint32_t codePoint) {
    if (codePoint < 0x80U) {
        out.push_back(static_cast<char>(codePoint));
    } else if (codePoint < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (codePoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
    } else if (codePoint < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (codePoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (codePoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codePoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
    }
}

/// A recursive-descent RFC 8259 reader. Every refusal carries the position of the offending byte.
class Parser {
public:
    explicit Parser(std::string_view text) noexcept : _text(text) {}

    [[nodiscard]] std::expected<Value, ParseError> run() {
        skipWhitespace();
        auto document = parseValue(0UZ);
        if (!document) {
            return document;
        }
        skipWhitespace();
        if (_pos != _text.size()) {
            return fail("trailing_content");
        }
        return document;
    }

private:
    std::string_view _text;
    std::size_t      _pos{0UZ};
    std::size_t      _line{1UZ};
    std::size_t      _column{1UZ};

    [[nodiscard]] char peek() const noexcept { return _pos < _text.size() ? _text[_pos] : '\0'; }

    [[nodiscard]] char peekAt(std::size_t ahead) const noexcept { return _pos + ahead < _text.size() ? _text[_pos + ahead] : '\0'; }

    void advance() noexcept {
        if (_pos >= _text.size()) {
            return;
        }
        if (_text[_pos] == '\n') {
            ++_line;
            _column = 1UZ;
        } else {
            ++_column;
        }
        ++_pos;
    }

    [[nodiscard]] std::unexpected<ParseError> fail(std::string_view name, std::string_view detail = {}) const { return std::unexpected(ParseError{_line, _column, std::string(name), std::string(detail)}); }

    void skipWhitespace() noexcept {
        while (_pos < _text.size()) {
            const char character = _text[_pos];
            if (character == ' ' || character == '\t' || character == '\n' || character == '\r') {
                advance();
            } else {
                return;
            }
        }
    }

    [[nodiscard]] bool startsWith(std::string_view literal) const noexcept { return _text.substr(_pos).starts_with(literal); }

    void advanceBy(std::size_t count) noexcept {
        for (std::size_t i = 0UZ; i < count; ++i) {
            advance();
        }
    }

    [[nodiscard]] std::expected<Value, ParseError> parseValue(std::size_t depth) {
        const char character = peek();
        switch (character) {
        case '{': return parseObject(depth + 1UZ);
        case '[': return parseArray(depth + 1UZ);
        case '"': {
            auto text = parseStringToken();
            if (!text) {
                return std::unexpected(text.error());
            }
            return Value::fromString(std::move(*text));
        }
        case '\'': return fail("single_quote");
        case '/': return fail("comment_not_allowed");
        case '+': return fail("leading_plus");
        default: break;
        }
        if (character == '-' || isDigit(character)) {
            return parseNumber();
        }
        if (startsWith("true")) {
            advanceBy(4UZ);
            return literalEnd(Value::fromBool(true));
        }
        if (startsWith("false")) {
            advanceBy(5UZ);
            return literalEnd(Value::fromBool(false));
        }
        if (startsWith("null")) {
            advanceBy(4UZ);
            return literalEnd(Value{});
        }
        if (startsWith("NaN") || startsWith("nan") || startsWith("Infinity") || startsWith("inf") || startsWith("-Infinity")) {
            return fail("bare_literal");
        }
        if (_pos >= _text.size()) {
            return fail("unexpected_end");
        }
        if (isAsciiLetter(character)) {
            return fail("bare_literal");
        }
        return fail("unexpected_character");
    }

    [[nodiscard]] std::expected<Value, ParseError> literalEnd(Value produced) const {
        if (isAsciiLetter(peek()) || isDigit(peek())) {
            return fail("bare_literal");
        }
        return produced;
    }

    [[nodiscard]] std::expected<Value, ParseError> parseObject(std::size_t depth) {
        if (depth > kMaxDepth) {
            return fail("nesting_too_deep");
        }
        advance(); // consume '{'
        Value object = Value::makeObject();
        skipWhitespace();
        if (peek() == '/') {
            return fail("comment_not_allowed");
        }
        if (peek() == '}') {
            advance();
            return object;
        }
        while (true) {
            skipWhitespace();
            if (peek() == '/') {
                return fail("comment_not_allowed");
            }
            if (peek() == '\'') {
                return fail("single_quote");
            }
            if (peek() == '}') {
                return fail("trailing_comma");
            }
            if (_pos >= _text.size()) {
                return fail("unexpected_end");
            }
            if (peek() != '"') {
                return fail("unquoted_key");
            }
            const std::size_t keyLine   = _line;
            const std::size_t keyColumn = _column;

            auto key = parseStringToken();
            if (!key) {
                return std::unexpected(key.error());
            }
            if (object.contains(*key)) {
                return std::unexpected(ParseError{keyLine, keyColumn, "duplicate_key", *key});
            }
            skipWhitespace();
            if (peek() != ':') {
                return fail("expected_colon");
            }
            advance();
            skipWhitespace();
            auto member = parseValue(depth);
            if (!member) {
                return member;
            }
            object.append(std::move(*key), std::move(*member));

            skipWhitespace();
            if (peek() == '/') {
                return fail("comment_not_allowed");
            }
            if (peek() == ',') {
                advance();
                continue;
            }
            if (peek() == '}') {
                advance();
                return object;
            }
            if (_pos >= _text.size()) {
                return fail("unexpected_end");
            }
            return fail("expected_comma_or_brace");
        }
    }

    [[nodiscard]] std::expected<Value, ParseError> parseArray(std::size_t depth) {
        if (depth > kMaxDepth) {
            return fail("nesting_too_deep");
        }
        advance(); // consume '['
        Value array = Value::makeArray();
        skipWhitespace();
        if (peek() == '/') {
            return fail("comment_not_allowed");
        }
        if (peek() == ']') {
            advance();
            return array;
        }
        while (true) {
            skipWhitespace();
            if (peek() == ']') {
                return fail("trailing_comma");
            }
            auto element = parseValue(depth);
            if (!element) {
                return element;
            }
            array.push(std::move(*element));

            skipWhitespace();
            if (peek() == '/') {
                return fail("comment_not_allowed");
            }
            if (peek() == ',') {
                advance();
                continue;
            }
            if (peek() == ']') {
                advance();
                return array;
            }
            if (_pos >= _text.size()) {
                return fail("unexpected_end");
            }
            return fail("expected_comma_or_bracket");
        }
    }

    [[nodiscard]] std::expected<std::string, ParseError> parseStringToken() {
        advance(); // consume the opening quote
        std::string out;
        while (true) {
            if (_pos >= _text.size()) {
                return fail("unterminated_string");
            }
            const unsigned char lead = static_cast<unsigned char>(_text[_pos]);
            if (lead == static_cast<unsigned char>('"')) {
                advance();
                return out;
            }
            if (lead == static_cast<unsigned char>('\\')) {
                auto escaped = parseEscape(out);
                if (escaped) {
                    return std::unexpected(*escaped);
                }
                continue;
            }
            if (lead < 0x20U) {
                return fail("bad_control_character");
            }
            const std::size_t sequence = utf8SequenceLength(lead);
            if (sequence == 0UZ || _pos + sequence > _text.size()) {
                return fail("invalid_utf8");
            }
            if (!validContinuation(lead, sequence)) {
                return fail("invalid_utf8");
            }
            out.append(_text.substr(_pos, sequence));
            advanceBy(sequence);
        }
    }

    [[nodiscard]] bool validContinuation(unsigned char lead, std::size_t sequence) const noexcept {
        if (sequence == 1UZ) {
            return true;
        }
        const unsigned char second = static_cast<unsigned char>(_text[_pos + 1UZ]);
        unsigned char       low    = 0x80U;
        unsigned char       high   = 0xBFU;
        if (lead == 0xE0U) {
            low = 0xA0U;
        } else if (lead == 0xEDU) {
            high = 0x9FU; // exclude the surrogate range
        } else if (lead == 0xF0U) {
            low = 0x90U;
        } else if (lead == 0xF4U) {
            high = 0x8FU;
        }
        if (second < low || second > high) {
            return false;
        }
        for (std::size_t i = 2UZ; i < sequence; ++i) {
            const unsigned char continuation = static_cast<unsigned char>(_text[_pos + i]);
            if (continuation < 0x80U || continuation > 0xBFU) {
                return false;
            }
        }
        return true;
    }

    /// Consume one escape sequence and append its expansion to `out`; returns the refusal on failure.
    [[nodiscard]] std::optional<ParseError> parseEscape(std::string& out) {
        advance(); // consume the backslash
        const char marker = peek();
        switch (marker) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
            advance();
            auto first = parseHexQuad();
            if (!first) {
                return ParseError{_line, _column, "bad_escape", {}};
            }
            std::uint32_t codePoint = *first;
            if (codePoint >= 0xD800U && codePoint <= 0xDBFFU) {
                if (peek() != '\\' || peekAt(1UZ) != 'u') {
                    return ParseError{_line, _column, "bad_escape", {}};
                }
                advance();
                advance();
                auto low = parseHexQuad();
                if (!low || *low < 0xDC00U || *low > 0xDFFFU) {
                    return ParseError{_line, _column, "bad_escape", {}};
                }
                codePoint = 0x10000U + ((codePoint - 0xD800U) << 10U) + (*low - 0xDC00U);
            } else if (codePoint >= 0xDC00U && codePoint <= 0xDFFFU) {
                return ParseError{_line, _column, "bad_escape", {}};
            }
            appendUtf8(out, codePoint);
            return std::nullopt;
        }
        default: return ParseError{_line, _column, "bad_escape", {}};
        }
        advance();
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::uint32_t> parseHexQuad() {
        std::uint32_t codePoint = 0U;
        for (std::size_t i = 0UZ; i < 4UZ; ++i) {
            const char    digit  = peek();
            std::uint32_t nibble = 0U;
            if (digit >= '0' && digit <= '9') {
                nibble = static_cast<std::uint32_t>(digit - '0');
            } else if (digit >= 'a' && digit <= 'f') {
                nibble = static_cast<std::uint32_t>(digit - 'a') + 10U;
            } else if (digit >= 'A' && digit <= 'F') {
                nibble = static_cast<std::uint32_t>(digit - 'A') + 10U;
            } else {
                return std::nullopt;
            }
            codePoint = (codePoint << 4U) | nibble;
            advance();
        }
        return codePoint;
    }

    [[nodiscard]] std::expected<Value, ParseError> parseNumber() {
        const std::size_t tokenStart  = _pos;
        const std::size_t tokenLine   = _line;
        const std::size_t tokenColumn = _column;
        Number            number{};

        if (peek() == '-') {
            number.negative = true;
            advance();
        }
        if (!isDigit(peek())) {
            return fail("bad_number");
        }
        const std::size_t integerStart = _pos;
        if (peek() == '0') {
            advance();
            if (peek() == 'x' || peek() == 'X') {
                return fail("hex_number");
            }
            if (isDigit(peek())) {
                return fail("leading_zero");
            }
        } else {
            while (isDigit(peek())) {
                advance();
            }
        }
        const std::size_t integerEnd = _pos;

        bool hasFraction     = false;
        bool fractionAllZero = true;
        bool hasExponent     = false;
        if (peek() == '.') {
            hasFraction = true;
            advance();
            if (!isDigit(peek())) {
                return fail("bad_number");
            }
            while (isDigit(peek())) {
                if (peek() != '0') {
                    fractionAllZero = false;
                }
                advance();
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            hasExponent = true;
            advance();
            if (peek() == '+' || peek() == '-') {
                advance();
            }
            if (!isDigit(peek())) {
                return fail("bad_number");
            }
            while (isDigit(peek())) {
                advance();
            }
        }
        if (peek() == 'x' || peek() == 'X') {
            return fail("hex_number");
        }
        if (isAsciiLetter(peek()) || isDigit(peek()) || peek() == '.') {
            return fail("bad_number");
        }

        number.isInteger       = !hasFraction && !hasExponent;
        number.isIntegralFloat = hasFraction && fractionAllZero && !hasExponent;
        if (number.isInteger || number.isIntegralFloat) {
            const std::string_view digits    = _text.substr(integerStart, integerEnd - integerStart);
            std::uint64_t          magnitude = 0U;
            const auto             converted = std::from_chars(digits.data(), digits.data() + digits.size(), magnitude);
            if (converted.ec != std::errc{} || converted.ptr != digits.data() + digits.size()) {
                number.magnitudeOverflow = true;
            } else {
                number.magnitude = magnitude;
            }
        }

        const std::string_view token = _text.substr(tokenStart, _pos - tokenStart);
        if (number.isInteger && number.magnitudeOverflow) {
            return std::unexpected(ParseError{tokenLine, tokenColumn, "index_overflow", std::string(token)});
        }

        double     real      = 0.0;
        const auto converted = std::from_chars(token.data(), token.data() + token.size(), real);
        if (converted.ec == std::errc::result_out_of_range) {
            return std::unexpected(ParseError{tokenLine, tokenColumn, "number_out_of_range", std::string(token)});
        }
        if (converted.ec != std::errc{}) {
            return std::unexpected(ParseError{tokenLine, tokenColumn, "bad_number", std::string(token)});
        }
        number.real = real;
        return Value::fromNumber(number);
    }
};

inline void writeEscapedString(std::string& out, std::string_view text) {
    out.push_back('"');
    for (const char character : text) {
        const unsigned char byte = static_cast<unsigned char>(character);
        switch (character) {
        case '"': out.append("\\\""); continue;
        case '\\': out.append("\\\\"); continue;
        case '\b': out.append("\\b"); continue;
        case '\f': out.append("\\f"); continue;
        case '\n': out.append("\\n"); continue;
        case '\r': out.append("\\r"); continue;
        case '\t': out.append("\\t"); continue;
        default: break;
        }
        if (byte < 0x20U) {
            constexpr std::string_view kHex = "0123456789abcdef";
            out.append("\\u00");
            out.push_back(kHex[(byte >> 4U) & 0x0FU]);
            out.push_back(kHex[byte & 0x0FU]);
            continue;
        }
        out.push_back(character); // every other code point passes through as UTF-8
    }
    out.push_back('"');
}

inline void writeNumber(std::string& out, const Number& number) {
    std::array<char, 64> buffer{};
    if (number.isInteger) {
        if (number.negative) {
            out.push_back('-');
        }
        const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), number.magnitude);
        out.append(buffer.data(), static_cast<std::size_t>(converted.ptr - buffer.data()));
        return;
    }
    // The shortest decimal that reads back to the same double, in whichever of fixed and scientific
    // notation is shorter. This is the precision-free `std::to_chars` overload; the `general` format
    // argument is a different rule that spells 433921337 as 4.33921337e+08.
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), number.real);
    out.append(buffer.data(), static_cast<std::size_t>(converted.ptr - buffer.data()));
}

inline void writeIndent(std::string& out, std::size_t level) { out.append(2UZ * level, ' '); }

inline void writeValue(std::string& out, const Value& value, std::size_t level) {
    switch (value.kind()) {
    case Value::Kind::Null: out.append("null"); return;
    case Value::Kind::Bool: out.append(value.boolValue() ? "true" : "false"); return;
    case Value::Kind::Number: writeNumber(out, value.number()); return;
    case Value::Kind::String: writeEscapedString(out, value.str()); return;
    case Value::Kind::Array: {
        if (value.size() == 0UZ) {
            out.append("[]");
            return;
        }
        out.append("[\n");
        for (std::size_t i = 0UZ; i < value.size(); ++i) {
            writeIndent(out, level + 1UZ);
            writeValue(out, value.at(i), level + 1UZ);
            if (i + 1UZ < value.size()) {
                out.push_back(',');
            }
            out.push_back('\n');
        }
        writeIndent(out, level);
        out.push_back(']');
        return;
    }
    case Value::Kind::Object: {
        if (value.size() == 0UZ) {
            out.append("{}");
            return;
        }
        out.append("{\n");
        for (std::size_t i = 0UZ; i < value.size(); ++i) {
            writeIndent(out, level + 1UZ);
            writeEscapedString(out, value.keyAt(i));
            out.append(": ");
            writeValue(out, value.valueAt(i), level + 1UZ);
            if (i + 1UZ < value.size()) {
                out.push_back(',');
            }
            out.push_back('\n');
        }
        writeIndent(out, level);
        out.push_back('}');
        return;
    }
    }
}

} // namespace detail

/// Parse one JSON document. No exception escapes; a failure carries the one-based line and column.
[[nodiscard]] inline std::expected<Value, ParseError> parse(std::string_view text) {
    detail::Parser parser(text);
    return parser.run();
}

/// Serialize `value` with a two-space indent, one member per line, and a single trailing newline.
[[nodiscard]] inline std::string write(const Value& value) {
    std::string out;
    detail::writeValue(out, value, 0UZ);
    out.push_back('\n');
    return out;
}

} // namespace gr::sigmf::json

#endif // GNURADIO_ALGORITHM_SIGMF_JSON_HPP
