#ifndef DATASETESTIMATORS_HPP
#define DATASETESTIMATORS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <format>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/math/UncertainValueFormatter.hpp>

#include "DataSetHelper.hpp"
#include "DataSetMath.hpp"

namespace gr::dataset {

template<typename T>
struct Point {
    std::size_t index;
    T           value;
};

namespace estimators {

/// @brief accumulator for the Welford updates: an integral sample type is promoted to double so that the per-sample division keeps the fractional part the update carries forward.
template<typename T>
using WelfordAccumulator = std::conditional_t<std::is_integral_v<T>, double, T>;

/// @brief arithmetic type of an estimator's intermediate results: an integral sample type is promoted to double, which
/// holds every 32-bit integral sample exactly, so that a fractional intermediate is not truncated to zero.
template<typename T>
using PromotedAccumulator = std::conditional_t<std::is_integral_v<T>, double, T>;

/// @brief converts an intermediate result to the sample type: an integral type takes the nearest sample value, halves
/// away from zero; a floating-point or uncertainty-carrying type keeps the value it already has.
template<typename T, typename TAcc>
[[nodiscard]] constexpr T fromAccumulator(const TAcc& value) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(value < TAcc(0) ? value - TAcc(0.5) : value + TAcc(0.5));
    } else {
        return static_cast<T>(value);
    }
}

template<typename T>
[[nodiscard]] constexpr std::expected<T, gr::Error> computeCentreOfMass(const DataSet<T>& ds, std::size_t minIndex = 0UZ, std::size_t maxIndex = max_size, std::size_t signalIndex = 0, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(ds, minIndex, maxIndex, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    maxIndex = *checkedMax;
    T com    = T(0);
    T mass   = T(0);

    for (std::size_t i = minIndex; i < maxIndex; i++) {
        const T x     = detail::xValue(ds, i);
        const T value = detail::yValue(ds, i, signalIndex);
        if (gr::math::isfinite(x) && gr::math::isfinite(value)) {
            com += x * value;
            mass += value;
        }
    }
    if (gr::value(mass) == gr::value(T(0))) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return com / mass;
}

/// @brief The width in samples of the peak at @p index, read where each flank falls to half its height;
/// NaN where either flank reaches the end of @p data still above half maximum.
template<std::ranges::random_access_range T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr TValue computeFWHM(const T& data, std::size_t index) {
    using value_t = gr::meta::fundamental_base_value_type_t<TValue>; // innermost value
    if (!(index > 0UZ && index < data.size() - 1UZ)) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    TValue      maxHalf = value_t(0.5) * gr::value(data[index]);
    std::size_t lowerLimit;
    std::size_t upperLimit;
    for (upperLimit = index; upperLimit < data.size() && data[upperLimit] > maxHalf; upperLimit++) {
        // done in condition
    }
    for (lowerLimit = index; lowerLimit > 0UZ && data[lowerLimit] > maxHalf; lowerLimit--) {
        // done in condition
    }
    if (upperLimit >= data.size() || data[lowerLimit] > maxHalf) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    return static_cast<TValue>(upperLimit - lowerLimit);
}

/// @copydoc computeFWHM
/// The two crossings are refined by linear interpolation between the samples they fall between.
template<std::ranges::random_access_range T, typename TValue = typename T::value_type>
[[nodiscard]] constexpr TValue computeInterpolatedFWHM(const T& data, std::size_t index) {
    using value_t = gr::meta::fundamental_base_value_type_t<TValue>; // innermost value
    if (!(index > 0 && index < data.size() - 1)) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    TValue      maxHalf = value_t(0.5) * data[index];
    std::size_t lowerLimit;
    std::size_t upperLimit;
    for (upperLimit = index; upperLimit < data.size() && data[upperLimit] > maxHalf; upperLimit++) {
        // done in condition
    }
    for (lowerLimit = index; lowerLimit > 0UZ && data[lowerLimit] > maxHalf; lowerLimit--) {
        // done in condition
    }
    if (upperLimit >= data.size() || data[lowerLimit] > maxHalf) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    TValue lowerRefined = detail::linearInterpolate(value_t(lowerLimit), value_t(lowerLimit + 1), data[lowerLimit], data[lowerLimit + 1], maxHalf);
    TValue upperRefined = detail::linearInterpolate(value_t(upperLimit - 1), value_t(upperLimit), data[upperLimit - 1], data[upperLimit], maxHalf);
    return upperRefined - lowerRefined;
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type>
[[nodiscard]] constexpr std::expected<std::optional<Point<T>>, gr::Error> getMaximum(D&& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    using TValue                  = gr::meta::fundamental_base_value_type_t<T>;
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    auto           checkedMax     = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;
    if (indexMin == indexMax) {
        return Point<T>{indexMin, detail::yValue(dataSet, indexMin, signalIndex)};
    }

    int locMax = -1;
    T   maxVal = std::numeric_limits<TValue>::lowest();
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = detail::yValue(dataSet, i, signalIndex); gr::math::isfinite(actual) && actual > maxVal) {
            maxVal = actual;
            locMax = static_cast<int>(i);
        }
    }
    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        if (locMax >= 0) {
            dataSet.timing_events[signalIndex].push_back({static_cast<std::ptrdiff_t>(locMax), {{"gr:maximum", gr::value(maxVal)}}});
        }
    }

    return locMax >= 0 ? std::make_optional(Point<T>{static_cast<std::size_t>(locMax), maxVal}) : std::nullopt;
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type>
[[nodiscard]] constexpr std::expected<std::optional<Point<T>>, gr::Error> getMinimum(D&& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    using TValue                  = gr::meta::fundamental_base_value_type_t<T>;
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    auto           checkedMax     = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;
    if (indexMin == indexMax) {
        return Point<T>{indexMin, detail::yValue(dataSet, indexMin, signalIndex)};
    }

    T   minVal = +std::numeric_limits<TValue>::max();
    int locMin = -1;
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = detail::yValue(dataSet, i, signalIndex); gr::math::isfinite(actual) && actual < minVal) {
            minVal = actual;
            locMin = static_cast<int>(i);
        }
    }
    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        if (locMin >= 0) {
            dataSet.timing_events[signalIndex].push_back({static_cast<std::ptrdiff_t>(locMin), {{"gr:minimum", gr::value(minVal)}}});
        }
    }

    return locMin >= 0 ? std::make_optional(Point<T>{static_cast<std::size_t>(locMin), minVal}) : std::nullopt;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<T, gr::Error> getMean(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    using TAcc = WelfordAccumulator<T>;

    TAcc count = 0;
    TAcc mean  = 0;
    // Welford: the working values stay at the scale of the deviations, so a large offset cannot swamp the sample spread
    for (const auto& val : finiteValues) {
        count += TAcc(1);
        mean += (static_cast<TAcc>(val) - mean) / count;
    }

    return count > TAcc(0) ? static_cast<T>(mean) : std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<T, gr::Error> getMedian(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    std::span<const T> values = dataSet.signalValues(signalIndex).subspan(indexMin, indexMax - indexMin);

    if (values.empty()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }

    std::vector<T> data(values.begin(), values.end()); // temporary mutable copy for in-place partitioning
    const auto     midIndex = data.size() / 2UZ;
    auto           mid      = data.begin() + static_cast<std::ptrdiff_t>(midIndex);
    std::ranges::nth_element(data, mid);

    if ((data.size() & 1UZ) != 0UZ) {
        return static_cast<T>(data.at(midIndex)); // odd-sized data, return the middle element
    }

    // even-sized data, calculate the mean of the two middle elements
    const auto lowerMidIndex = midIndex - 1UZ;
    auto       lowerMid      = data.begin() + static_cast<std::ptrdiff_t>(lowerMidIndex);
    std::ranges::nth_element(data.begin(), lowerMid, mid);
    using TAcc = PromotedAccumulator<T>;
    return fromAccumulator<T>(static_cast<TAcc>(0.5) * (static_cast<TAcc>(data.at(lowerMidIndex)) + static_cast<TAcc>(data.at(midIndex))));
}

template<typename T>
[[nodiscard]] constexpr std::expected<T, gr::Error> getRange(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    auto [minIt, maxIt] = std::ranges::minmax_element(finiteValues);
    if (minIt == finiteValues.end() || maxIt == finiteValues.end()) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    return *maxIt - *minIt;
}

/// @brief population standard deviation, computed by Welford accumulation.
template<typename T>
[[nodiscard]] constexpr std::expected<T, gr::Error> getStdDev(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    using TAcc = WelfordAccumulator<T>;

    TAcc count = 0;
    TAcc mean  = 0;
    TAcc m2    = 0;
    for (const auto& val : finiteValues) {
        count += TAcc(1);
        const TAcc sample = static_cast<TAcc>(val);
        const TAcc delta  = sample - mean;
        mean += delta / count;
        m2 += delta * (sample - mean);
    }

    if (!(count > TAcc(0))) {
        return T(0);
    }
    const TAcc variance = m2 / count;
    return variance > TAcc(0) ? static_cast<T>(gr::math::sqrt(variance)) : T(0);
}

/// @brief The standard deviation, i.e. the RMS of the mean-subtracted signal, despite the name.
/// Retained under this name for source compatibility; prefer getStdDev.
template<typename T>
[[nodiscard]] constexpr std::expected<T, gr::Error> getRms(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    return getStdDev(dataSet, indexMin, indexMax, signalIndex, location);
}

template<typename T>
[[nodiscard]] constexpr std::expected<T, gr::Error> getIntegral(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    // sign if reversed
    const T           sign_ = detail::sign(T(1), int(indexMax) - int(indexMin));
    const std::size_t start = std::min(indexMin, indexMax);
    const std::size_t stop  = std::max(indexMin, indexMax);
    if (stop <= start + 1UZ) {
        return T(0);
    }

    using TAcc = PromotedAccumulator<T>;

    auto idxRange = std::views::iota(start, stop - 1);
    TAcc integral = std::transform_reduce(std::ranges::begin(idxRange), std::ranges::end(idxRange), TAcc(0), std::plus<>{}, //
        [&](std::size_t i) -> TAcc {
            TAcc x0 = static_cast<TAcc>(detail::xValue(dataSet, i));
            TAcc x1 = static_cast<TAcc>(detail::xValue(dataSet, i + 1));
            TAcc y0 = static_cast<TAcc>(detail::yValue(dataSet, i, signalIndex));
            TAcc y1 = static_cast<TAcc>(detail::yValue(dataSet, i + 1, signalIndex));

            TAcc area = TAcc(0.5) * (x1 - x0) * (y0 + y1);
            return gr::math::isfinite(area) ? area : TAcc(0);
        });

    return fromAccumulator<T>(static_cast<TAcc>(sign_) * integral);
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<T, gr::Error> getEdgeDetect(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
    T range  = maxVal > minVal ? maxVal - minVal : minVal - maxVal;                 // abs
    if (!gr::math::isfinite(range) || range == TValue(0)) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    // check if falling or rising
    T    startVal  = detail::yValue(dataSet, indexMin, signalIndex);
    T    endVal    = detail::yValue(dataSet, indexMax - 1, signalIndex);
    bool inverted  = (startVal > endVal);
    T    startTime = detail::xValue(dataSet, indexMin);

    T halfCross = inverted ? (maxVal - TValue(0.5) * range) : (minVal + TValue(0.5) * range);
    for (std::size_t i = indexMin; i < indexMax; i++) {
        T actual = detail::yValue(dataSet, i, signalIndex);
        if (!gr::math::isfinite(actual)) {
            continue;
        }
        if ((!inverted && actual > halfCross) || (inverted && actual < halfCross)) {
            T xNow = detail::xValue(dataSet, i);
            return xNow - startTime;
        }
    }
    return std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<T, gr::Error> getDutyCycle(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above

    if (!gr::math::isfinite(minVal) || !gr::math::isfinite(maxVal) || minVal == maxVal) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    T       range        = maxVal > minVal ? maxVal - minVal : minVal - maxVal;
    const T thresholdMin = minVal + TValue(0.45) * range;
    const T thresholdMax = minVal + TValue(0.55) * range;
    int     countLow = 0, countHigh = 0;
    for (const auto& val : finiteValues) {
        if (val < thresholdMin) {
            countLow++;
        } else if (val > thresholdMax) {
            countHigh++;
        }
    }

    const int totalCount = countLow + countHigh;
    return totalCount > 0 ? static_cast<TValue>(countHigh) / static_cast<TValue>(totalCount) : std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<T, gr::Error> getFrequencyEstimate(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    // getFrequencyEstimate => naive/simple approach counting edges
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
    if (!gr::math::isfinite(minVal) || !gr::math::isfinite(maxVal) || maxVal == minVal) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    T range        = maxVal > minVal ? maxVal - minVal : minVal - maxVal; // abs
    T thresholdMin = minVal + TValue(0.45) * range;
    T thresholdMax = minVal + TValue(0.55) * range;

    T   startRisingEdge  = std::numeric_limits<TValue>::quiet_NaN();
    T   startFallingEdge = std::numeric_limits<TValue>::quiet_NaN();
    T   avgPeriod        = 0;
    int avgPeriodCount   = 0;

    // 0 => below threshold, 1 => above threshold
    float actualState = 0.f;
    for (std::size_t i = indexMin; i < indexMax; i++) {
        T actual = detail::yValue(dataSet, i, signalIndex);
        if (!gr::math::isfinite(actual)) {
            continue;
        }
        T x = detail::xValue(dataSet, i);

        if (actualState < 0.5f) {
            // was low, check if we cross thresholdMax => rising edge
            if (actual > thresholdMax) {
                actualState = 1.0f;
                if (gr::math::isfinite(startRisingEdge)) {
                    T period        = x - startRisingEdge;
                    startRisingEdge = x;
                    avgPeriod += period;
                    avgPeriodCount++;
                } else {
                    startRisingEdge = x;
                }
            }
        } else {
            // was high, check if we cross thresholdMin => falling edge
            if (actual < thresholdMin) {
                actualState = 0.0f;
                if (gr::math::isfinite(startFallingEdge)) {
                    T period         = x - startFallingEdge;
                    startFallingEdge = x;
                    avgPeriod += period;
                    avgPeriodCount++;
                } else {
                    startFallingEdge = x;
                }
            }
        }
    }
    if (avgPeriodCount == 0) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    return static_cast<TValue>(avgPeriodCount) / avgPeriod; // freq = # of periods / total time
}

template<std::ranges::random_access_range T, typename TValue = typename T::value_type>
[[nodiscard]] constexpr TValue interpolateGaussian(const T& data, std::size_t index) {
    using value_t = gr::meta::fundamental_base_value_type_t<TValue>;
    if (!(index > 0 && index < data.size() - 1)) {
        return static_cast<value_t>(index); // fallback
    }
    const TValue left   = data[index - 1];
    const TValue center = data[index];
    const TValue right  = data[index + 1];

    if (!gr::math::isfinite(left) || !gr::math::isfinite(right) || !gr::math::isfinite(center)) {
        return static_cast<value_t>(index);
    }

    if (gr::value(left) <= value_t(0) || gr::value(right) <= value_t(0) || gr::value(center) <= value_t(0)) {
        return static_cast<value_t>(index);
    }
    TValue val         = static_cast<value_t>(index);
    TValue numerator   = value_t(0.5f) * gr::math::log(right / left);
    TValue denominator = gr::math::log((center * center) / (left * right));
    if (gr::value(denominator) == TValue(0)) {
        return val;
    }
    return val + numerator / denominator;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<T, gr::Error> getLocationMaximumGaussInterpolated(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    // getFrequencyEstimate => naive/simple approach counting edges
    auto checkedMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    const auto locMax_ = *getMaximum(dataSet, indexMin, indexMax, signalIndex); // the range is checked above
    if (!locMax_.has_value()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    const std::size_t locMax = locMax_.value().index;

    T refinedIndex = interpolateGaussian(dataSet.signalValues(signalIndex), locMax);
    T x0           = detail::xValue(dataSet, locMax);
    // approximate step
    // we’ll do a naive approach: X_{locMax+1} - X_{locMax}
    if (locMax + 1UZ >= static_cast<std::size_t>(dataSet.axisValues(dim::X).size())) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    T x1       = detail::xValue(dataSet, locMax + 1UZ);
    T diff     = x1 - x0;
    T deltaBin = refinedIndex - static_cast<TValue>(locMax);
    return x0 + deltaBin * diff;
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<T, gr::Error> getZeroCrossing(D&& dataSet, TValue threshold, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0, std::source_location location = std::source_location::current()) {
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    auto           checkedMax     = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    T    initial  = detail::yValue(dataSet, 0UZ, signalIndex);
    int  index    = -1;
    T    value    = std::numeric_limits<TValue>::quiet_NaN();
    bool isRising = initial < threshold;
    if (isRising) { // rising edge
        for (std::size_t i = indexMin + 1UZ; i < indexMax; ++i) {
            T yPrev = detail::yValue(dataSet, i - 1UZ, signalIndex);
            T yCurr = detail::yValue(dataSet, i, signalIndex);
            if (gr::math::isfinite(yPrev) && gr::math::isfinite(yCurr) && yCurr >= threshold) {
                T xPrev = detail::xValue(dataSet, i - 1UZ);
                T xCurr = detail::xValue(dataSet, i);
                index   = static_cast<int>(i);
                value   = gr::dataset::detail::linearInterpolate(xPrev, xCurr, yPrev, yCurr, T(threshold));
                break;
            }
        }
    } else { // falling edge
        for (std::size_t i = indexMin + 1UZ; i < indexMax; ++i) {
            T yPrev = detail::yValue(dataSet, i - 1UZ, signalIndex);
            T yCurr = detail::yValue(dataSet, i, signalIndex);
            if (gr::math::isfinite(yPrev) && gr::math::isfinite(yCurr) && yCurr <= threshold) {
                T xPrev = detail::xValue(dataSet, i - 1UZ);
                T xCurr = detail::xValue(dataSet, i);
                index   = static_cast<int>(i);
                value   = gr::dataset::detail::linearInterpolate(xPrev, xCurr, yPrev, yCurr, T(threshold));
                break;
            }
        }
    }
    if (index < 0) {
        index = static_cast<int>(indexMax); // chose end-of-range
        value = detail::yValue(dataSet, indexMax - 1UZ, signalIndex);
    }

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        const auto          idx = static_cast<std::size_t>(index);
        std::string         context;
        const std::uint64_t period          = 1'000'000'000;
        const std::uint64_t time            = static_cast<std::uint64_t>(gr::value(value)) * period;
        const std::uint64_t timeUncertainty = static_cast<std::uint64_t>(gr::uncertainty(value)) * period;
        property_map        data            = property_map{                                                                                                 //
            {gr::tag::TRIGGER_NAME.shortKey(), std::format("{}_EDGE_LEVEL_{}", isRising ? "RISING" : "FALLING", threshold)},              //
            {gr::tag::TRIGGER_TIME.shortKey(), time}, {"trigger_time_error", timeUncertainty}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f}, //
            {gr::tag::CONTEXT.shortKey(), context}};
        dataSet.timing_events[signalIndex].push_back({idx, std::move(data)});
    }
    return value;
}

template<typename T>
struct StepStartDetectionResult {
    std::size_t index;        ///< Index where the step starts.
    T           initialValue; ///< Initial value of the signal.
    T           minValue;     ///< Minimum value of the signal.
    T           maxValue;     ///< Maximum value of the signal.
    bool        isRising;     ///< Indicates if the detected step is rising (true) or falling (false).
};

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] std::expected<std::optional<StepStartDetectionResult<T>>, gr::Error> detectStepStart(D& ds, TValue threshold = TValue(0.5), std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;

    std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(ds, "", location);
    if (!dsCheck.has_value()) {
        return std::unexpected(gr::Error(std::format("Invalid DataSet for step/pulse start detection: {}", dsCheck.error()), location));
    }
    auto checkedMax = detail::checkIndexRange(ds, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;

    auto signal = ds.signalValues(signalIndex);
    if (signal.empty()) {
        return std::unexpected(gr::Error("Signal is empty.", location));
    }

    T initial = signal.front();
    T max_val = estimators::getMaximum(ds, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
    T min_val = estimators::getMinimum(ds, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        ds.meta_information[signalIndex]["gr:minimum"] = gr::value(min_val);
        ds.meta_information[signalIndex]["gr:maximum"] = gr::value(max_val);
    }

    bool isRising  = initial < max_val;
    bool isFalling = initial > min_val;

    if (!isRising && !isFalling) {
        return std::nullopt; // no clear step detected
    }

    T thresholdCrossing = isRising ? initial + threshold * (max_val - initial) : initial - threshold * (initial - min_val);
    T triggerTime       = *estimators::getZeroCrossing<MetaInfo::None>(ds, thresholdCrossing, indexMin, indexMax, signalIndex); // the range is checked above
    if (std::isnan(gr::value(triggerTime))) {
        return std::nullopt; // failed to compute trigger time
    }

    // Find the closest index to the triggerTime
    const auto& xAxis = ds.axisValues(0);
    auto        it    = std::lower_bound(xAxis.begin(), xAxis.end(), triggerTime);
    if (it == xAxis.end()) {
        return std::nullopt; // no step detected
    }
    std::size_t index = static_cast<std::size_t>(std::distance(xAxis.begin(), it));

    // ensure the index is within bounds
    if (index >= signal.size()) {
        return std::nullopt;
    }

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        std::string         context;
        const std::uint64_t period          = 1'000'000'000;
        const std::uint64_t time            = static_cast<std::uint64_t>(gr::value(xAxis[index])) * period;
        const std::uint64_t timeUncertainty = 0UZ;
        property_map        data            = property_map{{gr::tag::TRIGGER_NAME.shortKey(), std::format("{}_EDGE_LEVEL_{}", isRising ? "RISING" : "FALLING", threshold)}, //
                              {gr::tag::TRIGGER_TIME.shortKey(), time}, {"trigger_time_error", timeUncertainty}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f},                 //
                              {gr::tag::CONTEXT.shortKey(), context}};
        ds.timing_events[signalIndex].push_back({index, std::move(data)});
    }
    return StepStartDetectionResult<T>{.index = index, .initialValue = initial, .minValue = min_val, .maxValue = max_val, .isRising = isRising};
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::expected<std::optional<Point<T>>, gr::Error> getSettingTime(D&& dataSet, std::size_t windowSize, TValue tolerance = TValue(0), std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    auto           checkedMax     = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;
    if (windowSize == 0UZ || (indexMax - indexMin) < windowSize) {
        return std::nullopt;
    }
    if (tolerance == TValue(0)) {
        T minVal  = getMinimum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
        T maxVal  = getMaximum(dataSet, indexMin, indexMax, signalIndex)->value().value; // the range is checked above
        tolerance = TValue(0.02) * gr::value(gr::math::abs(maxVal - minVal));
    }

    // estimate settling time by shifting RMS window until the RMS is below the required threshold
    int settlingTime = -1;
    for (std::size_t i = indexMin; i + windowSize <= indexMax; ++i) {
        T localRMS = *estimators::getRms(dataSet, i, i + windowSize, signalIndex); // i + windowSize <= indexMax
        if (gr::math::abs(localRMS) <= tolerance) {
            settlingTime = static_cast<int>(i);
            break;
        }
    }
    if (settlingTime < 0) {
        return std::nullopt;
    }

    // estimate flat-top level by increasing the window as long as RMS is getting smaller
    const std::size_t settlingTimeIdx = static_cast<std::size_t>(settlingTime);
    T                 prevRMS         = *estimators::getRms(dataSet, settlingTimeIdx, settlingTimeIdx + windowSize, signalIndex); // within the checked range
    std::size_t       maxIndex;
    for (maxIndex = settlingTimeIdx; maxIndex + windowSize < indexMax; ++maxIndex) {
        T localRMS = *estimators::getRms(dataSet, settlingTimeIdx, maxIndex + windowSize, signalIndex);
        if (localRMS >= prevRMS) {
            break;
        }
        prevRMS = localRMS;
    }
    T settlingLevel = *estimators::getMean(dataSet, settlingTimeIdx, maxIndex + windowSize, signalIndex);
    settlingTime += static_cast<int>(windowSize) / 2;
    if (settlingTime < 0) {
        return std::nullopt;
    }

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        const auto          idx = static_cast<std::size_t>(settlingTime);
        std::string         context;
        const auto          xAxis           = dataSet.axisValues(dim::X);
        const std::uint64_t period          = 1'000'000'000;
        const std::uint64_t time            = static_cast<std::uint64_t>(gr::value(xAxis[idx])) * period;
        const std::uint64_t timeUncertainty = 0UZ;
        property_map        data            = property_map{{gr::tag::TRIGGER_NAME.shortKey(), "SETTLING_TIME"}, {"gr:settling_level", gr::value(settlingLevel)}, //
                              {gr::tag::TRIGGER_TIME.shortKey(), time}, {"trigger_time_error", timeUncertainty}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f},      //
                              {gr::tag::CONTEXT.shortKey(), context}};
        dataSet.timing_events[signalIndex].push_back({idx, std::move(data)});
    }

    return Point<T>{static_cast<std::size_t>(settlingTime), settlingLevel};
}

template<typename T>
struct StepPulseResponseMetrics {
    bool isEstimate    = true;  ///< indicates that values are rough estimates
    bool isPulse       = false; ///< indicates if the response is a pulse.
    bool isRising      = true;  ///< indicates if the first edge is a rsiging edge
    T    triggerTime   = 0.0;   ///< time at which the trigger occurs (50% crossing).
    T    V1            = 0.0;   ///< steady-state level before the step.
    T    V2            = 0.0;   ///< steady-state level after the step.
    T    riseTime      = 0.0;   ///< rise time (e.g., 10% -> 90% of V2 - V1).
    T    peakAmplitude = 0.0;   ///< peak amplitude relative to V1.
    T    peakTime      = 0.0;   ///< time at which the peak occurs.
    T    overshoot     = 0.0;   ///< overshoot percentage.
    T    settlingTime  = 0.0;   ///< settling time within 2% of V2.
};

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type>
[[nodiscard]] std::expected<StepPulseResponseMetrics<T>, gr::Error> analyzeStepPulseResponse(D&& ds, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    using TValue                  = gr::meta::fundamental_base_value_type_t<T>;
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    auto           checkedMax     = detail::checkIndexRange(ds, indexMin, indexMax, signalIndex, location);
    if (!checkedMax) {
        return std::unexpected(checkedMax.error());
    }
    indexMax = *checkedMax;
    if (indexMax - indexMin < 2UZ) {
        return std::unexpected(gr::Error("Not enough data points for step/pulse analysis.", location));
    }

    // Step 1: detect initial rough edge
    std::array<StepStartDetectionResult<T>, 3UZ> roughEdges{};
    for (const auto& [edge, level] : std::views::zip(roughEdges, std::array{TValue(0.5), TValue(0.2), TValue(0.8)})) {
        auto detected = estimators::detectStepStart<MetaInfo::None>(ds, level, indexMin, indexMax, signalIndex, location);
        if (!detected) {
            return std::unexpected(detected.error());
        }
        if (!detected->has_value()) {
            return std::unexpected(gr::Error(std::format("no step detected in the signal at level {}.", level), location));
        }
        edge = **detected;
    }
    const auto& [firstRoughEdge, firstRoughLevel20, firstRoughLevel80] = roughEdges;
    const std::size_t roughRiseTime                                    = firstRoughLevel80.index - firstRoughLevel20.index;
    const TValue      roughMinThreshold                                = gr::value(TValue(0.02) * (firstRoughEdge.maxValue - firstRoughEdge.minValue));

    StepPulseResponseMetrics<T> metrics;
    metrics.triggerTime = gr::cast<T>(firstRoughEdge.index);
    metrics.riseTime    = gr::cast<T>(roughRiseTime);

    std::span<const T> signalValues = ds.signalValues(signalIndex);
    const bool         isRising     = firstRoughEdge.isRising;

    // Step 2: compute V1 using rough detected edge
    const std::size_t V1Start = 0UZ;
    const std::size_t V1End   = firstRoughEdge.index > roughRiseTime ? firstRoughEdge.index - roughRiseTime : 4UZ; // based on 20%-80% rise-time around ~50% level
    const auto        rms     = estimators::getRms(ds, V1Start, V1End, signalIndex, location);
    const auto        mean    = estimators::getMean(ds, V1Start, V1End, signalIndex, location);
    if (!rms || !mean) {
        return std::unexpected(rms ? mean.error() : rms.error());
    }
    const TValue rmsEstimate  = std::max(gr::value(*rms), roughMinThreshold);
    const TValue meanEstimate = gr::value(*mean);
    const auto   rampStart    = estimators::getZeroCrossing<MetaInfo::None>(ds, meanEstimate + TValue(2.5) * (isRising ? +rmsEstimate : -rmsEstimate), V1Start, firstRoughEdge.index, signalIndex, location);
    if (!rampStart) {
        return std::unexpected(rampStart.error());
    }

    const auto V1 = estimators::getMean(ds, V1Start, std::min(static_cast<std::size_t>(gr::value(*rampStart)), firstRoughEdge.index), signalIndex, location); // should be bias-free from over- and under-shoot
    if (!V1) {
        return std::unexpected(V1.error());
    }
    metrics.V1 = *V1;

    // Step 3: detect peak position
    const std::size_t peakMaxIndex = std::min(firstRoughEdge.index + 2UZ * roughRiseTime, indexMax);
    const auto        peak         = isRising ? estimators::getMaximum(ds, firstRoughEdge.index, peakMaxIndex, 0UZ, location) : estimators::getMinimum(ds, firstRoughEdge.index, peakMaxIndex, 0UZ, location);
    if (!peak) {
        return std::unexpected(peak.error());
    }
    const auto& peakIndex = *peak;
    if (!peakIndex.has_value()) { // no peak detected
        metrics.peakAmplitude = std::numeric_limits<TValue>::quiet_NaN();
        metrics.peakTime      = std::numeric_limits<TValue>::quiet_NaN();
    } else {
        metrics.peakAmplitude = ds.signalValues(signalIndex)[peakIndex.value().index] - metrics.V1;
        metrics.peakTime      = ds.axisValues(dim::X)[peakIndex.value().index];

        const auto settling = getSettingTime(ds, 4UZ * roughRiseTime, roughMinThreshold, peakIndex.value().index, indexMax, signalIndex, location);
        if (!settling) {
            return std::unexpected(settling.error());
        }
        const auto& settlingTimeOpt = *settling;
        if (settlingTimeOpt.has_value()) {
            metrics.isEstimate   = false;
            metrics.settlingTime = gr::cast<T>(settlingTimeOpt.value().index);
            metrics.V2           = gr::cast<T>(settlingTimeOpt.value().value);
        } else {
            metrics.settlingTime = std::numeric_limits<TValue>::quiet_NaN();
            metrics.V2           = firstRoughEdge.maxValue;
        }
    }

    // we have now a more precise V1 and V2 level available to minimise the bias of over- and undershoot

    // Step 4: compute bias-free overshoot

    metrics.overshoot = ((metrics.peakAmplitude) / (metrics.V2 - metrics.V1)) * TValue(100);

    // Step 5: compute bias-free trigger time and 10%->90% rise-time
    const T riseLevel   = isRising ? metrics.V1 + TValue(0.5) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.5) * (metrics.V1 - metrics.V2);
    const T firstEdge   = *estimators::getZeroCrossing<MetaInfo::Apply>(ds, riseLevel, indexMin, indexMax, signalIndex); // the range is checked above
    metrics.triggerTime = firstEdge;

    const T riseStart    = isRising ? metrics.V1 + TValue(0.1) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.1) * (metrics.V1 - metrics.V2);
    const T riseEnd      = isRising ? metrics.V1 + TValue(0.9) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.9) * (metrics.V1 - metrics.V2);
    const T firstLevel10 = *estimators::getZeroCrossing<MetaInfo::Apply>(ds, riseStart, indexMin, indexMax, signalIndex); // the range is checked above
    const T firstLevel90 = *estimators::getZeroCrossing<MetaInfo::Apply>(ds, riseEnd, indexMin, indexMax, signalIndex);   // the range is checked above
    metrics.riseTime     = firstLevel90 - firstLevel10;

    // Step 5: Determine if it's a pulse by checking for a falling or rising edge after the step
    const T lastValue = signalValues.back();
    metrics.isPulse   = gr::value(metrics.V2 - lastValue) > TValue(3) * roughMinThreshold;

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        ds.meta_information[signalIndex]["gr::detected"]       = metrics.isPulse ? "pulse" : "step";
        ds.meta_information[signalIndex]["gr::triggerTime"]    = gr::value(metrics.triggerTime);
        ds.meta_information[signalIndex]["gr::rise_time"]      = gr::value(metrics.riseTime);
        ds.meta_information[signalIndex]["gr::V1"]             = gr::value(metrics.V1);
        ds.meta_information[signalIndex]["gr::V2"]             = gr::value(metrics.V2);
        ds.meta_information[signalIndex]["gr::peak_amplitude"] = gr::value(metrics.peakAmplitude);
        ds.meta_information[signalIndex]["gr::overshoot"]      = gr::value(metrics.overshoot);
    }

    return metrics;
}

} // namespace estimators

} // namespace gr::dataset

#endif // DATASETESTIMATORS_HPP
