#include <boost/ut.hpp>
#include <format>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp> // for draw(...)
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

#include <gnuradio-4.0/algorithm/dataset/DataSetEstimators.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetHelper.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetMath.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetTestFunctions.hpp>

namespace test::detail {

template<class TLhs, class TRhs, class TEpsilon>
[[nodiscard]] constexpr auto approx(const TLhs& lhs, const TRhs& rhs, const TEpsilon& epsilon) {
    if constexpr (gr::meta::complex_like<TLhs>) {
        return boost::ut::detail::and_{boost::ut::detail::approx_{lhs.real(), rhs.real(), epsilon}, boost::ut::detail::approx_{lhs.imag(), rhs.imag(), epsilon}};
    } else {
        return boost::ut::detail::approx_{lhs, rhs, epsilon};
    }
}
} // namespace test::detail

const boost::ut::suite<"DataSet<T> visual test functions"> _DataSetTestFcuntions = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    constexpr static std::size_t nSamples = 201UZ;

    "triangular DataSet"_test = []<typename T = double> {
        gr::DataSet<T> ds = generate::from<T>("generic DataSet", std::vector{0, 1, 1, 2, 3, 5, 8, 13}).value();
        gr::dataset::draw(ds);
    };

    "triangular DataSet"_test = []<typename T = double> {
        gr::DataSet<T> ds = generate::triangular<T>("triagonal", nSamples);
        gr::dataset::draw(ds);

        std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(ds);
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;

        gr::DataSet<T> ds1 = generate::triangular<double>("triagonal - odd", 11);
        std::println("\"{:20}\": {}", ds1.signalName(0UZ), ds1.signal_values);
        expect(eq(ds1.signalValues().front(), ds1.signalValues().back()));
        expect(eq(ds1.signalValues()[5UZ], 1.0));

        gr::DataSet<T> ds2 = generate::triangular<double>("triagonal - even", 10);
        std::println("\"{:20}\": {}", ds2.signalName(0UZ), ds2.signal_values);
        expect(eq(ds2.signalValues().front(), ds2.signalValues().back()));
        expect(eq(ds2.signalValues()[4UZ], ds2.signalValues()[5UZ]));
    };

    "ramp DataSet"_test = []<typename T = double> {
        gr::DataSet<T>                 ds      = generate::ramp<T>("ramp", nSamples);
        std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(ds);
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
        gr::dataset::draw(ds);
    };

    "gaussFunction DataSet"_test = []<typename T = double> {
        constexpr T mean          = T(nSamples) / T(2);
        constexpr T sigma         = T(nSamples) / T(10);
        const T     normalisation = sigma * gr::math::sqrt(T(2) * std::numbers::pi_v<T>);

        gr::DataSet<T>                 ds      = generate::gaussFunction<T>("gaussFunction", nSamples, mean, sigma, T(0), normalisation).value();
        std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(ds);
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
        gr::dataset::draw(ds);
    };

    "ramp + gauss DataSet"_test = []<typename T = double> {
        using value_t             = gr::meta::fundamental_base_value_type_t<T>;
        constexpr T mean          = T(nSamples) / T(2);
        constexpr T sigma         = T(nSamples) / T(10);
        const T     normalisation = sigma * gr::math::sqrt(T(2) * std::numbers::pi_v<T>);

        gr::DataSet<T> ds1 = generate::ramp<T>("ramp", nSamples, value_t(0), value_t(0.2));
        gr::DataSet<T> ds2 = generate::gaussFunction<T>("gaussFunction", nSamples, mean, sigma, T(0), normalisation).value();
        gr::DataSet<T> ds  = addFunction(ds1, ds2);

        std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(ds);
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
        gr::dataset::draw(ds);
    };

    "randomStepFunction DataSet"_test = []<typename T = double> {
        gr::DataSet<T>                 ds      = generate::randomStepFunction<T>("randomStepFunction", nSamples).value();
        std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(ds);
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
        gr::dataset::draw(ds);
    };
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wimplicit-float-conversion" // GCC & Clang: Loss or promotion of floating-point precision, disabled only for unit-tests
#endif

const boost::ut::suite<"DataSet<T> element-wise accessor"> _dataSetAccessors = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    using test::detail::approx;

    "basic element-wise access API "_test = []<typename T> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds       = generate::triangular<T>("triag", 3UZ); // data ~ [0, 1, 0]

        expect(approx(getIndexValue(ds, dim::Y, 0).value(), value_t(0), value_t(1e-3f)));
        expect(approx(getIndexValue(ds, dim::Y, 1).value(), value_t(1), value_t(1e-3f)));
        expect(approx(getIndexValue(ds, dim::Y, 0).value(), value_t(0), value_t(1e-3f)));

        expect(!gr::math::isfinite(getIndexValue(ds, dim::X, 3).value())) << std::format("element is not NaN: {}", getIndexValue(ds, dim::X, 3).value());
        expect(!gr::math::isfinite(getIndexValue(ds, dim::Y, 3).value())) << std::format("element is not NaN: {}", getIndexValue(ds, dim::Y, 3).value());

        const auto noZ = getIndexValue(ds, dim::Z, 0);
        expect(fatal(!noZ.has_value()));
        expect(noZ.error().message.contains("dimIndex 2 is out of range")) << noZ.error().message;
        expect(!getValue(ds, dim::Z, value_t(0.5)).has_value()) << "getValue refuses dim::Z";
        expect(!getDistance(ds, dim::Z, 0UZ, 1UZ).has_value()) << "getDistance refuses dim::Z";
        const auto farRange = getDistance(ds, dim::X, 0UZ, 7UZ);
        expect(fatal(!farRange.has_value()));
        expect(farRange.error().message.contains("out of range")) << farRange.error().message;
        expect(!getSubArrayCopy(ds, 0UZ, 7UZ).has_value()) << "getSubArrayCopy refuses an index past the end";
        const auto meanPastEnd = estimators::getMean(ds, 0UZ, 7UZ);
        expect(fatal(!meanPastEnd.has_value())) << "an estimator returns the range check's error";
        expect(meanPastEnd.error().message.contains("out of range")) << meanPastEnd.error().message;

        expect(approx(getDistance(ds, dim::X, 0UZ, 2UZ).value(), value_t(2), value_t(1e-3f)));
        expect(approx(getDistance(ds, dim::X).value(), value_t(2), value_t(1e-3f)));

        expect(approx(getDistance(ds, dim::Y, 0UZ, 1UZ).value(), value_t(1), value_t(1e-3f))); // Y-distance of first to middle
        expect(approx(getDistance(ds, dim::Y).value(), T(0), value_t(1e-3f)));                 // Y-distance of first to last

        expect(approx(getValue(ds, dim::X, value_t(0.123f)).value(), static_cast<T>(0.123f), value_t(1e-3f))); // identity
        expect(approx(getValue(ds, dim::Y, value_t(0.5)).value(), static_cast<T>(0.5f), value_t(1e-3f)));

        std::vector<T> copyX = getSubArrayCopy(ds, dim::X, 0UZ, 2UZ).value();
        for (std::size_t i = 0; i < copyX.size(); i++) {
            expect(eq(copyX[i], getIndexValue(ds, dim::X, i).value())) << std::format("X-index {} mismatch", i);
        }

        std::vector<T> copyY = getSubArrayCopy(ds, dim::X, 0UZ, 2UZ).value();
        for (std::size_t i = 0; i < copyY.size(); i++) {
            expect(eq(copyY[i], getIndexValue(ds, dim::Y, i).value())) << std::format("X-index {} mismatch", i);
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
};

const boost::ut::suite<"DSP helper"> _dspHelper = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    using test::detail::approx;

    "common dsp helper functions"_test = []<typename T> {
        using value_t                     = T;
        const static std::string typeName = gr::meta::type_name<T>();

        "tenLog10"_test = [] {
            expect(approx(tenLog10(value_t(10)), value_t(10), value_t(1e-3f))) << "10 * log10(10) should be 10";
            expect(approx(tenLog10(value_t(1)), value_t(0), value_t(1e-3f))) << "10 * log10(1) should be 0";
            expect(approx(tenLog10(value_t(0.1)), value_t(-10), value_t(1e-3f))) << "10 * log10(0.1) should be -10";

            // a ratio at or below zero, or below the power floor, reads as the dB floor
            const value_t floorDb(gr::math::kDbFloor);
            expect(approx(tenLog10(value_t(0)), floorDb, value_t(1e-3f))) << std::format("tenLog10<{}>(0) = {} should be the dB floor", typeName, tenLog10(value_t(0)));
            expect(approx(tenLog10(value_t(-1)), floorDb, value_t(1e-3f))) << std::format("tenLog10<{}>(-1) = {} should be the dB floor", typeName, tenLog10(value_t(-1)));
            expect(approx(tenLog10(value_t(gr::math::kDbFloorPower / 10.)), floorDb, value_t(1e-3f))) << "a ratio below the floor reads as the floor";
            expect(approx(tenLog10(value_t(gr::math::kDbFloorPower)), floorDb, value_t(1e-2f))) << "the floor ratio itself is -300 dB";
        };

        "decibel"_test = [] {
            expect(approx(decibel(value_t(10)), value_t(20), value_t(1e-3f))) << "20 * log10(10) should be 20";
            expect(approx(decibel(value_t(1)), value_t(0), value_t(1e-3f))) << "20 * log10(1) should be 0";
            expect(approx(decibel(value_t(0.1)), value_t(-20), value_t(1e-3f))) << "20 * log10(0.1) should be -20";

            // an amplitude at or below zero, or below the amplitude floor, reads as the dB floor
            const value_t floorDb(gr::math::kDbFloor);
            expect(approx(decibel(value_t(0)), floorDb, value_t(1e-3f))) << "decibel(0) should be the dB floor";
            expect(approx(decibel(value_t(-1)), floorDb, value_t(1e-3f))) << "decibel(-1) should be the dB floor";
            expect(approx(decibel(value_t(gr::math::kDbFloorAmplitude / 10.)), floorDb, value_t(1e-3f))) << "an amplitude below the floor reads as the floor";
            expect(approx(decibel(value_t(gr::math::kDbFloorAmplitude)), floorDb, value_t(1e-2f))) << "the floor amplitude itself is -300 dB";
        };

        "inverseDecibel"_test = [] {
            expect(approx(inverseDecibel(value_t(20)), value_t(10), value_t(1e-3f))) << "10^(20 / 20) should be 10";
            expect(approx(inverseDecibel(value_t(0)), value_t(1), value_t(1e-3f))) << "10^(0 / 20) should be 1";
            expect(approx(inverseDecibel(value_t(-20)), value_t(0.1), value_t(1e-3f))) << "10^(-20 / 20) should be 0.1";

            // edge cases
            expect(approx(inverseDecibel(value_t(100)), gr::math::pow(value_t(10), value_t(5)), value_t(1e-3f))) << "10^(100 / 20) should be 10^5";
            expect(approx(inverseDecibel(value_t(-100)), gr::math::pow(value_t(10), value_t(-5)), value_t(1e-3f))) << "10^(-100 / 20) should be 10^-5";
        };
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
};

const boost::ut::suite<"DataSet<T> estimator"> _qaDataSetEstimators = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    using test::detail::approx;

    constexpr static size_t nSamples = 11;

    "basic estimators"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        // 10-sample triangle => ascending up to i=4..5 => data ~ [0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0]
        auto ds = generate::triangular<T>("triag", nSamples);

        expect(approx(estimators::computeCentreOfMass(ds).value(), T(5), T(1e-3)));
        expect(gr::math::isfinite(estimators::computeCentreOfMass(ds, 0UZ, nSamples).value())) << "should be finite for partial range";

        std::vector<T> data{1, 2, 3, 2, 1};
        expect(approx(estimators::computeFWHM(data, 2), T(4), T(1e-5)));
        expect(approx(estimators::computeInterpolatedFWHM(data, 2), T(3), T(1e-5)));

        // the triangle is zero at both ends, so the minimum's index is the first of the two, as the
        // maximum's is the first of two equal peaks
        expect(eq(estimators::getMaximum(ds, 0UZ, nSamples).value().value().index, 5UZ));
        expect(eq(estimators::getMinimum(ds, 0UZ, nSamples).value().value().index, 0UZ));
        expect(eq(estimators::getMaximum(ds).value().value().index, 5UZ));
        expect(eq(estimators::getMinimum(ds).value().value().index, 0UZ));

        expect(eq(gr::value(estimators::getMaximum(ds, 0UZ, nSamples).value().value().value), value_t(1)));
        expect(eq(gr::value(estimators::getMaximum(ds).value().value().value), value_t(1)));
        expect(eq(gr::value(estimators::getMinimum(ds, 0UZ, nSamples).value().value().value), value_t(0)));
        expect(eq(gr::value(estimators::getMinimum(ds).value().value().value), value_t(0)));

        expect(approx(estimators::getMean(ds, 0UZ, nSamples).value(), T(0.454545), T(1e-3f)));
        expect(approx(estimators::getMean(ds).value(), T(0.454545), T(1e-3f)));

        expect(approx(estimators::getMedian(ds, 0UZ, nSamples).value(), T(0.4), T(1e-3f)));
        expect(approx(estimators::getMedian(ds).value(), T(0.4), T(1e-3f)));

        expect(eq(gr::value(estimators::getRange(ds, 0UZ, nSamples).value()), value_t(1)));
        expect(eq(gr::value(estimators::getRange(ds).value()), value_t(1)));

        expect(approx(estimators::getRms(ds, 0UZ, nSamples).value(), T(0.320124), T(1e-3)));
        expect(approx(estimators::getRms(ds).value(), T(0.320124), T(1e-3)));
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getRms(ds, 0UZ, nSamples).value()), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getRms(ds).value()), value_t(0)));
        }

        expect(approx(estimators::getIntegral(ds, 0UZ, nSamples).value(), T(5.0), T(1e-3)));
        expect(approx(estimators::getIntegral(ds).value(), T(5.0), T(1e-3)));
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getIntegral(ds, 0UZ, nSamples).value()), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getIntegral(ds).value()), value_t(0)));
        }

        expect(approx(estimators::getEdgeDetect(ds, 0UZ, nSamples).value(), T(3), T(0.5))) << "50% is ~ 0.5 => crossing near i=3";
        expect(approx(estimators::getEdgeDetect(ds).value(), T(3), T(0.5))) << "50% is ~ 0.5 => crossing near i=3";
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
    ;

    // A record whose minimum is neither at an end nor at the last sample: the index has to come from the
    // comparison. A triangle, which is what the estimators are otherwise read with, cannot tell the two
    // apart, its minimum sitting at both ends.
    "the minimum is reported where the minimum is"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds       = generate::from<T>("interior minimum", std::vector<value_t>{5, 2, -3, 4, 1}).value();

        const auto minimum = estimators::getMinimum(ds).value();
        expect(eq(minimum.value().index, 2UZ));
        expect(eq(gr::value(minimum.value().value), value_t(-3)));
        expect(eq(ds.timing_events[0UZ].back().first, std::ptrdiff_t(2))) << "the annotation marks the sample the minimum was found at";

        expect(eq(estimators::getMaximum(ds).value().value().index, 0UZ));
        expect(eq(gr::value(estimators::getMaximum(ds).value().value().value), value_t(5)));

        expect(eq(estimators::getMinimum(ds, 3UZ, 5UZ).value().value().index, 4UZ)) << "a range that excludes the minimum reports the range's own";
        expect(eq(gr::value(estimators::getMinimum(ds, 3UZ, 5UZ).value().value().value), value_t(1)));

        auto withGaps = generate::from<T>("infinite tail", std::vector<value_t>{4, -2, 3, std::numeric_limits<value_t>::quiet_NaN(), std::numeric_limits<value_t>::infinity()}).value();
        expect(eq(estimators::getMinimum(withGaps).value().value().index, 1UZ)) << "the last finite sample is not the minimum";
        expect(eq(gr::value(estimators::getMinimum(withGaps).value().value().value), value_t(-2)));
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    // A peak whose left flank is still above half maximum at the first sample: the descending scan has
    // to stop there, and the width is as undefined as it is when the ascending scan reaches the last.
    "a flank that never falls to half maximum has no width"_test = []<typename T = double> {
        const std::vector<T> leftEdge{T(10), T(9), T(4), T(1)};
        expect(!gr::math::isfinite(estimators::computeFWHM(leftEdge, 1UZ))) << "the left flank runs off the front";
        expect(!gr::math::isfinite(estimators::computeInterpolatedFWHM(leftEdge, 1UZ)));

        const std::vector<T> rightEdge{T(1), T(4), T(9), T(10)};
        expect(!gr::math::isfinite(estimators::computeFWHM(rightEdge, 2UZ))) << "and the right flank off the back";

        const std::vector<T> peak{T(0), T(1), T(10), T(1), T(0)};
        expect(approx(estimators::computeFWHM(peak, 2UZ), T(2), T(1e-5))) << "a peak with both flanks inside the record still reads";
    } | std::tuple<float, double>{};

    // the half that a median or a trapezoid takes lives in the accumulator, so an integral sample type keeps it up to the return
    "median and integral of an integral sample type"_test = [] {
        static_assert(std::is_same_v<estimators::PromotedAccumulator<int>, double>);
        static_assert(std::is_same_v<estimators::PromotedAccumulator<float>, float>);
        static_assert(std::is_same_v<estimators::PromotedAccumulator<gr::UncertainValue<float>>, gr::UncertainValue<float>>);

        const gr::DataSet<int> odd = generate::from<int>("int odd", std::vector<int>{1, 5, 2}).value();
        expect(eq(estimators::getMedian(odd).value(), 2)) << std::format("median of an odd count: {}", estimators::getMedian(odd).value());

        const gr::DataSet<int> even = generate::from<int>("int even", std::vector<int>{1, 2, 4, 10}).value();
        expect(eq(estimators::getMedian(even).value(), 3)) << std::format("median of an even count: {}", estimators::getMedian(even).value());

        const gr::DataSet<int> half = generate::from<int>("int half", std::vector<int>{1, 2, 3, 4}).value();
        expect(eq(estimators::getMedian(half).value(), 3)) << "a median of 2.5 takes the nearest sample value, away from zero";

        const gr::DataSet<int> negative = generate::from<int>("int negative", std::vector<int>{-4, -3, -2, -1}).value();
        expect(eq(estimators::getMedian(negative).value(), -3)) << "a median of -2.5 takes the nearest sample value, away from zero";

        const gr::DataSet<unsigned> unsignedEven = generate::from<unsigned>("unsigned even", std::vector<unsigned>{10U, 20U, 30U, 40U}).value();
        expect(eq(estimators::getMedian(unsignedEven).value(), 25U)) << std::format("median of an unsigned even count: {}", estimators::getMedian(unsignedEven).value());

        const gr::DataSet<int> ramp = generate::from<int>("int ramp", std::vector<int>{0, 2, 4, 6}).value();
        expect(eq(estimators::getIntegral(ramp).value(), 9)) << std::format("trapezoids of 1, 3 and 5 over a unit index axis: {}", estimators::getIntegral(ramp).value());

        const gr::DataSet<int> halfRamp = generate::from<int>("int half ramp", std::vector<int>{0, 1, 2, 3}).value();
        expect(eq(estimators::getIntegral(halfRamp).value(), 5)) << "an integral of 4.5 takes the nearest sample value, away from zero";

        const gr::DataSet<unsigned> unsignedRamp = generate::from<unsigned>("unsigned ramp", std::vector<unsigned>{1U, 2U, 3U}).value();
        expect(eq(estimators::getIntegral(unsignedRamp).value(), 4U)) << std::format("trapezoids of 1.5 and 2.5 over a unit index axis: {}", estimators::getIntegral(unsignedRamp).value());
    };

    // a large DC offset must not swamp the sample spread: a 1 V sine riding on 1e6, accumulated in float
    "mean and standard deviation with a large offset"_test = [] {
        constexpr std::size_t nPoints   = 100000UZ;
        constexpr double      offset    = 1.e6;
        constexpr double      amplitude = 1.;
        constexpr float       expected  = static_cast<float>(amplitude / std::numbers::sqrt2);

        std::vector<float> values(nPoints);
        for (std::size_t i = 0UZ; i < nPoints; ++i) {
            values[i] = static_cast<float>(offset + amplitude * std::sin(2. * std::numbers::pi * 8. * static_cast<double>(i) / static_cast<double>(nPoints)));
        }
        const gr::DataSet<float> ds = generate::from<float>("offset sine", values).value();

        expect(approx(estimators::getMean(ds).value(), static_cast<float>(offset), 1.e-2f)) << std::format("mean: {}", estimators::getMean(ds).value());
        expect(approx(estimators::getStdDev(ds).value(), expected, 0.01f * expected)) << std::format("standard deviation: {}", estimators::getStdDev(ds).value());
        expect(eq(estimators::getRms(ds).value(), estimators::getStdDev(ds).value())) << "getRms delegates to getStdDev";
    };

    // an integral sample type is accumulated above itself, so neither the running division nor a descending unsigned update loses the answer
    "mean and standard deviation of an integral sample type"_test = [] {
        const gr::DataSet<int> ascending = generate::from<int>("int ascending", std::vector<int>{1, 2, 3}).value();
        expect(eq(estimators::getMean(ascending).value(), 2)) << std::format("mean: {}", estimators::getMean(ascending).value());
        expect(eq(estimators::getStdDev(ascending).value(), 0)) << std::format("standard deviation: {}", estimators::getStdDev(ascending).value());

        const gr::DataSet<unsigned> descending = generate::from<unsigned>("unsigned descending", std::vector<unsigned>{30U, 20U, 10U}).value();
        expect(eq(estimators::getMean(descending).value(), 20U)) << std::format("mean: {}", estimators::getMean(descending).value());
        expect(eq(estimators::getStdDev(descending).value(), 8U)) << std::format("standard deviation: {}", estimators::getStdDev(descending).value());
    };

    "getDutyCycle"_test = []<typename T = double> {
        using value_t         = gr::meta::fundamental_base_value_type_t<T>;
        std::vector<T> localY = {0, 0, 0, 1, 1, 1}; // simple data set with 0,0,0,1,1,1 => 50% high
        gr::DataSet<T> ds;
        ds.axis_names = {"Time"};
        ds.axis_units = {"s"};
        ds.axis_values.resize(1);
        ds.axis_values[0].resize(localY.size());
        ds.signal_names = {"simple step"};
        ds.signal_values.resize(localY.size());
        ds.extents = {1, static_cast<std::int32_t>(localY.size())};
        for (std::size_t i = 0; i < localY.size(); i++) {
            ds.axis_values[0][i] = T(i);
            ds.signal_values[i]  = localY[i];
        }
        ds.signal_ranges.push_back({T(0), T(1)});

        expect(eq(estimators::getDutyCycle(ds, 0UZ, localY.size()).value(), value_t(0.5))) << "simple 3-high/3-low => duty=0.5";
        expect(eq(estimators::getDutyCycle(ds).value(), value_t(0.5))) << "simple 3-high/3-low => duty=0.5";
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "getFrequencyEstimate"_test = []<typename T = double> {
        using value_t         = gr::meta::fundamental_base_value_type_t<T>;
        std::vector<T> localY = {0, 1, 0, 1, 0, 1};
        gr::DataSet<T> ds;
        ds.axis_names = {"Time"};
        ds.axis_units = {"s"};
        ds.axis_values.resize(1);
        ds.axis_values[0].resize(localY.size());
        ds.signal_names = {"oscillator"};
        ds.signal_values.resize(localY.size());
        ds.extents = {1, int(localY.size())};
        for (std::size_t i = 0; i < localY.size(); i++) {
            ds.axis_values[0][i] = T(i);
            ds.signal_values[i]  = localY[i];
        }
        ds.signal_ranges.push_back({T(0), T(1)});

        expect(eq(estimators::getFrequencyEstimate(ds, 0UZ, localY.size()).value(), value_t(0.5))) << "frequency ~ 0.5";
        expect(eq(estimators::getFrequencyEstimate(ds).value(), value_t(0.5))) << "frequency ~ 0.5";
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "getLocationMaximumGaussInterpolated"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds1      = generate::triangular<T>("triag", 7UZ); // symmetric peak around index 3

        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds1, 0UZ, 7UZ).value()), value_t(3), T(1e-3f))) << "Gauss interpolation ~3";
        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds1).value()), value_t(3), T(1e-3f))) << "Gauss interpolation ~3";
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds1, 0UZ, 7UZ).value()), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds1).value()), value_t(0)));
        }

        auto ds2 = generate::triangular<T>("triag", 6UZ); // symmetric peak around index 2.5
        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds2, 0UZ, 6UZ).value()), value_t(2.5), T(1e-3f))) << "Gauss interpolation ~2.5";
        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds2).value()), value_t(2.5), T(1e-3f))) << "Gauss interpolation ~2.5";
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds2, 0UZ, 6UZ).value()), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds2).value()), value_t(0)));
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "zeroCrossing test"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds       = generate::triangular<T>("ZC", nSamples);

        expect(approx(gr::value(estimators::getZeroCrossing(ds, value_t(0.5)).value()), value_t(2.5), value_t(1e-3))) << "zero crossing ~2.5";
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getZeroCrossing(ds, value_t(0.5)).value()), value_t(0)));
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
};

const boost::ut::suite<"DataSet<T> math "> _dataSetMath = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using namespace gr::dataset;
    using test::detail::approx;

    "basic math API "_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds1      = generate::triangular<T>("ds1", 5);
        auto ds2      = addFunction(ds1, value_t(2)); // add scalar '2'

        for (std::size_t i = 0; i < 5; i++) {
            T y1 = gr::dataset::getIndexValue(ds1, dim::Y, i).value();
            T y2 = gr::dataset::getIndexValue(ds2, dim::Y, i).value();
            expect(approx(y2, y1 + T(2), value_t(1e-3f)));
        }

        // add two dataSets
        auto ds3 = addFunction(ds1, ds1);
        for (std::size_t i = 0; i < 5; i++) {
            T y1 = gr::dataset::getIndexValue(ds1, 1, i).value();
            T y3 = gr::dataset::getIndexValue(ds3, 1, i).value();
            expect(approx(y3, value_t(2) * y1, value_t(1e-3f)));
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "mathFunction variants"_test = []<typename T = double> {
        auto ds1 = generate::triangular<T>("triag", 5);

        // 1) dataSet + dataSet
        auto dsAdd = mathFunction(ds1, ds1, MathOp::ADD);
        for (std::size_t i = 0UZ; i < 5UZ; i++) {
            T origVal = ds1.signal_values[i];
            T newVal  = dsAdd.signal_values[i];
            expect(eq(newVal, T(2) * origVal)) << std::format("Add op failed at i={}, got {}", i, newVal);
        }

        // 2) dataSet + double
        auto dsAdd2 = mathFunction(ds1, T(2), MathOp::ADD);
        for (std::size_t i = 0UZ; i < 5UZ; i++) {
            T origVal = ds1.signal_values[i];
            T newVal  = dsAdd2.signal_values[i];
            expect(eq(newVal, origVal + T(2)));
        }

        // 3) MULTIPLY
        auto dsMul = mathFunction(ds1, ds1, MathOp::MULTIPLY);
        for (std::size_t i = 0UZ; i < 5UZ; i++) {
            T origVal = ds1.signal_values[i];
            T newVal  = dsMul.signal_values[i];
            expect(approx(newVal, origVal * origVal, T(1e-5))) << "Multiply test";
        }

        // 4) SQR => (y1 + y2)^2 => ds1 + 2 => then squared
        auto dsSqr = mathFunction(ds1, T(2), MathOp::SQR);
        // check a single sample
        expect(eq(dsSqr.signal_values[0], T((ds1.signal_values[0] + T(2)) * (ds1.signal_values[0] + T(2)))));

        // 5) SQRT => sqrt(y1 + y2)
        auto dsSqrt = mathFunction(ds1, T(2.0), MathOp::SQRT);
        // if ds1.signal_values[0]=0 => sqrt(2) => ~1.4142
        expect(approx(dsSqrt.signal_values[0], T(1.4142f), T(1e-2)));

        // 6) LOG10 => 10*log10(y1 + y2)
        auto dsLog = mathFunction(ds1, T(2.0), MathOp::LOG10);
        // we can do a quick check for i=0 => 10 * log10(2.0) => 3.0103 dB
        expect(approx(dsLog.signal_values[0], T(3.01f), T(1e-2)));

        // 7) DB => 20*log10(y1 + y2)
        auto dsDb = mathFunction(ds1, T(2.0), MathOp::DB);
        // i=0 => 20*log10(2.0) => 6.0206 dB
        expect(approx(dsDb.signal_values[0], T(6.02f), T(1e-2)));

        // 8) INV_DB => 10^(y1/20), ignoring y2
        // if ds1.signal_values[0]=0 => => 10^(0/20)=>1
        auto dsInv = mathFunction(ds1, 2.0, MathOp::INV_DB);
        expect(eq(dsInv.signal_values[0], T(1))) << "inv_db test";
    };

    "computeDerivative"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;

        "ramp"_test = [] {
            auto ds = generate::ramp<T>("ramp", 5, value_t(0.), value_t(1.));

            auto           derivative = computeDerivative(ds).value();
            std::vector<T> expected   = {0.2, 0.2, 0.2, 0.2};

            for (std::size_t i = 0; i < expected.size(); ++i) {
                T val = derivative[i];
                expect(approx(val, expected[i], T(1e-3))) << std::format("Derivative at index {}: expected {}, got {}", i, expected[i], val);
            }
        };

        "step"_test = [] {
            auto ds_step         = generate::randomStepFunction<T>("step", 6, 3).value(); // [0, 0, 0, 1, 1, 1]
            auto derivative_step = computeDerivative(ds_step).value();

            std::vector<T> expected_step = {0.0, 0.0, 1.0, 0.0, 0.0};
            for (std::size_t i = 0; i < expected_step.size(); ++i) {
                T val = derivative_step[i];
                expect(approx(val, expected_step[i], T(1e-3))) << std::format("Step Derivative at index {}: expected {}, got {}", i, expected_step[i], val);
            }
        };
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "filter DataSet"_test = []<typename T = double> {
        using value_t             = gr::meta::fundamental_base_value_type_t<T>;
        auto ds_step              = generate::stepFunction<T>("step", 200, 25).value();
        auto responseCoefficients = gr::filter::iir::designResonatorPhysical(value_t(1), value_t(0.1), value_t(0.5)).value();
        auto filtered             = gr::dataset::filter::applyFilter(ds_step, responseCoefficients);
        if constexpr (std::is_same_v<T, float>) {
            gr::dataset::draw(ds_step, DefaultChartConfig{});
            gr::dataset::draw(filtered, DefaultChartConfig{.reset_view = gr::graphs::ResetChartView::RESET});
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "detectStepStart"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        // Create a step signal with a clear step at index 3
        auto ds_step         = generate::stepFunction<T>("step", 6, 3).value(); // [0, 0, 0, 1, 1, 1]
        auto detectionResult = estimators::detectStepStart(ds_step).value();

        expect(detectionResult.has_value()) << "step should be detected.";
        if (detectionResult.has_value()) {
            const auto& result = detectionResult.value();
            expect(eq(result.index, 3UZ)) << "detected step start";
            expect(eq(gr::value(result.initialValue), value_t(0.0))) << "initial value";
            expect(eq(gr::value(result.minValue), value_t(0.0))) << "min value";
            expect(eq(gr::value(result.maxValue), value_t(1.0))) << "max value";
            expect(result.isRising) << "step should be rising.";
        }

        auto ds_noisy_step = generate::randomStepFunction<T>("noisy_step", 6, 3).value(); // [0, 0, 0, 1, 1, 1]
        ds_noisy_step.signal_values[3] += value_t(0.1);                                   // slight overshoot

        auto detectionNoisyStep = estimators::detectStepStart(ds_noisy_step).value();
        expect(detectionNoisyStep.has_value()) << "noisy step should still be detected.";
        if (detectionNoisyStep.has_value()) {
            const auto& result = detectionNoisyStep.value();
            expect(eq(result.index, 3UZ)) << "detected step start";
            expect(eq(gr::value(result.initialValue), value_t(0.0))) << "initial value";
            expect(eq(gr::value(result.minValue), value_t(0.0))) << "min value";
            expect(eq(gr::value(result.maxValue), value_t(1.1))) << "max value";
            expect(result.isRising) << "step should be rising.";
        }

        auto ds_falling_step = generate::randomStepFunction<T>("falling_step", 6, 3).value(); // [0, 0, 0, 1, 1, 1]
        for (auto& val : ds_falling_step.signal_values) {
            val = value_t(1.0) - val;
        }
        auto detectionFallingStep = estimators::detectStepStart(ds_falling_step).value();
        expect(detectionFallingStep.has_value()) << "falling step should be detected.";
        if (detectionFallingStep.has_value()) {
            const auto& result = detectionFallingStep.value();
            expect(eq(result.index, 3UZ)) << "detected step start";
            expect(eq(gr::value(result.initialValue), T(1.0))) << "initial value";
            expect(eq(gr::value(result.minValue), T(0.0))) << "min value";
            expect(eq(gr::value(result.maxValue), T(1.0))) << "max value";
            expect(!result.isRising) << "step should be falling.";
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "analyzeStepPulseResponse - Step"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto response = gr::filter::iir::designResonatorPhysical(value_t(1), value_t(0.1), value_t(0.5)).value();
        auto ds_step  = gr::dataset::filter::applyFilter(generate::stepFunction<T>("step", 100, 20).value(), response);
        auto metrics  = estimators::analyzeStepPulseResponse(ds_step).value();

        if constexpr (std::is_same_v<T, float>) {
            gr::dataset::draw(ds_step, DefaultChartConfig{});
        }

        "basic metrics"_test = [&] {
            expect(!metrics.isPulse);
            expect(approx(metrics.V1, T(0.0), T(1e-1))) << "V1 initial level";
            expect(approx(metrics.V2, T(1.0), T(1e-1))) << "V2 flat-top level";
            expect(approx(metrics.triggerTime, T(20.4), T(1e-1)));
        };

        "rising-edge metrics"_test = [&] {
            expect(approx(metrics.riseTime, T(2.1), T(1e-1)));
            expect(approx(metrics.peakAmplitude, T(1.21), T(1e-2)));
            expect(approx(metrics.peakTime, T(23.0), T(2)));
            expect(approx(metrics.overshoot, T(123.0), T(1)));
            expect(approx(metrics.settlingTime, T(31.0), T(1e-1)));
        };
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "analyzeStepPulseResponse - Pulse"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;

        auto ds_pulse_proto = generate::stepFunction<T>("pulse", 100, 20).value();
        for (std::size_t i = 50; i < 70; ++i) {
            ds_pulse_proto.signal_values[i] = value_t(1.0) - value_t(0.01) * value_t(i - 50); // gradual decrease at end-of-flat top
        }
        for (std::size_t i = 70; i < 100; ++i) {
            ds_pulse_proto.signal_values[i] = 0.0; // final level
        }

        auto response = gr::filter::iir::designResonatorPhysical(value_t(1), value_t(0.1), value_t(0.5)).value();
        auto ds_pulse = gr::dataset::filter::applyFilter(ds_pulse_proto, response);
        auto metrics  = estimators::analyzeStepPulseResponse(ds_pulse).value();

        if constexpr (std::is_same_v<T, float>) {
            gr::dataset::draw(ds_pulse, DefaultChartConfig{});
        }

        "basic metrics"_test = [&] {
            expect(metrics.isPulse);
            expect(approx(metrics.V1, T(0.0), T(1e-1))) << "V1 initial level";
            expect(approx(metrics.V2, T(1.0), T(1e-1))) << "V2 flat-top level";
            expect(approx(metrics.triggerTime, T(20.4), T(1e-1)));
        };

        "rising-edge metrics"_test = [&] {
            expect(approx(metrics.riseTime, T(2.1), T(1e-1)));
            expect(approx(metrics.peakAmplitude, T(1.21), T(1e-2)));
            expect(approx(metrics.peakTime, T(23.0), T(2)));
            expect(approx(metrics.overshoot, T(123.0), T(1)));
            expect(approx(metrics.settlingTime, T(31.0), T(1e-1)));
        };
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "generator refusals"_test = []<typename T = double> {
        const auto empty = generate::from<T>("empty", std::vector<T>{});
        expect(fatal(!empty.has_value()));
        expect(empty.error().message.contains("must not be empty")) << empty.error().message;
        const auto fewPoints = generate::gaussFunction<T>("gauss", 2UZ, T(0));
        expect(fatal(!fewPoints.has_value()));
        expect(fewPoints.error().message.contains("greater than 2")) << fewPoints.error().message;
        const auto flatGauss = generate::gaussFunction<T>("gauss", 8UZ, T(0), T(0));
        expect(fatal(!flatGauss.has_value()));
        expect(flatGauss.error().message.contains("Sigma must be positive")) << flatGauss.error().message;
        const auto noStep = generate::stepFunction<T>("step", 0UZ);
        expect(fatal(!noStep.has_value()));
        expect(noStep.error().message.contains("greater than 0")) << noStep.error().message;
        const auto noRandomStep = generate::randomStepFunction<T>("step", 0UZ);
        expect(fatal(!noRandomStep.has_value()));
        expect(noRandomStep.error().message.contains("greater than 0")) << noRandomStep.error().message;
    };

    "step analysis refusals"_test = []<typename T = double> {
        auto       single   = generate::stepFunction<T>("single", 1UZ).value();
        const auto tooShort = estimators::analyzeStepPulseResponse(single);
        expect(fatal(!tooShort.has_value()));
        expect(tooShort.error().message.contains("Not enough data points")) << tooShort.error().message;

        auto       flat   = generate::stepFunction<T>("flat", 8UZ, 20U).value(); // the step lies past the end, so every sample is zero
        const auto noStep = estimators::analyzeStepPulseResponse(flat);
        expect(fatal(!noStep.has_value()));
        expect(noStep.error().message.contains("no step detected")) << noStep.error().message;

        auto inconsistent = generate::stepFunction<T>("inconsistent", 8UZ).value();
        inconsistent.signal_values.pop_back();
        const auto invalid = estimators::detectStepStart(inconsistent);
        expect(fatal(!invalid.has_value()));
        expect(invalid.error().message.contains("Invalid DataSet")) << invalid.error().message;
    };

    "analyzeStepPulseResponse - dampled and noisy Step"_test = []<typename T = double> {
        using value_t                = gr::meta::fundamental_base_value_type_t<T>;
        constexpr value_t noiseLevel = value_t(0.05);

        auto response = gr::filter::iir::designFilter<T, 0UZ>(gr::filter::Type::LOWPASS, {.order = 1UZ, .fLow = 0.05, .fs = 1.0}).value();
        auto val      = gr::dataset::filter::applyFilter<ProcessMode::InPlace>(generate::stepFunction<T>("damped step", 100, 20).value(), response);
        auto ds_step  = gr::dataset::addNoise<ProcessMode::InPlace>(val, noiseLevel, 0UZ, 42U).value();
        auto metrics  = estimators::analyzeStepPulseResponse(ds_step).value();

        if constexpr (std::is_same_v<T, float>) {
            gr::dataset::draw(ds_step, DefaultChartConfig{});
        }

        "basic metrics"_test = [&] {
            expect(!metrics.isPulse);
            expect(approx(metrics.V1, T(0.0), T(noiseLevel))) << "V1 initial level";
            expect(approx(metrics.V2, T(1.0), T(noiseLevel))) << "V2 flat-top level";
            expect(approx(metrics.triggerTime, T(21.5), T(0.5)));
        };

        "rising-edge metrics"_test = [&] {
            expect(approx(metrics.riseTime, T(7), T(1)));
            expect(approx(metrics.peakAmplitude, T(1.0), T(2 * noiseLevel)));
            // The noisy response is sensitive to compiler / platform differences, so keep this broad.
            expect(approx(metrics.peakTime, T(33.0), T(6)));
            expect(approx(metrics.overshoot, T(100.0), T(2 * noiseLevel * 100)));
            expect(metrics.isEstimate || gr::math::isfinite(metrics.settlingTime));
        };
    } | std::tuple<float /*, double, gr::UncertainValue<float>, gr::UncertainValue<double>*/>{};

    // WIP -- add more DataSet<T> math-related functionality here
};

const boost::ut::suite<"DataSet<T> filter"> _dataSetFilter = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    using test::detail::approx;

    "applyMovingAverage"_test = []<typename T> {
        using value_t     = gr::meta::fundamental_base_value_type_t<T>;
        gr::DataSet<T> ds = generate::ramp<T>("ramp", 5, value_t(0), value_t(1)); // [0, 0.2, 0.4, 0.6, 0.8]

        auto smoothed = filter::applyMovingAverage(ds, 3UZ).value();

        std::vector<T> expected = {value_t(0.1), value_t(0.2), value_t(0.4), value_t(0.6), value_t(0.7)};
        for (std::size_t i = 0; i < expected.size(); ++i) {
            T val = smoothed.signal_values[i];
            expect(approx(val, expected[i], T(1e-3))) << std::format("smoothed value at index {}: expected {}, got {}", i, expected[i], val);
        }

        const auto evenWindow = filter::applyMovingAverage(ds, 4UZ);
        expect(fatal(!evenWindow.has_value()));
        expect(evenWindow.error().message.contains("positive odd number")) << evenWindow.error().message;
        expect(!filter::applyMedian(ds, 0UZ).has_value()) << "applyMedian refuses an empty window";
        expect(!filter::applyRms(ds, 0UZ).has_value()) << "applyRms refuses an empty window";
        expect(!filter::applyPeakToPeak(ds, 0UZ).has_value()) << "applyPeakToPeak refuses an empty window";
        const auto negativeNoise = addNoise(ds, value_t(-1));
        expect(fatal(!negativeNoise.has_value()));
        expect(negativeNoise.error().message.contains("must be a positive number")) << negativeNoise.error().message;
        const auto oneSample = computeDerivative(generate::ramp<T>("one", 1, value_t(0), value_t(1)));
        expect(fatal(!oneSample.has_value()));
        expect(oneSample.error().message.contains("at least two samples")) << oneSample.error().message;
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "applyMedian"_test = []<typename T>() {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;

        auto ds1 = filter::applyMedian<ProcessMode::Copy>(generate::from<T>("median_test", std::vector<value_t>{1, 5, 3, 2, 0, 8}).value(), 3UZ).value();
        expect(eq(ds1.signal_values[1], T(3))); // median of {1,5,3} is 3
        expect(eq(ds1.signal_values[2], T(3))); // median of {5,3,2} is 3

        auto ds2 = filter::applyMedian<ProcessMode::InPlace>(generate::from<T>("median_test", std::vector<value_t>{3, 3, 4, 3, 3, 4, 3, 3}).value(), 3UZ).value();
        for (std::size_t i = 0UZ; i < ds2.signal_values.size(); ++i) {
            expect(eq(ds2.signal_values[i], T(3))) << "filter outlier around 3";
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "applyRms"_test = []<typename T>() {
        using value_t            = gr::meta::fundamental_base_value_type_t<T>;
        auto           ds        = generate::from<T>("rms_test", std::vector<value_t>{0, 10, 10, 10, 0}).value();
        gr::DataSet<T> ds_median = filter::applyRms<ProcessMode::Copy>(ds, 3UZ).value();
        expect(eq(ds_median.signal_values[2], T(0))) << "RMS with identical points => 0 stdev (around mean).";
        expect(eq(ds_median.signal_values[2], estimators::getRms(ds, 1UZ, 3UZ).value())) << "RMS with identical points => 0 stdev (around mean).";
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "applyPeakToPeak"_test = []<typename T>() {
        auto ds = filter::applyPeakToPeak(generate::from<T>("p2p_test", std::vector<T>{1, 2, 3, 5, 5, 0}).value(), 3UZ).value();

        expect(eq(ds.signal_values[0], T(1))); // window is {1, 2}    min=1, max=2 => range = 1
        expect(eq(ds.signal_values[1], T(2))); // window is {1, 2, 3} min=1, max=3 => range = 2
        expect(eq(ds.signal_values[2], T(3))); // window is {2, 3, 5} min=2, max=5 => range = 3
        expect(eq(ds.signal_values[3], T(2)));
        expect(eq(ds.signal_values[4], T(5)));
        expect(eq(ds.signal_values[5], T(5))); // window is {5, 0}    min=0, max=5 => ramge = 5
    } | std::tuple<float, double>{};

    "applySymmetricFilter"_test = []<typename T>() {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        // The zero-phase form is the forward pass and the backward pass averaged. A two-tap mean over a
        // leading impulse reads {0.5, 0.5, 0, 0} forwards and {0.5, 0, 0, 0} backwards, so the record
        // comes back as the mean of the two.
        const gr::filter::FilterCoefficients<value_t> twoTap{.b = {value_t(0.5), value_t(0.5)}, .a = {value_t(1)}};

        auto ds = filter::applySymmetricFilter(generate::from<T>("impulse", std::vector<value_t>{1, 0, 0, 0}).value(), twoTap);

        const std::vector<value_t> expected{value_t(0.5), value_t(0.25), value_t(0), value_t(0)};
        for (std::size_t i = 0UZ; i < expected.size(); ++i) {
            expect(approx(gr::value(ds.signal_values[i]), expected[i], value_t(1e-6))) << std::format("zero-phase filter at index {}", i);
        }
    } | std::tuple<float, double>{};

    "ProcessMode::InPlace works on the record it is given"_test = []<typename T>() {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        // In place means the record that comes back is the record that went in: its sample buffer is the
        // one the caller handed over, not a copy of it.
        auto movesTheBuffer = [](auto&& op) {
            auto       ds     = generate::from<T>("in place", std::vector<value_t>{3, 3, 4, 3, 3, 4, 3, 3}).value();
            const auto buffer = ds.signal_values.data();
            auto       out    = op(std::move(ds));
            return out.signal_values.data() == buffer;
        };
        expect(that % movesTheBuffer([](auto&& in) { return gr::dataset::addNoise<ProcessMode::InPlace>(std::move(in), value_t(0.1), 0UZ, 42U).value(); })) << "addNoise";
        expect(that % movesTheBuffer([](auto&& in) { return filter::applyMovingAverage<ProcessMode::InPlace>(std::move(in), 3UZ).value(); })) << "applyMovingAverage";
        expect(that % movesTheBuffer([](auto&& in) { return filter::applyMedian<ProcessMode::InPlace>(std::move(in), 3UZ).value(); })) << "applyMedian";
        expect(that % movesTheBuffer([](auto&& in) { return filter::applyRms<ProcessMode::InPlace>(std::move(in), 3UZ).value(); })) << "applyRms";
        expect(that % movesTheBuffer([](auto&& in) { return filter::applyPeakToPeak<ProcessMode::InPlace>(std::move(in), 3UZ).value(); })) << "applyPeakToPeak";

        // and the answer is the one the copying mode gives: a window that reads samples the pass has
        // already written would differ from it as soon as the pass writes into its own input
        auto       source  = generate::ramp<T>("ramp", 5, value_t(0), value_t(1));
        const auto viaCopy = filter::applyMovingAverage<ProcessMode::Copy>(source, 3UZ).value();
        const auto inPlace = filter::applyMovingAverage<ProcessMode::InPlace>(std::move(source), 3UZ).value();
        for (std::size_t i = 0UZ; i < viaCopy.signal_values.size(); ++i) {
            expect(eq(inPlace.signal_values[i], viaCopy.signal_values[i])) << std::format("moving average at index {}", i);
        }

        auto       kept   = generate::from<T>("kept", std::vector<value_t>{3, 3, 4, 3, 3, 4, 3, 3}).value();
        const auto buffer = kept.signal_values.data();
        const auto copied = filter::applyMedian<ProcessMode::Copy>(kept, 3UZ).value();
        expect(neq(copied.signal_values.data(), buffer)) << "the copying mode leaves the caller's record where it was";
        expect(eq(kept.signal_values[2], T(4))) << "and unfiltered";
    } | std::tuple<float, double>{};
};

#pragma GCC diagnostic pop

const boost::ut::suite<"DataSet consistency checks"> _dataSetConsistency = [] {
    using namespace boost::ut;

    "checkConsistency admits an axis-free record"_test = [] {
        // the packet convention's minimal shape: samples, one extent, one signal name, one
        // metadata map, one timing-events slot — no axis and no descriptive arrays
        gr::DataSet<std::uint8_t> packet;
        packet.signal_values = {1U, 2U, 3U};
        packet.extents       = {3};
        packet.signal_names.emplace_back("payload");
        packet.meta_information.resize(1UZ);
        packet.timing_events.resize(1UZ);
        expect(gr::dataset::checkConsistency(packet).has_value()) << "the minimal axis-free record validates; a payload carries no axis, quantity, unit or range";

        packet.signal_ranges.resize(2UZ); // a partially filled descriptive array is inconsistent, not unspecified
        expect(!gr::dataset::checkConsistency(packet).has_value());
        packet.signal_ranges.resize(1UZ);
        packet.signal_quantities.resize(1UZ);
        packet.signal_units.resize(1UZ);
        expect(gr::dataset::checkConsistency(packet).has_value()) << "one entry per signal is the fully specified form";

        packet.axis_names.emplace_back("time"); // an axis name without units or values is inconsistent, not axis-free
        expect(!gr::dataset::checkConsistency(packet).has_value());
    };

    "updateMinMax reduces per signal, component-wise for complex"_test = [] {
        gr::DataSet<float> two;
        two.signal_names  = {"a", "b"};
        two.extents       = {3};
        two.signal_values = {0.f, 2.f, 1.f, 10.f, 12.f, 11.f};
        gr::dataset::updateMinMax(two);
        expect(eq(two.signal_ranges.size(), 2UZ));
        expect(eq(two.signal_ranges[0].min, 0.f) and eq(two.signal_ranges[0].max, 2.f)) << "signal a's range must come from signal a alone";
        expect(eq(two.signal_ranges[1].min, 10.f) and eq(two.signal_ranges[1].max, 12.f)) << "signal b's range must come from signal b alone";

        gr::DataSet<std::complex<float>> z;
        z.signal_names  = {"z"};
        z.extents       = {2};
        z.signal_values = {{1.f, 5.f}, {-3.f, 2.f}};
        gr::dataset::updateMinMax(z);
        expect(eq(z.signal_ranges.size(), 1UZ));
        expect(z.signal_ranges[0].min == std::complex<float>(-3.f, 2.f) and z.signal_ranges[0].max == std::complex<float>(1.f, 5.f)) << "a complex range is the component-wise extremes, not samples ordered by magnitude";
    };
};

int main() { /* not needed for UT */ }
