#include "ops/softmax_attention/criterion.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

using namespace ninfer::test;

namespace {
constexpr ReductionCriterion profile{2.8e-3, 1.0e-3, 2.7e-3};

bool accepts(std::span<const double> actual, std::span<const double> reference) {
    const auto stats = compute_attention_output_stats(
        actual.data(), reference.data(), static_cast<std::int64_t>(actual.size()), profile);
    return attention_output_passes(stats, static_cast<std::int64_t>(actual.size()));
}
}

int main() {
    int failures = 0;
    const auto require = [&](bool condition, const char* message) {
        if (!condition) { std::cerr << message << '\n'; ++failures; }
    };

    // Exact BF16 outputs at midpoints from both signs and several binades must
    // be admissible. The former profile alone rejects this correct rounding.
    std::vector<double> exact, reference;
    for (int exponent : {-10, -1, 0, 1, 2, 10}) {
        for (double sign : {-1.0, 1.0}) {
            const double represented = sign * std::ldexp(1.0, exponent);
            exact.push_back(represented);
            reference.push_back(represented * (1.0 + 1.0/256.0 - 1.0e-12));
        }
    }
    require(accepts(exact, reference), "correct BF16 midpoint rounding was rejected");
    const auto previous = compute_reduction_stats(exact.data(), reference.data(), exact.size());
    require(!reduction_passes(previous, exact.size(), profile),
            "boundary fixture no longer proves the former criterion's representability gap");

    // Crossing a rounding boundary within the fixed absolute arithmetic budget
    // is distinct from a large reduction error. The raw relative-L2 check remains.
    const std::array<double, 5> mixed_reference{1.0, 2.0, 3.0, 5.5, 4.01625};
    const std::array<double, 5> mixed_actual{1.0, 2.0, 3.0, 5.5, 4.0};
    require(accepts(mixed_actual, mixed_reference),
            "rounding plus the existing absolute arithmetic budget was rejected");

    // One approximately 5% error among 1024 values passes relative-L2 alone.
    // Its BF16-representable outlier must still fail the per-element gross bound.
    std::vector<double> values(1024, 4.0), corrupted = values;
    corrupted[99] = 4.1875;
    const auto drift = compute_attention_output_stats(corrupted.data(), values.data(),
                                                       values.size(), profile);
    require(drift.raw.relative_l2 < drift.relative_l2_limit,
            "sparse corruption fixture should isolate the gross check");
    require(!attention_output_passes(drift, values.size()) && drift.first_gross_violation == 99,
            "approximately 5% accumulator drift was accepted");

    require(attention_bf16_half_ulp(4.0) == 1.0/64.0,
            "incorrect BF16 normal half-ULP");
    require(attention_bf16_half_ulp(0.0) == std::ldexp(1.0, -134),
            "incorrect BF16 subnormal half-ULP");
    const std::array<double, 1> zero{0.0}, infinity{std::numeric_limits<double>::infinity()};
    require(accepts(zero, zero), "exact zero was rejected");
    require(!accepts(infinity, zero), "non-finite output was accepted");

    std::cout << (failures ? "FAIL" : "PASS") << " BF16 attention output criterion CPU checks\n";
    return failures ? 1 : 0;
}
