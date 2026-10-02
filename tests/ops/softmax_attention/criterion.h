#pragma once

#include "ops/op_check.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ninfer::test {

// BF16 has eight significant binary digits, including the implicit leading bit.
// This is half its spacing in the reference value's binade. Below the smallest
// normal BF16 value, the subnormal spacing remains 2^-133. No Op staging cast
// or implementation-specific intermediate enters this output-format bound.
inline double attention_bf16_half_ulp(double reference) {
    const int exponent = reference == 0.0 ? -126 : std::max(-126, std::ilogb(std::abs(reference)));
    return std::ldexp(1.0, exponent - 8);
}

struct AttentionOutputStats {
    ReductionStats raw;
    double rounding_relative_l2 = 0.0;
    double relative_l2_limit = 0.0;
    double maximum_gross_ratio = 0.0;
    std::int64_t first_gross_violation = -1;
};

inline AttentionOutputStats compute_attention_output_stats(
    const double* actual, const double* reference, std::int64_t count,
    const ReductionCriterion& profile) {
    AttentionOutputStats stats;
    stats.raw = compute_reduction_stats(actual, reference, count);
    long double squared_reference = 0.0L;
    long double squared_rounding = 0.0L;
    const double profile_gross = profile.gross_relative_to_max_reference *
                                 stats.raw.maximum_absolute_reference;
    for (std::int64_t i = 0; i < count; ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) continue;
        const double rounding = attention_bf16_half_ulp(reference[i]);
        squared_reference += static_cast<long double>(reference[i]) * reference[i];
        squared_rounding += static_cast<long double>(rounding) * rounding;
        const double limit = profile.gross_absolute + std::max(profile_gross, rounding);
        const double error = std::abs(actual[i] - reference[i]);
        const double ratio = limit > 0.0 ? error / limit
                             : error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
        stats.maximum_gross_ratio = std::max(stats.maximum_gross_ratio, ratio);
        if (error > limit && stats.first_gross_violation < 0) stats.first_gross_violation = i;
    }
    stats.rounding_relative_l2 = std::sqrt(static_cast<double>(squared_rounding)) /
        std::max(std::sqrt(static_cast<double>(squared_reference)), 1.0e-30);
    stats.relative_l2_limit = std::max(profile.relative_l2, stats.rounding_relative_l2);
    return stats;
}

inline bool attention_output_passes(const AttentionOutputStats& stats, std::int64_t count) {
    return count > 0 && stats.raw.first_non_finite < 0 && stats.first_gross_violation < 0 &&
           stats.raw.relative_l2 <= stats.relative_l2_limit;
}

} // namespace ninfer::test
