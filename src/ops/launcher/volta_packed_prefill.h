#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Private SM70 implementation below causal_softmax_attention. All scratch belongs
// to the caller and remains live on stream. Returns false, without launching, for
// unsupported extents, insufficient existing scratch, or graph capture.
bool try_volta_packed_prefill(
    const float* q, const half* k, const half* v, const std::int32_t* positions,
    __nv_bfloat16* out, int tokens, int visible_keys, float scale,
    void* score_scratch, std::size_t score_bytes, void* state_scratch,
    std::size_t state_bytes, cudaStream_t stream);

} // namespace ninfer::ops::detail
