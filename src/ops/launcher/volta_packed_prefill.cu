// SPDX-License-Identifier: BSD-3-Clause
// Packed GQA / FP32 online-state dataflow adapted from 1Cat-vLLM v37.
// See third_party/onecat_v37/UPSTREAM.md for the exact source and license.

#include "ops/launcher/volta_packed_prefill.h"

#include "core/device.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"

#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kDim = 256;
constexpr int kQueryHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kGroup = kQueryHeads / kKVHeads;
constexpr int kTileM = 128;
constexpr int kTileN = 256;
constexpr int kTileK = 32;
constexpr int kMaxBlockN = 8192;
constexpr int kMaxTiles = kMaxBlockN / kTileN;
constexpr int kThreads = 256;
constexpr std::size_t kAlignment = 256;
using Element = cutlass::half_t;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using QKOutputOp = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
using PVOutputOp = cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;
using TileShape = cutlass::gemm::GemmShape<kTileM, kTileN, kTileK>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using QKDefault = typename cutlass::gemm::kernel::DefaultGemm<
    Element, RowMajor, 8, Element, ColumnMajor, 8, Element, RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70, TileShape, WarpShape,
    InstructionShape, QKOutputOp, Swizzle, 2, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using PVDefault = typename cutlass::gemm::kernel::DefaultGemm<
    Element, RowMajor, 8, Element, RowMajor, 8, float, RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70, TileShape, WarpShape,
    InstructionShape, PVOutputOp, Swizzle, 2, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using QKMma = typename QKDefault::Mma;
using PVBaseMma = typename PVDefault::Mma;
static_assert(QKDefault::kThreadCount == kThreads && PVDefault::kThreadCount == kThreads);

std::size_t align_up(std::size_t bytes) {
    return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}

// The baseline mask allocation holds all four GQA groups. Its lifetime spans
// every block below; no temporary allocation or second gathered KV is needed.
struct ScratchLayout {
    std::size_t query, probability, tile_maximum, tile_sum, maximum, denominator, bytes;

    ScratchLayout(int rows, int columns) {
        std::size_t cursor = 0;
        auto reserve = [&](std::size_t size) {
            const auto offset = align_up(cursor);
            cursor = offset + size;
            return offset;
        };
        const auto total_rows = std::size_t(kKVHeads) * rows;
        query = reserve(total_rows * kDim * sizeof(Element));
        probability = reserve(total_rows * columns * sizeof(Element));
        tile_maximum = reserve(total_rows * (columns / kTileN) * sizeof(float));
        tile_sum = reserve(total_rows * (columns / kTileN) * sizeof(float));
        maximum = reserve(total_rows * sizeof(float));
        denominator = reserve(total_rows * sizeof(float));
        bytes = cursor;
    }
};

// All launch state is passed by value. In particular the transform has no
// device globals, so independent streams cannot overwrite each other's state.
struct Arguments {
    const Element *query, *key, *value;
    const std::int32_t* positions;
    Element* probability;
    float *tile_maximum, *tile_sum, *maximum, *denominator, *numerator;
    int rows, columns, block_columns, key_begin;
    float scale;

    CUTLASS_DEVICE int tile_count() const { return (columns + kTileN - 1) / kTileN; }
    CUTLASS_DEVICE std::size_t metadata_stride() const {
        return std::size_t(rows) * (block_columns / kTileN);
    }
    CUTLASS_DEVICE std::size_t probability_stride() const {
        return std::size_t(rows) * block_columns;
    }
};

__global__ void pack_query(const float* __restrict__ source,
                           Element* __restrict__ packed, int rows) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= kKVHeads * rows * kDim) return;
    const int d = i % kDim;
    const int row = (i / kDim) % rows;
    const int group = i / (rows * kDim);
    const int token = row / kGroup;
    const int head = group * kGroup + row % kGroup;
    packed[i] = Element(source[(token * kQueryHeads + head) * kDim + d]);
}

// QK accumulates in FP32. This visitor applies the exact causal predicate
// before the maximum, retaining FP32 logits until the whole N256 tile's max
// is known. Only the tile-relative probabilities cross an FP16 boundary.
template <typename Iterator>
struct QKVisitor {
    static constexpr int kIterations = Iterator::kIterations;
    static constexpr int kElementsPerAccess = Iterator::kElementsPerAccess;
    static constexpr int kColumns = Iterator::ThreadMap::Iterations::kColumn;
    static constexpr int kThreadsPerRow = Iterator::ThreadMap::Detail::kAccessWidth;
    using AccumulatorFragment = cutlass::Array<float, kElementsPerAccess>;
    using Vector = cutlass::Array<Element, kElementsPerAccess>;
    Arguments args;
    Iterator destination;
    typename Iterator::Fragment fragment;
    AccumulatorFragment logits[kColumns];
    float row_maximum;
    int row, first_fragment;

    CUTLASS_DEVICE QKVisitor(Arguments const& a, int thread)
        : args(a), destination(typename Iterator::Params(RowMajor(a.columns)),
            a.probability + std::size_t(blockIdx.z) * a.probability_stride(),
            {a.rows, a.columns}, thread,
            {int(blockIdx.x) * kTileM, int(blockIdx.y) * kTileN}) {}
    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void end_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) { fragment.clear(); }
    CUTLASS_DEVICE void begin_row(int) { row_maximum = -CUDART_INF_F; }
    CUTLASS_DEVICE void visit(int, int, int column, int fragment_idx,
                              AccumulatorFragment const& accumulator) {
        const auto offset = destination.thread_start() +
                            Iterator::ThreadMap::iteration_offset(fragment_idx);
        row = offset.row();
        if (column == 0) first_fragment = fragment_idx;
        const int last_key = row < args.rows ? args.positions[row / kGroup] : -1;
#pragma unroll
        for (int e = 0; e < kElementsPerAccess; ++e) {
            const int key = offset.column() + e;
            const bool visible = row < args.rows && key < args.columns &&
                                 args.key_begin + key <= last_key;
            const float score = visible ? accumulator[e] * args.scale : -CUDART_INF_F;
            logits[column][e] = score;
            row_maximum = fmaxf(row_maximum, score);
        }
    }
    CUTLASS_DEVICE void end_row(int) {
#pragma unroll
        for (int delta = kThreadsPerRow / 2; delta > 0; delta /= 2)
            row_maximum = fmaxf(row_maximum,
                __shfl_xor_sync(0xffffffffu, row_maximum, delta));
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            Vector values;
#pragma unroll
            for (int e = 0; e < kElementsPerAccess; ++e) {
                const float p = logits[c][e] == -CUDART_INF_F ? 0.0f :
                                expf(logits[c][e] - row_maximum);
                values[e] = Element(p);
                // Follow v37's mass of the emitted FP16 probabilities.
                sum += static_cast<float>(values[e]);
            }
            reinterpret_cast<Vector*>(&fragment)[first_fragment + c] = values;
        }
#pragma unroll
        for (int delta = kThreadsPerRow / 2; delta > 0; delta /= 2)
            sum += __shfl_xor_sync(0xffffffffu, sum, delta);
        if (row < args.rows && threadIdx.x % kThreadsPerRow == 0) {
            const auto index = std::size_t(blockIdx.z) * args.metadata_stride() +
                               std::size_t(blockIdx.y) * args.rows + row;
            args.tile_maximum[index] = row_maximum;
            args.tile_sum[index] = sum;
        }
    }
    CUTLASS_DEVICE void end_step(int) { destination.store(fragment); ++destination; }
};

using QKEpilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    QKVisitor<typename QKDefault::Epilogue::OutputTileIterator>,
    typename QKDefault::Epilogue>::Epilogue;
union QKShared {
    typename QKMma::SharedStorage mma;
    typename QKEpilogue::SharedStorage epilogue;
};

__global__ void qk_probability_kernel(Arguments args) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<QKShared*>(storage);
    const int thread = threadIdx.x;
    const int warp = __shfl_sync(0xffffffffu, thread / 32, 0);
    const int lane = thread % 32;
    const auto kv_offset = std::size_t(args.key_begin) * kKVHeads * kDim + blockIdx.z * kDim;
    // CUTLASS's shared load/store iterator type requires mutable pointers;
    // the MMA invokes only its load methods for Q/K/V.
    typename QKMma::IteratorA a(typename QKMma::IteratorA::Params(RowMajor(kDim)),
        const_cast<Element*>(args.query) + std::size_t(blockIdx.z) * args.rows * kDim,
        {args.rows, kDim}, thread, {int(blockIdx.x) * kTileM, 0});
    typename QKMma::IteratorB b(typename QKMma::IteratorB::Params(ColumnMajor(kKVHeads * kDim)),
        const_cast<Element*>(args.key) + kv_offset,
        {kDim, args.columns}, thread, {0, int(blockIdx.y) * kTileN});
    QKMma mma(shared.mma, thread, warp, lane);
    typename QKMma::FragmentC accumulator;
    accumulator.clear();
    mma(kDim / kTileK, accumulator, a, b, accumulator);
    __syncthreads();
    QKVisitor<typename QKDefault::Epilogue::OutputTileIterator> visitor(args, thread);
    QKEpilogue epilogue(shared.epilogue, thread, warp, lane);
    epilogue(visitor, accumulator);
}

// One PV CTA covers all 256 value channels for 128 packed query rows. It is
// consequently the unique writer of each row's FP32 online state. The stats
// remain live outside the union that reuses MMA storage for the epilogue.
struct PVStatistics {
    float tile_scale[kMaxTiles * kTileM];
    float block_maximum[kTileM];
    float old_scale[kTileM];
    float block_scale[kTileM];
};

CUTLASS_DEVICE void prepare_statistics(Arguments const& args, PVStatistics& stats) {
    const int thread = threadIdx.x;
    const auto group_offset = std::size_t(blockIdx.z) * args.metadata_stride();
    if (thread < kTileM) {
        const int row = int(blockIdx.x) * kTileM + thread;
        float maximum = -CUDART_INF_F;
        float sum = 0.0f;
        if (row < args.rows) {
            for (int tile = 0; tile < args.tile_count(); ++tile)
                maximum = fmaxf(maximum, args.tile_maximum[group_offset + tile * args.rows + row]);
            for (int tile = 0; tile < args.tile_count(); ++tile) {
                const auto i = group_offset + tile * args.rows + row;
                const float tile_max = args.tile_maximum[i];
                if (tile_max != -CUDART_INF_F)
                    sum += args.tile_sum[i] * expf(tile_max - maximum);
            }
            const auto state_row = std::size_t(blockIdx.z) * args.rows + row;
            const float old_max = args.key_begin == 0 ? -CUDART_INF_F : args.maximum[state_row];
            const float next_max = fmaxf(old_max, maximum);
            const float alpha = old_max == -CUDART_INF_F ? 0.0f : expf(old_max - next_max);
            const float beta = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next_max);
            const float old_sum = args.key_begin == 0 ? 0.0f : args.denominator[state_row];
            args.maximum[state_row] = next_max;
            args.denominator[state_row] = fmaf(old_sum, alpha, sum * beta);
            stats.old_scale[thread] = alpha;
            stats.block_scale[thread] = beta;
        } else {
            stats.old_scale[thread] = 0.0f;
            stats.block_scale[thread] = 0.0f;
        }
        stats.block_maximum[thread] = maximum;
    }
    __syncthreads();
    for (int i = thread; i < args.tile_count() * kTileM; i += kThreads) {
        const int tile = i / kTileM;
        const int local_row = i % kTileM;
        const int row = int(blockIdx.x) * kTileM + local_row;
        const float tile_max = row < args.rows ?
            args.tile_maximum[group_offset + tile * args.rows + row] : -CUDART_INF_F;
        stats.tile_scale[i] = tile_max == -CUDART_INF_F ? 0.0f :
                             expf(tile_max - stats.block_maximum[local_row]);
    }
    __syncthreads();
}

// The v37 rescale is fused into A's global-to-shared load. The final masked
// look-ahead fragment never indexes beyond the valid tile metadata.
struct ProbabilityTransform {
    using Iterator = typename PVBaseMma::IteratorA;
    using InputFragment = typename Iterator::Fragment;
    using OutputFragment = cutlass::Array<typename PVBaseMma::SmemIteratorA::Element,
                                         InputFragment::kElements>;
    using ThreadMap = typename Iterator::ThreadMap;
    using Access = typename Iterator::AccessType;
    static constexpr int kAccesses = Iterator::UnderlyingIterator::kAccessesPerVector;
    static constexpr int kContiguous = ThreadMap::Iterations::kContiguous;
    static constexpr int kStrided = ThreadMap::Iterations::kStrided;
    const float* tile_scale;
    int tile_count, offset = 0;
    int row[kStrided];

    CUTLASS_DEVICE ProbabilityTransform(const float* scales = nullptr, int count = 0)
        : tile_scale(scales), tile_count(count) {
        const auto start = ThreadMap::initial_offset(threadIdx.x);
#pragma unroll
        for (int s = 0; s < kStrided; ++s)
            row[s] = start.strided() + s * ThreadMap::Delta::kStrided;
    }
    CUTLASS_DEVICE OutputFragment operator()(InputFragment const& input) {
        OutputFragment output;
        const auto* source = reinterpret_cast<const Access*>(&input);
        auto* destination = reinterpret_cast<Access*>(&output);
        const int tile = offset / kTileN;
#pragma unroll
        for (int s = 0; s < kStrided; ++s) {
            const float scale = tile < tile_count ? tile_scale[tile * kTileM + row[s]] : 0.0f;
#pragma unroll
            for (int c = 0; c < kContiguous; ++c) {
#pragma unroll
                for (int v = 0; v < kAccesses; ++v) {
                    const int i = v + kAccesses * (c + s * kContiguous);
#pragma unroll
                    for (int e = 0; e < Access::kElements; ++e)
                        destination[i][e] = Element(static_cast<float>(source[i][e]) * scale);
                }
            }
        }
        offset += kTileK;
        return output;
    }
};

using PVMma = cutlass::gemm::threadblock::MmaPipelined<
    typename PVBaseMma::Shape, typename PVBaseMma::IteratorA,
    typename PVBaseMma::SmemIteratorA, typename PVBaseMma::IteratorB,
    typename PVBaseMma::SmemIteratorB, float, RowMajor,
    typename PVBaseMma::Policy, ProbabilityTransform>;

template <typename Iterator>
struct PVVisitor {
    static constexpr int kIterations = Iterator::kIterations;
    static constexpr int kElementsPerAccess = Iterator::kElementsPerAccess;
    using AccumulatorFragment = cutlass::Array<float, kElementsPerAccess>;
    using Vector = cutlass::Array<float, kElementsPerAccess>;
    Arguments args;
    PVStatistics const& stats;
    Iterator destination;
    typename Iterator::Fragment fragment, source;

    CUTLASS_DEVICE PVVisitor(Arguments const& a, PVStatistics const& s, int thread)
        : args(a), stats(s), destination(typename Iterator::Params(RowMajor(kDim)),
            a.numerator + std::size_t(blockIdx.z) * a.rows * kDim,
            {a.rows, kDim}, thread, {int(blockIdx.x) * kTileM, 0}) {}
    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void end_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        fragment.clear();
        source.clear();
        if (args.key_begin != 0) destination.load(source);
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void visit(int, int, int, int fragment_idx,
                              AccumulatorFragment const& accumulator) {
        const auto coord = destination.thread_start() +
                           Iterator::ThreadMap::iteration_offset(fragment_idx);
        const int local_row = coord.row() - int(blockIdx.x) * kTileM;
        const float alpha = stats.old_scale[local_row];
        const float beta = stats.block_scale[local_row];
        const auto& previous = reinterpret_cast<const Vector*>(&source)[fragment_idx];
        auto& result = reinterpret_cast<Vector*>(&fragment)[fragment_idx];
#pragma unroll
        for (int e = 0; e < kElementsPerAccess; ++e)
            result[e] = fmaf(previous[e], alpha, accumulator[e] * beta);
    }
    CUTLASS_DEVICE void end_step(int) { destination.store(fragment); ++destination; }
};

using PVEpilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    PVVisitor<typename PVDefault::Epilogue::OutputTileIterator>,
    typename PVDefault::Epilogue>::Epilogue;
struct PVShared {
    union {
        typename PVMma::SharedStorage mma;
        typename PVEpilogue::SharedStorage epilogue;
    } operation;
    PVStatistics statistics;
};
static_assert(sizeof(QKShared) <= 96 * 1024 && sizeof(PVShared) <= 96 * 1024,
              "Volta prefill must fit V100 opt-in shared memory");

__global__ void pv_online_kernel(Arguments args) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<PVShared*>(storage);
    const int thread = threadIdx.x;
    const int warp = __shfl_sync(0xffffffffu, thread / 32, 0);
    const int lane = thread % 32;
    prepare_statistics(args, shared.statistics);
    typename PVMma::IteratorA a(typename PVMma::IteratorA::Params(RowMajor(args.columns)),
        args.probability + std::size_t(blockIdx.z) * args.probability_stride(),
        {args.rows, args.columns}, thread, {int(blockIdx.x) * kTileM, 0});
    const auto kv_offset = std::size_t(args.key_begin) * kKVHeads * kDim + blockIdx.z * kDim;
    typename PVMma::IteratorB b(typename PVMma::IteratorB::Params(RowMajor(kKVHeads * kDim)),
        const_cast<Element*>(args.value) + kv_offset,
        {args.columns, kDim}, thread, {0, 0});
    PVMma mma(shared.operation.mma, thread, warp, lane,
              ProbabilityTransform(shared.statistics.tile_scale, args.tile_count()));
    typename PVMma::FragmentC accumulator;
    accumulator.clear();
    mma((args.columns + kTileK - 1) / kTileK, accumulator, a, b, accumulator);
    __syncthreads();
    PVVisitor<typename PVDefault::Epilogue::OutputTileIterator> visitor(args, shared.statistics, thread);
    PVEpilogue epilogue(shared.operation.epilogue, thread, warp, lane);
    epilogue(visitor, accumulator);
}

__global__ void store_output(const float* __restrict__ numerator,
                             const float* __restrict__ denominator,
                             __nv_bfloat16* __restrict__ output, int rows) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= kKVHeads * rows * kDim) return;
    const int d = i % kDim;
    const int row = (i / kDim) % rows;
    const int group = i / (rows * kDim);
    const int token = row / kGroup;
    const int head = group * kGroup + row % kGroup;
    const float mass = denominator[group * rows + row];
    output[(token * kQueryHeads + head) * kDim + d] =
        __float2bfloat16_rn(mass > 0.0f ? numerator[i] / mass : 0.0f);
}

} // namespace

bool try_volta_packed_prefill(
    const float* q, const half* k, const half* v, const std::int32_t* positions,
    __nv_bfloat16* out, int tokens, int visible_keys, float scale,
    void* score_scratch, std::size_t score_bytes, void* state_scratch,
    std::size_t state_bytes, cudaStream_t stream) {
    // Compare with the corrected FP32 fallback: packed prefill also wins at
    // T64/T128 despite the smaller PV grid. Preserve partial production blocks.
    if (tokens < 64 || tokens > 1024 || visible_keys < 32768 || visible_keys > 262144 ||
        visible_keys % kTileK != 0 || std::abs(scale - 0.0625f) > 1.0e-8f) return false;
    const int rows = tokens * kGroup;
    const int elements = kKVHeads * rows * kDim;
    if (state_bytes < std::size_t(elements) * sizeof(float)) return false;
    int block_columns = kMaxBlockN;
    while (block_columns >= 512 && ScratchLayout(rows, block_columns).bytes > score_bytes)
        block_columns /= 2;
    if (block_columns < 512) return false;
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) return false;
    const ScratchLayout layout(rows, block_columns);
    auto* scratch = static_cast<std::uint8_t*>(score_scratch);
    auto* query = reinterpret_cast<Element*>(scratch + layout.query);
    Arguments args{};
    args.query = query;
    args.key = reinterpret_cast<const Element*>(k);
    args.value = reinterpret_cast<const Element*>(v);
    args.positions = positions;
    args.probability = reinterpret_cast<Element*>(scratch + layout.probability);
    args.tile_maximum = reinterpret_cast<float*>(scratch + layout.tile_maximum);
    args.tile_sum = reinterpret_cast<float*>(scratch + layout.tile_sum);
    args.maximum = reinterpret_cast<float*>(scratch + layout.maximum);
    args.denominator = reinterpret_cast<float*>(scratch + layout.denominator);
    args.numerator = static_cast<float*>(state_scratch);
    args.rows = rows;
    args.block_columns = block_columns;
    args.scale = scale;
    CUDA_CHECK(cudaFuncSetAttribute(qk_probability_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(QKShared)));
    CUDA_CHECK(cudaFuncSetAttribute(pv_online_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(PVShared)));
    pack_query<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(q, query, rows);
    CUDA_CHECK(cudaGetLastError());
    for (int begin = 0; begin < visible_keys; begin += block_columns) {
        args.columns = std::min(block_columns, visible_keys - begin);
        args.key_begin = begin;
        const dim3 qk_grid((rows + kTileM - 1) / kTileM,
                           (args.columns + kTileN - 1) / kTileN, kKVHeads);
        qk_probability_kernel<<<qk_grid, kThreads, sizeof(QKShared), stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
        const dim3 pv_grid((rows + kTileM - 1) / kTileM, 1, kKVHeads);
        pv_online_kernel<<<pv_grid, kThreads, sizeof(PVShared), stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
    store_output<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        args.numerator, args.denominator, out, rows);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace ninfer::ops::detail
