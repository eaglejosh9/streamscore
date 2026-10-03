#pragma once
#include "model.hpp"
#include "model_int8.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <cstdint>
#include <cstring>

namespace model {

// Batched int8 inference.
//
// WHY BATCHING EXISTS AT ALL
//
// Single-event inference loads all 2,691 weights to perform 2,691
// multiply-accumulates: one arithmetic operation per byte of weight
// fetched. That ratio -- arithmetic intensity -- is the number that
// decides whether a processor can be kept busy.
//
// At an intensity of 1, you are purely bandwidth-bound. A GPU with 500
// GB/s and 20 TFLOPs would spend all its time fetching weights it uses
// once, and the ~5 microsecond kernel launch would dwarf the 49 ns of
// actual work. That is why phases 1-8 never needed a GPU.
//
// Batching N events reuses each loaded weight N times, so the intensity
// becomes N. At N=1 the CPU wins trivially; at N=4096 the GPU's bandwidth
// finally has something to do. Somewhere between is a crossover, and
// finding it is the point of this phase.
//
// THE COST BATCHING CHARGES
//
// Latency. A batch of 32 means the first event waits for 31 more to
// arrive before anything is scored. At the measured 2.9M events/sec that
// is ~11 microseconds of queueing -- against a single-event inference
// cost of 49 ns. Throughput mode buys bandwidth with latency, which is
// exactly why the system keeps both modes rather than picking one.

constexpr int MAX_BATCH = 4096;

// Quantized inputs for a batch, stored FEATURE-MAJOR.
//
// The events arrive row-major -- each Record holds its 16 features
// contiguously. But the kernel multiplies weight element i against
// feature i of every event at once, and in row-major those N values sit
// 16 bytes apart: a strided gather, which NEON cannot do in one load.
//
// Transposing to feature-major puts them adjacent, so one vld1q_s8 pulls
// feature i for 16 consecutive events. The transpose costs one pass over
// N*16 bytes; the alternative costs a gather on every load of the inner
// loop. For N >= 16 the transpose wins comfortably, and below that the
// single-event kernel is the right tool anyway.
//
// Padded to MAX_BATCH so the stride between features is a compile-time
// constant and the loads stay aligned.
struct BatchInput {
  alignas(64) int8_t q[N_IN][MAX_BATCH];   // q[feature][event]
  int n = 0;
};

// Scalar batched reference. No intrinsics -- this is the readable
// definition of the arithmetic, and the oracle the vectorized version is
// diffed against. Identical results to running Int8Backend N times.
struct BatchScalarBackend {
  const WeightsInt8& w;
  explicit BatchScalarBackend(const WeightsInt8& weights) : w(weights) {}

  // in:  n * N_IN raw features, row-major (one event per row)
  // out: n * N_OUT logits, row-major
  void forward_batch(const float* in, float* out, int n) const {
    static thread_local BatchInput x;
    static thread_local int8_t h1[N_HIDDEN][MAX_BATCH];
    static thread_local int8_t h2[N_HIDDEN][MAX_BATCH];

    // Normalize and quantize, transposing on the way in. This is the one
    // pass that converts row-major arrivals into feature-major storage.
    x.n = n;
    for (int e = 0; e < n; ++e) {
      const float* f = in + size_t(e) * N_IN;
      for (int i = 0; i < N_IN; ++i) {
        const float v = (f[i] - w.norm_mean[i]) * w.inv_norm_std[i];
        x.q[i][e] = quantize_inv(v, w.inv_x_scale);
      }
    }

    // Layer 0: 16 -> 32.
    //
    // LOOP ORDER IS THE WHOLE POINT. j (output neuron) is outer and the
    // batch is inner, so w0's row j is loaded once and reused across all
    // n events. Swapping them -- batch outer, j inner -- would reload the
    // weights every event and the batch would buy nothing at all.
    for (int j = 0; j < N_HIDDEN; ++j) {
      for (int e = 0; e < n; ++e) {
        int32_t acc = 0;
        for (int i = 0; i < N_IN; ++i) {
          acc += int32_t(x.q[i][e]) * int32_t(w.w0[j * N_IN + i]);
        }
        float h = float(acc) * w.s0 + w.b0[j];
        h = h > 0.0f ? h : 0.0f;
        h1[j][e] = quantize_inv(h, w.inv_h1_scale);
      }
    }

    for (int j = 0; j < N_HIDDEN; ++j) {
      for (int e = 0; e < n; ++e) {
        int32_t acc = 0;
        for (int i = 0; i < N_HIDDEN; ++i) {
          acc += int32_t(h1[i][e]) * int32_t(w.w2[j * N_HIDDEN + i]);
        }
        float h = float(acc) * w.s2 + w.b2[j];
        h = h > 0.0f ? h : 0.0f;
        h2[j][e] = quantize_inv(h, w.inv_h2_scale);
      }
    }

    // Layer 4: 32 -> 3, logits stay float. Written back row-major so the
    // caller gets its results laid out per event, which is what every
    // consumer wants.
    for (int k = 0; k < N_OUT; ++k) {
      for (int e = 0; e < n; ++e) {
        int32_t acc = 0;
        for (int j = 0; j < N_HIDDEN; ++j) {
          acc += int32_t(h2[j][e]) * int32_t(w.w4[k * N_HIDDEN + j]);
        }
        out[size_t(e) * N_OUT + k] = float(acc) * w.s4 + w.b4[k];
      }
    }
  }
};


} // namespace model