#pragma once
#include "model_batch.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>

namespace model {

// Vectorized batched int8 inference.
//
// THE STRUCTURAL DIFFERENCE FROM THE SINGLE-EVENT KERNEL
//
// NeonBackend vectorized across FEATURES: one vdotq_s32 consumed all 16
// inputs of one event, and vaddvq_s32 then collapsed four lanes into one
// scalar. That horizontal sum was unavoidable, and it is the operation
// SIMD hardware is worst at -- lanes are built not to talk to each other.
// It is why vectorizing the dots only bought 1.58x: the tail could not
// follow, because hsum handed back one scalar at a time.
//
// Here the lanes hold DIFFERENT EVENTS. Each lane accumulates its own
// event's dot product and they never mix, so there is no horizontal sum
// anywhere. The feature loop becomes a plain sequence of multiply-adds,
// and the scale/bias/relu/requantize tail -- the part that dominated the
// single-event kernel -- vectorizes too, four events at a time.
struct BatchNeonBackend {
  const WeightsInt8& w;
  explicit BatchNeonBackend(const WeightsInt8& weights) : w(weights) {}

  // 16 events per register pass: one int8x16_t of inputs, accumulating
  // into four int32x4_t (int32 accumulators are 4 per register).
  static constexpr int LANES = 16;

  void forward_batch(const float* in, float* out, int n) const {
    static thread_local BatchInput x;
    static thread_local int8_t h1[N_HIDDEN][MAX_BATCH];
    static thread_local int8_t h2[N_HIDDEN][MAX_BATCH];

    // Transpose row-major arrivals into feature-major storage. This pass
    // is what makes every vld1q_s8 below a contiguous load instead of a
    // gather; it is the enabling step for the whole kernel.
    for (int e = 0; e < n; ++e) {
      const float* f = in + size_t(e) * N_IN;
      for (int i = 0; i < N_IN; ++i) {
        const float v = (f[i] - w.norm_mean[i]) * w.inv_norm_std[i];
        x.q[i][e] = quantize_inv(v, w.inv_x_scale);
      }
    }
    x.n = n;

    layer(&x.q[0][0], w.w0, N_IN,     w.b0, w.s0, w.inv_h1_scale,
          &h1[0][0],  N_HIDDEN, n);
    layer(&h1[0][0],  w.w2, N_HIDDEN, w.b2, w.s2, w.inv_h2_scale,
          &h2[0][0],  N_HIDDEN, n);

    // Layer 4 does not fit the shared helper: it emits float logits in
    // ROW-MAJOR order (the caller wants each event's three values
    // together), with no relu and no requantization.
    for (int k = 0; k < N_OUT; ++k) {
      const int8_t* wrow = &w.w4[size_t(k) * N_HIDDEN];
      for (int e = 0; e < n; e += LANES) {
        const int lanes = (n - e < LANES) ? (n - e) : LANES;

        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);

        for (int j = 0; j < N_HIDDEN; ++j) {
          const int8x16_t wv = vdupq_n_s8(wrow[j]);
          const int8x16_t xv = vld1q_s8(&h2[j][e]);
          accumulate(xv, wv, a0, a1, a2, a3);
        }

        int32_t acc[LANES];
        vst1q_s32(acc + 0,  a0);
        vst1q_s32(acc + 4,  a1);
        vst1q_s32(acc + 8,  a2);
        vst1q_s32(acc + 12, a3);
        for (int t = 0; t < lanes; ++t) {
          out[size_t(e + t) * N_OUT + k] = float(acc[t]) * w.s4 + w.b4[k];
        }
      }
    }
  }

 private:
  // Sixteen int8 products, widened and accumulated into four int32 lanes
  // WITHOUT any lane crossing. Each output lane belongs to one event and
  // stays that way for the whole feature loop.
  //
  // The trap here: vpadalq_s16 looks like exactly the right instruction
  // -- it widens int16x8 to int32x4 and accumulates -- but it adds
  // ADJACENT PAIRS. acc[0] += v[0] + v[1] merges two events into one
  // lane, which is precisely the crossing this design exists to avoid.
  // It compiles, it runs, and it produces plausible but wrong logits that
  // only a bit-exact diff catches.
  //
  // vaddw_s16 is the correct primitive: a plain widening add that keeps
  // each product in its own lane.
  //
  // vmull_s8 cannot overflow -- 127*127 = 16,129 fits in int16 -- but
  // accumulating 32 of those would reach 516k and overflow, so each
  // product is widened to int32 before it is accumulated.
  static inline void accumulate(int8x16_t xv, int8x16_t wv,
                                int32x4_t& a0, int32x4_t& a1,
                                int32x4_t& a2, int32x4_t& a3) {
    // vmull_s8 takes int8x8, so split the 16 lanes into halves.
    const int16x8_t lo = vmull_s8(vget_low_s8(xv),  vget_low_s8(wv));
    const int16x8_t hi = vmull_s8(vget_high_s8(xv), vget_high_s8(wv));

    a0 = vaddw_s16(a0, vget_low_s16(lo));    // events e+0  .. e+3
    a1 = vaddw_s16(a1, vget_high_s16(lo));   // events e+4  .. e+7
    a2 = vaddw_s16(a2, vget_low_s16(hi));    // events e+8  .. e+11
    a3 = vaddw_s16(a3, vget_high_s16(hi));   // events e+12 .. e+15
  }

  // Scale, bias, relu and requantize four events at a time.
  //
  // This is the step that could NOT be vectorized in the single-event
  // kernel: hsum produced one scalar, so the tail ran scalar 67 times per
  // forward pass and ended up ~50% of the runtime. With a batch there are
  // 16 accumulators and four fit in a register.
  //
  // vcvtnq_s32_f32 rounds to nearest with ties to even, which is what
  // lrintf does in quantize_inv -- the two paths must agree bit for bit
  // or the oracle diff fails.
  //
  // The clamp is explicit rather than relying on vqmovn's saturation:
  // vqmovn saturates at -128, quantize_inv clamps at -127. One value
  // apart, and it would show up as a handful of differing bytes.
  static inline int32x4_t requant(int32x4_t a, float32x4_t bias_v,
                                  float scale, float inv_out_scale,
                                  bool relu) {
    float32x4_t f = vcvtq_f32_s32(a);
    f = vfmaq_n_f32(bias_v, f, scale);               // acc * scale + bias
    if (relu) f = vmaxq_f32(f, vdupq_n_f32(0.0f));
    f = vmulq_n_f32(f, inv_out_scale);
    int32x4_t q = vcvtnq_s32_f32(f);
    q = vminq_s32(q, vdupq_n_s32(127));
    q = vmaxq_s32(q, vdupq_n_s32(-127));
    return q;
  }

  // One int8 layer: a feature-major input block times a weight matrix,
  // producing a feature-major output block. Both hidden layers share this
  // -- they differ only in how many inputs they take.
  //
  // Strides are MAX_BATCH on both sides, since every feature-major buffer
  // in this file is padded to that width so the stride is a compile-time
  // constant.
  void layer(const int8_t* in_t, const int8_t* wmat, int n_in,
             const float* bias, float scale, float inv_out_scale,
             int8_t* out_t, int n_out, int n) const {
    for (int j = 0; j < n_out; ++j) {
      // The weight row for neuron j, loaded once and reused across every
      // event in the batch. THIS REUSE IS THE BATCHING WIN. Swapping the
      // j and e loops would reload the matrix per event and gain nothing
      // over the unbatched kernel.
      const int8_t* wrow = &wmat[size_t(j) * n_in];
      const float32x4_t bias_v = vdupq_n_f32(bias[j]);

      for (int e = 0; e < n; e += LANES) {
        const int lanes = (n - e < LANES) ? (n - e) : LANES;

        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);

        for (int i = 0; i < n_in; ++i) {
          // One weight, broadcast to all 16 lanes: every event in this
          // block multiplies its feature i by the same number.
          const int8x16_t wv = vdupq_n_s8(wrow[i]);
          // Feature i for 16 consecutive events -- contiguous only
          // because of the feature-major transpose.
          const int8x16_t xv = vld1q_s8(&in_t[size_t(i) * MAX_BATCH + e]);
          accumulate(xv, wv, a0, a1, a2, a3);
        }

        const int32x4_t q0 = requant(a0, bias_v, scale, inv_out_scale, true);
        const int32x4_t q1 = requant(a1, bias_v, scale, inv_out_scale, true);
        const int32x4_t q2 = requant(a2, bias_v, scale, inv_out_scale, true);
        const int32x4_t q3 = requant(a3, bias_v, scale, inv_out_scale, true);

        // Narrow int32x4 x4 -> int8x16. The values are already clamped to
        // [-127, 127], so these moves are exact and the saturation in
        // vmovn never engages.
        const int16x8_t n01 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
        const int16x8_t n23 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
        const int8x16_t packed = vcombine_s8(vmovn_s16(n01), vmovn_s16(n23));

        int8_t* dst = &out_t[size_t(j) * MAX_BATCH + e];
        if (lanes == LANES) {
          vst1q_s8(dst, packed);
        } else {
          // Final partial block. The lanes past n hold garbage read from
          // the padded region -- harmless, but they must not be stored,
          // or the next layer would consume them.
          int8_t tmp[LANES];
          vst1q_s8(tmp, packed);
          for (int t = 0; t < lanes; ++t) dst[t] = tmp[t];
        }
      }
    }
  }
};

} // namespace model
#endif  // __ARM_NEON