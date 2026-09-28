#pragma once
#include "model_int8.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>

namespace model {

// NEON int8 forward pass.
//
// Identical arithmetic to Int8Backend -- same scales, same requantization,
// same saturation. Only the inner dot product changes, so any difference
// in output is a kernel bug, which is why this is diffed against
// Int8Backend rather than against float.
//
// Requires the dotprod extension (ARMv8.2+). Apple Silicon has it. Build
// with -march=armv8.2-a+dotprod or Apple's equivalent.
struct NeonBackend {
  const WeightsInt8& w;
  explicit NeonBackend(const WeightsInt8& weights) : w(weights) {}

  // Sum the four int32 lanes of a vector into one scalar.
  static inline int32_t hsum(int32x4_t v) {
    return vaddvq_s32(v);
  }

  // Dot product of two int8 arrays of length 16.
  static inline int32_t dot16(const int8_t* a, const int8_t* b) {
    // TODO: vld1q_s8 loads 16 int8s into an int8x16_t.
    //       vdupq_n_s32(0) makes a zeroed int32x4_t accumulator.
    //       vdotq_s32(acc, va, vb) does the work.
    //       hsum the result.
    int8x16_t va = vld1q_s8(a);        // load 16 bytes
    int8x16_t vb = vld1q_s8(b);        // load 16 bytes
    int32x4_t acc = vdupq_n_s32(0);    // zeroed accumulator
    acc = vdotq_s32(acc, va, vb);      // 16 multiply-accumulates
    return hsum(acc);                  // four lanes -> one number
  }

  // Dot product of two int8 arrays of length 32 -- two vdot calls into the
  // same accumulator, so the four lanes stay live across both and only one
  // horizontal sum is needed.
  static inline int32_t dot32(const int8_t* a, const int8_t* b) {  
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, vld1q_s8(a),      vld1q_s8(b));
    acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
    return hsum(acc);         
  }

  void forward(const float* in, float* out) const {
    alignas(16) int8_t x_q[N_IN];
    alignas(16) int8_t h1_q[N_HIDDEN];
    alignas(16) int8_t h2_q[N_HIDDEN];

    // TODO: normalize + quantize, same as Int8Backend. You could vectorize
    //       this too, but it is 16 elements and the divide dominates --
    //       measure before bothering.
    for (int i = 0; i < N_IN; ++i) {
      float v = (in[i] - w.norm_mean[i]) * w.inv_norm_std[i];
      x_q[i]  = quantize_inv(v, w.inv_x_scale);
    }

    // TODO: layer 0 -- for each j, acc = dot16(x_q, &w.w0[j * N_IN]);
    //       then the identical scale/bias/relu/requantize as Int8Backend.
    for (int j = 0; j < N_HIDDEN; ++j) {
      int32_t acc = dot16(x_q, &w.w0[j * N_IN]);
      float h = float(acc) * w.s0 + w.b0[j];
      h = h > 0.0f ? h : 0.0f;
      // Re-enter integer space for the next layer, at ITS scale.
      h1_q[j] = quantize_inv(h, w.inv_h1_scale);
    }

    // TODO: layer 2 -- dot32 against &w.w2[j * N_HIDDEN].
    for (int j = 0; j < N_HIDDEN; ++j) {
      int32_t acc = dot32(h1_q, &w.w2[j * N_HIDDEN]);
      float h = float(acc) * w.s2 + w.b2[j];
      h = h > 0.0f ? h : 0.0f;
      h2_q[j] = quantize_inv(h, w.inv_h2_scale);
    }

    // TODO: layer 4 -- dot32 against &w.w4[k * N_HIDDEN], no relu.
    for (int k = 0; k < N_OUT; ++k) {
      int32_t acc = dot32(h2_q, &w.w4[k * N_HIDDEN]);
      out[k] = float(acc) * w.s4 + w.b4[k];
    }
  }
};

} // namespace model
#endif  // __ARM_NEON