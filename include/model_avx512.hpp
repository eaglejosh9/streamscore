#pragma once
#include "model_int8.hpp"

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
#include <immintrin.h>

namespace model {

// AVX-512 VNNI int8 forward pass -- the x86 counterpart of NeonBackend.
//
// Identical arithmetic to Int8Backend: same scales, same requantization
// (ties-to-even, clamp to [-127, 127]), same float op order. Only the
// integer dot products and the elementwise float work are vectorized, so
// any difference from Int8Backend is a kernel bug.
//
// Bit-exactness relies on -ffp-contract=off: GCC implements _mm512_mul_ps
// and _mm512_add_ps as plain vector * and +, which it will happily fuse
// into an FMA otherwise -- and Int8Backend's mul-then-add would no longer
// match.
//
// --- Layout ------------------------------------------------------------
//
// NeonBackend computes one dot product per neuron and horizontally sums
// it. Here the work is turned sideways so no horizontal sum is ever
// needed: each int32 lane of a zmm accumulates ONE neuron, and the inputs
// are fed four at a time by broadcasting a 4-byte group to all 16 lanes.
//
//   vpdpbusd acc, bcast(x[4g..4g+3]), W_g      W_g lane n = w[n][4g..4g+3]
//
// After K/4 such steps, lane n holds neuron n's full dot product, and the
// bias / scale / relu / requantize run 16 neurons at a time. The weights
// are repacked once, in the constructor, into that per-group layout.
//
// --- Signedness ----------------------------------------------------------
//
// vpdpbusd multiplies UNSIGNED bytes by SIGNED bytes. After relu, h1 and
// h2 are in [0, 127], so layers 2 and 4 feed them in as-is.
//
// Layer 0's input x is signed. Rather than fall back to maddubs + madd,
// it is shifted into unsigned range and the shift is subtracted out:
//
//   sum_i (x_i + 128) * w_i  =  sum_i x_i * w_i  +  128 * sum_i w_i
//
// x_i + 128 is in [1, 255] (x is clamped to [-127, 127]), and the second
// term depends only on the weights, so it is precomputed per neuron. All
// of this is exact integer arithmetic: per lane each vpdpbusd adds at most
// 4 * 255 * 128 = 130560, nowhere near int32 overflow, and the
// non-saturating vpdpbusd (not vpdpbusds) is used. So layer 0 stays on
// VNNI with one extra vpsubd per 16 neurons.
struct Avx512Backend {
  const WeightsInt8& w;

  static constexpr int L0_GROUPS = N_IN / 4;       // 4
  static constexpr int L2_GROUPS = N_HIDDEN / 4;   // 8
  static constexpr int BLOCKS    = N_HIDDEN / 16;  // 2 zmm of neurons

  static_assert(N_IN == 16, "input quantize assumes one zmm of features");
  static_assert(N_HIDDEN % 16 == 0 && N_HIDDEN % 4 == 0, "hidden layout");
  static_assert(N_OUT <= 16, "output layer packs into one zmm");

  // wN_p[g][b] is the 64-byte vector for input group g, neuron block b.
  alignas(64) int8_t  w0_p[L0_GROUPS][BLOCKS][64];
  alignas(64) int8_t  w2_p[L2_GROUPS][BLOCKS][64];
  alignas(64) int8_t  w4_p[L2_GROUPS][64];          // N_OUT lanes used, rest 0
  alignas(64) int32_t corr0[N_HIDDEN];              // 128 * sum_i w0[j][i]

  explicit Avx512Backend(const WeightsInt8& weights) : w(weights) {
    for (int g = 0; g < L0_GROUPS; ++g)
      for (int b = 0; b < BLOCKS; ++b)
        for (int n = 0; n < 16; ++n)
          for (int k = 0; k < 4; ++k)
            w0_p[g][b][4 * n + k] = w.w0[(16 * b + n) * N_IN + 4 * g + k];

    for (int g = 0; g < L2_GROUPS; ++g)
      for (int b = 0; b < BLOCKS; ++b)
        for (int n = 0; n < 16; ++n)
          for (int k = 0; k < 4; ++k)
            w2_p[g][b][4 * n + k] = w.w2[(16 * b + n) * N_HIDDEN + 4 * g + k];

    for (int g = 0; g < L2_GROUPS; ++g)
      for (int n = 0; n < 16; ++n)
        for (int k = 0; k < 4; ++k)
          w4_p[g][4 * n + k] = n < N_OUT ? w.w4[n * N_HIDDEN + 4 * g + k] : 0;

    for (int j = 0; j < N_HIDDEN; ++j) {
      int32_t s = 0;
      for (int i = 0; i < N_IN; ++i) s += w.w0[j * N_IN + i];
      corr0[j] = 128 * s;
    }
  }

  // Sixteen quantize_inv()s at once, already multiplied by 1/scale:
  // returns lrintf(v) clamped to [-127, 127], as int32 lanes.
  //
  // Clamping in float BEFORE rounding gives the same answer as rounding
  // then clamping (rounding is monotone and +-127 are integers), and keeps
  // vcvtps2dq away from its 32-bit overflow. NaN also lands on -127 via
  // max_ps's second-operand rule, which is what lrintf's 0x8000... gives.
  //
  // The one remaining divergence is v >= 2^63 or +inf: lrintf overflows
  // to LONG_MIN there, so the scalar path yields -127, not +127. That mask
  // reproduces it. It never fires on real data; it is here so "bit-exact"
  // holds for every input, not just the ones we have tested.
  static inline __m512i quantize16(__m512 v) {
    const __m512 lo = _mm512_set1_ps(-127.0f);
    const __m512 hi = _mm512_set1_ps(127.0f);
    __mmask16 ovf = _mm512_cmp_ps_mask(v, _mm512_set1_ps(0x1p63f), _CMP_NLT_UQ);
    __m512 c = _mm512_min_ps(_mm512_max_ps(v, lo), hi);
    __m512i q = _mm512_cvt_roundps_epi32(c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm512_mask_blend_epi32(ovf, q, _mm512_set1_epi32(-127));
  }

  // The 4-byte input group g of a byte array, broadcast to all 16 lanes.
  static inline __m512i bcast4(const uint8_t* a, int g) {
    int32_t v;
    __builtin_memcpy(&v, a + 4 * g, 4);
    return _mm512_set1_epi32(v);
  }

  static inline __m512i ld(const int8_t* p) { return _mm512_load_si512(p); }

  // float(acc) * s + b, then relu, then requantize to int8 at inv_scale.
  // Same op order as Int8Backend; max_ps(h, 0) returns +0 for -0 and NaN,
  // exactly like `h > 0 ? h : 0`.
  static inline __m128i requant16(__m512i acc, __m512 s, const float* b,
                                  __m512 inv_scale) {
    __m512 h = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc), s),
                             _mm512_loadu_ps(b));
    h = _mm512_max_ps(h, _mm512_setzero_ps());
    return _mm512_cvtepi32_epi8(quantize16(_mm512_mul_ps(h, inv_scale)));
  }

  void forward(const float* in, float* out) const {
    alignas(16) uint8_t x_u[N_IN];       // x_q + 128
    alignas(32) uint8_t h1_q[N_HIDDEN];  // [0, 127]
    alignas(32) uint8_t h2_q[N_HIDDEN];  // [0, 127]

    // Normalize + quantize the 16 inputs in one zmm. Same two multiplies
    // as Int8Backend, in the same order: (in - mean) * inv_std, then
    // * inv_x_scale.
    {
      __m512 v = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(in),
                                             _mm512_loadu_ps(w.norm_mean)),
                               _mm512_loadu_ps(w.inv_norm_std));
      __m512i q = quantize16(_mm512_mul_ps(v, _mm512_set1_ps(w.inv_x_scale)));
      q = _mm512_add_epi32(q, _mm512_set1_epi32(128));   // [1, 255]
      _mm_store_si128(reinterpret_cast<__m128i*>(x_u), _mm512_cvtepi32_epi8(q));
    }

    // Layer 0: 16 -> 32. Four VNNI steps per block of 16 neurons; the two
    // blocks are independent chains.
    {
      __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
      for (int g = 0; g < L0_GROUPS; ++g) {
        __m512i xg = bcast4(x_u, g);
        a0 = _mm512_dpbusd_epi32(a0, xg, ld(w0_p[g][0]));
        a1 = _mm512_dpbusd_epi32(a1, xg, ld(w0_p[g][1]));
      }
      a0 = _mm512_sub_epi32(a0, _mm512_load_si512(corr0));
      a1 = _mm512_sub_epi32(a1, _mm512_load_si512(corr0 + 16));

      const __m512 s0 = _mm512_set1_ps(w.s0), inv = _mm512_set1_ps(w.inv_h1_scale);
      _mm_store_si128(reinterpret_cast<__m128i*>(h1_q),      requant16(a0, s0, w.b0,      inv));
      _mm_store_si128(reinterpret_cast<__m128i*>(h1_q + 16), requant16(a1, s0, w.b0 + 16, inv));
    }

    // Layer 2: 32 -> 32. Eight steps per block; even and odd groups go to
    // separate accumulators so each dependency chain is 4 deep, not 8.
    {
      __m512i a0e = _mm512_setzero_si512(), a0o = _mm512_setzero_si512();
      __m512i a1e = _mm512_setzero_si512(), a1o = _mm512_setzero_si512();
      for (int g = 0; g < L2_GROUPS; g += 2) {
        __m512i xe = bcast4(h1_q, g), xo = bcast4(h1_q, g + 1);
        a0e = _mm512_dpbusd_epi32(a0e, xe, ld(w2_p[g][0]));
        a1e = _mm512_dpbusd_epi32(a1e, xe, ld(w2_p[g][1]));
        a0o = _mm512_dpbusd_epi32(a0o, xo, ld(w2_p[g + 1][0]));
        a1o = _mm512_dpbusd_epi32(a1o, xo, ld(w2_p[g + 1][1]));
      }
      __m512i a0 = _mm512_add_epi32(a0e, a0o), a1 = _mm512_add_epi32(a1e, a1o);

      const __m512 s2 = _mm512_set1_ps(w.s2), inv = _mm512_set1_ps(w.inv_h2_scale);
      _mm_store_si128(reinterpret_cast<__m128i*>(h2_q),      requant16(a0, s2, w.b2,      inv));
      _mm_store_si128(reinterpret_cast<__m128i*>(h2_q + 16), requant16(a1, s2, w.b2 + 16, inv));
    }

    // Layer 4: 32 -> 3. Same scheme with N_OUT live lanes; no relu, no
    // requantization. Masked load/store so b4 and out are touched only
    // in their N_OUT valid floats.
    {
      __m512i ae = _mm512_setzero_si512(), ao = _mm512_setzero_si512();
      for (int g = 0; g < L2_GROUPS; g += 2) {
        ae = _mm512_dpbusd_epi32(ae, bcast4(h2_q, g),     ld(w4_p[g]));
        ao = _mm512_dpbusd_epi32(ao, bcast4(h2_q, g + 1), ld(w4_p[g + 1]));
      }
      constexpr __mmask16 m = (1u << N_OUT) - 1;
      __m512 o = _mm512_add_ps(
          _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(ae, ao)),
                        _mm512_set1_ps(w.s4)),
          _mm512_maskz_loadu_ps(m, w.b4));
      _mm512_mask_storeu_ps(out, m, o);
    }
  }
};

} // namespace model
#endif  // __AVX512F__ && __AVX512VNNI__
