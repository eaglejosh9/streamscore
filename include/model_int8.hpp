#pragma once
#include "model.hpp"   // N_IN, N_HIDDEN, N_OUT

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cmath>

namespace model {

constexpr uint32_t INT8_MAGIC   = 0x53544D51;  // "STMQ"
constexpr uint32_t INT8_VERSION = 1;

// Mirrors python/calibrate.py's file layout exactly.
//
// Weights are int8 with one scale per tensor -- quantize.py showed per-row
// scaling halves the weight reconstruction error but makes no measurable
// accuracy difference at this size, so this takes the simpler kernel.
//
// Biases stay float32: 67 values total, and quantizing them would add
// error for a saving of 200 bytes.
//
// Activations are quantized too, with ranges observed by calibration.
// That is what lets the inner loops run entirely in integers, which is the
// whole point -- weight-only quantization would save the same memory and
// buy no speed.
struct WeightsInt8 {
  float norm_mean[N_IN];
  float norm_std[N_IN];
  float inv_norm_std[N_IN]; 

  float x_scale, h1_scale, h2_scale;   // activation scales, from calibration
  float w0_scale, w2_scale, w4_scale;  // weight scales, per tensor

    alignas(16) int8_t w0[N_HIDDEN * N_IN];
    float  b0[N_HIDDEN];
    alignas(16) int8_t w2[N_HIDDEN * N_HIDDEN];
    float  b2[N_HIDDEN];
    alignas(16) int8_t w4[N_OUT * N_HIDDEN];
    float  b4[N_OUT];

  // (input activation scale) * (weight scale), one per layer. Constant
  // across every neuron in a layer, so it is computed once in load()
  // rather than 32 times in the hot path.
  //
  // Note which activation scale pairs with which weight matrix: s0 uses
  // the scale of what FEEDS layer 0, not what it produces.
  float s0, s2, s4;

  // Reciprocals of the activation scales. quantize() divides by these 67
  // times per forward pass; a divide is roughly 10 cycles against 3 for a
  // multiply, and the scales are constant, so the reciprocal is computed
  // once here instead.
  float inv_x_scale, inv_h1_scale, inv_h2_scale;

  bool load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror("weights_int8"); return false; }

    uint32_t hdr[5];
    if (std::fread(hdr, sizeof(uint32_t), 5, f) != 5) {
      std::fprintf(stderr, "weights_int8: truncated header\n");
      std::fclose(f);
      return false;
    }

    if (hdr[0] != INT8_MAGIC) {
      std::fprintf(stderr, "weights_int8: bad magic 0x%08x (want 0x%08x)\n",
                   hdr[0], INT8_MAGIC);
      std::fclose(f); return false;
    }
    if (hdr[1] != INT8_VERSION) {
      std::fprintf(stderr, "weights_int8: version %u, this build expects %u\n",
                   hdr[1], INT8_VERSION);
      std::fclose(f); return false;
    }
    if (hdr[2] != N_IN) {
      std::fprintf(stderr, "weights_int8: n_in %u, this build expects %d\n",
                   hdr[2], N_IN);
      std::fclose(f); return false;
    }
    if (hdr[3] != N_HIDDEN) {
      std::fprintf(stderr, "weights_int8: n_hidden %u, this build expects %d\n",
                   hdr[3], N_HIDDEN);
      std::fclose(f); return false;
    }
    if (hdr[4] != N_OUT) {
      std::fprintf(stderr, "weights_int8: n_out %u, this build expects %d\n",
                   hdr[4], N_OUT);
      std::fclose(f); return false;
    }

    // Two readers because this file interleaves float32 and int8 blocks.
    // Order must match calibrate.py exactly; nothing enforces that but the
    // version number.
    bool ok = true;
    auto rd = [&](float* dst, size_t n, const char* what) {
      if (!ok) return;
      if (std::fread(dst, sizeof(float), n, f) != n) {
        std::fprintf(stderr, "weights_int8: truncated reading %s\n", what);
        ok = false;
      }
    };
    auto rq = [&](int8_t* dst, size_t n, const char* what) {
      if (!ok) return;
      if (std::fread(dst, sizeof(int8_t), n, f) != n) {
        std::fprintf(stderr, "weights_int8: truncated reading %s\n", what);
        ok = false;
      }
    };

    rd(norm_mean, N_IN, "norm_mean");
    rd(norm_std,  N_IN, "norm_std");
    for (int i = 0; i < N_IN; ++i) inv_norm_std[i] = 1.0f / norm_std[i];

    // The three activation scales are consecutive float members, so this
    // fills all of x_scale, h1_scale, h2_scale. Same for the weight scales.
    rd(&x_scale,  3, "activation scales");
    rd(&w0_scale, 3, "weight scales");

    rq(w0, N_HIDDEN * N_IN,     "w0");
    rd(b0, N_HIDDEN,            "b0");
    rq(w2, N_HIDDEN * N_HIDDEN, "w2");
    rd(b2, N_HIDDEN,            "b2");
    rq(w4, N_OUT * N_HIDDEN,    "w4");
    rd(b4, N_OUT,               "b4");

    if (!ok) { std::fclose(f); return false; }

    // Only meaningful once every read succeeded.
    s0 = x_scale  * w0_scale;
    s2 = h1_scale * w2_scale;
    s4 = h2_scale * w4_scale;

    inv_x_scale  = 1.0f / x_scale;
    inv_h1_scale = 1.0f / h1_scale;
    inv_h2_scale = 1.0f / h2_scale;

    // A short file is caught above; this catches the opposite, a writer
    // emitting more than this reader knows about.
    char extra;
    if (std::fread(&extra, 1, 1, f) != 0) {
      std::fprintf(stderr, "weights_int8: trailing bytes -- layout mismatch\n");
      std::fclose(f); return false;
    }

    std::fclose(f);
    return true;
  }
};

// Quantize one float to int8, saturating.
//
// Clamp in the wide type BEFORE narrowing: a value of 300 cast straight to
// int8_t wraps to 44, which would sail through a post-cast range check.
// Calibration deliberately set scales so ~0.01% of activations exceed the
// range, so this path is exercised on real data.
//
// lrintf rounds to nearest (ties to even), matching numpy's np.round. A
// plain cast truncates toward zero and would not match the Python side.
inline int8_t quantize(float v, float scale) {
  long q = std::lrintf(v / scale);
  if (q < -127) q = -127;
  else if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}

// Same as quantize(), but takes 1/scale so the hot path multiplies rather
// than divides.
//
// NOTE: v * (1/s) is not always bitwise equal to v / s -- the reciprocal
// is itself rounded. Both Int8Backend and NeonBackend must switch together
// or the bitwise-equality check between them breaks.
inline int8_t quantize_inv(float v, float inv_scale) {
  long q = std::lrintf(v * inv_scale);
  if (q < -127) q = -127;
  else if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}

// Scalar int8 forward pass. No intrinsics -- this is the readable form of
// the arithmetic the NEON kernel vectorizes, and the thing that kernel is
// diffed against.
struct Int8Backend {
  const WeightsInt8& w;
  explicit Int8Backend(const WeightsInt8& weights) : w(weights) {}

  void forward(const float* in, float* out) const {
    int8_t x_q[N_IN];
    int8_t h1_q[N_HIDDEN];
    int8_t h2_q[N_HIDDEN];

    // Normalize in float, then quantize. The intermediate must be a float:
    // writing the normalized value into an int8_t first would truncate it
    // before quantization ever ran.
    for (int i = 0; i < N_IN; ++i) {
      float v = (in[i] - w.norm_mean[i]) * w.inv_norm_std[i];
      x_q[i]  = quantize_inv(v, w.inv_x_scale);
    }

    // Layer 0: 16 -> 32.
    //
    // acc is int32 because 16 terms of at most 127*127 = 16129 reach
    // ~258k; layer 2's 32 terms reach ~516k. int16 would overflow both.
    //
    // The stride is always the number of INPUTS to the layer: 16 here,
    // 32 in both layers below.
    for (int j = 0; j < N_HIDDEN; ++j) {
      int32_t acc = 0;
      for (int i = 0; i < N_IN; ++i) {
        acc += int32_t(x_q[i]) * int32_t(w.w0[j * N_IN + i]);
      }
      // Leave integer space once per neuron: acc is the true dot product
      // divided by (x_scale * w0_scale), so multiplying by s0 recovers it.
      float h = float(acc) * w.s0 + w.b0[j];
      h = h > 0.0f ? h : 0.0f;
      // Re-enter integer space for the next layer, at ITS scale.
      h1_q[j] = quantize_inv(h, w.inv_h1_scale);

    }

    // Layer 2: 32 -> 32.
    for (int j = 0; j < N_HIDDEN; ++j) {
      int32_t acc = 0;
      for (int i = 0; i < N_HIDDEN; ++i) {
        acc += int32_t(h1_q[i]) * int32_t(w.w2[j * N_HIDDEN + i]);
      }
      float h = float(acc) * w.s2 + w.b2[j];
      h = h > 0.0f ? h : 0.0f;
      h2_q[j] = quantize_inv(h, w.inv_h2_scale);
    }

    // Layer 4: 32 -> 3. No relu and no requantization -- logits stay float.
    for (int k = 0; k < N_OUT; ++k) {
      int32_t acc = 0;
      for (int j = 0; j < N_HIDDEN; ++j) {
        acc += int32_t(h2_q[j]) * int32_t(w.w4[k * N_HIDDEN + j]);
      }
      out[k] = float(acc) * w.s4 + w.b4[k];
    }
  }
};

} // namespace model