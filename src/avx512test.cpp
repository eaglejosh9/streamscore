// Avx512Backend vs Int8Backend: bit-exactness, then single-event speed.
//
//   ./build/avx512test data/weights_int8.bin data/aapl.bin
//
// 1. quantize16 against quantize_inv on edge values (ties, clamp bounds,
//    NaN, inf, lrintf overflow) -- the cases real data never reaches.
// 2. Both backends on N_CHECK market-hours records; logits must match
//    to the bit.
// 3. Best-of-REPS timing of ~1M sequential forward() calls each.

#include "features.hpp"
#include "model_avx512.hpp"
#include "model_int8.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#if !(defined(__AVX512F__) && defined(__AVX512VNNI__))
#error "avx512test needs AVX-512F + VNNI (build with -march=native on a VNNI CPU)"
#endif

namespace {

using clk = std::chrono::steady_clock;

constexpr int N_CHECK = 4096;
constexpr int REPS    = 7;
constexpr int PASSES  = 256;   // N_CHECK * PASSES = 1,048,576 events per rep

// Market-hours feature rows from a replay dump, flattened to n * N_IN.
// Same filter as cudabench: outside market hours every activation clamps.
bool load_features(const char* path, std::vector<float>& out, int want) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::perror(path); return false; }

  uint32_t hdr[2];
  if (std::fread(hdr, sizeof(uint32_t), 2, f) != 2 ||
      hdr[0] != feat::DUMP_MAGIC || hdr[1] != feat::DUMP_VERSION) {
    std::fprintf(stderr, "%s: bad header (want magic 0x%08x version %u)\n",
                 path, feat::DUMP_MAGIC, feat::DUMP_VERSION);
    std::fclose(f);
    return false;
  }

  std::vector<feat::Record> chunk(1 << 16);
  size_t total = 0, kept = 0;
  out.clear();
  out.reserve(size_t(want) * model::N_IN);
  size_t got;
  while (kept < size_t(want) &&
         (got = std::fread(chunk.data(), sizeof(feat::Record), chunk.size(), f)) > 0) {
    total += got;
    for (size_t i = 0; i < got && kept < size_t(want); ++i) {
      const auto& r = chunk[i];
      if (r.ts < feat::MARKET_OPEN || r.ts >= feat::MARKET_CLOSE) continue;
      out.insert(out.end(), r.f, r.f + model::N_IN);
      ++kept;
    }
  }
  std::fclose(f);
  std::printf("features: %zu market-hours rows (of %zu read) from %s\n",
              kept, total, path);
  if (kept < size_t(want)) {
    std::fprintf(stderr, "need %d market-hours rows, have %zu\n", want, kept);
    return false;
  }
  return true;
}

bool check_quantize() {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> v = {
      0.0f, -0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 0.49999997f,
      126.5f, 127.0f, 127.49f, 127.5f, 128.0f, -126.5f, -127.0f, -127.5f, -128.0f,
      1e9f, -1e9f, 0x1p31f, -0x1p31f, 0x1p62f, 0x1p63f, -0x1p63f, 0x1p64f, -0x1p64f,
      std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::denorm_min(), inf, -inf, nan, -nan,
  };
  // Plus a dense sweep across the clamp range, ties included.
  for (int i = -600; i <= 600; ++i) v.push_back(i * 0.25f);
  while (v.size() % 16) v.push_back(0.0f);

  size_t bad = 0;
  for (size_t base = 0; base < v.size(); base += 16) {
    alignas(64) int32_t q[16];
    _mm512_store_si512(q, model::Avx512Backend::quantize16(_mm512_loadu_ps(&v[base])));
    for (int i = 0; i < 16; ++i) {
      int8_t want = model::quantize_inv(v[base + i], 1.0f);
      if (q[i] == want) continue;
      if (bad < 10)
        std::printf("  quantize %.9g: scalar %d  avx512 %d\n", v[base + i], want, q[i]);
      ++bad;
    }
  }
  std::printf("quantize:    %s -- %zu values (ties, clamps, NaN, inf, 2^63)\n",
              bad ? "FAIL" : "PASS", v.size());
  return bad == 0;
}

template <class B>
double ns_per_event(const B& b, const std::vector<float>& feats, std::vector<float>& out) {
  double best = 1e30;
  for (int r = 0; r < REPS; ++r) {
    auto t0 = clk::now();
    for (int p = 0; p < PASSES; ++p)
      for (int i = 0; i < N_CHECK; ++i)
        b.forward(&feats[size_t(i) * model::N_IN], &out[size_t(i) * model::N_OUT]);
    auto t1 = clk::now();
    best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  return best / (double(N_CHECK) * PASSES);
}

} // namespace

int main(int argc, char** argv) {
  static_assert(model::N_IN == feat::N_FEATURES, "model and features disagree");
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <weights_int8.bin> <features.bin>\n", argv[0]);
    return 1;
  }

  model::WeightsInt8 wq;
  if (!wq.load(argv[1])) return 1;

  std::vector<float> feats;
  if (!load_features(argv[2], feats, N_CHECK)) return 1;

  model::Int8Backend   ref(wq);
  model::Avx512Backend fast(wq);

  if (!check_quantize()) return 1;

  // ---- bit-exactness -----------------------------------------------------
  const size_t n_vals = size_t(N_CHECK) * model::N_OUT;
  std::vector<float> out_ref(n_vals), out_fast(n_vals);
  for (int i = 0; i < N_CHECK; ++i) {
    ref.forward(&feats[size_t(i) * model::N_IN],  &out_ref[size_t(i) * model::N_OUT]);
    fast.forward(&feats[size_t(i) * model::N_IN], &out_fast[size_t(i) * model::N_OUT]);
  }
  if (std::memcmp(out_ref.data(), out_fast.data(), n_vals * sizeof(float)) != 0) {
    size_t bad = 0;
    for (size_t i = 0; i < n_vals; ++i) {
      uint32_t a, b;
      std::memcpy(&a, &out_ref[i], 4);
      std::memcpy(&b, &out_fast[i], 4);
      if (a == b) continue;
      if (bad < 10)
        std::printf("  event %zu logit %zu: int8 %.9g (0x%08x)  avx512 %.9g (0x%08x)\n",
                    i / model::N_OUT, i % model::N_OUT, out_ref[i], a, out_fast[i], b);
      ++bad;
    }
    std::printf("correctness: FAIL -- %zu of %zu logits differ\n", bad, n_vals);
    return 1;
  }
  std::printf("correctness: PASS -- %d events, %zu logits bit-exact\n\n", N_CHECK, n_vals);

  // ---- speed -------------------------------------------------------------
  std::vector<float> sink(n_vals);
  const double ns_ref  = ns_per_event(ref,  feats, sink);
  const double ns_fast = ns_per_event(fast, feats, sink);

  std::printf("best of %d, %d events each, one event per call, single thread\n",
              REPS, N_CHECK * PASSES);
  std::printf("  %-28s %8.2f ns/event\n", "Int8Backend (scalar)", ns_ref);
  std::printf("  %-28s %8.2f ns/event\n", "Avx512Backend (vpdpbusd)", ns_fast);
  std::printf("  speedup                      %8.2fx\n", ns_ref / ns_fast);

  // Keep the timed writes observable.
  float acc = 0.0f;
  for (float x : sink) acc += x;
  std::printf("  (checksum %g)\n", acc);
  return 0;
}
