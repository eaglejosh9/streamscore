// Correctness and throughput for the batched backends.
//
// Three implementations must agree BIT FOR BIT:
//
//   NeonBackend          one event at a time, lanes = features
//   BatchScalarBackend   n events, plain loops, the readable oracle
//   BatchNeonBackend     n events, lanes = events
//
// They are the same integer arithmetic in different orders, and integer
// addition is associative, so there is no rounding to excuse a
// difference. Anything other than exact equality is a kernel bug -- most
// likely vpadalq_s16 in place of vaddw_s16, which merges adjacent lanes
// and produces plausible but wrong logits.
//
// Then the sweep: throughput against batch size. At n=1 the batch
// machinery is pure overhead; it should climb steeply and then flatten
// once the weights no longer fit in L1 and the reuse batching bought
// stops paying.

#include "features.hpp"
#include "model.hpp"
#include "model_int8.hpp"
#include "model_neon.hpp"
#include "model_batch.hpp"
#include "model_batch_neon.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <time.h>
static inline uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
#else
#include <time.h>
static inline uint64_t now_ns() {
  timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec;
}
#endif

namespace {

// Compare two logit arrays bitwise. memcmp rather than a tolerance:
// these must be identical, not close.
bool bit_equal(const float* a, const float* b, size_t count) {
  return std::memcmp(a, b, count * sizeof(float)) == 0;
}

void report_mismatch(const char* lhs, const char* rhs,
                     const float* a, const float* b, int n) {
  int shown = 0;
  for (int e = 0; e < n && shown < 5; ++e) {
    for (int k = 0; k < model::N_OUT; ++k) {
      const size_t i = size_t(e) * model::N_OUT + k;
      if (a[i] != b[i]) {
        std::printf("    event %d logit %d:  %s=%.9g  %s=%.9g  (delta %.3g)\n",
                    e, k, lhs, a[i], rhs, b[i], double(a[i] - b[i]));
        ++shown;
        break;
      }
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <weights_int8.bin> <features.bin>\n", argv[0]);
    return 1;
  }

  model::WeightsInt8 w;
  if (!w.load(argv[1])) return 1;

  model::NeonBackend       single(w);
  model::BatchScalarBackend bscalar(w);
  model::BatchNeonBackend   bneon(w);

  // Load market-hours records: the model is only calibrated for that
  // window, and feeding it out-of-distribution inputs saturates every
  // int8 activation (see docs/05-backends.md).
  FILE* fin = std::fopen(argv[2], "rb");
  if (!fin) { std::perror("features"); return 1; }
  uint32_t hdr[2];
  if (std::fread(hdr, sizeof(uint32_t), 2, fin) != 2 ||
      hdr[0] != feat::DUMP_MAGIC || hdr[1] != feat::DUMP_VERSION) {
    std::fprintf(stderr, "features: bad header\n");
    return 1;
  }

  std::vector<float> inputs;
  inputs.reserve(size_t(model::MAX_BATCH) * model::N_IN);
  {
    feat::Record r;
    int kept = 0;
    while (kept < model::MAX_BATCH &&
           std::fread(&r, sizeof(feat::Record), 1, fin) == 1) {
      if (r.ts < feat::MARKET_OPEN || r.ts >= feat::MARKET_CLOSE) continue;
      inputs.insert(inputs.end(), r.f, r.f + model::N_IN);
      ++kept;
    }
  }
  std::fclose(fin);

  const int loaded = int(inputs.size() / model::N_IN);
  std::printf("loaded %d market-hours records\n\n", loaded);
  if (loaded < 64) { std::fprintf(stderr, "not enough records\n"); return 1; }

  std::vector<float> out_single(size_t(loaded) * model::N_OUT);
  std::vector<float> out_scalar(size_t(loaded) * model::N_OUT);
  std::vector<float> out_neon(size_t(loaded) * model::N_OUT);

  // ---- correctness ----------------------------------------------------
  //
  // Sizes chosen to exercise the block edges: 1 and 15 are entirely
  // partial blocks, 16 and 64 are exact multiples of LANES, 17 and 100
  // leave a remainder. The partial-block path is where an off-by-one
  // hides, because it only runs on the last iteration.
  const int sizes[] = {1, 7, 15, 16, 17, 31, 32, 64, 100, 255, 256, 1000};
  bool all_ok = true;

  std::printf("%-8s %-14s %-14s\n", "batch", "scalar vs 1x1", "neon vs scalar");
  for (int n : sizes) {
    if (n > loaded) continue;

    for (int e = 0; e < n; ++e) {
      single.forward(&inputs[size_t(e) * model::N_IN],
                     &out_single[size_t(e) * model::N_OUT]);
    }
    bscalar.forward_batch(inputs.data(), out_scalar.data(), n);
    bneon.forward_batch(inputs.data(), out_neon.data(), n);

    const size_t count = size_t(n) * model::N_OUT;
    const bool ok_scalar = bit_equal(out_single.data(), out_scalar.data(), count);
    const bool ok_neon   = bit_equal(out_scalar.data(), out_neon.data(), count);

    std::printf("%-8d %-14s %-14s\n", n,
                ok_scalar ? "exact" : "MISMATCH",
                ok_neon   ? "exact" : "MISMATCH");

    if (!ok_scalar) {
      report_mismatch("1x1", "scalar", out_single.data(), out_scalar.data(), n);
      all_ok = false;
    }
    if (!ok_neon) {
      report_mismatch("scalar", "neon", out_scalar.data(), out_neon.data(), n);
      all_ok = false;
    }
  }

  if (!all_ok) {
    std::printf("\nFAIL. Check, in order: vaddw_s16 rather than vpadalq_s16 in\n"
                "accumulate (pairwise addition merges two events into one\n"
                "lane); vcvtnq rather than vcvtq for round-to-nearest; the\n"
                "clamp at -127 rather than vqmovn's -128; and the partial\n"
                "final block not storing past n.\n");
    return 1;
  }
  std::printf("\nall three backends agree bit for bit.\n");

  // ---- throughput -----------------------------------------------------
  //
  // Each configuration scores the same total number of events, so the
  // comparison is like for like: a batch of 1 run 4096 times against a
  // batch of 4096 run once.
  std::printf("\n%-8s %-12s %-12s %-12s %s\n",
              "batch", "ns/event", "M events/s", "vs 1x1", "");

  constexpr int TOTAL = 1 << 20;   // events scored per configuration
  constexpr int REPS  = 5;
  double base_ns = 0.0;

  for (int n : sizes) {
    if (n > loaded) continue;

    // Warm up: the first pass faults pages and pulls weights into cache,
    // and timing that would measure memory rather than the kernel.
    bneon.forward_batch(inputs.data(), out_neon.data(), n);

    uint64_t best = UINT64_MAX;
    for (int r = 0; r < REPS; ++r) {
      const uint64_t t0 = now_ns();
      for (int done = 0; done < TOTAL; done += n) {
        bneon.forward_batch(inputs.data(), out_neon.data(), n);
      }
      const uint64_t dt = now_ns() - t0;
      if (dt < best) best = dt;
    }

    const double ns_per_event = double(best) / double(TOTAL);
    if (base_ns == 0.0) base_ns = ns_per_event;
    std::printf("%-8d %-12.1f %-12.2f %-12.2fx\n",
                n, ns_per_event, 1000.0 / ns_per_event,
                base_ns / ns_per_event);
  }

  // Single-event NEON for reference: the latency-mode number from phase 5.
  {
    float logits[model::N_OUT];
    uint64_t best = UINT64_MAX;
    for (int r = 0; r < REPS; ++r) {
      const uint64_t t0 = now_ns();
      for (int e = 0; e < TOTAL; ++e) {
        single.forward(&inputs[size_t(e % loaded) * model::N_IN], logits);
      }
      const uint64_t dt = now_ns() - t0;
      if (dt < best) best = dt;
    }
    const double ns = double(best) / double(TOTAL);
    std::printf("\nsingle-event NeonBackend (latency mode): %.1f ns/event, "
                "%.2f M events/s\n", ns, 1000.0 / ns);
  }

  return 0;
}