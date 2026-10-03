// CPU vs GPU crossover for the int8 MLP.
//
// The CPU scores events one at a time with model::FastBackend; the GPU
// scores a whole batch per launch. For each batch size this prints:
//
//   cpu        n sequential FastBackend::forward calls
//   gpu kern   kernel only (cudaEvent, averaged over many launches)
//   gpu total  H2D + kernel + D2H -- what a caller actually pays
//
// The gap between the two GPU columns is the PCIe round trip, and at small
// batches it dominates everything.
//
// First, a bit-exactness check: both sides run the same integer arithmetic,
// so the logits must match to the bit.
//
//   ./build/cudabench data/weights_int8.bin data/aapl.bin

#include "backend_select.hpp"
#include "features.hpp"
#include "model_cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

constexpr int N_CHECK = 4096;
constexpr int REPS    = 7;
constexpr int BATCHES[] = {1, 8, 32, 128, 256, 512, 1024, 2048,
                           4096, 8192, 16384, 32768, 65536};
constexpr int MAX_BATCH = 65536;

// Market-hours feature rows from a replay dump, flattened to n * N_IN.
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

  // Outside market hours the inputs are out of distribution and saturate
  // every int8 activation -- a check on those would compare clamps.
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

template <class F>
double best_of_us(F&& fn) {
  double best = 1e30;
  for (int r = 0; r < REPS; ++r) {
    auto t0 = clk::now();
    fn();
    auto t1 = clk::now();
    best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return best;
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
  if (!load_features(argv[2], feats, MAX_BATCH)) return 1;

  auto info = model::CudaBackend::device_info();
  std::printf("gpu: %s  sm_%d%d  %d SMs  %.0f GB/s peak DRAM\n",
              info.name, info.cc_major, info.cc_minor, info.sm_count,
              info.mem_bandwidth_gbs);
  std::printf("cpu: %s, one event per call, single thread\n\n",
              model::fast_backend_name());

  model::FastBackend cpu(wq);
  model::CudaBackend gpu(wq, MAX_BATCH);

  auto* h_in  = static_cast<float*>(model::CudaBackend::alloc_pinned(
      size_t(MAX_BATCH) * model::N_IN * sizeof(float)));
  auto* h_out = static_cast<float*>(model::CudaBackend::alloc_pinned(
      size_t(MAX_BATCH) * model::N_OUT * sizeof(float)));
  std::memcpy(h_in, feats.data(), feats.size() * sizeof(float));
  std::vector<float> cpu_out(size_t(MAX_BATCH) * model::N_OUT);

  // ---- correctness -------------------------------------------------------
  for (int i = 0; i < N_CHECK; ++i)
    cpu.forward(h_in + size_t(i) * model::N_IN, cpu_out.data() + size_t(i) * model::N_OUT);
  gpu.forward_batch(h_in, h_out, N_CHECK);

  const size_t n_vals = size_t(N_CHECK) * model::N_OUT;
  if (std::memcmp(cpu_out.data(), h_out, n_vals * sizeof(float)) == 0) {
    std::printf("correctness: PASS -- %d events, %zu logits bit-exact\n\n",
                N_CHECK, n_vals);
  } else {
    size_t bad = 0;
    for (size_t i = 0; i < n_vals; ++i) {
      uint32_t a, b;
      std::memcpy(&a, &cpu_out[i], 4);
      std::memcpy(&b, &h_out[i], 4);
      if (a == b) continue;
      if (bad < 10)
        std::printf("  event %zu logit %zu: cpu %.9g (0x%08x)  gpu %.9g (0x%08x)\n",
                    i / model::N_OUT, i % model::N_OUT, cpu_out[i], a, h_out[i], b);
      ++bad;
    }
    std::printf("correctness: FAIL -- %zu of %zu logits differ\n", bad, n_vals);
    model::CudaBackend::free_pinned(h_in);
    model::CudaBackend::free_pinned(h_out);
    return 1;
  }

  // ---- sweep -------------------------------------------------------------
  std::printf("%8s %12s %12s %12s %10s %10s\n",
              "batch", "cpu us", "gpu kern us", "gpu tot us", "kern x", "total x");
  std::printf("%8s %12s %12s %12s %10s %10s\n",
              "", "", "", "", "(cpu/kern)", "(cpu/tot)");

  for (int n : BATCHES) {
    double cpu_us = best_of_us([&] {
      for (int i = 0; i < n; ++i)
        cpu.forward(h_in + size_t(i) * model::N_IN, cpu_out.data() + size_t(i) * model::N_OUT);
    });

    gpu.forward_batch(h_in, h_out, n);   // warm-up; also leaves inputs resident
    double tot_us = best_of_us([&] { gpu.forward_batch(h_in, h_out, n); });

    // Enough launches that event resolution (~0.5 us) is noise.
    double kern_us = 1e3 * gpu.time_kernel_ms(n, 200);

    std::printf("%8d %12.2f %12.2f %12.2f %10.2f %10.2f\n",
                n, cpu_us, kern_us, tot_us, cpu_us / kern_us, cpu_us / tot_us);
  }

  model::CudaBackend::free_pinned(h_in);
  model::CudaBackend::free_pinned(h_out);
  return 0;
}
