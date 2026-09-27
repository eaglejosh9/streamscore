#pragma once
#include <cstdint>
#include <cstdio>
#include <cstddef>

namespace model {

constexpr uint32_t WEIGHTS_MAGIC   = 0x53544D57;  // "STMW"
constexpr uint32_t WEIGHTS_VERSION = 1;

constexpr int N_IN     = 16;
constexpr int N_HIDDEN = 32;
constexpr int N_OUT    = 3;

// Weights, loaded once at startup and never modified afterward.
//
// Layout matches python/export.py exactly: a 20-byte header (magic,
// version, n_in, n_hidden, n_out) followed by eight float32 arrays in the
// order below. Nothing enforces that agreement but the version number, so
// any change here needs a matching change there and a version bump.
//
// Matrices are row-major with shape (out, in), which is how PyTorch stores
// nn.Linear.weight -- no transpose is needed at export. Row j holds
// neuron j's weights contiguously, which is the order the kernel reads
// them in.
struct Weights {
  float norm_mean[N_IN];
  float norm_std[N_IN];
  float w0[N_HIDDEN * N_IN];        // w0[j * N_IN + i]
  float b0[N_HIDDEN];
  float w2[N_HIDDEN * N_HIDDEN];    // w2[j * N_HIDDEN + i]
  float b2[N_HIDDEN];
  float w4[N_OUT * N_HIDDEN];       // w4[k * N_HIDDEN + j]
  float b4[N_OUT];

  bool load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror("weights"); return false; }

    uint32_t hdr[5];
    if (std::fread(hdr, sizeof(uint32_t), 5, f) != 5) {
      std::fprintf(stderr, "weights: truncated header\n");
      std::fclose(f);
      return false;
    }

    // Each check names the specific field that mismatched -- a bad magic
    // and a bad n_in are different bugs and want different next steps.
    if (hdr[0] != WEIGHTS_MAGIC) {
      std::fprintf(stderr, "weights: bad magic 0x%08x (want 0x%08x)\n",
                   hdr[0], WEIGHTS_MAGIC);
      std::fclose(f); return false;
    }
    if (hdr[1] != WEIGHTS_VERSION) {
      std::fprintf(stderr, "weights: version %u, this build expects %u\n",
                   hdr[1], WEIGHTS_VERSION);
      std::fclose(f); return false;
    }
    if (hdr[2] != N_IN) {
      std::fprintf(stderr, "weights: n_in %u, this build expects %d\n",
                   hdr[2], N_IN);
      std::fclose(f); return false;
    }
    if (hdr[3] != N_HIDDEN) {
      std::fprintf(stderr, "weights: n_hidden %u, this build expects %d\n",
                   hdr[3], N_HIDDEN);
      std::fclose(f); return false;
    }
    if (hdr[4] != N_OUT) {
      std::fprintf(stderr, "weights: n_out %u, this build expects %d\n",
                   hdr[4], N_OUT);
      std::fclose(f); return false;
    }

    // Order must match export.py's key list. Once one read fails the rest
    // become no-ops, so a truncated file produces one clear message rather
    // than a cascade.
    bool ok = true;
    auto rd = [&](float* dst, size_t n, const char* what) {
      if (!ok) return;
      if (std::fread(dst, sizeof(float), n, f) != n) {
        std::fprintf(stderr, "weights: truncated reading %s\n", what);
        ok = false;
      }
    };

    rd(norm_mean, N_IN,                "norm_mean");
    rd(norm_std,  N_IN,                "norm_std");
    rd(w0,        N_HIDDEN * N_IN,     "w0");
    rd(b0,        N_HIDDEN,            "b0");
    rd(w2,        N_HIDDEN * N_HIDDEN, "w2");
    rd(b2,        N_HIDDEN,            "b2");
    rd(w4,        N_OUT * N_HIDDEN,    "w4");
    rd(b4,        N_OUT,               "b4");

    if (!ok) { std::fclose(f); return false; }

    // The file must end exactly here. A short file is caught above; this
    // catches the opposite -- a writer that emits more than this reader
    // knows about, which means the layouts have diverged.
    char extra;
    if (std::fread(&extra, 1, 1, f) != 0) {
      std::fprintf(stderr, "weights: trailing bytes -- layout mismatch\n");
      std::fclose(f); return false;
    }

    std::fclose(f);
    return true;
  }
};

// Scalar float32 forward pass.
//
// This is the correctness oracle. It is diffed against PyTorch once to
// prove the weights and indexing are right; after that, every optimized
// backend is diffed against THIS rather than against PyTorch, because it
// runs in the same process on the same inputs and isolates the kernel
// change from everything else.
struct ScalarBackend {
  const Weights& w;
  explicit ScalarBackend(const Weights& weights) : w(weights) {}

  // in:  N_IN raw features, straight from feat::Record.f
  // out: N_OUT logits (no softmax -- argmax is unchanged by it, and the
  //      PyTorch model returns logits too, so the diff compares like
  //      against like)
  void forward(const float* in, float* out) const {
    float x[N_IN];
    float h1[N_HIDDEN];
    float h2[N_HIDDEN];

    // Must match dataset.normalize exactly, including the stats used.
    // A divergence here is train/serve skew: the model sees inputs on a
    // different scale than it was fit on, and nothing errors.
    for (int i = 0; i < N_IN; ++i) {
      x[i] = (in[i] - w.norm_mean[i]) / w.norm_std[i];
    }

    // Layer 0: 16 -> 32. j indexes the output neuron, i the input.
    // Starting the accumulator at the bias avoids a separate zero-then-add
    // step and keeps the running sum in a register.
    for (int j = 0; j < N_HIDDEN; ++j) {
      float sum = w.b0[j];
      for (int i = 0; i < N_IN; ++i) {
        sum += x[i] * w.w0[j * N_IN + i];
      }
      h1[j] = sum > 0.0f ? sum : 0.0f;   // relu
    }

    // Layer 2: 32 -> 32.
    for (int j = 0; j < N_HIDDEN; ++j) {
      float sum = w.b2[j];
      for (int i = 0; i < N_HIDDEN; ++i) {
        sum += h1[i] * w.w2[j * N_HIDDEN + i];
      }
      h2[j] = sum > 0.0f ? sum : 0.0f;   // relu
    }

    // Layer 4: 32 -> 3. No relu -- these are logits.
    for (int k = 0; k < N_OUT; ++k) {
      float sum = w.b4[k];
      for (int j = 0; j < N_HIDDEN; ++j) {
        sum += h2[j] * w.w4[k * N_HIDDEN + j];
      }
      out[k] = sum;
    }
  }
};

} // namespace model