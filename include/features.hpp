#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace feat {

constexpr int N_FEATURES = 16;
constexpr int N_LEVELS   = 5;   // book depth we look at

// Top-N levels of one side of the book, filled by the caller.
struct Levels {
  uint32_t px[N_LEVELS]  = {};
  uint64_t qty[N_LEVELS] = {};
  int      n = 0;               // how many levels actually exist (0..N_LEVELS)
};

#pragma pack(push, 1)
struct Record {
  uint64_t ts;
  int32_t  mid;                 // label target, not a feature
  int32_t  spread;              // filter key, not a feature
  float    f[N_FEATURES];       // contiguous — this is what the kernel reads
};
#pragma pack(pop)
static_assert(sizeof(Record) == 80, "Record layout changed — bump DUMP_VERSION");

constexpr uint32_t DUMP_MAGIC   = 0x53544D32;  // "STM2"
constexpr uint32_t DUMP_VERSION = 2;

struct Ema {
  float value = 0.0f;
  float alpha;
  bool  primed = false;
  explicit Ema(float a) : alpha(a) {}

  float update(float x) {
    if (!primed) { value = x; primed = true; return value; }
    value += alpha * (x - value);
    return value;
  }
};

// Sum quantity over the first k levels (or fewer, if the book is thinner).
static float sum_qty(const Levels& lv, int k) {
  uint64_t res = 0;
  for (int i = 0; i < std::min(lv.n, k); i++) {
    res += lv.qty[i];
  }
  return float(res);
}

//requires at least one input on levels
static float imbalance_at(const Levels& bid, const Levels& ask, int k) {
  float bq = sum_qty(bid, k);
  float aq = sum_qty(ask, k);
  return (bq - aq) / (bq + aq);
}

struct Engine {
  Ema fast_ema{0.10f};
  Ema mid_ema{0.03f};
  Ema slow_ema{0.01f};
  Ema abs_move{0.01f};          // realized-vol proxy

  float signed_vol = 0.0f;
  float trade_rate = 0.0f;
  int32_t prev_mid = 0;
  bool have_prev = false;

  static constexpr float VOL_DECAY  = 0.999f;
  static constexpr float RATE_DECAY = 0.999f;

  void on_trade(uint32_t shares, char resting_side) {
    if (resting_side == 'S') signed_vol += float(shares);
    else                     signed_vol -= float(shares);
    trade_rate += 1.0f;
  }

  Record compute(uint64_t ts, const Levels& bid, const Levels& ask) {
    // Caller guarantees bid.n >= 1 and ask.n >= 1.

    signed_vol *= VOL_DECAY;
    trade_rate *= RATE_DECAY;

    int32_t best_bid = int32_t(bid.px[0]);
    int32_t best_ask = int32_t(ask.px[0]);
    int32_t mid      = best_bid + (best_ask - best_bid) / 2;
    int32_t spread   = best_ask - best_bid;

    Record r;
    r.ts = ts; r.mid = mid; r.spread = spread;

    // TODO 0-2. Imbalance at depth 1, 3, and 5.
    //   Write one helper that sums qty over the first k levels of each side
    //   (capped at .n) and returns (bq - aq) / (bq + aq) as a float.
    //   Cast out of uint64_t before subtracting.
    //   Deeper imbalance is smoother and slower-moving than top-of-book.
    r.f[0] = imbalance_at(bid, ask, 1);
    r.f[1] = imbalance_at(bid, ask, 3);
    r.f[2] = imbalance_at(bid, ask, 5);

    // TODO 3. Normalized spread: spread relative to mid, e.g.
    //   float(spread) / float(mid) — a scale-free version so the feature
    //   doesn't encode "AAPL costs $290".
    //   Consider scaling by 1e4 so it isn't a tiny number near zero.
    r.f[3] = (float(spread) / float(mid)) * 1e4f;

    // TODO 4. Microprice deviation.
    //   microprice = (best_bid*ask_qty + best_ask*bid_qty) / (bid_qty+ask_qty)
    //   Note the CROSS weighting: heavy bid size pulls the fair price UP
    //   toward the ask. Then emit (microprice - mid) / spread, which is
    //   bounded roughly to [-0.5, 0.5].
    float bq0 = float(bid.qty[0]);
    float aq0 = float(ask.qty[0]);
    float microprice = (float(best_bid) * aq0 + float(best_ask) * bq0) / (bq0 + aq0);
    r.f[4] = (microprice - float(mid)) / float(spread);

        // 5-6. Momentum at two timescales. All three EMAs update every event in
    // a fixed order; skipping one would desync them and make the
    // differences meaningless.
    float fast = fast_ema.update(float(mid));
    float med  = mid_ema.update(float(mid));
    float slow = slow_ema.update(float(mid));
    float inv_spread = 1.0f / float(spread);
    r.f[5] = (fast - med)  * inv_spread;
    r.f[6] = (med  - slow) * inv_spread;

    // 7. Signed aggressive volume, scaled into the same rough range as the
    // other features. The divisor is arbitrary but must be identical in the
    // online path — an offline/online mismatch here is exactly the kind of
    // train/serve skew the phase-5 harness checks for.
    r.f[7] = signed_vol / 1000.0f;

    // 8. Trade arrival rate, same decay family as signed_vol.
    r.f[8] = trade_rate;

    // 9-10. Total resting depth per side. Log because depth is heavy-tailed:
    // a few levels hold enormous size and the raw value would dominate the
    // input scale.
    r.f[9]  = std::log1p(sum_qty(bid, N_LEVELS));
    r.f[10] = std::log1p(sum_qty(ask, N_LEVELS));

    // 11-12. Book slope: how far price walks across the available levels,
    // in units of spread. Steeper means thinner. Both positive by
    // construction since bids descend and asks ascend from the touch.
    r.f[11] = float(int32_t(bid.px[0]) - int32_t(bid.px[bid.n - 1])) * inv_spread;
    r.f[12] = float(int32_t(ask.px[ask.n - 1]) - int32_t(ask.px[0])) * inv_spread;

    // 13-14. How many of the N_LEVELS slots are actually populated. A thin
    // book (n < 5) is a different regime and the model should see that
    // directly rather than inferring it from the slope.
    r.f[13] = float(bid.n) / float(N_LEVELS);
    r.f[14] = float(ask.n) / float(N_LEVELS);

    // 15. Realized volatility proxy: EMA of absolute mid changes.
    float move = have_prev ? std::fabs(float(mid - prev_mid)) : 0.0f;
    r.f[15] = abs_move.update(move) * inv_spread;

    // State carried to the next event. Must be the last thing compute does.
    prev_mid  = mid;
    have_prev = true;
    return r;
  }
};

} // namespace feat