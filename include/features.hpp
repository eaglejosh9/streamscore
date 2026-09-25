#pragma once
#include <cstddef>
#include <cstdint>

namespace feat {

// One feature vector, written verbatim to the dump file.
// Fixed layout, no padding surprises, no pointers — this struct IS the
// on-disk format, so changing it changes the file version.
#pragma pack(push, 1)
struct Record {
  uint64_t ts;          // ns since midnight, from the message header
  int32_t  mid;         // (best_bid + best_ask) / 2, price units
  int32_t  spread;      // best_ask - best_bid
  float    imbalance;   // -1.0 .. +1.0
  float    ema_diff;    // short EMA - long EMA of mid
  float    signed_vol;  // decayed signed aggressive volume
};
#pragma pack(pop)
static_assert(sizeof(Record) == 28, "Record layout changed — bump the dump version");

// Magic + version so a reader can reject a stale dump.
constexpr uint32_t DUMP_MAGIC   = 0x53544D31;  // "STM1"
constexpr uint32_t DUMP_VERSION = 1;

// An exponential moving average.
//   ema += alpha * (x - ema)
// alpha near 1 reacts fast, near 0 reacts slowly.
struct Ema {
  float value = 0.0f;
  float alpha;
  bool  primed = false;

  explicit Ema(float a) : alpha(a) {}

  // TODO: update(x) — on the first call, set value = x and prime it
  //       (otherwise the EMA spends thousands of events climbing from 0).
  //       After that, apply the recurrence above. Return the new value.
  float update(float x) {
    if (!primed) {
        value = x;
        primed = true;
        return value;
    }
    value += alpha * (x -  value);
    return value;
  }
};

struct Engine {
  Ema short_ema{0.10f};
  Ema long_ema{0.01f};
  float signed_vol = 0.0f;

  // Decay applied to signed_vol on every event, so old trades fade.
  // Without decay this is an unbounded running sum and drifts forever.
  static constexpr float VOL_DECAY = 0.999f;

  // TODO: on_trade(shares, resting_side)
  //   Called from E/C messages. If the RESTING order was a sell, the
  //   aggressor was a buyer -> positive. Resting buy -> negative.
  //   Add to signed_vol.
  void on_trade(uint32_t shares, char resting_side) {
    if (resting_side == 'S') 
        signed_vol += shares;
    else
        signed_vol -= shares;
  }


  // TODO: compute(ts, best_bid, best_ask, bid_qty, ask_qty) -> Record
  //   Fill every field. Decay signed_vol here (once per event, not per
  //   trade). Decide what to emit when one side of the book is empty —
  //   imbalance has a 0/0 case and mid is meaningless.
  Record compute (uint64_t ts, uint32_t best_bid, uint32_t best_ask, uint64_t bid_qty, uint64_t ask_qty) {
    int32_t mid    = int32_t(best_bid + (best_ask - best_bid) / 2);
    int32_t spread = int32_t(best_ask) - int32_t(best_bid);
    float bq = float(bid_qty);
    float aq = float(ask_qty);
    float imbalance = (bq - aq) / (bq + aq);
    float s = short_ema.update(float(mid));
    float l = long_ema.update(float(mid));
    float ema_diff = s - l;
    signed_vol *= VOL_DECAY;
    Record r;
    r.ts         = ts;
    r.mid        = mid;
    r.spread     = spread;
    r.imbalance  = imbalance;
    r.ema_diff   = ema_diff;
    r.signed_vol = signed_vol;
    return r;
  }

};

} // namespace feat