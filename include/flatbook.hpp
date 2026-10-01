#pragma once
#include <cstdint>
#include <cstring>
#include <map>

namespace feed {

// Flat order book side: a fixed array near the touch, a map for the rest.
//
// Designed against measured access patterns over a full session
// (268M messages, 8,892 symbols, see docs/08-bookprofile.md):
//
//   operation mix     71.8% modify an existing level
//                     16.7% insert, 11.5% erase
//   rank of change    47.2% at the touch, 82% within the top 8, mean 7.5
//   depth             median 63, p99 2,047, max 4,227 (AAPL)
//
// Two conclusions, both counterintuitive enough that guessing would have
// got them wrong:
//
// 1. THE ARRAY IS REVERSED. The obvious layout puts the best price at
//    index 0. But 47% of all changes happen at the touch, and inserting
//    at index 0 shifts the entire array -- so the most common operation
//    would be the most expensive one. Measured mean shift under that
//    layout: 120 elements, 74M times over a session, ~107 GB of memmove.
//
//    Storing worst-at-0, best-at-the-end inverts it: a new best price is
//    an append and shifts nothing. The cost becomes `rank` instead of
//    `n - rank`, and the measured mean rank is 7.5 -- a 16x reduction
//    from the index order alone.
//
// 2. THE ARRAY IS CAPPED. AAPL holds 4,227 levels but the feature engine
//    reads five. 2,024 of 8,892 symbols ever exceed 64 levels, and
//    changes beyond rank 64 are 0.8% of traffic. So 64 entries stay flat
//    and contiguous; the cold tail spills to a map where its O(log n) is
//    paid rarely.
//
// Structure-of-arrays rather than array-of-structs: the top five prices
// land in one 20-byte run, which is one cache line, and nothing from the
// quantity array is dragged in alongside them.
//
// Cmp is the "is better" relation and does double duty -- std::greater
// for bids (higher is better) and std::less for asks (lower is better) --
// so one template covers both sides with no branching on side.
template <typename Cmp>
struct FlatSide {
  static constexpr int CAP = 64;

  // Promotion low-water mark. Erasing from the array does NOT immediately
  // pull a level up from the spill: every spilled level is worse than
  // everything in the array, so the array still holds the best n and
  // queries stay correct. Promotion is only needed before the array runs
  // short enough to stop covering the top five, and batching it amortises
  // the one shift it costs.
  static constexpr int LOW_WATER = 8;
  static constexpr int PROMOTE   = 32;

  alignas(64) uint32_t px[CAP];
  alignas(64) uint64_t qty[CAP];
  int n = 0;                              // entries in the array
  std::map<uint32_t, uint64_t, Cmp> spill;  // strictly worse than px[0]

  static constexpr Cmp better{};

  bool   empty()  const { return n == 0 && spill.empty(); }
  size_t levels() const { return size_t(n) + spill.size(); }

  // The touch lives at the high-index end.
  uint32_t best_price() const { return n ? px[n - 1] : 0; }
  uint64_t best_qty()   const { return n ? qty[n - 1] : 0; }

  // Walk out from the touch: index n-1 is best, n-2 next, and so on.
  // Contiguous and descending in betterness, which is what the feature
  // engine wants.
  int top(int k, uint32_t* out_px, uint64_t* out_qty) const {
    int got = 0;
    for (int i = n - 1; i >= 0 && got < k; --i, ++got) {
      out_px[got]  = px[i];
      out_qty[got] = qty[i];
    }
    return got;
  }

  // Find `price` in the array, or the index where it would be inserted.
  //
  // Scans DOWN from the touch rather than binary searching: 82% of
  // changes are within 8 of the touch, so a few linear steps through one
  // cache line beats log2(64) = 6 unpredictable branches.
  int locate(uint32_t price, bool* found) const {
    for (int i = n - 1; i >= 0; --i) {
      if (px[i] == price)          { *found = true;  return i; }
      if (better(price, px[i]))    { *found = false; return i + 1; }
    }
    *found = false;
    return 0;   // worse than everything in the array
  }

  void set(uint32_t price, uint64_t q) {
    bool found = false;
    const int k = locate(price, &found);

    if (found) {
      if (q != 0) { qty[k] = q; return; }     // 71.8% case: one store
      erase_at(k);
      return;
    }

    // Not in the array. It belongs there only if it beats the worst
    // entry, or if there is room.
    const bool belongs_in_array = (k > 0) || (n < CAP && spill.empty());
    if (belongs_in_array) {
      if (q == 0) { spill.erase(price); return; }
      insert_at(k, price, q);
      return;
    }

    // Worse than everything in a full array: the cold tail.
    if (q == 0) spill.erase(price);
    else        spill[price] = q;
  }

  uint64_t qty_at(uint32_t price) const {
    bool found = false;
    const int k = locate(price, &found);
    if (found) return qty[k];
    auto it = spill.find(price);
    return it == spill.end() ? 0 : it->second;
  }

 private:
  void insert_at(int k, uint32_t price, uint64_t q) {
    int at = k;
    if (n == CAP) {
        spill[px[0]] = qty[0];
        std::memmove(px,  px  + 1, size_t(CAP - 1) * sizeof(px[0]));
        std::memmove(qty, qty + 1, size_t(CAP - 1) * sizeof(qty[0]));
        --n;
        --at;
    }

    // TODO 2. Make room at `at` and write the new level.
    //         Elements [at, n) shift up by one: that is `n - at` moves,
    //         and for an append (at == n) it is zero -- which is the
    //         whole point of the reversed layout.
    //
    //         memmove(px  + at + 1, px  + at, (n - at) * sizeof(px[0]));
    //         memmove(qty + at + 1, qty + at, (n - at) * sizeof(qty[0]));
    //         then px[at] = price; qty[at] = q; ++n;
    const int move = n - at;
    if (move > 0) {
        std::memmove(px  + at + 1, px  + at, size_t(move) * sizeof(px[0]));
        std::memmove(qty + at + 1, qty + at, size_t(move) * sizeof(qty[0]));
    }
    px[at]  = price;
    qty[at] = q;
    ++n;
    (void)at; (void)price; (void)q;
  }

  void erase_at(int k) {
    const int move = n - k - 1;
    if (move > 0) {
      std::memmove(px  + k, px  + k + 1, size_t(move) * sizeof(px[0]));
      std::memmove(qty + k, qty + k + 1, size_t(move) * sizeof(qty[0]));
    }
    --n;
    if (n < LOW_WATER && !spill.empty()) promote();
  }

  // Pull the best entries out of the spill back into the array.
  //
  // Done in a batch rather than one at a time: each promotion shifts the
  // whole array up to make room at the bottom, so moving 32 levels costs
  // one shift instead of 32.
  void promote() {
    int take = 0;
    for (auto it = spill.begin(); it != spill.end() && take < PROMOTE; ++it) ++take;
    if (take == 0 || n + take > CAP) take = CAP - n;
    if (take <= 0) return;

    std::memmove(px  + take, px,  size_t(n) * sizeof(px[0]));
    std::memmove(qty + take, qty, size_t(n) * sizeof(qty[0]));

    // spill is ordered best-first, and the array runs worst-first, so the
    // best promoted level lands at index take-1 and fills downward.
    int slot = take - 1;
    auto it = spill.begin();
    for (int i = 0; i < take; ++i) {
      px[slot]  = it->first;
      qty[slot] = it->second;
      --slot;
      it = spill.erase(it);
    }
    n += take;
  }
};

// Bids: higher is better, so std::greater is the "is better" relation and
// also orders the spill map best-first.
// Asks: lower is better, so std::less does both jobs.
struct FlatBook {
  FlatSide<std::greater<uint32_t>> bids;
  FlatSide<std::less<uint32_t>>    asks;

  void apply(char side, uint32_t price, uint64_t qty) {
    if (side == 'B') bids.set(price, qty);
    else             asks.set(price, qty);
  }

  uint32_t best_bid() const { return bids.best_price(); }
  uint32_t best_ask() const { return asks.best_price(); }
  bool two_sided()    const { return bids.n > 0 && asks.n > 0; }
  size_t levels()     const { return bids.levels() + asks.levels(); }

  bool consistent() const {
    if (!two_sided()) return true;
    return best_bid() < best_ask();
  }

  uint64_t qty_at(char side, uint32_t price) const {
    return side == 'B' ? bids.qty_at(price) : asks.qty_at(price);
  }

  void clear() {
    bids.n = 0; bids.spill.clear();
    asks.n = 0; asks.spill.clear();
  }
};

} // namespace feed