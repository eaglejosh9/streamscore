#pragma once
#include "feed.hpp"
#include "features.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "flatbook.hpp"

namespace feed {

// Everything tracked per symbol.
//
// Each symbol needs its own feature state: the EMAs, the decayed signed
// volume, prev_mid. Those are running values over that symbol's own event
// stream and sharing them across symbols would be meaningless -- AAPL's
// momentum says nothing about ZVZZT's.
struct SymbolState {
  FlatBook book;
  feat::Engine engine;
  uint64_t events = 0;
  uint64_t emitted = 0;
  uint32_t crossings = 0;      // times this symbol's book inverted
  int64_t  worst_cross = 0;    // deepest inversion, in price units
  // Trading state from the last 'H' message. 'T' = trading normally.
  // Defaults to 'T' because a symbol with no H message before its first
  // quote is assumed live -- the alternative, assuming halted, would
  // silently drop every symbol NASDAQ never sends an action for.
  char state = 'T';
  uint32_t cross_while_halted = 0;
  uint64_t last_state_change = 0;
  uint64_t cross_near_resume = 0;   // crossed within 1ms of a state change
};

// The whole tradable universe for one session.
//
// Indexed by a DENSE VECTOR, not a hash map. ITCH stock locates are
// assigned sequentially from 1 at the start of each session, so they are
// dense small integers with no gaps -- exactly the case where an array
// beats a hash table. Lookup is one bounds check and a pointer offset,
// against a hash, a modulo, and a probable cache miss.
//
// This is the same argument as counts[256] in the milestone-1 parser:
// when the key space is small and dense, index it directly.
struct Universe {
  std::vector<SymbolState> states;
  std::vector<std::string> names;   // locate -> ticker, for reporting

  explicit Universe(size_t max_locate) : states(max_locate), names(max_locate) {}

  bool valid(uint16_t locate) const { return locate < states.size(); }
  SymbolState& operator[](uint16_t locate) { return states[locate]; }

  void name(uint16_t locate, const std::string& s) {
    if (valid(locate)) names[locate] = s;
  }

  size_t active() const {
    size_t n = 0;
    for (const auto& s : states) if (s.events) ++n;
    return n;
  }
};

// Translates ITCH messages into LevelUpdates.
//
// ONE adapter for the entire feed, not one per symbol. ITCH order
// reference numbers are unique across the whole session rather than per
// instrument, so the reference map cannot be partitioned by symbol -- and
// a delete arrives carrying nothing but the reference, so you cannot even
// tell which book it belongs to until you have looked it up.
//
// That is why the Order record stores `locate`: it is the only way a
// bare-reference message can be routed to the right book.
struct ItchAdapter {
  struct Order {
    uint32_t price;
    uint32_t shares;
    uint16_t locate;   // which symbol -- a delete has no other way to know
    char     side;
  };

  Universe& uni;
  std::unordered_map<uint64_t, Order> orders;

  uint64_t unknown_ref = 0;
  uint64_t bad_reduce  = 0;
  uint64_t bad_locate  = 0;

  explicit ItchAdapter(Universe& u) : uni(u) {
    // ~1-2M orders rest simultaneously across the universe at peak.
    // Reserving avoids rehashing the whole table mid-run, which would
    // show up as a large latency outlier at unpredictable moments.
    orders.reserve(4u << 20);
  }

  LevelUpdate on_add(uint64_t ts, uint16_t locate, uint64_t ref, char side,
                     uint32_t price, uint32_t shares) {
    if (side != 'B' && side != 'S') {
      std::fprintf(stderr, "fatal: bad side byte 0x%02x for ref %llu\n",
                   (unsigned)side, (unsigned long long)ref);
      std::exit(1);
    }
    orders[ref] = Order{price, shares, locate, side};
    const uint64_t cur = uni[locate].book.qty_at(side, price);
    return LevelUpdate{ts, locate, price, cur + shares, side};
  }

  // Cancel or execute. Returns false if the reference is unknown, in
  // which case no update should be applied.
  bool on_reduce(uint64_t ts, uint64_t ref, uint32_t shares, LevelUpdate* out) {
    auto it = orders.find(ref);
    if (it == orders.end()) { unknown_ref++; return false; }
    Order& o = it->second;
    if (o.shares < shares) { bad_reduce++; return false; }

    // Copy before any erase: `o` is a reference into the map and dies
    // the moment the entry is removed.
    const char     side   = o.side;
    const uint32_t price  = o.price;
    const uint16_t locate = o.locate;

    o.shares -= shares;
    if (o.shares == 0) orders.erase(it);

    const uint64_t cur = uni[locate].book.qty_at(side, price);
    *out = LevelUpdate{ts, locate, price, cur >= shares ? cur - shares : 0, side};
    return true;
  }

  bool on_delete(uint64_t ts, uint64_t ref, LevelUpdate* out) {
    auto it = orders.find(ref);
    if (it == orders.end()) { unknown_ref++; return false; }
    return on_reduce(ts, ref, it->second.shares, out);
  }

  // Replace is a delete plus an add under a NEW reference, so it produces
  // TWO updates. The U message carries no side; it is inherited from the
  // original order and must be read before the delete erases it.
  //
  // The caller MUST apply *out_remove* to the book before this computes
  // *out_add* -- otherwise, when the old and new price are the same, the
  // add would read a level total that never had the old order subtracted,
  // and the second update would overwrite the first with an inflated
  // quantity. Hence the two-phase interface: prepare, apply, finish.
  bool replace_begin(uint64_t ts, uint64_t old_ref, LevelUpdate* out_remove,
                     uint16_t* locate_out, char* side_out) {
    auto it = orders.find(old_ref);
    if (it == orders.end()) { unknown_ref++; return false; }
    *side_out   = it->second.side;
    *locate_out = it->second.locate;
    return on_delete(ts, old_ref, out_remove);
  }

  LevelUpdate replace_finish(uint64_t ts, uint16_t locate, uint64_t new_ref,
                             char side, uint32_t price, uint32_t shares) {
    return on_add(ts, locate, new_ref, side, price, shares);
  }

  size_t resting() const { return orders.size(); }
};

} // namespace feed