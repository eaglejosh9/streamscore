#pragma once
#include <cstdint>
#include <map>

namespace feed {

// The one thing every market data source produces, whatever its wire format.
//
// ITCH is ORDER-based: a delete carries only a reference number, so the
// adapter must remember every resting order to know which level changed.
// Coinbase level2 is LEVEL-based: it states the new quantity directly.
// The common ground is the price level, and the quantity is ABSOLUTE.
//
// `locate` says WHICH symbol. Multi-symbol is why this field exists: one
// feed carries ~8,900 interleaved instruments, and every update has to be
// routed to the right book.
struct LevelUpdate {
  uint64_t ts;       // source timestamp, ns
  uint16_t locate;   // symbol id, dense and small: 1..~8900
  uint32_t price;    // integer, 4 implied decimals
  uint64_t qty;      // absolute size at this level; 0 removes it
  char     side;     // 'B' or 'S'
};

// The two sides are different TYPES, not just different variables: bids
// sort descending so begin() is the best bid, asks ascending so begin()
// is the best ask. That makes best_bid and best_ask the same expression,
// at the cost of the sides not being interchangeable at the type level --
// hence the template below. Aliasing one map type as the other would be
// undefined behaviour.
using BidMap = std::map<uint32_t, uint64_t, std::greater<uint32_t>>;
using AskMap = std::map<uint32_t, uint64_t>;

// Price levels only.
//
// No order-reference map here: that is an artifact of ITCH's wire format,
// not a property of an order book, and it lives in the adapter. That
// separation matters more at 8,900 books than it did at one -- otherwise
// every symbol would carry a hash table, and ITCH order references are
// unique across the WHOLE feed rather than per symbol, so they could not
// be partitioned per book anyway.
struct Book {
  BidMap bids;
  AskMap asks;

  template <typename MapT>
  static void set_level(MapT& side, uint32_t price, uint64_t qty) {
    if (qty == 0) side.erase(price);
    else          side[price] = qty;
  }     

  void apply(const LevelUpdate& u) {
    if (u.side == 'B') set_level(bids, u.price, u.qty);
    else               set_level(asks, u.price, u.qty);
  }

  uint64_t qty_at(char s, uint32_t price) const {
    if (s == 'B') {
      auto it = bids.find(price);
      return it == bids.end() ? 0 : it->second;
    }
    auto it = asks.find(price);
    return it == asks.end() ? 0 : it->second;
  }

  uint32_t best_bid() const { return bids.empty() ? 0 : bids.begin()->first; }
  uint32_t best_ask() const { return asks.empty() ? 0 : asks.begin()->first; }
  bool two_sided()   const { return !bids.empty() && !asks.empty(); }
  size_t levels()    const { return bids.size() + asks.size(); }

  bool consistent() const {
    if (!two_sided()) return true;
    return best_bid() < best_ask();
  }

  // A level-based feed resynchronises by replacing the book wholesale
  // after a sequence gap. ITCH has no snapshot -- only the delta stream
  // from the session start -- so a gap there is unrecoverable.
  void clear() { bids.clear(); asks.clear(); }
};

} // namespace feed