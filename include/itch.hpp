#pragma once
#include <cstdint>
#include <cstddef>

namespace itch{

// Every message body starts with this 11-byte header.
// offset 0 : type (1)   offset 1 : stock_locate (2)
// offset 3 : tracking (2)   offset 5 : timestamp (6)

// ---------------------------------------------------------------------------
// ITCH 5.0 message layouts.
//
// Framing: each message is preceded by a 2-byte big-endian length prefix that
// is NOT part of the message. Offsets below are from the start of the message
// body (offset 0 = type byte), so in a buffer the body begins at pos + 2.
//
// All integers are big-endian. Prices are u32 with 4 implied decimals
// (1500200 == $150.02) and must stay integers — never convert to float.
// Symbols are 8 bytes, space-padded, not null-terminated.
//
// Shared header, present on every message:
//    0   1   type (ASCII)
//    1   2   stock locate
//    3   2   tracking number
//    5   6   timestamp, ns since midnight
//
// Offset 11 is the order reference on every order-lifecycle message.
//
// 'R'  stock directory                          39 bytes
//   11   8   symbol
//   (market category, lot size, etc. follow — unused)
//
// 'A'  add order                                36 bytes
//   11   8   order reference
//   19   1   side: 'B' buy, 'S' sell
//   20   4   shares
//   24   8   symbol
//   32   4   price
//
// 'F'  add order with MPID                      40 bytes
//   identical to 'A' through offset 35, then:
//   36   4   market participant id (unused)
//
// 'E'  order executed                           31 bytes
//   11   8   order reference
//   19   4   executed shares
//   23   8   match number (unused)
//
// 'C'  order executed with price                36 bytes
//   11   8   order reference
//   19   4   executed shares
//   23   8   match number (unused)
//   31   1   printable flag (unused)
//   32   4   execution price (unused for book maintenance)
//
// 'X'  order cancel                             23 bytes
//   11   8   order reference
//   19   4   canceled shares
//
// 'D'  order delete                             19 bytes
//   11   8   order reference
//   (no other fields — this is why the order map must exist)
//
// 'U'  order replace                            35 bytes
//   11   8   original order reference
//   19   8   new order reference
//   27   4   shares
//   31   4   price
//   Semantics: delete the original entirely, then add a NEW order under the
//   new reference. Side is not carried — look it up from the original before
//   erasing it.
//
// 'P'  trade, non-cross                         44 bytes
//   11   8   order reference (often 0)
//   19   1   side
//   20   4   shares
//   24   8   symbol
//   32   4   price
//   36   8   match number
//   Hidden liquidity: never displayed, so it does NOT modify the book.
//   Useful in milestone 3 for volume features.
//
// Handled via the length prefix and otherwise ignored: H (trading action),
// Y (reg SHO), L (market participant position), I (NOII), Q (cross trade),
// S (system event), and the rest.
// ---------------------------------------------------------------------------

constexpr size_t HEADER_SIZE = 11;

enum class MsgType : uint8_t {
  SystemEvent    = 'S',
  StockDirectory = 'R',
  AddOrder       = 'A',
  AddOrderMPID   = 'F',
  Execute        = 'E',
  ExecuteWithPx  = 'C',
  Cancel         = 'X',
  Delete         = 'D',
  Replace        = 'U',
  Trade          = 'P',
};

inline uint16_t read_be16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) << 8 | p[1];
}

inline uint32_t read_be32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) << 24 | static_cast<uint32_t>(p[1]) << 16 |
         static_cast<uint32_t>(p[2]) << 8  | p[3];
}

// 6-byte big-endian timestamp, nanoseconds since midnight.
inline uint64_t read_be48(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 6; ++i) v = (v << 8) | p[i];
  return v;
}

inline uint64_t read_be64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
  return v;
}

} //namespace itch