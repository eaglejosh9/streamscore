#pragma once
#include <cstdint>

namespace itch{

// Every message body starts with this 11-byte header.
// offset 0 : type (1)   offset 1 : stock_locate (2)
// offset 3 : tracking (2)   offset 5 : timestamp (6)
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

} //namespace itch