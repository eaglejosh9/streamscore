#pragma once
// Picks the fastest single-event CPU backend this target can compile.
//
// NeonBackend needs ARMv8.2 dotprod and compiles to nothing elsewhere, so
// on x86 this falls back to the scalar Int8Backend. Both run identical
// integer arithmetic, so outputs are bit-identical whichever is chosen.

#include "model_int8.hpp"
#include "model_neon.hpp"

namespace model {

#if defined(__ARM_NEON)
using FastBackend = NeonBackend;
constexpr const char* fast_backend_name() { return "NeonBackend (ARM int8 vdotq)"; }
#else
using FastBackend = Int8Backend;
constexpr const char* fast_backend_name() { return "Int8Backend (scalar int8, no NEON)"; }
#endif

} // namespace model
