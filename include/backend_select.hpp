#pragma once
// Picks the fastest single-event CPU backend this target can compile.
//
//   x86 with AVX-512F + VNNI  -> Avx512Backend (vpdpbusd)
//   ARM with NEON             -> NeonBackend   (vdotq_s32, needs dotprod)
//   anything else             -> Int8Backend   (scalar)
//
// Each SIMD header compiles to nothing on targets it does not support. All
// three run identical integer arithmetic, so outputs are bit-identical
// whichever is chosen.

#include "model_int8.hpp"
#include "model_neon.hpp"
#include "model_avx512.hpp"

namespace model {

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
using FastBackend = Avx512Backend;
constexpr const char* fast_backend_name() { return "Avx512Backend (x86 int8 vpdpbusd)"; }
#elif defined(__ARM_NEON)
using FastBackend = NeonBackend;
constexpr const char* fast_backend_name() { return "NeonBackend (ARM int8 vdotq)"; }
#else
using FastBackend = Int8Backend;
constexpr const char* fast_backend_name() { return "Int8Backend (scalar int8, no SIMD)"; }
#endif

} // namespace model
