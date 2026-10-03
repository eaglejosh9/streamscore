#pragma once
// CUDA int8 forward pass, one thread per event.
//
// Plain C++ header: no CUDA types leak out, so ordinary .cpp files can
// include it and link against model_cuda without being compiled by nvcc.
//
// Arithmetic is identical to Int8Backend in model_int8.hpp -- same scales,
// same requantization, same saturation -- so its logits are bit-exact
// against the CPU and any difference is a kernel bug.

#include <cstddef>

namespace model {

struct WeightsInt8;

struct CudaDeviceInfo {
  char   name[256];
  int    cc_major, cc_minor;
  int    sm_count;
  double mem_bandwidth_gbs;   // theoretical peak, from clock * bus width
};

class CudaBackend {
 public:
  // Copies the weights into __constant__ memory and allocates device buffers
  // for up to max_batch events. All allocation happens here: cudaMalloc costs
  // ~100 us and would swamp any per-call measurement.
  CudaBackend(const WeightsInt8& w, int max_batch);
  ~CudaBackend();

  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;

  // What a caller pays: H2D copy + kernel + D2H copy, synchronous.
  // in is n * N_IN floats, out is n * N_OUT floats. n <= max_batch.
  void forward_batch(const float* in, float* out, int n);

  // Kernel only, on whatever inputs the last forward_batch left resident on
  // the device. Averaged over iters back-to-back launches, timed with
  // cudaEvents. Returns milliseconds per launch.
  float time_kernel_ms(int n, int iters);

  int max_batch() const { return max_batch_; }

  static CudaDeviceInfo device_info();

  // Page-locked host memory. Pageable memory forces the driver through an
  // extra staging copy on every transfer, so a latency-sensitive caller
  // would hold its buffers pinned.
  static void* alloc_pinned(size_t bytes);
  static void  free_pinned(void* p);

 private:
  int    max_batch_;
  float* d_in_  = nullptr;
  float* d_out_ = nullptr;
  void*  start_ = nullptr;   // cudaEvent_t, opaque here
  void*  stop_  = nullptr;
};

} // namespace model
