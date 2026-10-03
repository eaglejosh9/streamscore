// CUDA int8 forward pass. Mirrors model::Int8Backend::forward exactly.
//
// Layout of the work:
//   - one thread per event; a 128-thread block handles 128 consecutive events
//   - weights live in __constant__ memory: every thread in a warp reads the
//     same weight word in the same step, which constant memory broadcasts
//   - inputs and outputs are staged through shared memory so global traffic
//     is fully coalesced (a thread's own row is 64 B, so direct per-thread
//     loads would scatter each warp instruction over 2 KB)
//   - dot products use __dp4a: four int8 MACs per instruction, with weights
//     packed four-per-int32 on the host and activations packed the same way
//     in registers
//
// Bit-exactness against the CPU depends on three things:
//   1. __float2int_rn + clamp to [-127, 127], the same as quantize_inv.
//   2. No FMA contraction. nvcc fuses a*b + c by default, which skips the
//      intermediate rounding the CPU does; __fmul_rn/__fadd_rn forbid that.
//   3. Integer accumulation order doesn't matter -- int32 addition is exact.

#include "model_cuda.h"
#include "model_int8.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,   \
                   cudaGetErrorString(err_));                              \
      std::abort();                                                        \
    }                                                                      \
  } while (0)

namespace model {
namespace {

constexpr int BLOCK = 128;

// Packed words per row: four int8 per int32.
constexpr int W_IN  = N_IN / 4;       // 4
constexpr int W_HID = N_HIDDEN / 4;   // 8
static_assert(N_IN % 4 == 0 && N_HIDDEN % 4 == 0, "dp4a packing needs /4");

// Padded shared-memory row stride. Rows of 16 floats would put every
// thread's element i on one of two banks; 17 is coprime with 32, so each
// thread in a warp hits its own bank.
constexpr int IN_STRIDE = N_IN + 1;

struct ConstWeights {
  float norm_mean[N_IN];
  float inv_norm_std[N_IN];
  float inv_x_scale, inv_h1_scale, inv_h2_scale;
  float s0, s2, s4;
  float b0[N_HIDDEN];
  float b2[N_HIDDEN];
  float b4[N_OUT];
  int32_t w0[N_HIDDEN * W_IN];
  int32_t w2[N_HIDDEN * W_HID];
  int32_t w4[N_OUT * W_HID];
};

__constant__ ConstWeights c_w;

__device__ __forceinline__ int32_t quantize_inv(float v, float inv_scale) {
  int q = __float2int_rn(__fmul_rn(v, inv_scale));
  return q < -127 ? -127 : (q > 127 ? 127 : q);
}

// Pack four int8 values (held as int32) into one dp4a operand, low byte first
// -- the same order the host gets by memcpy-ing an int8 array into an int32.
__device__ __forceinline__ int32_t pack4(int32_t a, int32_t b, int32_t c, int32_t d) {
  return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | (int32_t(uint32_t(d) << 24));
}

__global__ void __launch_bounds__(BLOCK)
forward_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s_in[BLOCK * IN_STRIDE];
  __shared__ float s_out[BLOCK * N_OUT];

  const int base = blockIdx.x * BLOCK;
  const int cnt  = min(BLOCK, n - base);
  const int tid  = threadIdx.x;

  // Coalesced load: the block's rows are one contiguous run of cnt*16 floats.
  const float4* in4 = reinterpret_cast<const float4*>(in + size_t(base) * N_IN);
  for (int e = tid; e < cnt * (N_IN / 4); e += BLOCK) {
    float4 v = in4[e];
    float* dst = &s_in[(e / (N_IN / 4)) * IN_STRIDE + (e % (N_IN / 4)) * 4];
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
  }
  __syncthreads();

  if (tid < cnt) {
    const float* row = &s_in[tid * IN_STRIDE];

    // Normalize and quantize the input, packing as we go.
    int32_t x[W_IN];
#pragma unroll
    for (int p = 0; p < W_IN; ++p) {
      int32_t q[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        int i = p * 4 + k;
        float v = __fmul_rn(__fsub_rn(row[i], c_w.norm_mean[i]), c_w.inv_norm_std[i]);
        q[k] = quantize_inv(v, c_w.inv_x_scale);
      }
      x[p] = pack4(q[0], q[1], q[2], q[3]);
    }

    // Layer 0: 16 -> 32, relu, requantize at h1's scale.
    int32_t h1[W_HID];
#pragma unroll
    for (int p = 0; p < W_HID; ++p) {
      int32_t q[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        int j = p * 4 + k;
        int32_t acc = 0;
#pragma unroll
        for (int i = 0; i < W_IN; ++i) acc = __dp4a(x[i], c_w.w0[j * W_IN + i], acc);
        float h = __fadd_rn(__fmul_rn(__int2float_rn(acc), c_w.s0), c_w.b0[j]);
        h = h > 0.0f ? h : 0.0f;
        q[k] = quantize_inv(h, c_w.inv_h1_scale);
      }
      h1[p] = pack4(q[0], q[1], q[2], q[3]);
    }

    // Layer 2: 32 -> 32, relu, requantize at h2's scale.
    int32_t h2[W_HID];
#pragma unroll
    for (int p = 0; p < W_HID; ++p) {
      int32_t q[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        int j = p * 4 + k;
        int32_t acc = 0;
#pragma unroll
        for (int i = 0; i < W_HID; ++i) acc = __dp4a(h1[i], c_w.w2[j * W_HID + i], acc);
        float h = __fadd_rn(__fmul_rn(__int2float_rn(acc), c_w.s2), c_w.b2[j]);
        h = h > 0.0f ? h : 0.0f;
        q[k] = quantize_inv(h, c_w.inv_h2_scale);
      }
      h2[p] = pack4(q[0], q[1], q[2], q[3]);
    }

    // Layer 4: 32 -> 3. Logits stay float.
#pragma unroll
    for (int k = 0; k < N_OUT; ++k) {
      int32_t acc = 0;
#pragma unroll
      for (int i = 0; i < W_HID; ++i) acc = __dp4a(h2[i], c_w.w4[k * W_HID + i], acc);
      s_out[tid * N_OUT + k] = __fadd_rn(__fmul_rn(__int2float_rn(acc), c_w.s4), c_w.b4[k]);
    }
  }
  __syncthreads();

  // Coalesced store: stride-3 shared reads are conflict-free (3 is coprime
  // with 32), and the global writes are one contiguous run.
  float* dst = out + size_t(base) * N_OUT;
  for (int e = tid; e < cnt * N_OUT; e += BLOCK) dst[e] = s_out[e];
}

void pack_rows(int32_t* dst, const int8_t* src, int rows, int cols) {
  // Row-major int8 [rows][cols] -> int32 [rows][cols/4]. memcpy keeps the
  // host's byte order, which is what pack4 reproduces on the device.
  std::memcpy(dst, src, size_t(rows) * cols);
}

} // namespace

CudaBackend::CudaBackend(const WeightsInt8& w, int max_batch) : max_batch_(max_batch) {
  ConstWeights h{};
  std::memcpy(h.norm_mean, w.norm_mean, sizeof h.norm_mean);
  std::memcpy(h.inv_norm_std, w.inv_norm_std, sizeof h.inv_norm_std);
  h.inv_x_scale = w.inv_x_scale;
  h.inv_h1_scale = w.inv_h1_scale;
  h.inv_h2_scale = w.inv_h2_scale;
  h.s0 = w.s0; h.s2 = w.s2; h.s4 = w.s4;
  std::memcpy(h.b0, w.b0, sizeof h.b0);
  std::memcpy(h.b2, w.b2, sizeof h.b2);
  std::memcpy(h.b4, w.b4, sizeof h.b4);
  pack_rows(h.w0, w.w0, N_HIDDEN, N_IN);
  pack_rows(h.w2, w.w2, N_HIDDEN, N_HIDDEN);
  pack_rows(h.w4, w.w4, N_OUT, N_HIDDEN);
  CUDA_CHECK(cudaMemcpyToSymbol(c_w, &h, sizeof h));

  CUDA_CHECK(cudaMalloc(&d_in_, size_t(max_batch) * N_IN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_, size_t(max_batch) * N_OUT * sizeof(float)));

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  start_ = a;
  stop_ = b;
}

CudaBackend::~CudaBackend() {
  cudaEventDestroy(static_cast<cudaEvent_t>(start_));
  cudaEventDestroy(static_cast<cudaEvent_t>(stop_));
  cudaFree(d_in_);
  cudaFree(d_out_);
}

static void launch(const float* d_in, float* d_out, int n) {
  int grid = (n + BLOCK - 1) / BLOCK;
  forward_kernel<<<grid, BLOCK>>>(d_in, d_out, n);
}

void CudaBackend::forward_batch(const float* in, float* out, int n) {
  if (n <= 0) return;
  if (n > max_batch_) {
    std::fprintf(stderr, "CudaBackend: batch %d exceeds max %d\n", n, max_batch_);
    std::abort();
  }
  CUDA_CHECK(cudaMemcpy(d_in_, in, size_t(n) * N_IN * sizeof(float), cudaMemcpyHostToDevice));
  launch(d_in_, d_out_, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(out, d_out_, size_t(n) * N_OUT * sizeof(float), cudaMemcpyDeviceToHost));
}

float CudaBackend::time_kernel_ms(int n, int iters) {
  if (n <= 0 || iters <= 0) return 0.0f;
  auto a = static_cast<cudaEvent_t>(start_);
  auto b = static_cast<cudaEvent_t>(stop_);

  launch(d_in_, d_out_, n);   // warm-up
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) launch(d_in_, d_out_, n);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms / float(iters);
}

CudaDeviceInfo CudaBackend::device_info() {
  CudaDeviceInfo info{};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::snprintf(info.name, sizeof info.name, "%s", p.name);
  info.cc_major = p.major;
  info.cc_minor = p.minor;
  info.sm_count = p.multiProcessorCount;

  // memoryClockRate left cudaDeviceProp in CUDA 13; the attribute remains.
  int clock_khz = 0, bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  // x2 for double data rate.
  info.mem_bandwidth_gbs = 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  return info;
}

void* CudaBackend::alloc_pinned(size_t bytes) {
  void* p = nullptr;
  CUDA_CHECK(cudaMallocHost(&p, bytes));
  return p;
}

void CudaBackend::free_pinned(void* p) { cudaFreeHost(p); }

} // namespace model
