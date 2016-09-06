/**
* Given a input tensor x with shape (N, C, D), compute x.mean(2).mean(0)
* This function is useful in batch normalization.
* Refer to https://people.maths.ox.ac.uk/gilesm/cuda/prac4/reduction.pdf.
* But the unrolling warps seems to be not working correctly for now.
*/

#include <cstdio>

#include "common.hpp"
#include "timer.hpp"

const int N = 256;
const int C = 1024;
const int D = 28*28;

#define BENCHMARK(title, callfunc) \
  do { \
    CUDA_CHECK(cudaMemset(d_ret, 0, ret_size)); \
    Timer timer; \
    timer.Start(); \
    (callfunc); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
    timer.Stop(); \
    printf("%-50s %10.5fms\n", (title), timer.Milliseconds()); \
  } while (0)


__global__ void reduce0(const float* in, float* out) {
  __shared__ float buffer[CUDA_NUM_THREADS];
  const unsigned int tid = threadIdx.x;
  const unsigned int c = blockIdx.x;

  // load and accumulate data to buffer
  buffer[tid] = 0;
  for (int i = tid; i < N * D; i += blockDim.x) {
    const unsigned int n = i / D;
    const unsigned int d = i % D;
    const unsigned int index = n * C * D + c * D + d;
    buffer[tid] += in[index];
  }
  __syncthreads();

  // do tree reduction in buffer
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {  // <-- bad: divergent branching
      buffer[tid] += buffer[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) out[c] = buffer[0] / (N * D);
}

__global__ void reduce1(const float* in, float* out) {
  __shared__ float buffer[CUDA_NUM_THREADS];
  const unsigned int tid = threadIdx.x;
  const unsigned int c = blockIdx.x;

  // load and accumulate data to buffer
  buffer[tid] = 0;
  for (int i = tid; i < N * D; i += blockDim.x) {
    const unsigned int n = i / D;
    const unsigned int d = i % D;
    const unsigned int index = n * C * D + c * D + d;
    buffer[tid] += in[index];
  }
  __syncthreads();

  // do tree reduction in buffer
  for (int s = 1; s < blockDim.x; s *= 2) {
    const int index = 2 * s * tid;
    if (index < blockDim.x) {  // <-- bad: shared memory bank conflicts
      buffer[index] += buffer[index + s];
    }
    __syncthreads();
  }

  if (tid == 0) out[c] = buffer[0] / (N * D);
}

__global__ void reduce2(const float* in, float* out) {
  __shared__ float buffer[CUDA_NUM_THREADS];
  const unsigned int tid = threadIdx.x;
  const unsigned int c = blockIdx.x;

  // load and accumulate data to buffer
  buffer[tid] = 0;
  for (int i = tid; i < N * D; i += blockDim.x) {
    const unsigned int n = i / D;
    const unsigned int d = i % D;
    const unsigned int index = n * C * D + c * D + d;
    buffer[tid] += in[index];
  }
  __syncthreads();

  // do tree reduction in buffer
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer[tid] += buffer[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) out[c] = buffer[0] / (N * D);
}

int main() {
  const int count = N * C * D;
  const int data_size = count * sizeof(float);
  const int ret_size = C * sizeof(float);

  float* h_data = new float[count];
  for (int i = 0; i < count; ++i) {
    const int c = (i / D) % C;
    h_data[i] = static_cast<float>(c);
  }

  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, data_size));
  float* d_ret;
  CUDA_CHECK(cudaMalloc(&d_ret, ret_size));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));

  BENCHMARK("interleaved addressing with divergent branching",
            (reduce0<<<C, CUDA_NUM_THREADS>>>(d_data, d_ret)));
  BENCHMARK("interleaved addressing with bank conflicts",
            (reduce1<<<C, CUDA_NUM_THREADS>>>(d_data, d_ret)));
  BENCHMARK("sequential addressing",
            (reduce2<<<C, CUDA_NUM_THREADS>>>(d_data, d_ret)));

  float* h_ret = new float[C];
  CUDA_CHECK(cudaMemcpy(h_ret, d_ret, ret_size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_ret));

  delete[] h_data;
  delete[] h_ret;

  return 0;
}