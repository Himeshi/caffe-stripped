#include "caffe/fp16.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

void copy_posit_globals_to_gpu(int nbits, int esize, int useed, int useed_zeros, int posit_shift_amount,
  int maxrealexp, POSIT_TYPE maxrealp, POSIT_TYPE minrealp, POSIT_TYPE infp, float maxreal, float minreal) {
  cudaMemcpyToSymbol(_g_nbits_gpu, &nbits, sizeof(int));
  cudaMemcpyToSymbol(_g_esize_gpu, &esize, sizeof(int));
  cudaMemcpyToSymbol(_g_useed_gpu, &useed, sizeof(int));
  cudaMemcpyToSymbol(_g_useed_zeros_gpu, &useed_zeros, sizeof(int));
  cudaMemcpyToSymbol(_g_posit_shift_amount_gpu, &posit_shift_amount, sizeof(int));
  cudaMemcpyToSymbol(_g_maxrealexp_gpu, &maxrealexp, sizeof(int));
  cudaMemcpyToSymbol(_g_maxrealp_gpu, &maxrealp, sizeof(posit));
  cudaMemcpyToSymbol(_g_minrealp_gpu, &minrealp, sizeof(posit));
  cudaMemcpyToSymbol(_g_infp_gpu, &infp, sizeof(posit));
  cudaMemcpyToSymbol(_g_maxreal_gpu, &maxreal, sizeof(float));
  cudaMemcpyToSymbol(_g_minreal_gpu, &minreal, sizeof(float));
}

__global__ void convert_to_fp16(const int n, float* in, fp16* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fp32tofp16_gpu(in[index]);
  }
}

__global__ void convert_to_fp16(const int n, double* in, fp16* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp32tofp16_gpu(in[index]);
  }
}
__global__ void convert_to_float(const int n,  fp16* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__global__ void convert_to_float(const int n,  fp16* in, double* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__global__ void convert_to_float(const int n, const fp16* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__global__ void convert_to_float(const int n, const fp16* in, double* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
  
}

__global__ void outputweights(const int n, float* in) {
  CUDA_KERNEL_LOOP(index, n) {
printf("%d %f\n", index, in[index]);
  }
}

__global__ void outputweights(const int n, double* in) {
  CUDA_KERNEL_LOOP(index, n) {
printf("%d %f\n", index, in[index]);
  }
}

}
