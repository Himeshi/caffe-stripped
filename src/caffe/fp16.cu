#include "caffe/fp16.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

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
