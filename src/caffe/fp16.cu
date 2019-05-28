#include "caffe/fp16.hpp"

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

__global__ void convert_to_float(const int n, fp16* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__global__ void convert_to_float(const int n, fp16* in, double* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__device__ float fp16tofp32_gpu(fp16 f16value) {
  union Bits v;
  v.ui = f16value;
  int32_t sign = v.si & signC;
  v.si ^= sign;
  sign <<= shiftSign;
  v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
  v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
  union Bits s;
  s.si = mulC;
  s.f *= v.si;
  int32_t mask = -(norC > v.si);
  v.si <<= fp16shift;
  v.si ^= (s.si ^ v.si) & mask;
  v.si |= sign;
  return v.f;
}

__device__ fp16 fp32tofp16_gpu(float f) {
  union Bits v, s;
  v.f = f;
  uint32_t sign = v.si & signN;
  v.si ^= sign;
  sign >>= shiftSign; // logical shift
  s.si = mulN;
  s.si = s.f * v.f; // correct subnormals
  v.si ^= (s.si ^ v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxfp16N));
  v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
  v.ui >>= fp16shift; // logical shift
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  return v.ui | sign;
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
