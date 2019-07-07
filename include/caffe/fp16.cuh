/*
 * fp16.hpp
 * Code taken from https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#ifndef INCLUDE_CAFFE_FP16_CUH_
#define INCLUDE_CAFFE_FP16_CUH_
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include "caffe/fp16.hpp"

namespace caffe {
	
__device__ __inline__ float fp16tofp32_gpu(fp16 f16value) {
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

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
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

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
