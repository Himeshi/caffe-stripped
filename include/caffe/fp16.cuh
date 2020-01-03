/*
 * fp16.hpp
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
#include "caffe/fp16.hpp"

namespace caffe {
	
__device__ __inline__ float fp16tofp32_gpu(fp16 f16value) {
  union Bits v;
  v.ui = f16value;
  int32_t sign = v.si & signC;
  v.si ^= sign;
  sign <<= shiftSign;
  v.si ^= ((v.si + minD) ^ v.si) & -(v.si > 0);
  v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
  v.si <<= fp16shift;
  v.si |= sign;
  return v.f;
}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  union Bits v;
  uint32_t round_bit = 0;
  v.f = f;
  uint32_t sign = v.si & signN;
  v.si ^= sign;
  sign >>= shiftSign; // logical shift
  v.si ^= (tempN ^v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -(v.si > maxfp16N);
  round_bit = (v.si & 0x00002000) >> roundShift;
  v.ui >>= fp16shift; // logical shift
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  v.si += round_bit;
  return v.ui | sign;
}
}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
