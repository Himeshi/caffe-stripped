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
	
__device__ __inline__ float fp16tofp32_gpu(fp16 p) {
  union Bits v;

  // get sign
  bool sign = p & SIGN_MASK;

  //get exponent
  v.ui = p << POSIT_LENGTH_PLUS_ONE;
  int exponent = (__clz(v.ui));

  //get fraction
  v.ui <<= exponent;

  // assemble
  v.ui >>= FLOAT_SIGN_PLUS_EXP_LENGTH;
  v.ui += ((SINGLE_PRECISION_BIAS - exponent) << FLOAT_EXPONENT_SHIFT);

  v.si ^= (FLOAT_INF ^ v.si) & -(p == _G_INFP);
  v.si ^= (0 ^ v.si) & -(p == 0);

  v.ui |= (sign << FLOAT_SIGN_SHIFT);
  return (v.f * SCALING_FACTOR);
}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  fp16 p = 0;
  union Bits v;
  v.f = f / SCALING_FACTOR;
  bool sign = v.ui & FLOAT_SIGN_MASK;
  v.ui &= 0x7FFFFFFF;

  p ^= (p ^_G_MAXREALP) & -(v.si >= _G_MAXREAL_INT);
  p ^= (p ^ _G_INFP) & -(v.si >= FLOAT_INF);
  p ^= (p ^ _G_MINREALP) & -(v.si != 0 && v.si <= _G_MINREAL_INT);

  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  //get exponent and fraction
  int exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  uint32_t fraction = (v.si << FLOAT_SIGN_PLUS_EXP_LENGTH);

  //assemble
  uint32_t temp = fraction >> (exp + 1);
  int rb = (temp & POSIT_HALFWAY_BIT_MASK) && ((temp & 0x01000000) || (temp & 0x007fffff));
  temp = (temp >> 24) + rb;
  temp |= (sign << 7);
  p ^= (temp ^ p) & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

  return p;
}

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
