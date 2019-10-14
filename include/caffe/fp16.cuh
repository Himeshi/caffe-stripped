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
	
  // handle zero
  if (p == 0)
    return 0.0;

  // handle infinity
  if (p == _G_INFP)
    return INFINITY;

  if (p == _G_MAXREALP)
    return _G_MAXREAL;

  if (p == _G_MINREALP)
    return _G_MINREAL;

  double f = 1.0;

  // check sign bit
  FP16_TYPE sign = p & SIGN_MASK;
  // if negative, get the two's complement
  if (sign) {
    p = ~p + 1;
    f = -f;
  }

  // get the regime
  FP16_TYPE second_bit = p & SECOND_BIT_MASK;
  // remove the sign
  p <<= 1;
  int regime = 0;
  int regime_length = 0;
  if (second_bit) {
    // sign of regime is +ve, find first 0
	// Here we have to subtract the posit limb size, because clz takes an
	// int which aligns the short to the right
    FP16_TYPE flipped = ~p;
    regime = __clz(flipped) - FP16_LIMB_SIZE - 1;
    regime_length = regime + 2;
  } else {
    // sign of regime is -ve, find first 1
	regime = FP16_LIMB_SIZE - __clz(p);
    regime_length = 1 - regime;
  }

  // remove regime and get exponent
  p <<= regime_length;
  int exponent = p >> (FP16_LIMB_SIZE - _G_ESIZE);

  // remove exponent and get fraction
  p <<= _G_ESIZE;
  int running_length = (regime_length + 1 + _G_ESIZE);
  int fraction_size = ((_G_NBITS - running_length) + abs((_G_NBITS - running_length))) >> 1;
  int fraction = p >> (FP16_LIMB_SIZE - fraction_size);
  fraction = fraction | (1 << fraction_size);

  return f * ((float)fraction / (float)(1 << fraction_size)) *
         powf(_G_USEED, regime) * (1 << exponent);

}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  fp16 p = 0;
  union Bits v;
  v.f = f;
  uint32_t sign = v.ui & FLOAT_SIGN_MASK;
  v.si ^= sign;
  sign >>= FLOAT_SIGN_SHIFT;

  if (v.f == 0.0) {
    return p;
  }

  if (v.f == INFINITY || f != f) {
    return _G_INFP;
  }

  if (v.f >= _G_MAXREAL) {
    p = _G_MAXREALP;
    p = (p ^ -sign) + sign;
    return p;
  }

  if (v.f <= _G_MINREAL) {
    p = _G_MINREALP;
    p = (p ^ -sign) + sign;
    return p;
  }

  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  // get absolute exponent
  bool exp_sign = !(v.ui >> 30);

  //get regime and exponent
  uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  TEMP_TYPE regime_and_exp = (((1 << ((exp >> _G_ESIZE) + 1)) - 1) << (_G_ESIZE + 1)) | (exp & POSIT_EXPONENT_MASK);;
  //if exponent is negative
  regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign) >> ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);
  int regime_and_exp_length = (exp >> _G_ESIZE) + 2 + _G_ESIZE - ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);

  //assemble
  regime_and_exp <<= (UNSIGNED_LONG_LONG_SIZE - regime_and_exp_length);
  regime_and_exp |= ((TEMP_TYPE) (v.ui & FLOAT_FRACTION_MASK) << (POSIT_EXP_SHIFT - regime_and_exp_length));
  p = (regime_and_exp >> POSIT_EXTRA_BITS_SHIFT);

  //round
  p += (bool) (regime_and_exp & POSIT_HALFWAY_BIT_MASK) && ((p & 1) | (regime_and_exp & POSIT_EXTRA_BITS_MASK));
  p <<= _G_POSIT_SHIFT_AMOUNT;

  p = (p ^ -sign) + sign;

  return p;
}

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
