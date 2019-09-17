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

__device__ __inline__ fp16 get_posit_from_parts_gpu(int exponent, unsigned int fraction, unsigned int fraction_size) {
  // assume the fraction is normalized and it's MSB is hidden already
  fp16 p = 0;
  int regime, regime_length, exponentp;
  TEMP_TYPE temp, ob, hb, sb;

  // compute regime and exponent
  int abs_exp = abs(exponent);
  regime = abs_exp >> _G_USEED_ZEROS_SHIFT;
  exponentp = abs_exp - (_G_USEED_ZEROS * regime);
  regime_length = regime + 2;
  bool sign = exponent & 0x80000000;
  regime = ((((1 << (regime + 1)) - 1) << 1) * !sign) + (1 * sign);
  regime_length += (sign * -1);		//if the regime is negative subtract one
  regime_length += (((bool) (abs_exp & ((1 << _G_USEED_ZEROS_SHIFT) - 1)))
      & sign);//if the regime is negative and is divisible by _G_USEED_ZEROS_SHIFT add one

  //assemble regime and exponent
  exponentp = (exponentp ^ -sign) + sign;
  int temp_assemble = regime << _G_ESIZE;
  temp_assemble |= (exponentp & ((1 << _G_ESIZE) - 1));
  int running_size = 1 + regime_length + _G_ESIZE;	//add one for sign

  // assemble the fraction
  temp = temp_assemble;
  temp <<= fraction_size;
  temp |= fraction;
  running_size += fraction_size;

  //left align temp
  temp = temp << (UNSIGNED_LONG_LONG_SIZE - running_size);

  //round
  int extra_bits = (UNSIGNED_LONG_LONG_SIZE - _G_NBITS);
  p = temp >> extra_bits;
  ob = p & 1;
  TEMP_TYPE hb_mask = (1ULL << (extra_bits - 1));
  hb = temp & hb_mask;
  sb = temp & (hb_mask - 1);
  p += ((ob && hb) | (hb && sb));

  return p;
}
	
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
  if (f == 0.0) {
    return p;
  }

  if (f == INFINITY || f == -INFINITY) {
    return _G_INFP;
  }

  if (fabs(f) >= _G_MAXREAL) {
    p = _G_MAXREALP;
    if (f < 0)
      p = ~p + 1;
    return p;
  }

  if (fabs(f) <= _G_MINREAL) {
    p = _G_MINREALP;
    if (f < 0)
      p = ~p + 1;
    return p;
  }

  if (f != f)
    return _G_INFP;

  // get sign, exponent and fraction from float
  unsigned int temp;
  memcpy(&temp, &f, sizeof(temp));
  int exponent = (temp & FLOAT_EXPONENT_MASK) >> FLOAT_EXPONENT_SHIFT;
  unsigned int fraction = (temp & FLOAT_FRACTION_MASK);
  if (exponent) {
    exponent -= SINGLE_PRECISION_BIAS;
  } else {
    exponent = FLOAT_DENORMAL_EXPONENT;
    int normalization =
        __clz(fraction) - FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE;
    exponent -= normalization;
    // hide the most significant bit
    fraction &= ~(1 << normalization);
  }

  p = get_posit_from_parts_gpu(exponent, fraction, FLOAT_EXPONENT_SHIFT);

  if (f < 0)
    p = ~p + 1;

  return p << _G_POSIT_SHIFT_AMOUNT;
}

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
