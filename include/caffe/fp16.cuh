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

namespace caffe {

#ifndef CPU_ONLY
__constant__ int _g_nbits_gpu;
__constant__ int _g_esize_gpu;
__constant__ int _g_useed_gpu;
__constant__ int _g_useed_zeros_gpu;
__constant__ int _g_posit_shift_amount_gpu;
__constant__ int _g_maxrealexp_gpu;
__constant__ POSIT_TYPE _g_maxrealp_gpu;
__constant__ POSIT_TYPE _g_minrealp_gpu;
__constant__ POSIT_TYPE _g_infp_gpu;
__constant__ float _g_maxreal_gpu;
__constant__ float _g_minreal_gpu;
#endif

__device__ __inline__ fp16 get_posit_from_parts_gpu(int exponent, unsigned int fraction, unsigned int fraction_size) {
    //assume the fraction is normalized and it's MSB is hidden already
	fp16 p = 0;
	int regime, regime_length, exponentp, ob, hb, sb, rb;
	TEMP_TYPE temp;

	//find regime and exponent
	if (exponent >= 0) {
		regime = exponent / _g_useed_zeros_gpu;
		exponentp = exponent - (_g_useed_zeros_gpu * regime);
		regime_length = regime + 2;
		regime = ((1 << (regime + 1)) - 1) << 1;
	} else {
		regime = abs(exponent / _g_useed_zeros_gpu);
		if (exponent % _g_useed_zeros_gpu)
			regime += 1;
		regime_length = regime + 1;
		exponentp = exponent + (_g_useed_zeros_gpu * regime);
		regime = 1;
	}

	//assemble the regime
	temp = regime;
	int running_size = regime_length + 1;

	//assemble the exponent
	temp <<= _g_esize_gpu;
	int exponent_length = FLOAT_SIZE - __clz(abs(exponentp));
	exponentp >>= (((exponent_length - _g_esize_gpu)
			+ abs((exponent_length - _g_esize_gpu))) >> 1);
	temp |= exponentp;
	running_size += _g_esize_gpu;

	//assemble the fraction
	temp <<= fraction_size;
	temp |= fraction;
	running_size += fraction_size;

	int extra_bits = (running_size - _g_nbits_gpu);

	if (extra_bits > 0) {
		//round
		p = temp >> extra_bits;
		ob = p & 0x0001;
		hb = (temp >> (extra_bits - 1)) & 1ULL;
		sb = ((1ULL << (extra_bits - 1)) - 1) & temp;
		rb = (ob && hb) || (hb && sb);
		p = p + rb;
	} else {
		// no need to round
		p = temp << -extra_bits;
	}

	return p;
}
	
__device__ __inline__ float fp16tofp32_gpu(fp16 p) {
	
  // handle zero
  if (p == 0)
    return 0.0;

  // handle infinity
  if (p == _g_infp_gpu)
    return INFINITY;

  if (p == _g_maxrealp_gpu)
    return _g_maxreal_gpu;

  if (p == _g_minrealp_gpu)
    return _g_minreal_gpu;

  double f = 1.0;

  // check sign bit
  POSIT_TYPE sign = p & SIGN_MASK;
  // if negative, get the two's complement
  if (sign) {
    p = ~p + 1;
    f = -f;
  }

  // get the regime
  POSIT_TYPE second_bit = p & SECOND_BIT_MASK;
  // remove the sign
  p <<= 1;
  int regime = 0;
  int regime_length = 0;
  if (second_bit) {
    // sign of regime is +ve, find first 0
	// Here we have to subtract the posit limb size, because clz takes an
	// int which aligns the short to the right
    POSIT_TYPE flipped = ~p;
    regime = __clz(flipped) - POSIT_LIMB_SIZE - 1;
    regime_length = regime + 2;
  } else {
    // sign of regime is -ve, find first 1
	regime = POSIT_LIMB_SIZE - __clz(p);
    regime_length = 1 - regime;
  }

  // remove regime and get exponent
  p <<= regime_length;
  int exponent = p >> (POSIT_LIMB_SIZE - _g_esize_gpu);

  // remove exponent and get fraction
  p <<= _g_esize_gpu;
  int running_length = (regime_length + 1 + _g_esize_gpu);
  int fraction_size = ((_g_nbits_gpu - running_length) + abs((_g_nbits_gpu - running_length))) >> 1;
  int fraction = p >> (POSIT_LIMB_SIZE - fraction_size);
  fraction = fraction | (1 << fraction_size);

  return f * ((float)fraction / (float)(1 << fraction_size)) *
         powf(_g_useed_gpu, regime) * (1 << exponent);

}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {

  fp16 p = 0;
  if (f == 0.0) {
    return p;
  }

  if (f == INFINITY || f == -INFINITY) {
    return _g_infp_gpu;
  }

  if (fabs(f) >= _g_maxreal_gpu) {
    p = _g_maxrealp_gpu;
    if (f < 0)
      p = ~p + 1;
    return p;
  }

  if (fabs(f) <= _g_minreal_gpu) {
    p = _g_minrealp_gpu;
    if (f < 0)
      p = ~p + 1;
    return p;
  }

  if (f == NAN)
    return _g_infp_gpu;

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

  return p << _g_posit_shift_amount_gpu;
}

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
