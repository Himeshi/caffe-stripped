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
  p = (p ^ -sign) + sign;

  // get the regime sign
  bool regime_sign = p & SECOND_BIT_MASK;

  // get regime
  v.ui = p << POSIT_LENGTH_PLUS_ONE;
  //int regime_length = (__clz(v.ui) & -!regime_sign) + (__clz(~v.ui) & -regime_sign);
  int regime_length;
  if(regime_sign)
    regime_length = (__clz(~v.ui));
  else
    regime_length = (__clz(v.ui));
  int regime = (regime_length - regime_sign) << _G_ESIZE;
  regime = (regime ^ -regime_sign) + regime_sign;

  // assemble
  v.ui <<= (regime_length + 1);
  v.ui >>= (FLOAT_SIGN_PLUS_EXP_LENGTH - _G_ESIZE);
  v.ui += ((SINGLE_PRECISION_BIAS - regime) << FLOAT_EXPONENT_SHIFT);

  v.si ^= (FLOAT_INF ^ v.si) & -(p == _G_INFP);
  v.si ^= (0 ^ v.si) & -(p == 0);

  v.ui |= (sign << FLOAT_SIGN_SHIFT);
  return v.f;

}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  fp16 p = 0;
  union Bits v;
  v.f = f;
  bool sign = v.ui & FLOAT_SIGN_MASK;
  v.ui &= 0x7FFFFFFF;

#ifdef FLOAT_ROUNDING
	uint16_t roundSign = sign << 15;
	if(v.ui > _G_MAXREAL_INT)
		return _G_INFP | roundSign;
	if(v.ui < _G_MINREAL_INT)
		return 0;
#endif
  p ^= (p ^_G_MAXREALP) & -(v.si >= _G_MAXREAL_INT);
  p ^= (p ^ _G_INFP) & -(v.si >= FLOAT_INF);
  p ^= (p ^ _G_MINREALP) & -(v.si <= _G_MINREAL_INT);

  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  // get exponent sign
  bool exp_sign = !(v.ui >> FLOAT_EXP_SIGN_SHIFT);

  //get regime and exponent
  uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  TEMP_TYPE regime_and_exp = (((1 << ((exp >> _G_ESIZE) + 1)) - 1) << (_G_ESIZE + 1)) | (exp & POSIT_EXPONENT_MASK);;
  //if exponent is negative
  regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign) >> ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);
  int regime_and_exp_length = (exp >> _G_ESIZE) + 2 + _G_ESIZE - ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);

  //assemble
  regime_and_exp <<= (UNSIGNED_LONG_LONG_SIZE - regime_and_exp_length);
  regime_and_exp |= ((TEMP_TYPE) (v.ui & FLOAT_FRACTION_MASK) << (POSIT_EXP_SHIFT - regime_and_exp_length));
  fp16 temp_p = (regime_and_exp >> POSIT_EXTRA_BITS_SHIFT);

  //round
  temp_p += (bool) (regime_and_exp & POSIT_HALFWAY_BIT_MASK) && ((temp_p & 1) | (regime_and_exp & POSIT_EXTRA_BITS_MASK));
#if _G_NBITS != 16
  temp_p <<= _G_POSIT_SHIFT_AMOUNT;
#endif
  p ^= (temp_p ^ p) & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

  p = (p ^ -sign) + sign;

  return p;
}

__device__ __inline__ struct decomposed_posit decompose_posit_gpu(fp16 p) {
	struct decomposed_posit dp;

	//get the sign
	dp.sign = p >> (FP16_LIMB_SIZE - 1);
	if (dp.sign) {
		p = ~p + 1;
	}

	//handling it here so as to avoid error later when
	//getting the regime later by fliping p
	if (p == _G_MAXREALP) {
		dp.exponent = _G_USEED_ZEROS * (_G_NBITS - 2);
		dp.fraction = 1;
		dp.fraction_size = 0;
		return dp;
	}

	//get the regime
	fp16 second_bit = p & SECOND_BIT_MASK;
	int regime = 0;
	int regime_length = 0;
	p <<= 1;
	if (second_bit) {
		//sign of regime is +ve, find first 0
		fp16 flipped = ~p;
		regime = __clz(flipped) - FP16_LIMB_SIZE;
		regime_length = regime + 1;
		regime -= 1;
	} else {
		//sign of regime is -ve, find first 1
		regime = __clz(p) - FP16_LIMB_SIZE;
		regime_length = regime + 1;
		regime = -regime;
	}

	//remove the regime and get exponent
	p <<= regime_length;
	dp.exponent = p >> (FP16_LIMB_SIZE - _G_ESIZE);
	dp.exponent = _G_USEED_ZEROS * regime + dp.exponent;

	//remove exponent and get the fraction
	p <<= _G_ESIZE;
	int running_length = 1 + regime_length + _G_ESIZE;
	dp.fraction_size = GET_MAX((_G_NBITS - running_length), 0);
	dp.fraction = p >> (FP16_LIMB_SIZE - dp.fraction_size);
	dp.fraction = dp.fraction | (1 << dp.fraction_size);
	return dp;
}

__device__ __inline__ fp16 get_posit_from_parts_gpu(int exponent, unsigned int fraction, unsigned int fraction_size) {
	//assume the fraction is normalized and it's MSB is hidden already
	fp16 p = 0;
	int regime, regime_length, exponentp, ob, hb, sb, rb;
	TEMP_TYPE temp;

	//find regime and exponent
	if (exponent >= 0) {
		regime = exponent / _G_USEED_ZEROS;
		exponentp = exponent - (_G_USEED_ZEROS * regime);
		regime_length = regime + 2;
		regime = ((1 << (regime + 1)) - 1) << 1;
	} else {
		regime = abs(exponent / _G_USEED_ZEROS);
		if (exponent % _G_USEED_ZEROS)
			regime += 1;
		regime_length = regime + 1;
		exponentp = exponent + (_G_USEED_ZEROS * regime);
		regime = 1;
	}

	//assemble the regime
	temp = regime;
	int running_size = regime_length + 1;

	//assemble the exponent
	temp <<= _G_ESIZE;
	int exponent_length = FLOAT_SIZE - __builtin_clz(abs(exponentp));
	exponentp >>= (((exponent_length - _G_ESIZE)
			+ abs((exponent_length - _G_ESIZE))) >> 1);
	temp |= exponentp;
	running_size += _G_ESIZE;

	//assemble the fraction
	temp <<= fraction_size;
	temp |= fraction;
	running_size += fraction_size;

	int extra_bits = (running_size - _G_NBITS);

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

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
