/*
 * posit.cpp
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#include "caffe/fp16.hpp"

namespace caffe {

fp16 get_posit_from_parts(int exponent, unsigned int fraction,
                           unsigned int fraction_size) {
  // assume the fraction is normalized and it's MSB is hidden already
  fp16 p = 0;
  int regime, regime_length, exponentp, ob, hb, sb, rb;
  TEMP_TYPE temp;

  // find regime and exponent
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

  // assemble the regime
  temp = regime;
  int running_size = regime_length + 1;

  // assemble the exponent
  temp <<= _G_ESIZE;
  int exponent_length = FLOAT_SIZE - __builtin_clz(abs(exponentp));
  exponentp >>= (((exponent_length - _G_ESIZE)
			+ abs((exponent_length - _G_ESIZE))) >> 1);
  temp |= exponentp;
  running_size += _G_ESIZE;

  // assemble the fraction
  temp <<= fraction_size;
  temp |= fraction;
  running_size += fraction_size;

  int extra_bits = (running_size - _G_NBITS);

  if (extra_bits > 0) {
    // round
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

float fp16tofp32(fp16 p) {
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
		regime = __builtin_clz(flipped) - FP16_LIMB_SIZE - 1;
		regime_length = regime + 2;
	} else {
		// sign of regime is -ve, find first 1
		regime = FP16_LIMB_SIZE - __builtin_clz(p);
		regime_length = 1 - regime;
	}

	// remove regime and get exponent
	p <<= regime_length;
	int exponent = p >> (FP16_LIMB_SIZE - _G_ESIZE);

	// remove exponent and get fraction
	p <<= _G_ESIZE;
	int running_length = (regime_length + 1 + _G_ESIZE);
	int fraction_size = ((_G_NBITS - running_length)
			+ abs((_G_NBITS - running_length))) >> 1;
	int fraction = p >> (FP16_LIMB_SIZE - fraction_size);
	fraction = fraction | (1 << fraction_size);

	return f * ((float) fraction / (float) (1 << fraction_size))
			* pow(_G_USEED, regime) * (1 << exponent);
}

fp16 fp32tofp16(float f) {
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

	if (f == NAN)
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
		int normalization = __builtin_clz(fraction)
				- FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE;
		exponent -= normalization;
		// hide the most significant bit
		fraction &= ~(1 << normalization);
	}

	p = get_posit_from_parts(exponent, fraction, FLOAT_EXPONENT_SHIFT);

	if (f < 0)
		p = ~p + 1;

	return p << _G_POSIT_SHIFT_AMOUNT;
}
}
