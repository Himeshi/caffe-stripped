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
	int regime, regime_length, exponentp;
	TEMP_TYPE temp, ob, hb, sb;

	// compute regime and exponent
	int abs_exp = abs(exponent);
	regime = abs_exp >> _G_USEED_ZEROS_SHIFT;
	exponentp = abs_exp & ((1 << _G_ESIZE) - 1);
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
	union Bits v;
	v.f = f;
	uint32_t sign = v.si & FLOAT_SIGN_MASK;
	v.si ^= sign;
	sign >>= FLOAT_SIGN_SHIFT;

	if (v.f == 0.0) {
		return p;
	}

	if (v.f == INFINITY || std::isnan(v.f)) {
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

	// get sign, exponent and fraction from float

	// min posit exponent in 16, 3 is 112
	// therefore all the float subnormals will be handled
	// in the previous if statement
	int exponent = (v.ui & FLOAT_EXPONENT_MASK) >> FLOAT_EXPONENT_SHIFT;
	unsigned int fraction = (v.ui & FLOAT_FRACTION_MASK);
	exponent -= SINGLE_PRECISION_BIAS;

	p = get_posit_from_parts(exponent, fraction, FLOAT_EXPONENT_SHIFT);

	p = (p ^ -sign) + sign;

	return p << _G_POSIT_SHIFT_AMOUNT;
}
}
