/*
 * posit.cpp
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#include "caffe/fp16.hpp"

namespace caffe {

float fp16tofp32(fp16 p) {
	union Bits v;

	// get sign
	bool sign = p & SIGN_MASK;

	//get exponent
	v.ui = p << POSIT_LENGTH_PLUS_ONE;
	int exponent = (__builtin_clz(v.ui));

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

fp16 fp32tofp16(float f) {
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
	int rb = (temp & POSIT_HALFWAY_BIT_MASK) && ((temp & 0x00010000) || (temp & 0x00007fff));
	temp = (temp >> 24) + rb;
	temp <<= 8;
	p ^= (temp ^ p) & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));
	p |= (sign << 15);

	return p;
}
}
