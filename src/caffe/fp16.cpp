/*
 * posit.cpp
 *
 * Code taken and modified from https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#include "caffe/fp16.hpp"

namespace caffe {

float fp16tofp32(fp16 f16value) {
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

fp16 fp32tofp16(float f) {
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
