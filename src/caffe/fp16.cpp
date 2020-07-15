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
	v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
	v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
	union Bits s;
	s.si = mulC;
	s.f *= v.si;
	int32_t mask = -(norC > v.si);
	v.si <<= fp16shift;
	v.si ^= (s.si ^ v.si) & mask;
	v.si |= sign;
	return v.f;
}

fp16 fp32tofp16(float f) {
	union Bits v, s;
	v.f = f;
	uint32_t sign = v.si & signN;
	v.si ^= sign;
	sign >>= shiftSign; // logical shift
	s.si = mulN;
	s.si = s.f * v.f; // correct subnormals
	// get the bits that could potentially be cut off for rounding
	int32_t bits = (v.si & 0x000003FF) & -(minN > v.si);
	v.si ^= (s.si ^ v.si) & -(minN > v.si);
	v.si ^= (nanN ^ v.si) & -(v.si > infN);
	v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxfp16N));
	s.si = (v.si & 0x00001000) && ((v.si & 0x00002000) | (v.si & 0x00000FFF) | bits);
	v.ui >>= fp16shift; // logical shift
	v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
	v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
	v.si += s.si;
	return v.ui | sign;
}
}
