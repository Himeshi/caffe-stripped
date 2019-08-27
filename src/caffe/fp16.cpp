/*
 * posit.cpp
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#include "caffe/fp16.hpp"

namespace caffe {

float fp16tofp32(fp16 f16value) {
	union Bits v;
	v.ui = f16value << 16;
	return v.f;
}

fp16 fp32tofp16(float f) {
	union Bits v;
	v.f = f;
	fp16 result = v.ui >> 16;
	//round to nearest even
	if((0x00008000 & v.ui) && ((0x00007FFF & v.ui) || (1 & result))) {
		result++;
	}
	return result;
}
}
