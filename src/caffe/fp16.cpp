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
	union Bits v, s;
	v.f = f;
	fp16 result = v.ui >> 16;
	return result;
}
}
