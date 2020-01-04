/*
 * posit.cpp
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#include "cuda_fp16.h"
#include "caffe/fp16.hpp"
namespace caffe {

float fp16tofp32(fp16 f16value) {
    __half temp = *((half*)&f16value);
    return __half2float(temp);
/*	union Bits v;
	v.ui = f16value << 16;
	return v.f;
*/
}

fp16 fp32tofp16(float f) {
/*
	union Bits v;
	v.f = f;
	fp16 result = v.ui >> 16;
	//round to nearest even
	result += ((0x00008000 & v.ui) && ((0x00007FFF & v.ui) || (1 & result)));
	return result;
*/
      __half temp =  __float2half_rn(f);
      fp16 result = *((fp16*)&temp);
        return result;
//    return __float2half_rn(f);
}
}
