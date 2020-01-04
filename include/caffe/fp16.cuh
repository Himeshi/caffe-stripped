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
/*	
__device__ __inline__ float fp16tofp32_gpu(fp16 f16value) {
  union Bits v;
  v.ui = f16value << 16;
  return v.f;
}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  union Bits v;
  v.f = f;
  fp16 result = v.ui >> 16;
  //round to nearest even
  result += ((0x00008000 & v.ui) && ((0x00007FFF & v.ui) || (1 & result)));
  return result;
}
*/
__device__ __inline__ float fp16tofp32_gpu(fp16 f16value) {
    __half temp = *((half*)&f16value);
    return __half2float(temp);
}
__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
    __half temp =__float2half_rn(f);
    fp16 result =  *((fp16*)&temp);
    return result;
}
}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
