/*
 * fp16.hpp
 * Code taken from https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#ifndef INCLUDE_CAFFE_FP16_HPP_
#define INCLUDE_CAFFE_FP16_HPP_

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>

#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};

#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t

#define fp16shift 13
#define shiftSign 16

#define infN 0x7F800000 // flt32 infinity
#define maxfp16N 0x477FE000 // max flt16 normal as a flt32
#define minN 0x38800000 // min flt16 normal as a flt32
#define signN 0x80000000 // flt32 sign bit

#define infC 0x3FC00
#define nanN 0x7F802000 // minimum flt16 nan as a flt32
#define maxC 0x23BFF
#define minC 0x1C400
#define signC 0xFFFF8000 // flt16 sign bit

#define mulN 0x52000000 // (1 << 23) / minN
#define mulC 0x33800000 // minN / (1 << (23 - shift))

#define subC 0x003FF // max flt32 subnormal down shifted
#define norC 0x00400 // min flt32 normal down shifted

#define maxD 0x1C000
#define minD 0x1C000

namespace caffe {

typedef FP16_TYPE fp16;

float fp16tofp32(fp16 f16value);

fp16 fp32tofp16(float f);

__global__ void convert_to_fp16_and_back(const int n, float* in);

__global__ void convert_to_fp16_and_back(const int n, double* in);

__device__ float fp16tofp32_gpu(fp16 f16value);

__device__ fp16 fp32tofp16_gpu(float f);

__global__ void outputweights(const int n, float* in);

__global__ void outputweights(const int n, double* in);

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
