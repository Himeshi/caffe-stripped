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
//#include "caffe/util/mkl_alternate.hpp"

//#define SATURATION_ROUNDING

union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};

#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t

#define fp16shift 14
#define shiftSign 16

#define infN 0x7F800000 // flt32 infinity
#define maxfp16N 0x4F7FC000 // max flt16 normal as a flt32
#define minN 0x30800000 // min flt16 normal as a flt32
#define signN 0x80000000 // flt32 sign bit

#define infC 0x1FE00
#define nanN 0x7FC00000 // a flt16 nan as a flt32
#define maxC 0x13DFF
#define minC 0x0C200
#define signC 0xFFFF8000 // flt16 sign bit

#define mulN 0x5A000000 // (1 << 23) / minN
#define mulC 0x2C000000 // minN / (1 << (23 - shift))

#define subC 0x001FF // max flt32 subnormal down shifted
#define norC 0x00200 // min flt32 normal down shifted

#define maxD 0xC000
#define minD 0xC000

#define maxfp16roundN 0x4F7FE000
#define minfp16roundN 0x2B800000
#define maxfp16 0x7DFF
#define minfp16 0x0001

namespace caffe {

typedef FP16_TYPE fp16;

float fp16tofp32(fp16 f16value);

fp16 fp32tofp16(float f);

void print_gpu_float_array(const float* d_data, int size);

void print_gpu_float_array(const double* d_data, int size);

void print_gpu_fp16_array(const fp16* d_data, int size);

__global__ void convert_to_fp16(const int n, float* in, fp16* out);

__global__ void convert_to_fp16(const int n, double* in, fp16* out);

__global__ void convert_to_float(const int n, fp16* in, float* out);

__global__ void convert_to_float(const int n, fp16* in, double* out);

__global__ void convert_to_fp16(const int n, const float* in, fp16* out);

__global__ void convert_to_fp16(const int n, const double* in, fp16* out);

__global__ void convert_to_float(const int n, const fp16* in, float* out);

__global__ void convert_to_float(const int n, const fp16* in, double* out);

__global__ void outputweights(const int n, float* in);

__global__ void outputweights(const int n, double* in);

}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
