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
#include "posit_constants.hpp"
#include "glog/logging.h"

//#define FLOAT_ROUNDING

namespace caffe {


union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};


struct decomposed_posit{
    short sign, exponent, fraction, fraction_size;
};

typedef FP16_TYPE fp16;

fp16 get_posit_from_parts(int exponent, unsigned int fraction,
                           unsigned int fraction_size);

float fp16tofp32(fp16 p);

fp16 fp32tofp16(float f);

struct decomposed_posit decompose_posit(fp16 p);

fp16 get_posit_from_parts(int exponent, unsigned int fraction, unsigned int fraction_size);

fp16 fused_multiply_add(fp16* a, fp16* b, int count);

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

__global__ void convert_to_float_3in1out(const int n1, const int n2, const int n3, const fp16* in1, const fp16* in2, const fp16* in3, float* out);

__global__ void convert_to_float_2in1out(const int n1, const int n2, const fp16* in1, const fp16* in2, float* out);

__global__ void convert_to_float(const int n, const fp16* in, double* out);

__global__ void outputweights(const int n, float* in);

__global__ void outputweights(const int n, double* in);

void init_cuda_buffer(void);
}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
