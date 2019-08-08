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

#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t

#define _G_NBITS 16
#define _G_ESIZE 2

#define SIGN_MASK 0x8000
#define SECOND_BIT_MASK 0x4000
#define POSIT_INF 0x0000
#define POSIT_LIMB_ALL_BITS_SET 0xffff
#define SINGLE_PRECISION_BIAS 127
#define FLOAT_SIZE 32
#define FLOAT_EXPONENT_MASK 0x7f800000
#define FLOAT_FRACTION_MASK 0x007fffff
#define FLOAT_SIGN_SHIFT 31
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_DENORMAL_EXPONENT -126
#define FLOAT_HIDDEN_BIT_SET_MASK 0x00800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE 8
#define TEMP_TYPE unsigned long long
#define UNSIGNED_LONG_LONG_SIZE 64
#define EDP_ACC_SIZE 63

#define GET_MAX(a, b)                                                          \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#if _G_NBITS == 16
#if _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_POSIT_SHIFT_AMOUNT 0
#define _G_MAXREALP 32767
#define _G_MINREALP 1
#define _G_INFP 32768
#define _G_MAXREAL 7.205759e+16
#define _G_MINREAL 1.387779e-17
#endif

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_POSIT_SHIFT_AMOUNT (FP16_LIMB_SIZE - _G_NBITS)
#define _G_MAXREALP ((1 << (_G_NBITS - 1)) - 1) << _G_POSIT_SHIFT_AMOUNT
#define _G_MINREALP (1 << _G_POSIT_SHIFT_AMOUNT)
#define _G_INFP 1 << (FP16_LIMB_SIZE - 1)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

namespace caffe {

typedef FP16_TYPE fp16;

fp16 get_posit_from_parts(int exponent, unsigned int fraction,
                           unsigned int fraction_size);

float fp16tofp32(fp16 p);

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
