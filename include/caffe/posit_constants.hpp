/*
 * posit_constants.hpp
 *
 *  Created on: Sep 7, 2019
 *      Author: himeshi
 */

#ifndef INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_
#define INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_

#define FP16_LIMB_SIZE 32
#define FP16_TYPE uint32_t

#define SCALING_FACTOR 0.25

#define _G_NBITS 17
#define _G_ESIZE 1

#define SIGN_MASK 0x80000000
#define FLOAT_SIGN_MASK 0x80000000
#define FLOAT_SIGN_RESET_MASK 0x7FFFFFFF
#define SECOND_BIT_MASK 0x40000000
#define POSIT_INF 0x00000000
#define POSIT_LIMB_ALL_BITS_SET 0xffffffff
#define SINGLE_PRECISION_BIAS 127
#define FLOAT_SIZE 32
#define FLOAT_EXPONENT_MASK 0x7f800000
#define FLOAT_FRACTION_MASK 0x007fffff
#define FLOAT_SIGN_SHIFT 31
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_DENORMAL_EXPONENT -126
#define FLOAT_HIDDEN_BIT_SET_MASK 0x00800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE 8
#define TEMP_TYPE uint64_t
#define UNSIGNED_LONG_LONG_SIZE 64
#define EDP_ACC_SIZE 63
#define POSIT_EXP_SHIFT 41 //64-23
#define FLOAT_EXP_SIGN_SHIFT 30
#define FLOAT_INF 0x7F800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9

#define GET_MAX(a, b)                                                          \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define _G_INFP 0x80000000

#if _G_NBITS == 17
#define _G_POSIT_SHIFT_AMOUNT 15
#define _G_MAXREALP 0x7FFF8000
#define _G_MINREALP 0x00008000
#define POSIT_EXTRA_BITS_SHIFT 48 // 64 - _G_NBITS + 1
#define POSIT_EXTRA_BITS_MASK 0x00007FFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0000800000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 1.073741824e+9
#define _G_MINREAL 9.31322574615e-10
#define _G_MAXREAL_INT 0x4E800000
#define _G_MINREAL_INT 0x30800000

#endif
#endif

#endif /* INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_ */
