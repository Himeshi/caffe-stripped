/*
 * posit_constants.hpp
 *
 *  Created on: Sep 7, 2019
 *      Author: himeshi
 */

#ifndef INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_
#define INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_

#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t

#define _G_NBITS 8
#define _G_ESIZE 1

#define SCALING_FACTOR 1.0

#define SIGN_MASK 0x8000
#define POSIT_LENGTH_PLUS_ONE 17
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9
#define SINGLE_PRECISION_BIAS 127
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_INF 0x7F800000
#define _G_INFP 32768
#define FLOAT_SIGN_SHIFT 31
#define _G_MAXREAL_INT 0x3fff0000
#define _G_MINREAL_INT 0x3cc00000
#define _G_MAXREALP 32512
#define _G_MINREALP 256
#define FLOAT_SIGN_MASK 0x80000000
#define POSIT_HALFWAY_BIT_MASK 0x00008000

#endif /* INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_ */
