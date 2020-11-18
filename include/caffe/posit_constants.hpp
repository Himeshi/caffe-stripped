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
#define _G_MAX_REGIME_SIZE 7
#define EXPONENT_PERCENTILE 0.9

#define _G_ESIZE_BWD 2
#define _G_MAX_REGIME_SIZE_BWD 7
#define EXPONENT_PERCENTILE_BWD 0.5

#define SIGN_MASK 0x8000
#define FLOAT_SIGN_MASK 0x80000000
#define FLOAT_SIGN_RESET_MASK 0x7FFFFFFF
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
#define TEMP_TYPE uint64_t
#define UNSIGNED_LONG_LONG_SIZE 64
#define EDP_ACC_SIZE 63
#define POSIT_EXP_SHIFT 41 //64-23
#define FLOAT_EXP_SIGN_SHIFT 30
#define FLOAT_INF 0x7F800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9
#define POSIT_LENGTH_PLUS_ONE 17

#define GET_MAX(a, b)                                                          \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define _G_INFP 32768

#if _G_NBITS == 8
#define _G_POSIT_SHIFT_AMOUNT 8
#define _G_MAXREALP 32512
#define _G_MINREALP 256
#define POSIT_EXTRA_BITS_SHIFT 57
#define POSIT_EXTRA_BITS_MASK 0x00FFFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0100000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1

#if _G_MAX_REGIME_SIZE == 1
#define _G_MAXREAL_INT 0x407c0000
#define _G_MINREAL_INT 0x39800000

#elif _G_MAX_REGIME_SIZE == 2
#define _G_MAXREAL_INT 0x41780000
#define _G_MINREAL_INT 0x39800000

#elif _G_MAX_REGIME_SIZE == 3
#define _G_MAXREAL_INT 0x42700000
#define _G_MINREAL_INT 0x39800000

#elif _G_MAX_REGIME_SIZE == 4
#define _G_MAXREAL_INT 0x43600000
#define _G_MINREAL_INT 0x39800000

#elif _G_MAX_REGIME_SIZE == 5
#define _G_MAXREAL_INT 0x44400000
#define _G_MINREAL_INT 0x39800000

#elif _G_MAX_REGIME_SIZE == 6
#define _G_MAXREAL_INT 0x45000000
#define _G_MINREAL_INT 0x39800000

#elif _G_MAX_REGIME_SIZE == 7
#define _G_MAXREAL_INT 0x45800000
#define _G_MINREAL_INT 0x39800000
#endif

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3

#if _G_MAX_REGIME_SIZE == 1
#define _G_MAXREAL_INT 0x41780000
#define _G_MINREAL_INT 0x33800000

#elif _G_MAX_REGIME_SIZE == 2
#define _G_MAXREAL_INT 0x43700000
#define _G_MINREAL_INT 0x33800000

#elif _G_MAX_REGIME_SIZE == 3
#define _G_MAXREAL_INT 0x45600000
#define _G_MINREAL_INT 0x33800000

#elif _G_MAX_REGIME_SIZE == 4
#define _G_MAXREAL_INT 0x47400000
#define _G_MINREAL_INT 0x33800000

#elif _G_MAX_REGIME_SIZE == 5
#define _G_MAXREAL_INT 0x49000000
#define _G_MINREAL_INT 0x33800000

#elif _G_MAX_REGIME_SIZE == 6
#define _G_MAXREAL_INT 0x4A800000
#define _G_MINREAL_INT 0x33800000

#elif _G_MAX_REGIME_SIZE == 7
#define _G_MAXREAL_INT 0x4B800000
#define _G_MINREAL_INT 0x33800000
#endif

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7

#if _G_MAX_REGIME_SIZE == 2
#define _G_MAXREAL_INT 0x47600000
#define _G_MINREAL_INT 0x27800000

#elif _G_MAX_REGIME_SIZE == 3
#define _G_MAXREAL_INT 0x4B400000
#define _G_MINREAL_INT 0x27800000

#elif _G_MAX_REGIME_SIZE == 4
#define _G_MAXREAL_INT 0x4F000000
#define _G_MINREAL_INT 0x27800000

#elif _G_MAX_REGIME_SIZE == 5
#define _G_MAXREAL_INT 0x52800000
#define _G_MINREAL_INT 0x27800000

#elif _G_MAX_REGIME_SIZE == 6
#define _G_MAXREAL_INT 0x55800000
#define _G_MINREAL_INT 0x27800000

#elif _G_MAX_REGIME_SIZE == 7
#define _G_MAXREAL_INT 0x57800000
#define _G_MINREAL_INT 0x27800000
#endif

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 7.922816251e+28
#define _G_MINREAL 1.262177448e-29

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#if _G_ESIZE_BWD == 1
#define _G_USEED_BWD 4
#define _G_USEED_ZEROS_BWD 2
#define POSIT_EXPONENT_MASK_BWD 1

#if _G_MAX_REGIME_SIZE_BWD == 1
#define _G_MAXREAL_INT_BWD 0x407c0000
#define _G_MINREAL_INT_BWD 0x39800000

#elif _G_MAX_REGIME_SIZE_BWD == 2
#define _G_MAXREAL_INT_BWD 0x41780000
#define _G_MINREAL_INT_BWD 0x39800000

#elif _G_MAX_REGIME_SIZE_BWD == 3
#define _G_MAXREAL_INT_BWD 0x42700000
#define _G_MINREAL_INT_BWD 0x39800000

#elif _G_MAX_REGIME_SIZE_BWD == 4
#define _G_MAXREAL_INT_BWD 0x43600000
#define _G_MINREAL_INT_BWD 0x39800000

#elif _G_MAX_REGIME_SIZE_BWD == 5
#define _G_MAXREAL_INT_BWD 0x44400000
#define _G_MINREAL_INT_BWD 0x39800000

#elif _G_MAX_REGIME_SIZE_BWD == 6
#define _G_MAXREAL_INT_BWD 0x45000000
#define _G_MINREAL_INT_BWD 0x39800000

#elif _G_MAX_REGIME_SIZE_BWD == 7
#define _G_MAXREAL_INT_BWD 0x45800000
#define _G_MINREAL_INT_BWD 0x39800000
#endif

#elif _G_ESIZE_BWD == 2
#define _G_USEED_BWD 16
#define _G_USEED_ZEROS_BWD 4
#define POSIT_EXPONENT_MASK_BWD 3

#if _G_MAX_REGIME_SIZE_BWD == 1
#define _G_MAXREAL_INT_BWD 0x41780000
#define _G_MINREAL_INT_BWD 0x33800000

#elif _G_MAX_REGIME_SIZE_BWD == 2
#define _G_MAXREAL_INT_BWD 0x43700000
#define _G_MINREAL_INT_BWD 0x33800000

#elif _G_MAX_REGIME_SIZE_BWD == 3
#define _G_MAXREAL_INT_BWD 0x45600000
#define _G_MINREAL_INT_BWD 0x33800000

#elif _G_MAX_REGIME_SIZE_BWD == 4
#define _G_MAXREAL_INT_BWD 0x47400000
#define _G_MINREAL_INT_BWD 0x33800000

#elif _G_MAX_REGIME_SIZE_BWD == 5
#define _G_MAXREAL_INT_BWD 0x49000000
#define _G_MINREAL_INT_BWD 0x33800000

#elif _G_MAX_REGIME_SIZE_BWD == 6
#define _G_MAXREAL_INT_BWD 0x4A800000
#define _G_MINREAL_INT_BWD 0x33800000

#elif _G_MAX_REGIME_SIZE_BWD == 7
#define _G_MAXREAL_INT_BWD 0x4B800000
#define _G_MINREAL_INT_BWD 0x33800000
#endif

#elif _G_ESIZE_BWD == 3
#define _G_USEED_BWD 256
#define _G_USEED_ZEROS_BWD 8
#define POSIT_EXPONENT_MASK_BWD 7

#if _G_MAX_REGIME_SIZE_BWD == 2
#define _G_MAXREAL_INT_BWD 0x47600000
#define _G_MINREAL_INT_BWD 0x27800000

#elif _G_MAX_REGIME_SIZE_BWD == 3
#define _G_MAXREAL_INT_BWD 0x4B400000
#define _G_MINREAL_INT_BWD 0x27800000

#elif _G_MAX_REGIME_SIZE_BWD == 4
#define _G_MAXREAL_INT_BWD 0x4F000000
#define _G_MINREAL_INT_BWD 0x27800000

#elif _G_MAX_REGIME_SIZE_BWD == 5
#define _G_MAXREAL_INT_BWD 0x52800000
#define _G_MINREAL_INT_BWD 0x27800000

#elif _G_MAX_REGIME_SIZE_BWD == 6
#define _G_MAXREAL_INT_BWD 0x55800000
#define _G_MINREAL_INT_BWD 0x27800000

#elif _G_MAX_REGIME_SIZE_BWD == 7
#define _G_MAXREAL_INT_BWD 0x57800000
#define _G_MINREAL_INT_BWD 0x27800000
#endif

#elif _G_ESIZE_BWD == 4
#define _G_USEED_BWD 512
#define _G_USEED_ZEROS_BWD 16
#define _G_MAXREAL_BWD 7.922816251e+28
#define _G_MINREAL_BWD 1.262177448e-29

#else
#define _G_USEED_BWD 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS_BWD (1 << _G_ESIZE)
#define _G_MAXREAL_BWD pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL_BWD (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#endif

#endif /* INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_ */
