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

#define _G_NBITS 16
#define _G_ESIZE 2

#define SIGN_MASK 0x8000
#define FLOAT_SIGN_MASK 0x80000000
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

#define GET_MAX(a, b)                                                          \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define _G_INFP 32768

#if _G_NBITS == 16
#define _G_POSIT_SHIFT_AMOUNT 0
#define _G_MAXREALP 32767
#define _G_MINREALP 1
#define POSIT_EXTRA_BITS_SHIFT 49 // 64 - _G_NBITS + 1
#define POSIT_EXTRA_BITS_MASK 0x0000FFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0001000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 2.684354560e+8
#define _G_MINREAL 3.725290298e-9

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3
#define _G_MAXREAL 7.205759e+16
#define _G_MINREAL 1.387779e-17

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7
#define _G_MAXREAL 5.192296859e+33
#define _G_MINREAL 1.925929944e-34

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 2.695994667e+67
#define _G_MINREAL 3.709206151e-68

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

#elif _G_NBITS == 8
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
#define _G_MAXREAL 4096
#define _G_MINREAL 0.0002441406250

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3
#define _G_MAXREAL 1.677721600e+7
#define _G_MINREAL 5.960464478e-8

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7
#define _G_MAXREAL 2.814749767e+14
#define _G_MINREAL 3.552713679e-15

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

#elif _G_NBITS == 9
#define _G_POSIT_SHIFT_AMOUNT 7
#define _G_MAXREALP 32640
#define _G_MINREALP 128

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 16384
#define _G_MINREAL 0.00006103515625

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 2.684354560e+8
#define _G_MINREAL 3.725290298e-9

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 7.205759404e+16
#define _G_MINREAL 1.387778781e-17

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 5.192296859e+33
#define _G_MINREAL 1.925929944e-34

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 10
#define _G_POSIT_SHIFT_AMOUNT 6
#define _G_MAXREALP 32704
#define _G_MINREALP 64

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 65536
#define _G_MINREAL 0.00001525878906

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 4.294967296e+9
#define _G_MINREAL 2.328306437e-10

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 1.844674407e+19
#define _G_MINREAL 5.421010862e-20

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 3.402823669e+38
#define _G_MINREAL 2.938735877e-39

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 11
#define _G_POSIT_SHIFT_AMOUNT 5
#define _G_MAXREALP 32736
#define _G_MINREALP 32

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 262144
#define _G_MINREAL 3.814697266e-6

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 6.871947674e+10
#define _G_MINREAL 1.455191523e-11

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 4.722366483e+21
#define _G_MINREAL 2.117582368e-22

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 2.230074520e+43
#define _G_MINREAL 4.484155086e-44

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 12
#define _G_POSIT_SHIFT_AMOUNT 4
#define _G_MAXREALP 32752
#define _G_MINREALP 16

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 1.048576000e+6
#define _G_MINREAL 9.536743164e-7

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 1.099511628e+12
#define _G_MINREAL 9.094947018e-13

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 1.208925820e+24
#define _G_MINREAL 8.271806126e-25

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 1.461501637e+48
#define _G_MINREAL 6.842277658e-49

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 13
#define _G_POSIT_SHIFT_AMOUNT 3
#define _G_MAXREALP 32760
#define _G_MINREALP 8

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 4.194304000e+6
#define _G_MINREAL 2.384185791e-7

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 1.759218604e+13
#define _G_MINREAL 5.684341886e-14

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 3.094850098e+26
#define _G_MINREAL 3.231174268e-27

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 9.578097130e+52
#define _G_MINREAL 1.044048715e-53

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 14
#define _G_POSIT_SHIFT_AMOUNT 2
#define _G_MAXREALP 32764
#define _G_MINREALP 4

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 1.677721600e+7
#define _G_MINREAL 5.960464478e-8

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 2.814749767e+14
#define _G_MINREAL 3.552713679e-15

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 7.922816251e+28
#define _G_MINREAL 1.262177448e-29

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 6.277101735e+57
#define _G_MINREAL 1.593091911e-58

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 15
#define _G_POSIT_SHIFT_AMOUNT 1
#define _G_MAXREALP 32766
#define _G_MINREALP 2

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 6.710886400e+7
#define _G_MINREAL 1.490116119e-8

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 4.503599627e+15
#define _G_MINREAL 2.220446049e-16

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 2.028240960e+31
#define _G_MINREAL 4.930380658e-32

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 4.113761393e+62
#define _G_MINREAL 2.430865343e-63

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#endif

#endif /* INCLUDE_CAFFE_POSIT_CONSTANTS_HPP_ */
