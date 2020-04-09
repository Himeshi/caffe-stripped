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
	
__device__ __inline__ float fp16tofp32_gpu(fp16 p) {
  union Bits v;

  // get sign
  bool sign = p & SIGN_MASK;
  p = (p ^ -sign) + sign;

  // get the regime sign
  bool regime_sign = p & SECOND_BIT_MASK;

  // get regime
  v.ui = p << POSIT_LENGTH_PLUS_ONE;
  //int regime_length = (__clz(v.ui) & -!regime_sign) + (__clz(~v.ui) & -regime_sign);
  int regime_length;
  if(regime_sign)
    regime_length = (__clz(~v.ui));
  else
    regime_length = (__clz(v.ui));
  int regime = (regime_length - regime_sign) << _G_ESIZE;
  regime = (regime ^ -regime_sign) + regime_sign;

  // assemble
  v.ui <<= (regime_length + 1);
  v.ui >>= (FLOAT_SIGN_PLUS_EXP_LENGTH - _G_ESIZE);
  v.ui += ((SINGLE_PRECISION_BIAS - regime) << FLOAT_EXPONENT_SHIFT);

  v.si ^= (FLOAT_INF ^ v.si) & -(p == _G_INFP);
  v.si ^= (0 ^ v.si) & -(p == 0);

  v.ui |= (sign << FLOAT_SIGN_SHIFT);
  return v.f;

}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  fp16 p = 0;
  union Bits v;
  v.f = f;
  bool sign = v.ui & FLOAT_SIGN_MASK;
  v.ui &= 0x7FFFFFFF;

#ifdef FLOAT_ROUNDING
	uint16_t roundSign = sign << 15;
	if(v.ui > _G_MAXREAL_INT)
		return _G_INFP | roundSign;
	if(v.ui < _G_MINREAL_INT)
		return 0;
#endif
  p ^= (p ^_G_MAXREALP) & -(v.si >= _G_MAXREAL_INT);
  p ^= (p ^ _G_INFP) & -(v.si >= FLOAT_INF);
  p ^= (p ^ _G_MINREALP) & -(v.si != 0 && v.si <= _G_MINREAL_INT);

  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  // get exponent sign
  bool exp_sign = !(v.ui >> FLOAT_EXP_SIGN_SHIFT);

  //get regime and exponent
  uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  TEMP_TYPE regime_and_exp = (((1 << ((exp >> _G_ESIZE) + 1)) - 1) << (_G_ESIZE + 1)) | (exp & POSIT_EXPONENT_MASK);;
  //if exponent is negative
  regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign) >> ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);
  int regime_and_exp_length = (exp >> _G_ESIZE) + 2 + _G_ESIZE - ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);

  //assemble
  regime_and_exp <<= (UNSIGNED_LONG_LONG_SIZE - regime_and_exp_length);
  regime_and_exp |= ((TEMP_TYPE) (v.ui & FLOAT_FRACTION_MASK) << (POSIT_EXP_SHIFT - regime_and_exp_length));
  fp16 temp_p = (regime_and_exp >> POSIT_EXTRA_BITS_SHIFT);

  //round
  temp_p += (bool) (regime_and_exp & POSIT_HALFWAY_BIT_MASK) && ((temp_p & 1) | (regime_and_exp & POSIT_EXTRA_BITS_MASK));
#if _G_NBITS != 16
  temp_p <<= _G_POSIT_SHIFT_AMOUNT;
#endif
  p ^= (temp_p ^ p) & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

  p = (p ^ -sign) + sign;

  return p;
}

__device__ __inline__ struct decomposed_posit decompose_posit_gpu(fp16 p) {
	struct decomposed_posit dp;

	//get the sign
	dp.sign = p >> (FP16_LIMB_SIZE - 1);
	if (dp.sign) {
		p = ~p + 1;
	}

	//handling it here so as to avoid error later when
	//getting the regime later by fliping p
	if (p == _G_MAXREALP) {
		dp.exponent = _G_USEED_ZEROS * (_G_NBITS - 2);
		dp.fraction = 1;
		dp.fraction_size = 0;
		return dp;
	}

	//get the regime
	fp16 second_bit = p & SECOND_BIT_MASK;
	int regime = 0;
	int regime_length = 0;
	p <<= 1;
	if (second_bit) {
		//sign of regime is +ve, find first 0
		fp16 flipped = ~p;
		regime = __clz(flipped) - FP16_LIMB_SIZE;
		regime_length = regime + 1;
		regime -= 1;
	} else {
		//sign of regime is -ve, find first 1
		regime = __clz(p) - FP16_LIMB_SIZE;
		regime_length = regime + 1;
		regime = -regime;
	}

	//remove the regime and get exponent
	p <<= regime_length;
	dp.exponent = p >> (FP16_LIMB_SIZE - _G_ESIZE);
	dp.exponent = _G_USEED_ZEROS * regime + dp.exponent;

	//remove exponent and get the fraction
	p <<= _G_ESIZE;
	int running_length = 1 + regime_length + _G_ESIZE;
	dp.fraction_size = GET_MAX((_G_NBITS - running_length), 0);
	dp.fraction = p >> (FP16_LIMB_SIZE - dp.fraction_size);
	dp.fraction = dp.fraction | (1 << dp.fraction_size);
	return dp;
}

__device__ __inline__ fp16 get_posit_from_parts_gpu(int exponent, unsigned int fraction, unsigned int fraction_size) {
	//assume the fraction is normalized and it's MSB is hidden already
	fp16 p = 0;
	int regime, regime_length, exponentp, ob, hb, sb, rb;
	TEMP_TYPE temp;

	//find regime and exponent
	if (exponent >= 0) {
		regime = exponent / _G_USEED_ZEROS;
		exponentp = exponent - (_G_USEED_ZEROS * regime);
		regime_length = regime + 2;
		regime = ((1 << (regime + 1)) - 1) << 1;
	} else {
		regime = abs(exponent / _G_USEED_ZEROS);
		if (exponent % _G_USEED_ZEROS)
			regime += 1;
		regime_length = regime + 1;
		exponentp = exponent + (_G_USEED_ZEROS * regime);
		regime = 1;
	}

	//assemble the regime
	temp = regime;
	int running_size = regime_length + 1;

	//assemble the exponent
	temp <<= _G_ESIZE;
	int exponent_length = FLOAT_SIZE - __clz(abs(exponentp));
	exponentp >>= (((exponent_length - _G_ESIZE)
			+ abs((exponent_length - _G_ESIZE))) >> 1);
	temp |= exponentp;
	running_size += _G_ESIZE;

	//assemble the fraction
	temp <<= fraction_size;
	temp |= fraction;
	running_size += fraction_size;

	int extra_bits = (running_size - _G_NBITS);

	if (extra_bits > 0) {
		//round
		p = temp >> extra_bits;
		ob = p & 0x0001;
		hb = (temp >> (extra_bits - 1)) & 1ULL;
		sb = ((1ULL << (extra_bits - 1)) - 1) & temp;
		rb = (ob && hb) || (hb && sb);
		p = p + rb;
	} else {
		// no need to round
		p = temp << -extra_bits;
	}

	return p;
}

__device__ __inline__ fp16 add_posit_gpu(fp16 a, fp16 b) {
	fp16 result = 0;

	int exponent, sign, fraction_size;
	unsigned long long temp_a, temp_b, fraction;

	//handle special cases
	//NaR and inf
	if (a == _G_INFP || b == _G_INFP)
		return _G_INFP;

	//zero
	if (a == 0 || b == 0)
		return a | b;

	//normal case
	struct decomposed_posit a_decomposed = decompose_posit_gpu(a);
	struct decomposed_posit b_decomposed = decompose_posit_gpu(b);

	//align the binary point
	int fraction_diff = a_decomposed.fraction_size - b_decomposed.fraction_size;
	if (fraction_diff > 0) {
		temp_a = a_decomposed.fraction;
		temp_b = b_decomposed.fraction << fraction_diff;
		fraction_size = a_decomposed.fraction_size;
	} else {
		temp_a = a_decomposed.fraction << -fraction_diff;
		temp_b = b_decomposed.fraction;
		fraction_size = b_decomposed.fraction_size;
	}

	//make the exponents the same
	int exponent_diff = a_decomposed.exponent - b_decomposed.exponent;
	if(abs(exponent_diff) > 13) {
		//11 + 1 is the maximum number of possible fraction bits
		//if you shift beyond this, then there is no overlap
		// this is to make the addition possible within 64 bits
		if(exponent_diff > 0) {
			exponent = a_decomposed.exponent - 2;
			temp_b = 1;
			temp_a <<= 2;
		} else {
			exponent = b_decomposed.exponent - 2;
			temp_a = 1;
			temp_b <<= 2;
		}
	} else {
		if (exponent_diff > 0) {
			temp_a <<= exponent_diff;
			exponent = b_decomposed.exponent;
		} else {
			temp_b <<= -exponent_diff;
			exponent = a_decomposed.exponent;
		}
	}

	//add the fractions
	if (a_decomposed.sign == b_decomposed.sign) {
		fraction = temp_a + temp_b;
		sign = a_decomposed.sign;
	} else if (temp_a > temp_b) {
		fraction = temp_a - temp_b;
		sign = a_decomposed.sign;
	} else {
		fraction = temp_b - temp_a;
		sign = b_decomposed.sign;
	}

	if(fraction == 0)
		return 0;

	//normalize fraction and hide leading bit
	unsigned int size_of_fraction = UNSIGNED_LONG_LONG_SIZE
			- __clzll(fraction);
	int fraction_size_diff = fraction_size - size_of_fraction + 1;
	fraction_size = size_of_fraction - 1;
	exponent -= fraction_size_diff;
	fraction = fraction & ~(1 << fraction_size);

	if (exponent >= MAX_REGIME) {
		result = _G_MAXREALP;
		if (sign)
			result = ~result + 1;
		return result;
	}

	if(exponent <= -MAX_REGIME) {
		result = _G_MINREALP;
		if (sign)
			result = ~result + 1;
		return result;
	}

	result = get_posit_from_parts_gpu(exponent, fraction, fraction_size);

	if (sign)
		result = ~result + 1;

	return (result << _G_POSIT_SHIFT_AMOUNT);
}

__device__ __inline__ fp16 subtract_posit_gpu(fp16 a, fp16 b) {
	// a - b
	b = ~b + 1;
	return add_posit_gpu(a, b);
}

__device__ __inline__ fp16 multiply_posit_gpu(fp16 a, fp16 b) {
	fp16 result = 0;

	//handle special cases
	//NaR and inf
	if (a == _G_INFP || b == _G_INFP)
		return _G_INFP;

	//zero
	if (a == 0 || b == 0)
		return 0;

	if(a == 0x4000)
		return b;

	if(b == 0x4000)
		return a;

	//normal case
	struct decomposed_posit a_decomposed = decompose_posit_gpu(a);
	struct decomposed_posit b_decomposed = decompose_posit_gpu(b);

	//get the sign of the result
	int sign = a_decomposed.sign ^ b_decomposed.sign;

	//add the exponents
	int exponent = a_decomposed.exponent + b_decomposed.exponent;

	//multiply the fractions
	unsigned int fraction = a_decomposed.fraction * b_decomposed.fraction;

	//get the fraction size
	unsigned int fraction_size = a_decomposed.fraction_size + b_decomposed.fraction_size;

	//normalize fraction and hide the leading bit
	unsigned int size_of_fraction = FLOAT_SIZE - __clz(fraction);
	int fraction_size_diff = fraction_size - size_of_fraction + 1;
	fraction_size = size_of_fraction - 1;
	exponent -= fraction_size_diff;
	fraction = fraction & ~(1 << fraction_size);

	if (exponent >= MAX_REGIME) {
		result = _G_MAXREALP;
		if (sign)
			result = ~result + 1;
		return result;
	}

	if(exponent <= -MAX_REGIME) {
		result = _G_MINREALP;
		if (sign)
			result = ~result + 1;
		return result;
	}

	result = get_posit_from_parts_gpu(exponent, fraction, fraction_size);

	if (sign)
		result = ~result + 1;

	return (result << _G_POSIT_SHIFT_AMOUNT);
}

__device__ __inline__ fp16 divide_posit_gpu(fp16 a, fp16 b) {
	//a / b
	fp16 result = 0;

	//handle special cases
	//NaR and inf
	if (a == _G_INFP || b == _G_INFP)
		return _G_INFP;

	//zero
	if (a == 0)
		return 0;

	if(b == 0)
		return _G_INFP;

	if(b == 0x4000)
		return a;

	//normal case
	struct decomposed_posit a_decomposed = decompose_posit_gpu(a);
	struct decomposed_posit b_decomposed = decompose_posit_gpu(b);

	//get the sign of the result
	int sign = a_decomposed.sign ^ b_decomposed.sign;

	//subtract the exponents
	int exponent = a_decomposed.exponent - b_decomposed.exponent;

	//get the fraction size
	unsigned int fraction_size = b_decomposed.fraction_size - a_decomposed.fraction_size;
	exponent += fraction_size;

	//divide the fractions
	//get at least 14 bits (for posit 16, 2)
	uint32_t fraction = 0;
	int shift;
	if(fraction_size > 0) {
		shift = fraction_size + 14;
	} else {
		shift = 14;
	}
	uint32_t a_fraction = a_decomposed.fraction << shift;
	fraction = a_fraction / b_decomposed.fraction;
	if(a_fraction % b_decomposed.fraction)
		fraction |= 1;
	exponent -= shift;
	fraction_size = 0;

	//normalize fraction and hide the leading bit
	unsigned int size_of_fraction = FLOAT_SIZE - __clz(fraction);
	int fraction_size_diff = fraction_size - size_of_fraction + 1;
	fraction_size = size_of_fraction - 1;
	exponent -= fraction_size_diff;
	fraction = fraction & ~(1 << fraction_size);

	if (exponent >= MAX_REGIME) {
		result = _G_MAXREALP;
		if (sign)
			result = ~result + 1;
		return result;
	}

	if(exponent <= -MAX_REGIME) {
		result = _G_MINREALP;
		if (sign)
			result = ~result + 1;
		return result;
	}

	result = get_posit_from_parts_gpu(exponent, fraction, fraction_size);

	if (sign)
		result = ~result + 1;

	return (result << _G_POSIT_SHIFT_AMOUNT);
}

__device__ __inline__ fp16 fused_multiply_add_gpu(fp16* a, fp16* b, int count) {
	fp16 a_i = 0, b_i = 0;
	uint64_t acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
	uint64_t temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0;
	int first_shift, second_shift, third_shift, fourth_shift, carry2, carry3, carry4, acc_sign = 0;
	fp16 result = 0;
	int sign, exponent, fraction_size;
	unsigned int fraction;
	for (int i = 0; i < count; i++) {
		a_i = a[i];
		b_i = b[i];

		if (a_i == _G_INFP || b_i == _G_INFP)
			return _G_INFP;

		if (a == 0 || b == 0) {
			sign = 0;
			exponent = 0;
			fraction = 0;
			fraction_size = 0;
		} else if (a_i == 0x4000) {
			struct decomposed_posit decomposed = decompose_posit_gpu(b_i);
			sign = decomposed.sign;
			exponent = decomposed.exponent;
			fraction = decomposed.fraction;
			fraction_size = decomposed.fraction_size;
		} else if (b_i == 0x4000) {
			struct decomposed_posit decomposed = decompose_posit_gpu(a_i);
			sign = decomposed.sign;
			exponent = decomposed.exponent;
			fraction = decomposed.fraction;
			fraction_size = decomposed.fraction_size;
		} else {
			//multiply
			struct decomposed_posit a_decomposed = decompose_posit_gpu(a_i);
			struct decomposed_posit b_decomposed = decompose_posit_gpu(b_i);

			//get the sign of the result
			sign = a_decomposed.sign ^ b_decomposed.sign;

			//add the exponents
			exponent = a_decomposed.exponent + b_decomposed.exponent;

			//multiply the fractions
			fraction = a_decomposed.fraction
					* b_decomposed.fraction;

			//get the fraction size
			fraction_size = a_decomposed.fraction_size
					+ b_decomposed.fraction_size;

			//normalize fraction
			unsigned int size_of_fraction = FLOAT_SIZE - __clz(fraction);
			int fraction_size_diff = fraction_size - size_of_fraction + 1;
			fraction_size = size_of_fraction - 1;
			exponent -= fraction_size_diff;
		}

		if(fraction != 0) {
			// 2 * (56 + 1) = 114
			//subtract fraction size from exponent
			exponent -= fraction_size;
			exponent += 114;

			//114 * 2 = 228 64*4 = 256
			temp1 = 0; temp2 = 0; temp3 = 0; temp4 = 0;
			temp4 = fraction << exponent;

			// 64 - (fraction_size + 1)
			first_shift = (63 - fraction_size);
			second_shift = exponent - first_shift;
			third_shift = exponent - (first_shift + second_shift);
			fourth_shift = exponent - (first_shift + second_shift + third_shift);
			if(exponent > first_shift) {
				//temp3 needs to be populated
				//calculate how many bits will go to temp3
				temp3 = fraction << second_shift;
			}

			if(exponent > (first_shift + second_shift)) {
				temp2 = fraction << third_shift;
			}

			if (exponent > (first_shift + second_shift + third_shift)) {
				temp1 = fraction << fourth_shift;
			}

		    //update accumulator
			carry4 = 0; carry3 = 0; carry2 = 0;
			if(sign == acc_sign) {
				carry4 = acc4 > (0xFFFFFFFFFFFFFFFF - temp4);
				acc4 += temp4;
				carry3 = acc3 > (0xFFFFFFFFFFFFFFFF - temp3 - carry4);
				acc3 += (temp3 + carry4);
				carry2 = acc2 > (0xFFFFFFFFFFFFFFFF - temp2 - carry3);
				acc2 += (temp2 + carry3);
				acc1 += (temp1 + carry2);
			} else {
				//figure out which is the larger value
				int acc_larger = 0;
				if(acc1 > temp1)
					acc_larger = 1;
				else if(acc1 == 0 && temp1 == 0 && acc2 > temp2)
					acc_larger = 1;
				else if(acc1 == 0 && temp1 == 0 && acc2 == 0 && temp2 == 0 && acc3 > temp3)
					acc_larger = 1;
				else if(acc1 == 0 && temp1 == 0 && acc2 == 0 && temp2 == 0 && acc3 == 0 && temp3 == 0 && acc4 > temp4)
					acc_larger = 1;

				if(acc_larger) {
					//convert temp to 2's complement
					temp4 = ~temp4;
					temp3 = ~temp3;
					temp2 = ~temp2;
					temp1 = ~temp1;

					temp4 += 1;
					if (temp4 == 0)
						temp3 += 1;
					if (temp3 == 0)
						temp2 += 1;
					if (temp2 == 0)
						temp1 += 1;
				} else {
					acc_sign = sign;

					acc4 = ~acc4;
					acc3 = ~acc3;
					acc2 = ~acc2;
					acc1 = ~acc1;

					acc4 += 1;
					if (acc4 == 0)
						acc3 += 1;
					if (acc3 == 0)
						acc2 += 1;
					if (acc2 == 0)
						acc1 += 1;
				}

				carry4 = acc4 > (0xFFFFFFFFFFFFFFFF - temp4);
				acc4 += temp4;
				carry3 = acc3 > (0xFFFFFFFFFFFFFFFF - temp3 - carry4);
				acc3 += (temp3 + carry4);
				carry2 = acc2 > (0xFFFFFFFFFFFFFFFF - temp2 - carry3);
				acc2 += (temp2 + carry3);
				acc1 += (temp1 + carry2);

			}
		}
	}

	//get exponent
	//find leading 1
	int local_pos, leading_pos, final_exp;
	if(acc1 != 0) {
		local_pos = 64 - __clzll(acc1);
		leading_pos = local_pos + (64 * 3);
		final_exp = leading_pos - 115;
		acc1 = acc1 & ~(1 << (local_pos - 1));
	} else if(acc2 != 0) {
		local_pos = 64 - __clzll(acc2);
		leading_pos = local_pos + (64 * 2);
		final_exp = leading_pos - 115;
		acc2 = acc2 & ~(1 << (local_pos - 1));
	} else if(acc3 != 0) {
		local_pos = 64 - __clzll(acc3);
		leading_pos = local_pos + 64;
		final_exp = leading_pos - 115;
		acc3 = acc3 & ~(1 << (local_pos - 1));
	} else if(acc4 != 0) {
		local_pos = 64 - __clzll(acc4);
		leading_pos = local_pos;
		final_exp = leading_pos - 115;
		acc4 = acc4 & ~(1 << (local_pos - 1));
	} else {
		return result;
	}


	if (final_exp >= MAX_REGIME) {
		result = _G_MAXREALP;
		if (acc_sign)
			result = ~result + 1;
		return result;
	}

	if(final_exp <= -MAX_REGIME) {
		result = _G_MINREALP;
		if (acc_sign)
			result = ~result + 1;
		return result;
	}

	//convert acc into posit result
	int regime, regime_length, exponentp, hb, sb, rb;
	TEMP_TYPE temp;

	//find regime and exponent
	if (final_exp >= 0) {
		regime = final_exp / _G_USEED_ZEROS;
		exponentp = final_exp - (_G_USEED_ZEROS * regime);
		regime_length = regime + 2;
		regime = ((1 << (regime + 1)) - 1) << 1;
	} else {
		regime = abs(final_exp / _G_USEED_ZEROS);
		if (final_exp % _G_USEED_ZEROS)
			regime += 1;
		regime_length = regime + 1;
		exponentp = final_exp + (_G_USEED_ZEROS * regime);
		regime = 1;
	}

	//assemble the regime
	temp = regime;
	int running_size = regime_length + 1;

	//assemble the exponent
	temp <<= _G_ESIZE;
	int exponent_length = FLOAT_SIZE - __clz(abs(exponentp));
	exponentp >>= (((exponent_length - _G_ESIZE)
			+ abs((exponent_length - _G_ESIZE))) >> 1);
	temp |= exponentp;
	running_size += _G_ESIZE;

	//assemble the fraction
	uint64_t halfway_mask;
	int remaining_space = _G_NBITS - running_size;
	int hb_pos;
	if(remaining_space > 0) {
		int fraction_start = leading_pos - 1;
		int fraction_end = leading_pos - remaining_space;
		hb_pos = fraction_end - 1;
		temp <<= remaining_space;
		//fraction start can't be > 192
		if (fraction_start > 128 && fraction_end > 128) {
			result = temp | (acc2 >> (fraction_end - 129));
		} else if (fraction_start > 128 && fraction_end <= 128) {
			int dangling_bits = remaining_space - fraction_start + 128;
			temp |= (acc2 << dangling_bits);
			temp |= (acc3 >> (64 - dangling_bits));
		} else if (fraction_start > 64 && fraction_end > 64) {
			result = temp | (acc3 >> (fraction_end - 65));
		} else if (fraction_start > 64 && fraction_end <= 64) {
			int dangling_bits = remaining_space - fraction_start + 64;
			temp |= (acc3 << dangling_bits);
			temp |= (acc4 >> (64 - dangling_bits));
		} else {
			result = temp | (acc4 >> (fraction_end - 1));
		}

		if(hb_pos > 128) {
			hb_pos -= 128;
			hb = acc2 & (1 << (hb_pos - 1));
			sb = (acc2 << (64 - hb_pos)) | acc3 | acc4;
		} else if (hb_pos > 64) {
			hb_pos -= 64;
			hb = acc3 & (1 << (hb_pos - 1));
			sb = (acc3 << (64 - hb_pos)) | acc4;
		} else {
			hb = acc4 & (1 << (hb_pos - 1));
			sb = (acc4 << (64 - hb_pos));
		}
	} else if(remaining_space == 0) {
		result = temp;
		hb_pos = local_pos - 1;
		halfway_mask = (1 << (hb_pos - 1));
		if (hb_pos > 192) {
			hb = acc1 & halfway_mask;
			acc1 = acc1 & ~halfway_mask;
		} else if (hb_pos > 128) {
			hb = acc2 & halfway_mask;
			acc2 = acc2 & ~halfway_mask;
		} else if (hb_pos > 64) {
			hb = acc3 & halfway_mask;
			acc3 = acc3 & ~halfway_mask;
		} else {
			hb = acc4 & halfway_mask;
			acc4 = acc4 & ~halfway_mask;
		}
		sb = acc1 | acc2 | acc3 | acc4;
	}
	else {
		remaining_space = -remaining_space;
		result = temp >> remaining_space;
		halfway_mask = (1 << (remaining_space - 1));
		hb = temp & halfway_mask;
		sb = (temp & (halfway_mask - 1)) | acc1 | acc2 | acc3 | acc4;
	}

	rb = hb && ((result & 1) | sb);
	result += rb;

	if(acc_sign)
		result = ~result + 1;

	return result;
}


}
#endif /* INCLUDE_CAFFE_FP16_HPP_ */
