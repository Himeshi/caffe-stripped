/*
 * posit.cpp
 *
 *  Created on: Jan 11, 2019
 *      Author: himeshi
 */

#include "caffe/fp16.hpp"

namespace caffe {

float fp16tofp32(fp16 p) {
	union Bits v;

	// get sign
	bool sign = p & SIGN_MASK;
	p = (p ^ -sign) + sign;

	// get the regime sign
	bool regime_sign = p & SECOND_BIT_MASK;

	// get regime
	v.ui = p << POSIT_LENGTH_PLUS_ONE;
	int regime_length = (__builtin_clz(v.ui) & -!regime_sign) + (__builtin_clz(~v.ui) & -regime_sign);
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

fp16 fp32tofp16(float f) {
	fp16 p = 0;
	union Bits v;
	v.f = f;
	bool sign = v.ui & FLOAT_SIGN_MASK;
	v.ui &= 0x7FFFFFFF;

	p = _G_MAXREALP & -(v.si >= _G_MAXREAL_INT);
	p = _G_INFP & -(v.si >= FLOAT_INF);
	p = _G_MINREALP & -(v.si <= _G_MINREAL_INT);

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
	p = temp_p & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

	p = (p ^ -sign) + sign;

	return p;
}
}
