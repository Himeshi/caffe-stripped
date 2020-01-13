/*
 * sampling.cpp
 *
 *  Created on: Jan 13, 2020
 *      Author: himeshi
 */

#include "caffe/sampling.hpp"

namespace caffe {

void sample_blob(const fp16* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, int sampling_frequency) {
	fp16 p;
	union Bits v;
	for (int i = 0; i < blob_count; i+= sampling_frequency) {
		cudaMemcpy(&p, &blob[i], sizeof(fp16), cudaMemcpyDeviceToHost);
		if(p == 0) {
			exp_map[0]++;
			frac_map[0]++;
		} else {
			bool sign = p & SIGN_MASK;
			p = (p ^ -sign) + sign;
			bool regime_sign = p & SECOND_BIT_MASK;

			// get regime
			v.ui = p << 17;
			//int regime_length = (__builtin_clz(v.ui) & -!regime_sign) + (__builtin_clz(~v.ui) & -regime_sign);
			int regime_length;
			  if(regime_sign)
			    regime_length = (__builtin_clz(~v.ui));
			  else
			    regime_length = (__builtin_clz(v.ui));
			int regime = (regime_length - regime_sign) << _G_ESIZE;
			regime = (regime ^ -regime_sign) + regime_sign;

			v.ui <<= (regime_length + 1);
			v.ui >>= (9 - _G_ESIZE);
			v.ui += ((SINGLE_PRECISION_BIAS - regime) << FLOAT_EXPONENT_SHIFT);
			int exponent = (v.ui & 0x7F800000) >> 23;
			exp_map[exponent - 127]++;
			frac_map[((v.ui & 0x007FFFFF) | 0x00800000)]++;
		}
	}
}

void sample_blob(const double* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, int sampling_frequency) {
	//printf("sampling\n");
}

void print_map(std::map<int, int> sample_map) {
	std::map<int,int>::const_iterator it;
	for (it = sample_map.begin(); it!= sample_map.end(); it++){
		std::cout << it->first<<" =>"<< it->second;
	}
	printf("\n");
}

}
