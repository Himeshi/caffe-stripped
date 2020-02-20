/*
 * sampling.cpp
 *
 *  Created on: Jan 10, 2020
 *      Author: himeshi
 */

#include "caffe/sampling.hpp"

namespace caffe {

void sample_blob(const float* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map, std::vector<uint32_t> val_vector, int sampling_frequency) {
	float temp;
	union Bits v;
	for (int i = 0; i < blob_count; i+= sampling_frequency) {
		cudaMemcpy(&temp, &blob[i], sizeof(float), cudaMemcpyDeviceToHost);
		v.f = temp;
#ifdef SAMPLE_EXP
		if(v.ui == 0) {
			exp_map[0]++;
			frac_map[0]++;
		} else {
			int exponent = (v.ui & 0x7F800000) >> 23;
			if(exponent) {
				exp_map[exponent - 127]++;
				frac_map[((v.ui & 0x007FFFFF) | 0x00800000)]++;
			} else {
				exp_map[-126]++;
				frac_map[v.ui & 0x007FFFFF]++;
			}
		}
#endif

#ifdef SAMPLE_VALUES
		val_map[(v.ui & 0x7FFFFFFF) >> 6]++;
#endif

#ifdef SAMPLE_FOR_ERROR
		val_vector.push_back(v.ui);
#endif
	}
}

void sample_blob(const double* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map, std::vector<uint32_t> val_vector, int sampling_frequency) {
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
