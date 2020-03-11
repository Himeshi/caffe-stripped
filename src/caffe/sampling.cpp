/*
 * sampling.cpp
 *
 *  Created on: Jan 13, 2020
 *      Author: himeshi
 */

#include "caffe/sampling.hpp"

namespace caffe {

void sample_blob(const fp16* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map, std::vector<float> &val_vector, int sampling_frequency) {
	fp16 temp;
	for (int i = 0; i < blob_count; i += i * 2) {
		cudaMemcpy(&temp, &blob[i], sizeof(fp16), cudaMemcpyDeviceToHost);
#ifdef SAMPLE_EXP
		if(temp == 0) {
			exp_map[0]++;
			frac_map[0]++;
		} else {
			int exponent = (temp & 0x7E00) >> 9;
			if(exponent) {
				exp_map[exponent - 31]++;
				frac_map[((temp & 0x01FF) | 0x0200)]++;
			} else {
				exp_map[-30]++;
				frac_map[temp & 0x01FF]++;
			}
		}
#endif

#ifdef SAMPLE_VALUES
		temp = temp & 0x7FFF;
		if(temp != 0)
		  val_map[temp >> 2]++;
		else
		  val_map[0]++;
#endif

#ifdef SAMPLE_FOR_ERROR
		val_vector.push_back(fp16tofp32(temp));
#endif
	}
}

void sample_blob(const double* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map, std::vector<float> &val_vector, int sampling_frequency) {
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
