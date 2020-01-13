/*
 * sampling.cpp
 *
 *  Created on: Jan 13, 2020
 *      Author: himeshi
 */
#include "caffe/sampling.hpp"

namespace caffe {

void sample_blob(const fp16* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, int sampling_frequency) {
	fp16 temp;
	for (int i = 0; i < blob_count; i+= sampling_frequency) {
		cudaMemcpy(&temp, &blob[i], sizeof(fp16), cudaMemcpyDeviceToHost);
		if(temp == 0) {
			exp_map[0]++;
			frac_map[0]++;
		} else {
			int exponent = (temp & 0x7E00) >> 9;
			exp_map[exponent - 31]++;
			frac_map[((temp & 0x01FF) | 0x0200)]++;
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

