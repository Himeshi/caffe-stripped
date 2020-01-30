/*
 * sampling.cpp
 *
 *  Created on: Jan 13, 2020
 *      Author: himeshi
 */
#include "caffe/sampling.hpp"

namespace caffe {

void sample_blob(const fp16* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map, int sampling_frequency) {
	fp16 temp;
	for (int i = 0; i < blob_count; i+= sampling_frequency) {
		cudaMemcpy(&temp, &blob[i], sizeof(fp16), cudaMemcpyDeviceToHost);
#ifdef SAMPLE_EXP
		if(temp == 0) {
			exp_map[0]++;
			frac_map[0]++;
		} else {
			int exponent = (temp & 0x7C00) >> 10;
			if(exponent) {
				exp_map[exponent - 15]++;
				frac_map[((temp & 0x03FF) | 0x0500)]++;
			} else {
				exp_map[-14]++;
				frac_map[temp & 0x03FF]++;
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
	}
}

void sample_blob(const double* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map, int sampling_frequency) {
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

