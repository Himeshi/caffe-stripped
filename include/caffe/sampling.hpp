/*
 * sampling.hpp
 *
 *  Created on: Jan 13, 2020
 *      Author: himeshi
 */

#ifndef INCLUDE_CAFFE_SAMPLING_HPP_
#define INCLUDE_CAFFE_SAMPLING_HPP_

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <map>
#include <vector>
#include "caffe/util/device_alternate.hpp"
#include "caffe/fp16.hpp"

#define SAMPLE_FLOATS

//#define SAMPLE_VALUES

//#define SAMPLE_EXP

#define SAMPLE_FOR_ERROR

#define SAMPLING_FREQ 100

#define WEIGHT_SAMPLING_FREQ 10

#define BIAS_SAMPLING_FREQ 1

#define LOSS_SAMPLING_FREQ 1

namespace caffe {

void sample_blob(const fp16* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map,  std::vector<uint32_t> &val_vector, int sampling_frequency);

void sample_blob(const double* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map,  std::vector<uint32_t> &val_vector, int sampling_frequency);

void sample_blob(const fp16* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, std::map<int, int> &val_map,  std::vector<uint32_t> &val_vector, int sampling_frequency);

void print_map(std::map<int, int> sample_map);
}

#endif /* INCLUDE_CAFFE_SAMPLING_HPP_ */
