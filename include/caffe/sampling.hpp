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
#include "caffe/util/device_alternate.hpp"

#define SAMPLE_FLOATS

#define SAMPLING_FREQ 50000

namespace caffe {

void sample_blob(const float* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, int sampling_frequency);

void sample_blob(const double* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, int sampling_frequency);

void sample_blob(const fp16* blob, int blob_count, std::map<int, int> &exp_map, std::map<int, int> &frac_map, int sampling_frequency);

void print_map(std::map<int, int> sample_map);
}

#endif /* INCLUDE_CAFFE_SAMPLING_HPP_ */
