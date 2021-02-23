#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  const fp16* bottom_data = bottom[0]->gpu_data();
  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* temp_bottom_data = this->temp_bottom_->mutable_gpu_data();
  const int count = bottom[0]->count();
  caffe_expand_blob_activations(count, temp_bottom_data, bottom_data, bottom[0]->data_bias);

  fp16* top_data = top[0]->mutable_gpu_data();
  this->temp_top_->Reshape(top[0]->shape());
  Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();

  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, temp_bottom_data, mask, uint_thres_, scale_, temp_top_data);
    caffe_compress_blob_activations(count, temp_top_data, top_data, &(top[0]->data_bias));
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
    top[0]->data_bias = bottom[0]->data_bias;
  }
#ifdef SAMPLE_FLOATS
    if(this->phase_ == TRAIN && this->sample_iter_) {
      sample_blob(top[0]->gpu_data(), top[0]->count(), this->activation_exp, this->activation_frac, this->activation, this->activation_vector, SAMPLING_FREQ);
    }
#endif
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (propagate_down[0]) {
    const fp16* top_diff = top[0]->gpu_diff();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* temp_top_diff = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(top[0]->count(), temp_top_diff, top_diff, top[0]->diff_bias);

    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    this->temp_bottom_->Reshape(bottom[0]->shape());
    Dtype* temp_bottom_diff = this->temp_bottom_->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, temp_top_diff, mask, uint_thres_, scale_, temp_bottom_diff);
      caffe_compress_blob_ag(count, temp_bottom_diff, bottom_diff, &(bottom[0]->diff_bias));
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
      bottom[0]->diff_bias = top[0]->diff_bias;
    }
#ifdef SAMPLE_FLOATS
     if (this->phase_ == TRAIN && this->sample_iter_) {
       sample_blob(bottom[0]->gpu_diff(), bottom[0]->count(), this->activation_gradient_exp, this->activation_gradient_frac, this->activation_gradient, this->activation_gradient_vector, SAMPLING_FREQ);
     }
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
