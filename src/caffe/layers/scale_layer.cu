#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"

namespace caffe {

template <typename Dtype>
__global__ void ScaleForward(const int n, const fp16* in,
    const fp16* scale, const int scale_dim, const int inner_dim,
    fp16* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = fp32tofp16_gpu(fp16tofp32_gpu(in[index]) * fp16tofp32_gpu(scale[scale_index]));
  }
}

template <typename Dtype>
__global__ void ScaleBiasForward(const int n, const fp16* in,
    const fp16* scale, const fp16* bias,
    const int scale_dim, const int inner_dim, fp16* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = fp32tofp16_gpu(fp16tofp32_gpu(in[index]) * fp16tofp32_gpu(scale[scale_index]) + fp16tofp32_gpu(bias[scale_index]));
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
      const vector<Blob<Dtype>*>& top_dtype){
  const int count = top[0]->count();
  const fp16* bottom_data = bottom[0]->gpu_data();
  if (bottom[0] == top[0]) {
    // in-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we could skip this if not
    // doing Backward, but Caffe currently provides no way of knowing whether
    // we'll need to do Backward at the time of the Forward call.
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
               temp_.mutable_gpu_data());
  }
  const fp16* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  fp16* top_data = top[0]->mutable_gpu_data();
  if (bias_layer_) {
    const fp16* bias_data = this->blobs_[bias_param_id_]->gpu_data();
    ScaleBiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, bias_data, scale_dim_, inner_dim_,
        top_data);
  } else {
    ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, scale_dim_, inner_dim_, top_data);
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	  const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype){
  if (bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_, top_dtype, bottom_dtype);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob<fp16>* scale = scale_param ? this->blobs_[0].get() : bottom[1];
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) {
    const fp16* top_diff = top[0]->gpu_diff();
    const bool in_place = (bottom[0] == top[0]);
    const fp16* bottom_data = (in_place ? &temp_ : bottom[0])->gpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in temp_.
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    fp16* product = (is_eltwise ? scale->mutable_gpu_diff() :
        (in_place ? temp_.mutable_gpu_data() : bottom[0]->mutable_gpu_diff()));
    float product_bias = (is_eltwise ? scale->diff_bias :
            (in_place ? temp_.data_bias : bottom[0]->diff_bias));
    caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      fp16* sum_result = NULL;
      float sum_result_bias = 1.0;
      if (inner_dim_ == 1) {
        sum_result = product;
        sum_result_bias =  product_bias;
      } else if (sum_result_.count() == 1) {
        const fp16* sum_mult = sum_multiplier_.gpu_data();
        fp16* scale_diff = scale->mutable_cpu_diff();
        if (scale_param) {
          Dtype result;
          caffe_gpu_dot_half(inner_dim_, product, sum_mult, &result, product_bias, sum_multiplier_.data_bias);
          *scale_diff += result;
        } else {
          caffe_gpu_dot(inner_dim_, product, sum_mult, scale_diff);
        }
      } else {
        const fp16* sum_mult = sum_multiplier_.gpu_data();
        sum_result = (outer_dim_ == 1) ?
            scale->mutable_gpu_diff() : sum_result_.mutable_gpu_data();
        sum_result_bias = (outer_dim_ == 1) ?
                scale->diff_bias : sum_result_.data_bias;
        caffe_gpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       fp32tofp16(1), product, sum_mult, fp32tofp16(0), sum_result);
      }
      if (outer_dim_ != 1) {
        const fp16* sum_mult = sum_multiplier_.gpu_data();
        if (scale_dim_ == 1) {
          fp16* scale_diff = scale->mutable_cpu_diff();
          if (scale_param) {
            Dtype result;
            caffe_gpu_dot_half(outer_dim_, sum_mult, sum_result, &result, sum_multiplier_.data_bias, sum_result_bias);
            *scale_diff += result;
          } else {
            caffe_gpu_dot(outer_dim_, sum_mult, sum_result, scale_diff);
          }
        } else {
          fp16* scale_diff = scale->mutable_gpu_diff();
          caffe_gpu_gemv(CblasTrans, outer_dim_, scale_dim_,
                         fp32tofp16(1), sum_result, sum_mult, fp32tofp16(scale_param),
                         scale_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const fp16* top_diff = top[0]->gpu_diff();
    const fp16* scale_data = scale->gpu_data();
    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, scale_data, scale_dim_, inner_dim_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);

}  // namespace caffe
