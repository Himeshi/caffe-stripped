#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
      const vector<Blob<Dtype>*>& top_dtype) {
  const fp16* bottom_data = bottom[0]->gpu_data();
  fp16* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
    top[0]->data_bias = bottom[0]->data_bias;
  }

  this->temp_top_->Reshape(top[0]->shape());
  Dtype* top_data_dtype = this->temp_top_->mutable_gpu_data();
  caffe_expand_blob_activations(top[0]->count(), top_data_dtype, top_data, top[0]->data_bias);

  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* bottom_data_dtype = this->temp_bottom_->mutable_gpu_data();
  caffe_expand_blob_activations(bottom[0]->count(), bottom_data_dtype, bottom_data, bottom[0]->data_bias);

  const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  caffe_expand_blob_w(weight_count, weight_temp, weight, this->blobs_[0]->data_bias);

  const fp16* bias = this->blobs_[1]->gpu_data();
  Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_data();
  int bias_count = this->blobs_[1]->count();
  caffe_expand_blob_w(bias_count, bias_temp, bias, this->blobs_[1]->data_bias);

  this->blobs_dtype_[2]->mutable_cpu_data()[0] = fp16tofp32_W(this->blobs_[2]->cpu_data()[0]);

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_dtype_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_dtype_[2]->cpu_data()[0];
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_dtype_[0]->gpu_data(), mean_.mutable_gpu_data());
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_dtype_[1]->gpu_data(), variance_.mutable_gpu_data());
  } else {
    // compute mean
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data_dtype,
        spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        mean_.mutable_gpu_data());
  }

  // subtract mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., top_data_dtype);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_mul(top[0]->count(), top_data_dtype, top_data_dtype,
        temp_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), temp_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, Dtype(1.),
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), Dtype(0.),
        variance_.mutable_gpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_dtype_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_dtype_[2]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
        moving_average_fraction_, this->blobs_dtype_[0]->mutable_gpu_data());
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_gpu_axpby(variance_.count(), bias_correction_factor,
        variance_.gpu_data(), moving_average_fraction_,
        this->blobs_dtype_[1]->mutable_gpu_data());
  }

  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(),
      variance_.mutable_gpu_data());

  // replicate variance to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), top_data_dtype, temp_.gpu_data(), top_data_dtype);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data_dtype,
      x_norm_.mutable_gpu_data());

  caffe_compress_blob_activations(top[0]->count(), top_data_dtype, top_data, &(top[0]->data_bias));
  this->blobs_[2]->mutable_cpu_data()[0] = fp32tofp16_W(this->blobs_dtype_[2]->cpu_data()[0]);
  caffe_compress_blob_w(this->blobs_[0]->count(), this->blobs_dtype_[0]->mutable_gpu_data(), this->blobs_[0]->mutable_gpu_data(), &((this->blobs_[0])->data_bias));
  caffe_compress_blob_w(this->blobs_[1]->count(), this->blobs_dtype_[1]->mutable_gpu_data(), this->blobs_[1]->mutable_gpu_data(), &((this->blobs_[1])->data_bias));
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	  const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  const Dtype* top_diff;
  this->temp_top_->Reshape(top[0]->shape());
  if (bottom[0] != top[0]) {
    Dtype* top_diff_temp = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(top[0]->count(), top_diff_temp, top[0]->gpu_diff(), top[0]->diff_bias);
    top_diff = this->temp_top_->gpu_diff();
  } else {
    caffe_copy(x_norm_.count(), this->temp_top_->gpu_diff(), x_norm_.mutable_gpu_diff());
    top_diff = x_norm_.gpu_diff();
  }

  fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* bottom_diff_temp = this->temp_bottom_->mutable_gpu_diff();
  caffe_expand_blob_ag(bottom[0]->count(), bottom_diff_temp, bottom_diff, bottom[0]->diff_bias);

  if (use_global_stats_) {
    caffe_gpu_div(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff_temp);
    return;
  }
  const Dtype* top_data = x_norm_.gpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff_temp);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
		  bottom_diff_temp, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());

  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., bottom_diff_temp);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(temp_.count(), top_data, bottom_diff_temp, bottom_diff_temp);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., bottom_diff_temp);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff_temp);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_gpu_div(temp_.count(), bottom_diff_temp, temp_.gpu_data(), bottom_diff_temp);
  caffe_compress_blob_ag(bottom[0]->count(), bottom_diff_temp, bottom_diff, &(bottom[0]->diff_bias));
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);


}  // namespace caffe
