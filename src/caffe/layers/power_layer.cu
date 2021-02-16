#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PowerLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  fp16* top_data = top[0]->mutable_gpu_data();
  this->temp_top_->Reshape(top[0]->shape());
  Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();

  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_gpu_set(count, value, temp_top_data);
    caffe_compress_blob_activations(top[0]->count(), temp_top_data, top_data, &(top[0]->data_bias));
    return;
  }

  const fp16* bottom_data = bottom[0]->gpu_data();
  caffe_copy(count, bottom_data, top_data);
  top[0]->data_bias = bottom[0]->data_bias;
  caffe_expand_blob_activations(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);

  if (scale_ != Dtype(1)) {
    caffe_gpu_scal(count, scale_, temp_top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_gpu_add_scalar(count, shift_, temp_top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_gpu_powx(count, temp_top_data, power_, temp_top_data);
  }
  caffe_compress_blob_activations(top[0]->count(), temp_top_data, top_data, &(top[0]->data_bias));
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (propagate_down[0]) {
    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    this->temp_bottom_->Reshape(bottom[0]->shape());
    Dtype* temp_bottom_diff = this->temp_bottom_->mutable_gpu_diff();
    caffe_expand_blob_ag(count, temp_bottom_diff, bottom_diff, bottom[0]->diff_bias);

    const fp16* top_diff = top[0]->gpu_diff();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* temp_top_diff = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(top[0]->count(), temp_top_diff, top_diff, top[0]->diff_bias);

    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_gpu_set(count, diff_scale_, temp_bottom_diff);
    } else {
      const fp16* bottom_data = bottom[0]->gpu_data();
      Dtype* temp_bottom_data = this->temp_bottom_->mutable_gpu_data();
      caffe_expand_blob_activations(count, temp_bottom_data, bottom_data, bottom[0]->data_bias);
      // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
      //               = diff_scale * y / (shift + scale * x)
      if (power_ == Dtype(2)) {
        // Special case for y = (shift + scale * x)^2
        //     -> dy/dx = 2 * scale * (shift + scale * x)
        //              = diff_scale * shift + diff_scale * scale * x
        caffe_gpu_axpby(count, diff_scale_ * scale_, temp_bottom_data,
            Dtype(0), temp_bottom_diff);
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(count, diff_scale_ * shift_, temp_bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for y = (scale * x)^power
        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
        //              = power * y / x
        const fp16* top_data = top[0]->gpu_data();
        Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();
        caffe_expand_blob_activations(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);

        caffe_gpu_div(count, temp_top_data, temp_bottom_data, temp_bottom_diff);
        caffe_gpu_scal(count, power_, temp_bottom_diff);
      } else {
        caffe_copy(count, bottom_data, bottom_diff);
        bottom[0]->diff_bias = bottom[0]->data_bias;
        caffe_expand_blob_activations(count, temp_bottom_diff, bottom_diff, bottom[0]->diff_bias);
        if (scale_ != Dtype(1)) {
          caffe_gpu_scal(count, scale_, temp_bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(count, shift_, temp_bottom_diff);
        }
        const fp16* top_data = top[0]->gpu_data();
        Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();
        caffe_expand_blob_activations(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);
        caffe_gpu_div(count, temp_top_data, temp_bottom_diff, temp_bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_gpu_scal(count, diff_scale_, temp_bottom_diff);
        }
      }
    }
    caffe_gpu_mul(count, temp_top_diff, temp_bottom_diff, temp_bottom_diff);
    caffe_compress_blob_ag(bottom[0]->count(), temp_bottom_diff, bottom_diff, &(bottom[0]->diff_bias));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PowerLayer);


}  // namespace caffe
