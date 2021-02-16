#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  int* mask = NULL;
  const int count = top[0]->count();
  fp16* top_data = top[0]->mutable_gpu_data();
  this->temp_top_->Reshape(top[0]->shape());
  Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();
  caffe_expand_blob_activations(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);

  Dtype* temp_bottom_data_0;
  Dtype* temp_bottom_data_1;

  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
	bottom_dtype[0]->Reshape(bottom[0]->shape());
	temp_bottom_data_0 = bottom_dtype[0]->mutable_gpu_data();
	caffe_expand_blob_activations(bottom[0]->count(), temp_bottom_data_0, bottom[0]->gpu_data(), bottom[0]->data_bias);

	this->temp_bottom_->Reshape(bottom[1]->shape());
	temp_bottom_data_1 = this->temp_bottom_->mutable_gpu_data();
	caffe_expand_blob_activations(bottom[1]->count(), temp_bottom_data_1, bottom[1]->gpu_data(), bottom[1]->data_bias);

	caffe_gpu_mul(count, temp_bottom_data_0, temp_bottom_data_1,
	  temp_top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      this->temp_bottom_->Reshape(bottom[i]->shape());
      Dtype* temp_bottom_data = this->temp_bottom_->mutable_gpu_data();
      caffe_expand_blob_activations(bottom[i]->count(), temp_bottom_data, bottom[i]->gpu_data(), bottom[i]->data_bias);
      caffe_gpu_mul(count, temp_top_data, temp_bottom_data, temp_top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set_half(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      this->temp_bottom_->Reshape(bottom[i]->shape());
      Dtype* temp_bottom_data = this->temp_bottom_->mutable_gpu_data();
      caffe_expand_blob_activations(bottom[i]->count(), temp_bottom_data, bottom[i]->gpu_data(), bottom[i]->data_bias);
      caffe_gpu_axpy(count, coeffs_[i], temp_bottom_data, temp_top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    bottom_dtype[0]->Reshape(bottom[0]->shape());
    temp_bottom_data_0 = bottom_dtype[0]->mutable_gpu_data();
    caffe_expand_blob_activations(bottom[0]->count(), temp_bottom_data_0, bottom[0]->gpu_data(), bottom[0]->data_bias);

    this->temp_bottom_->Reshape(bottom[1]->shape());
    temp_bottom_data_1 = this->temp_bottom_->mutable_gpu_data();
    caffe_expand_blob_activations(bottom[1]->count(), temp_bottom_data_1, bottom[1]->gpu_data(), bottom[1]->data_bias);

    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, temp_bottom_data_0, temp_bottom_data_1, 0, temp_top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      this->temp_bottom_->Reshape(bottom[i]->shape());
      Dtype* temp_bottom_data = this->temp_bottom_->mutable_gpu_data();
      caffe_expand_blob_activations(bottom[i]->count(), temp_bottom_data, bottom[i]->gpu_data(), bottom[i]->data_bias);
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, temp_top_data, temp_bottom_data, i-1, temp_top_data, mask);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
  caffe_compress_blob_activations(count, temp_top_data, top_data, &(top[0]->data_bias));
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  const int* mask = NULL;
  const int count = top[0]->count();
  this->temp_top_->Reshape(top[0]->shape());

  const fp16* top_data = top[0]->gpu_data();
  Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();
  caffe_expand_blob_activations(count, temp_top_data, top_data, top[0]->data_bias);

  const fp16* top_diff = top[0]->gpu_diff();
  Dtype* temp_top_diff = this->temp_top_->mutable_gpu_diff();
  caffe_expand_blob_ag(count, temp_top_diff, top_diff, top[0]->diff_bias);

  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      bottom_dtype[0]->Reshape(bottom[i]->shape());
      const fp16* bottom_data = bottom[i]->gpu_data();
      Dtype* temp_bottom_data_i = bottom_dtype[0]->mutable_gpu_data();
      caffe_expand_blob_activations(bottom[i]->count(), temp_bottom_data_i, bottom_data, bottom[i]->data_bias);

      fp16* bottom_diff = bottom[i]->mutable_gpu_diff();
      Dtype* temp_bottom_diff_i = bottom_dtype[0]->mutable_gpu_data();
      caffe_expand_blob_ag(bottom[i]->count(), temp_bottom_diff_i, bottom_diff, bottom[i]->diff_bias);

      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              this->temp_bottom_->Reshape(bottom[j]->shape());
              Dtype* temp_bottom_data_j = this->temp_bottom_->mutable_gpu_data();
              caffe_expand_blob_activations(bottom[j]->count(), temp_bottom_data_j, bottom[j]->gpu_data(), bottom[j]->data_bias);
              caffe_copy(count, temp_bottom_data_j, temp_bottom_diff_i);
              initialized = true;
            } else {
              this->temp_bottom_->Reshape(bottom[j]->shape());
              Dtype* temp_bottom_data_j = this->temp_bottom_->mutable_gpu_data();
              caffe_expand_blob_activations(bottom[j]->count(), temp_bottom_data_j, bottom[j]->gpu_data(), bottom[j]->data_bias);
			  caffe_gpu_mul(count, temp_bottom_data_j, temp_bottom_diff_i,
					  temp_bottom_diff_i);
            }
          }
        } else {
          caffe_gpu_div(count, temp_top_data, temp_bottom_data_i, temp_bottom_diff_i);
        }
        caffe_gpu_mul(count, temp_bottom_diff_i, temp_top_diff, temp_bottom_diff_i);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, temp_top_diff, temp_bottom_diff_i);
        } else {
          caffe_gpu_scale(count, coeffs_[i], temp_top_diff, temp_bottom_diff_i);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.gpu_data();
        caffe_expand_blob_ag(bottom[i]->count(), temp_bottom_diff_i, bottom_diff, bottom[i]->diff_bias);
        MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, temp_top_diff, i, mask, temp_bottom_diff_i);
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
      caffe_compress_blob_ag(bottom[i]->count(), temp_bottom_diff_i, bottom_diff, &(bottom[i]->diff_bias));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
