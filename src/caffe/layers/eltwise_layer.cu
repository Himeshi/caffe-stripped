#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const fp16* bottom_data_a,
    const fp16* bottom_data_b, const int blob_idx, fp16* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    float temp_bottom_data_a = fp16tofp32_gpu(bottom_data_a[index]);
    float temp_bottom_data_b = fp16tofp32_gpu(bottom_data_b[index]);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (temp_bottom_data_a > temp_bottom_data_b) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = temp_bottom_data_a;
        top_data[index] = fp32tofp16_gpu(maxval);
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = temp_bottom_data_b;
      top_data[index] = fp32tofp16_gpu(maxval);
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& top_dtype) {
  int* mask = NULL;
  const int count = top[0]->count();
  fp16* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set_half(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy_half(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const fp16* top_diff,
    const int blob_idx, const int* mask, fp16* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += fp16tofp32_gpu(top_diff[index]);
    }
    bottom_diff[index] = fp32tofp16_gpu(gradient);
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const fp16* top_data = top[0]->gpu_data();
  const fp16* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const fp16* bottom_data = bottom[i]->gpu_data();
      fp16* bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                            bottom_diff);
            }
          }
        } else {
          caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_gpu_scale_half(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.gpu_data();
        MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, i, mask, bottom_diff);
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
