#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<fp16>*>& bottom, const vector<Blob<fp16>*>& top,
    const vector<Blob<Dtype>*>& bottom_dtype, const vector<Blob<Dtype>*>& top_dtype) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_, softmax_bottom_vec_dtype_, softmax_top_vec_dtype_);
  const fp16* prob_data = prob_.gpu_data();
  int prob_data_count = prob_.count();
  this->temp_top_->Reshape(prob_.shape());
  Dtype* prob_data_dtype = this->temp_top_->mutable_gpu_data();
  caffe_expand_blob(prob_data_count, prob_data_dtype, prob_data, prob_.data_bias);

  const Dtype* label = bottom_dtype[1]->gpu_data();
  const int dim = prob_data_count / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  fp16* loss_data = bottom[0]->mutable_gpu_diff();
  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* loss_data_dtype = this->temp_bottom_->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  fp16* counts = prob_.mutable_gpu_diff();
  Dtype* counts_dtype = this->temp_top_->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data_dtype, label, loss_data_dtype,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts_dtype);
#ifdef SAMPLE_FLOATS
    if(this->phase_ == TRAIN && this->sample_iter_) {
      sample_blob(prob_.gpu_data(), prob_.count(), this->activation_exp, this->activation_frac, this->activation, this->activation_vector, LOSS_SAMPLING_FREQ);
    }
#endif
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data_dtype, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts_dtype, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = fp32tofp16(loss / get_normalizer(normalization_,
                                                        valid_count));
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }

  // Clear scratch memory to prevent interfering with backward (see #6202).
  caffe_gpu_set_half(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const fp16* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    const fp16* prob_data = prob_.gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(fp16), prob_data, bottom_diff);
    int bottom_diff_count = prob_.count();
    this->temp_bottom_->Reshape(prob_.shape());
    Dtype* bottom_diff_dtype = this->temp_bottom_->mutable_gpu_diff();
    caffe_expand_blob(bottom_diff_count, bottom_diff_dtype, bottom_diff, prob_.data_bias);

    const fp16* top_data = top[0]->gpu_data();

    const Dtype* label = bottom_dtype[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    fp16* counts = prob_.mutable_gpu_diff();
    Dtype* counts_dtype = this->temp_bottom_->mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff_dtype,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts_dtype);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts_dtype, &valid_count);
    }
    const Dtype loss_weight = fp16tofp32(top[0]->cpu_diff()[0]) /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff_dtype);
    caffe_compress_blob_ag(prob_.count(), bottom_diff_dtype, bottom_diff, &(bottom[0]->diff_bias));
#ifdef SAMPLE_FLOATS
    if(this->phase_ == TRAIN && this->sample_iter_) {
      sample_blob(bottom[0]->gpu_diff(), bottom[0]->count(), this->activation_gradient_exp, this->activation_gradient_frac, this->activation_gradient, this->activation_gradient_vector, LOSS_SAMPLING_FREQ);
    }
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
