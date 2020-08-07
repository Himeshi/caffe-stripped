#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const fp16* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const fp16* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += (fp16tofp32_gpu(in_off[head * step]) * fp16tofp32_gpu(in_off[head * step]));
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += (fp16tofp32_gpu(in_off[head * step]) * fp16tofp32_gpu(in_off[head * step]));
      if (head - size >= 0) {
        accum_scale -= (fp16tofp32_gpu(in_off[(head - size) * step])
                       * fp16tofp32_gpu(in_off[(head - size) * step]));
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= (fp16tofp32_gpu(in_off[(head - size) * step])
                       * fp16tofp32_gpu(in_off[(head - size) * step]));
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}


template <typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_gpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top, bottom_dtype, top_dtype);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
#ifdef SAMPLE_FLOATS
  if(this->phase_ == TRAIN && this->sample_iter_) {
    sample_blob(top[0]->gpu_data(), top[0]->count(), this->activation_exp, this->activation_frac, this->activation, this->activation_vector, SAMPLING_FREQ);
  }
#endif
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const fp16* const in,
    const Dtype* const scale, const Dtype negative_beta, fp16* const out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = fp32tofp16_gpu(fp16tofp32_gpu(in[index]) * pow(scale[index], negative_beta));
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<fp16>*>& bottom, const vector<Blob<fp16>*>& top) {
  // First, compute scale
  const fp16* bottom_data = bottom[0]->gpu_data();
  fp16* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScale<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num_, channels_, height_, width_, size_,
      alpha_ / size_, k_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -beta_, top_data);
  CUDA_POST_KERNEL_CHECK;
}
template void LRNLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<fp16>*>& bottom, const vector<Blob<fp16>*>& top);
template void LRNLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<fp16>*>& bottom, const vector<Blob<fp16>*>& top);


template <typename Dtype>
void LRNLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
#ifdef SAMPLE_FLOATS
  if(this->phase_ == TRAIN && this->sample_iter_) {
    sample_blob(bottom[0]->gpu_diff(), bottom[0]->count(), this->activation_gradient_exp, this->activation_gradient_frac, this->activation_gradient, this->activation_gradient_vector, SAMPLING_FREQ);
  }
#endif
}

template <typename Dtype>
__global__ void LRNComputeDiff(const int nthreads,
    const fp16* const bottom_data, const fp16* const top_data,
    const Dtype* const scale, const fp16* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio, fp16* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const fp16* const bottom_off = bottom_data + offset;
    const fp16* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const fp16* const top_diff_off = top_diff + offset;
    fp16* const bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += (fp16tofp32_gpu(top_diff_off[head * step]) * fp16tofp32_gpu(top_off[head * step]) /
          scale_off[head * step]);
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += (fp16tofp32_gpu(top_diff_off[head * step]) * fp16tofp32_gpu(top_off[head * step]) /
          scale_off[head * step]);
      if (head - size >= 0) {
        accum_ratio -= (fp16tofp32_gpu(top_diff_off[(head - size) * step]) *
            fp16tofp32_gpu(top_off[(head - size) * step]) / scale_off[(head - size) * step]);
      }
      bottom_diff_off[(head - post_pad) * step] =
          fp32tofp16_gpu(fp16tofp32_gpu(top_diff_off[(head - post_pad) * step])
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * fp16tofp32_gpu(bottom_off[(head - post_pad) * step]) * accum_ratio);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= fp16tofp32_gpu(top_diff_off[(head - size) * step]) *
            fp16tofp32_gpu(top_off[(head - size) * step]) / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          fp32tofp16_gpu(fp16tofp32_gpu(top_diff_off[(head - post_pad) * step])
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * fp16tofp32_gpu(bottom_off[(head - post_pad) * step]) * accum_ratio);
      ++head;
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<fp16>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<fp16>*>& bottom) {
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
      scale_.gpu_data(), top[0]->gpu_diff(), num_, channels_, height_, width_,
      size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
      bottom[0]->mutable_gpu_diff());
}
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<fp16>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<fp16>*>& bottom);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<fp16>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<fp16>*>& bottom);



INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe
