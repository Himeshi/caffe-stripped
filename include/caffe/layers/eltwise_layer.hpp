#ifndef CAFFE_ELTWISE_LAYER_HPP_
#define CAFFE_ELTWISE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseLayer : public Layer<Dtype> {
 public:
  explicit EltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top);
  virtual void Reshape(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top);

  virtual inline const char* type() const { return "Eltwise"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top);
  virtual void Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& top_dtype);
  virtual void Backward_cpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom);

  EltwiseParameter_EltwiseOp op_;
  vector<Dtype> coeffs_;
  Blob<int> max_idx_;

  bool stable_prod_grad_;
};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_
