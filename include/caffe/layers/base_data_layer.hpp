#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};

template <typename Dtype>
class Batch {
 public:
  Blob<fp16> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top);

  virtual void Forward_cpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top);
  virtual void Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& top_dtype);

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<fp16>* batch) = 0;

  vector<shared_ptr<Batch<fp16> > > prefetch_;
  BlockingQueue<Batch<fp16>*> prefetch_free_;
  BlockingQueue<Batch<fp16>*> prefetch_full_;
  Batch<fp16>* prefetch_current_;

  Blob<fp16> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
