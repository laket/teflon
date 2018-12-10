/*
Copyright (c) 2018 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <iostream>
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "persistent_tensor_ex.h"

using namespace tensorflow;
using namespace std;

typedef std::vector<PersistentTensorEx> PersistentTuple;


/*
 リプレイバッファー
 リングバッファを使っているため、読み込み速度が遅いと新しい値で上書きされる

 encode(idx, &tuple): idx番目のデータをtuple(引数)に追加する
 enqueue(components): データを追加する。componentsはひと揃いのデータ
 encode_history(QueueInterface::Tuple* history): 一番古いデータから順にすべてhistoryに足す
 wait_for_samples(num_samples) : num_samplesたまるまで待つ
*/

class ExperienceBuffer : public ResourceBase
{
  public:
    ExperienceBuffer(int64 capacity, DataTypeVector component_types, std::vector<TensorShape> component_shapes, PersistentTensor* min_seg_tree, PersistentTensor* sum_seg_tree)
        : capacity_(capacity), component_dtypes_(component_types), next_idx_(0), component_shapes_(component_shapes), closed_(false)
    {
        buffer_.resize(component_types.size());
        for(auto& c : buffer_)
            c.resize(capacity);
        min_seg_tree_ = min_seg_tree;
        sum_seg_tree_ = sum_seg_tree;
    }

    std::string DebugString() override {
        return "ExperienceBuffer";
    }
    virtual int num_components() const
    {
        return component_dtypes_.size();
    }
    virtual const DataTypeVector& component_dtypes() const
    {
        return component_dtypes_;
    }

    /*
      渡されたTupleに指定したidxの要素を追加する
       buffer_[idx_component][idx_batch]
      bufferの添字が直感と順番が逆な点に注意
      std::vector<std::vector<PersistentTensorEx>>
    */
    virtual void encode(OpKernelContext* ctx, int64 idx, QueueInterface::Tuple* tuple) {
        tf_shared_lock l(mu_);

        for (int i = 0; i < num_components(); ++i)
        {
            tuple->push_back(*buffer_[i][idx].AccessTensor(ctx));
        }
    }

    virtual tensorflow::Status encode(OpKernelContext* ctx, int64 idx, PersistentTuple* tuple) {
        tf_shared_lock l(mu_);
        if(idx >= size())
            return Status(error::Code::OUT_OF_RANGE, "ExperienceBuffer encoding OOR!");

        for (int i = 0; i < num_components(); ++i)
        {
            tuple->push_back(buffer_[i][idx]);
        }
        return Status::OK();
    }

    virtual void encode_history(OpKernelContext* ctx, QueueInterface::Tuple* history) {
        tf_shared_lock l(mu_);
        for (auto h = 0; h < size(); ++h)
        {
            //size = min(next_idx_, capacity_)
            // 端のidxまでいったら先頭に戻す
            if(size() == capacity())
            {
                const auto idx = (next_idx_ + h) % capacity_;
                encode(ctx, idx, history);
            }
            else
                encode(ctx, h, history);
        }
    }

    /*
      リングバッファにあらたにcomponentsを追加する
    */
    virtual int64 enqueue(const QueueInterface::Tuple& components)
    {
        mutex_lock l(mu_);
        int64 idx = next_idx_++ % capacity_;
        int c = 0;
        for (const auto &Tcomponent : components)
        {
            buffer_[c++][idx] = PersistentTensorEx(Tcomponent);
        }
        cv_.notify_all();
        return idx;
    }
    virtual int64 enqueue(const std::vector<PersistentTensorEx>& components)
    {
        mutex_lock l(mu_);
        int64 idx = next_idx_++ % capacity_;
        int c = 0;
        for (const auto &Tcomponent : components)
        {
            buffer_[c++][idx] = PersistentTensorEx(Tcomponent);
        }
        cv_.notify_all();
        return idx;
    }
    Status wait_for_samples(CancellationManager* cm, int64 num_samples)
    {
        mutex_lock l(mu_);
        while(size() < num_samples)
        {
            if(closed_ || cm->IsCancelled())
                return Status(error::Code::CANCELLED, "ExperienceBuffer Closed!");
            cv_.wait(l);
        }
        return Status::OK();
    }

    TensorShape ManyOutShape(int i, int64 batch_size) {
        TensorShape shape({batch_size});
        shape.AppendShape(component_shapes_[i]);
        return shape;
    }

    // global_index is used to avoid updating a sample that has been discarded
    virtual int64 global_index(int64 idx) const
    {
        idx += next_idx_ - (next_idx_ % capacity_);
        if(idx > next_idx_)
            return idx - capacity_;
        return idx;
    }
    virtual int64 oldest_index() const
    {
        return max(next_idx_ - capacity_, (int64)0);
    }
    virtual int64 size() const
    {
        return min(next_idx_, capacity_);
    }
    virtual int64 capacity() const
    {
        return capacity_;
    }
    virtual void close()
    {
        closed_ = true;
        cv_.notify_all();
    }

    // TODO: Fix this mess
    PersistentTensor* min_seg_tree_;
    PersistentTensor* sum_seg_tree_;
    tensorflow::mutex seg_tree_mu_;

  protected:
    tensorflow::mutex mu_;
    tensorflow::condition_variable cv_;
    int64 next_idx_;
    const int64 capacity_;
    const DataTypeVector component_dtypes_;
    const std::vector<TensorShape> component_shapes_;
    std::vector<std::vector<PersistentTensorEx>> buffer_;
    bool closed_;
};

class ExperienceBufferOp : public ResourceOpKernel<ExperienceBuffer> {
  public:
    explicit ExperienceBufferOp(OpKernelConstruction *ctx) : ResourceOpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("capacity", &capacity_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("component_types", &component_types_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &component_shapes_));

        Tensor *temp_min = nullptr;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_persistent(DT_FLOAT, TensorShape({capacity_ * 2}),
                                                &min_seg_tree_, &temp_min));

        Tensor *temp_sum = nullptr;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_persistent(DT_FLOAT, TensorShape({capacity_ * 2}),
                                                &sum_seg_tree_, &temp_sum));
        auto min_flat = temp_min->flat<float>();
        auto sum_flat = temp_sum->flat<float>();
        for (int64 i = 0; i < capacity_ * 2; ++i)
        {
            min_flat(i) = std::numeric_limits<float>::infinity();
            sum_flat(i) = 0;
        }
    }

  private:
    virtual Status CreateResource(ExperienceBuffer **ret)
    {
        *ret = new ExperienceBuffer(capacity_, component_types_, component_shapes_, &min_seg_tree_, &sum_seg_tree_);
        if (*ret == nullptr) {
            return errors::ResourceExhausted("Failed to allocate ExperienceBuffer");
        }
        return Status::OK();
    }
    int64 capacity_;
    DataTypeVector component_types_;
    std::vector<TensorShape> component_shapes_;
    PersistentTensor min_seg_tree_;
    PersistentTensor sum_seg_tree_;
};

REGISTER_OP("ExperienceBuffer")
    .Attr("capacity: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("shapes: list(shape) >= 1")
    .Attr("component_types: list(type) >= 1")
    .Output("handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("ExperienceBuffer").Device(DEVICE_CPU), ExperienceBufferOp);

class ExperienceBufferAccessOp : public OpKernel
{
  public:
    explicit ExperienceBufferAccessOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    }
    void Compute(OpKernelContext* ctx) override {
        ExperienceBuffer *buffer;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &buffer));
        core::ScopedUnref s(buffer);
        Compute(ctx, buffer);
    }
    virtual void Compute(OpKernelContext *ctx, ExperienceBuffer *buffer) = 0;
};

class ExperienceBufferEnqueueManyOp : public ExperienceBufferAccessOp {
    public:
    explicit ExperienceBufferEnqueueManyOp(OpKernelConstruction* ctx) : ExperienceBufferAccessOp(ctx) {
    }
    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer) override {
        DataTypeVector expected_inputs;
        if (ctx->input_dtype(0) == DT_RESOURCE) {
            expected_inputs.push_back(DT_RESOURCE);
        } else {
            expected_inputs.push_back(DT_STRING_REF);
        }
        for (DataType dt : buffer->component_dtypes()) {
            expected_inputs.push_back(dt);
        }
        OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {DT_INT64}));
        OpInputList components;
        OP_REQUIRES_OK(ctx, ctx->input_list("components", &components));
        const auto num_elements = components[0].dim_size(0);

        Tensor* indices_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                                            TensorShape({num_elements}),
                                            &indices_tensor));
        auto indices_tensor_flat = indices_tensor->flat<int64>();

        QueueInterface::Tuple tuple;
        for (const Tensor& Tcomponent : components)
            tuple.emplace_back(Tcomponent);

        for (auto index = 0; index < num_elements; ++index)
        {
            std::vector<PersistentTensorEx> element_tuple;
            for (auto i = 0; i < buffer->num_components(); ++i)
            {
                PersistentTensorEx element;
                OP_REQUIRES_OK(ctx, GetElementComponentFromBatch(tuple, index, i, ctx, &element));
                element_tuple.push_back(element);
            }
            indices_tensor_flat(index) = buffer->enqueue(element_tuple);
        }
    }
    static Status GetElementComponentFromBatch(const QueueInterface::Tuple& tuple,
                                                int64 index, int component,
                                                OpKernelContext* ctx,
                                                PersistentTensorEx* out_tensor) {
        *out_tensor = PersistentTensorEx(tuple[component], index);
        return Status::OK();
    }
};

REGISTER_OP("ExperienceBufferEnqueueMany")
    .Input("handle: resource")
    .Input("components: Tcomponents")
    .Output("indices: int64")
    .Attr("Tcomponents: list(type) >= 1")
    .SetShapeFn(::tensorflow::shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("ExperienceBufferEnqueueMany").Device(DEVICE_CPU), ExperienceBufferEnqueueManyOp);

template<typename T>
class ExperienceBufferWithPriorityAccessOp : public ExperienceBufferAccessOp {
    public:
    explicit ExperienceBufferWithPriorityAccessOp(OpKernelConstruction* ctx) : ExperienceBufferAccessOp(ctx) {
    }

    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer) override {
        Tensor* Tsum_tree = buffer->sum_seg_tree_->AccessTensor(ctx); // true because it will be acquired later

        Tensor* Tmin_tree = buffer->min_seg_tree_->AccessTensor(ctx); // true because it will be acquired later

        const int64 capacity = Tsum_tree->dim_size(0) / 2;
        OP_REQUIRES(ctx, Tsum_tree->dim_size(0) == Tmin_tree->dim_size(0), errors::Internal("Priority trees have different capacity."));
        OP_REQUIRES(ctx, capacity == buffer->capacity(), errors::Internal("ExperienceBufferWithPriorityAccessOp priority trees have different capacity from replay buffer."));

        Compute(ctx, buffer, Tsum_tree, Tmin_tree);
    }

    virtual void Compute(OpKernelContext *ctx, ExperienceBuffer *buffer, Tensor* Tsum_tree, Tensor* Tmin_tree) = 0;
};


/*
ReplayBufferから優先度に比例した重み付けでサンプリングを行う。
prefixsumは[0,1]の範囲で与えられる。
内部では、
1. prefixsumと合計重みの積が計算される。
2. Bufferの先頭から自分までの重みの合計が上記の積と一致するサンプルを見つける
をバッチサイズだけ繰り返す。これにより重みに比例した確率でサンプリングされる。

サンプリングされた点は取り除かれない。
取り除くかどうかは完全に新規性により評価される。

呼び出し例
prefixsum = tf.random_uniform((size,), dtype=tf.float32)
idxes, weights, total, p_min, _num_elements, components = custom_modules.replay_buffer_priority_sample(
              handle=self._buffer,
              prefixsum=prefixsum,
              Tcomponents=self.dtypes,
              minimum_sample_size=minimum_sample_size)

REGISTER_OP("ReplayBufferPrioritySample")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("minimum_sample_size: int = 1")
    .Attr("T: type")
    .Input("handle: resource")
    .Input("prefixsum: T")
    .Output("indices: int64")
    .Output("weights: T")          : sum由来。ここのindicesの重み
    .Output("sum: T")              : total priority
    .Output("min: T")              : min priority
    .Output("size: int64")         :
    .Output("components: Tcomponents") : 実際の値
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);
*/
template<typename T>
class ReplayBufferPrioritySampleOp : public ExperienceBufferWithPriorityAccessOp<T> {
    public:
    explicit ReplayBufferPrioritySampleOp(OpKernelConstruction* ctx) : ExperienceBufferWithPriorityAccessOp<T>(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("minimum_sample_size", &minimum_sample_size));
    }

    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer, Tensor* Tsum_tree, Tensor* Tmin_tree) override {
        auto sum_tree = Tsum_tree->flat<T>();
        auto min_tree = Tmin_tree->flat<T>();

        const auto capacity = buffer->capacity();
        // prefix_sumに該当
        const Tensor &search_values = ctx->input(1);
        auto search_values_flat = search_values.flat<T>();
        const int64 batch_size = search_values.dim_size(0);

        CancellationManager* cm = ctx->cancellation_manager();
        OP_REQUIRES_OK(ctx, buffer->wait_for_samples(cm, minimum_sample_size));
        Tensor* indices_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                                            TensorShape({static_cast<int>(batch_size)}),
                                            &indices_tensor));
        auto indices_flat = indices_tensor->flat<int64>();

        Tensor* values_out_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1,
                                            TensorShape({static_cast<int>(batch_size)}),
                                            &values_out_tensor));
        auto values_out_flat = values_out_tensor->flat<T>();

        Tensor* total_priority = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &total_priority));

        Tensor* min_priority = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, TensorShape({}), &min_priority));

        Tensor* Tsize = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, TensorShape({}), &Tsize));

        std::vector<PersistentTuple> tensors;
        tensors.resize(batch_size);
        {
            // This scope should be kept to a minimum as it is a critical section
            tf_shared_lock l(buffer->seg_tree_mu_);

            // Set the value of the sum output
            // 0は番兵？ 全体のtotal
            total_priority->flat<T>().setConstant(sum_tree(1));
            // 全体のmin
            min_priority->flat<T>().setConstant(min_tree(1));
            Tsize->flat<int64>().setConstant(buffer->size());

            for (auto b = 0; b < batch_size; ++b)
            {
                auto prefixsum = search_values_flat(b) * sum_tree(1);
                int64 idx = 1;
                while(idx < capacity)
                {
                    if (sum_tree(2 * idx) > prefixsum)
                        idx = 2 * idx;
                    else
                    {
                        prefixsum -= sum_tree(2 * idx);
                        idx = 2 * idx + 1;
                    }
                }
                indices_flat(b) = idx - capacity;
                values_out_flat(b) = sum_tree(idx);
                OP_REQUIRES_OK(ctx, buffer->encode(ctx, indices_flat(b), &tensors[b]));
                // This global_idx ensures that we won't update after we discarded a sample
                indices_flat(b) = buffer->global_index(indices_flat(b));
            }
        }

        // Now that we know indices we can gather the data
        QueueInterface::Tuple output_tuple;
        output_tuple.reserve(buffer->num_components());
        for (int i = 0; i < buffer->num_components(); ++i) {
            const TensorShape shape = buffer->ManyOutShape(i, batch_size);
            Tensor element;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(buffer->component_dtypes()[i], shape, &element));
            output_tuple.emplace_back(element);
        }

        // Finally we can copy the data
        for (auto e = 0; e < batch_size; ++e)
        {
            for (int i = 0; i < buffer->num_components(); ++i) {
                OP_REQUIRES_OK(ctx, tensors[e][i].CopyToSlice(ctx, &output_tuple[i], e));
            }
        }
        OpOutputList output_components;
        OP_REQUIRES_OK(ctx, ctx->output_list("components", &output_components));
        for (int i = 0; i < output_tuple.size(); ++i) {
            output_components.set(i, output_tuple[i]);
        }
    }
    private:
      int64 minimum_sample_size;
};

REGISTER_OP("ReplayBufferPrioritySample")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("minimum_sample_size: int = 1")
    .Attr("T: type")
    .Input("handle: resource")
    .Input("prefixsum: T")
    .Output("indices: int64")
    .Output("weights: T")
    .Output("sum: T")
    .Output("min: T")
    .Output("size: int64")
    .Output("components: Tcomponents")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

#define REGISTER_SEGMENT_LOOKUP_CPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ReplayBufferPrioritySample").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ReplayBufferPrioritySampleOp<TYPE>)

TF_CALL_float(REGISTER_SEGMENT_LOOKUP_CPU);



/*
ReplayBufferに優先度つきでデータをつむ
    return custom_modules.replay_buffer_priority_enqueue_many(self._buffer,
                                                              priorities,
                                                              components)


*/

template<typename T>
class ReplayBufferPriorityEnqueueManyOp : public ExperienceBufferWithPriorityAccessOp<T> {
    public:
    explicit ReplayBufferPriorityEnqueueManyOp(OpKernelConstruction* ctx) : ExperienceBufferWithPriorityAccessOp<T>(ctx) {
    }

    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer, Tensor* Tsum_tree, Tensor* Tmin_tree) override {
        DataTypeVector expected_inputs;
        if (ctx->input_dtype(0) == DT_RESOURCE) {
            expected_inputs.push_back(DT_RESOURCE);
        } else {
            expected_inputs.push_back(DT_STRING_REF);
        }

        auto sum_tree = Tsum_tree->flat<T>();
        auto min_tree = Tmin_tree->flat<T>();
        const auto capacity = buffer->capacity();

        // TODO Check shape
        expected_inputs.push_back(DT_FLOAT);
        const Tensor &Tpriority_values = ctx->input(1);
        const auto priority_values = Tpriority_values.flat<T>();

        for (DataType dt : buffer->component_dtypes()) {
            expected_inputs.push_back(dt);
        }
        OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
        OpInputList components;
        OP_REQUIRES_OK(ctx, ctx->input_list("components", &components));
        const auto num_elements = components[0].dim_size(0);

        QueueInterface::Tuple tuple;
        for (const Tensor& Tcomponent : components)
            tuple.emplace_back(Tcomponent);

        std::vector<std::vector<PersistentTensorEx>> batch;
        batch.resize(num_elements);
        for (auto index = 0; index < num_elements; ++index)
        {
            for (auto i = 0; i < buffer->num_components(); ++i)
            {
                PersistentTensorEx element;
                OP_REQUIRES_OK(ctx, ExperienceBufferEnqueueManyOp::GetElementComponentFromBatch(tuple, index, i, ctx, &element));
                batch[index].push_back(element);
            }
        }

        // Data has been copied, move to critical section
        {
            // This scope should be kept to a minimum as it is a critical section
            // TODO: Change locks
            mutex_lock l(buffer->seg_tree_mu_);
            for (int64 index = 0; index < num_elements; ++index)
            {
                // データを追加し, segment treeを更新する
                const int64 element_idx = buffer->enqueue(batch[index]);
                {
                    // Update sum
                    // element_idxが含まれる区間に全部足す
                    auto idx = element_idx + capacity;
                    sum_tree(idx) = priority_values(index);
                    for (int64 i = idx; i > 1; i >>= 1)
                    {
                        sum_tree(i >> 1) = sum_tree(i) + sum_tree(i ^ 1);
                    }
                }
                {
                    // Update min
                    auto idx = element_idx + capacity;
                    min_tree(idx) = priority_values(index);
                    for (int64 i = idx; i > 1; i >>= 1)
                    {
                        min_tree(i >> 1) = std::min(min_tree(i), min_tree(i ^ 1));
                    }
                }
            }
        }
    }
};

REGISTER_OP("ReplayBufferPriorityEnqueueMany")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("T: type")
    .Input("handle: resource")
    .Input("priorities: T")
    .Input("components: Tcomponents")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

#define REGISTER_PRIORITY_ENQUEUE_MANY(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ReplayBufferPriorityEnqueueMany").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ReplayBufferPriorityEnqueueManyOp<TYPE>)

TF_CALL_float(REGISTER_PRIORITY_ENQUEUE_MANY);



template<typename T>
class ReplayBufferUpdatePriorityOp : public ExperienceBufferWithPriorityAccessOp<T> {
    public:
    explicit ReplayBufferUpdatePriorityOp(OpKernelConstruction* ctx) : ExperienceBufferWithPriorityAccessOp<T>(ctx) {
    }

    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer, Tensor* Tsum_tree, Tensor* Tmin_tree) override {
        auto sum_tree = Tsum_tree->flat<T>();
        auto min_tree = Tmin_tree->flat<T>();
        const auto capacity = buffer->capacity();

        const Tensor& Tindices = ctx->input(1);
        auto indices = Tindices.flat<int64>();
        const int64 num_elements = Tindices.dim_size(0);

        // TODO: Check that we are not updating a discarded transition

        // TODO: Check matching shapes
        const Tensor& Tpriorities = ctx->input(2);
        auto priority_values = Tpriorities.flat<T>();
        {
            // This scope should be kept to a minimum as it is a critical section
            mutex_lock l(buffer->seg_tree_mu_);
            for (auto index = 0; index < num_elements; ++index)
            {
                //はみでてるときよう???
                if(indices(index) >= buffer->oldest_index())
                {
                    const auto element_idx = indices(index) % capacity;
                    {
                        // Update sum
                        auto idx = element_idx + capacity;
                        sum_tree(idx) = priority_values(index);
                        for (int64 i = idx; i > 1; i >>= 1)
                        {
                            sum_tree(i >> 1) = sum_tree(i) + sum_tree(i ^ 1);
                        }
                    }
                    {
                        // Update min
                        auto idx = element_idx + capacity;
                        min_tree(idx) = priority_values(index);
                        for (int64 i = idx; i > 1; i >>= 1)
                        {
                            min_tree(i >> 1) = std::min(min_tree(i), min_tree(i ^ 1));
                        }
                    }
                }
            }
        }
    }
};

REGISTER_OP("ReplayBufferUpdatePriority")
    .Attr("T: type")
    .Input("handle: resource")
    .Input("indices: int64")
    .Input("priorities: T")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

#define REGISTER_PRIORITY_UPDATE_PRIORITY(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ReplayBufferUpdatePriority").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ReplayBufferUpdatePriorityOp<TYPE>)

TF_CALL_float(REGISTER_PRIORITY_UPDATE_PRIORITY);


class ExperienceBufferSizeOp : public ExperienceBufferAccessOp {
    public:
    explicit ExperienceBufferSizeOp(OpKernelConstruction* ctx) : ExperienceBufferAccessOp(ctx) {
    }
    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer) override {
        Tensor* Tsize = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &Tsize));
        Tsize->flat<int64>().setConstant(buffer->size());
    }
};

REGISTER_OP("ExperienceBufferSize")
    .Input("handle: resource")
    .Output("size: int64")
    .SetShapeFn(::tensorflow::shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("ExperienceBufferSize").Device(DEVICE_CPU), ExperienceBufferSizeOp);

class ExperienceBufferCloseOp : public ExperienceBufferAccessOp {
    public:
    explicit ExperienceBufferCloseOp(OpKernelConstruction* ctx) : ExperienceBufferAccessOp(ctx) {
    }
    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer) override {
        buffer->close();
    }
};

REGISTER_OP("ExperienceBufferClose")
    .Input("handle: resource")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("ExperienceBufferClose").Device(DEVICE_CPU), ExperienceBufferCloseOp);


class ExperienceBufferEnqueueRecentOp : public ExperienceBufferAccessOp {
    public:
    explicit ExperienceBufferEnqueueRecentOp(OpKernelConstruction* ctx) : ExperienceBufferAccessOp(ctx) {
    }
    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer) override {
        DataTypeVector expected_inputs;
        DataTypeVector expected_outputs;
        if (ctx->input_dtype(0) == DT_RESOURCE) {
            expected_inputs.push_back(DT_RESOURCE);
        } else {
            expected_inputs.push_back(DT_STRING_REF);
        }
        for (DataType dt : buffer->component_dtypes())
            expected_inputs.push_back(dt);
        for (DataType dt : buffer->component_dtypes())
            expected_inputs.push_back(dt);

        expected_outputs.push_back(DT_BOOL);
        for (auto h = 0; h < buffer->capacity() + 1; ++h)
            for (DataType dt : buffer->component_dtypes())
                expected_outputs.push_back(dt);

        OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));
        OpInputList blank_components;
        OP_REQUIRES_OK(ctx, ctx->input_list("blank_components", &blank_components));
        OpInputList components;
        OP_REQUIRES_OK(ctx, ctx->input_list("components", &components));

        // typedef std::vector<Tensor> Tuple なぜ一度これを介するのか不明
        QueueInterface::Tuple output_tuple;

        // If buffer is not full, repeat blank_components
        for (auto h = 0; h < buffer->capacity() - buffer->size(); ++h)
            for (const Tensor& Tcomponent : blank_components)
                output_tuple.emplace_back(Tcomponent);

        buffer->encode_history(ctx, &output_tuple);

        QueueInterface::Tuple tuple;
        for (const Tensor& Tcomponent : components)
        {
            tuple.emplace_back(Tcomponent);
            output_tuple.emplace_back(Tcomponent);
        }

        Tensor* Tvalid = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &Tvalid));
        // capacityと一致しているか見る
        Tvalid->flat<bool>().setConstant(buffer->size() == buffer->capacity());

        buffer->enqueue(tuple);

        OpOutputList output_components;
        OP_REQUIRES_OK(ctx, ctx->output_list("history", &output_components));
        for (int i = 0; i < output_tuple.size(); ++i) {
            output_components.set(i, output_tuple[i]);
        }
    }
};

REGISTER_OP("ExperienceBufferEnqueueRecent")
    .Input("handle: resource")
    .Input("blank_components: Tcomponents")
    .Input("components: Tcomponents")
    .Output("valid: bool")
    .Output("history: Tout_components")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("Tout_components: list(type) >= 1")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("ExperienceBufferEnqueueRecent").Device(DEVICE_CPU), ExperienceBufferEnqueueRecentOp);


/*
呼び出し例

    def encode_history(self):
        valid, history=custom_modules.experience_buffer_encode_recent(self._buffer,
                                                                      blank_components=self.blank_components,
                                                                      Tout_components=self.dtypes * (self._length-1))

validはExperienceBufferが長さ全体まで埋まっているかどうか
historyは古い順からBufferのデータを全部返り値にいれたもの
*/

class ExperienceBufferEncodeRecentOp : public ExperienceBufferAccessOp {
    public:
    explicit ExperienceBufferEncodeRecentOp(OpKernelConstruction* ctx) : ExperienceBufferAccessOp(ctx) {
    }
    void Compute(OpKernelContext* ctx, ExperienceBuffer *buffer) override {
        DataTypeVector expected_inputs;
        DataTypeVector expected_outputs;
        if (ctx->input_dtype(0) == DT_RESOURCE) {
            expected_inputs.push_back(DT_RESOURCE);
        } else {
            expected_inputs.push_back(DT_STRING_REF);
        }
        for (DataType dt : buffer->component_dtypes())
            expected_inputs.push_back(dt);

        expected_outputs.push_back(DT_BOOL);
        for (auto h = 0; h < buffer->capacity(); ++h)
            for (DataType dt : buffer->component_dtypes())
                expected_outputs.push_back(dt);

        OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));
        OpInputList blank_components;
        OP_REQUIRES_OK(ctx, ctx->input_list("blank_components", &blank_components));

        QueueInterface::Tuple output_tuple;

        // If buffer is not full, repeat blank_components
        for (auto h = 0; h < buffer->capacity() - buffer->size(); ++h)
            for (const Tensor& Tcomponent : blank_components)
                output_tuple.emplace_back(Tcomponent);

        buffer->encode_history(ctx, &output_tuple);

        Tensor* Tvalid = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &Tvalid));
        Tvalid->flat<bool>().setConstant(buffer->size() == buffer->capacity());

        OpOutputList output_components;
        OP_REQUIRES_OK(ctx, ctx->output_list("history", &output_components));
        for (int i = 0; i < output_tuple.size(); ++i) {
            output_components.set(i, output_tuple[i]);
        }
    }
};

REGISTER_OP("ExperienceBufferEncodeRecent")
    .Input("handle: resource")
    .Input("blank_components: Tcomponents")
    .Output("valid: bool")
    .Output("history: Tout_components")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("Tout_components: list(type) >= 1")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("ExperienceBufferEncodeRecent").Device(DEVICE_CPU), ExperienceBufferEncodeRecentOp);
