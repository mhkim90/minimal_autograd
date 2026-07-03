#pragma once
// variable.h — Var / VarPtr.
//
// Phase 2 redesign: backward() does a topological sort of the graph, then
// runs each node's back_fn in reverse. Shared nodes are visited once, so
// a node used in two branches gets its gradient accumulated exactly once.

#include "autograd/tensor.h"
#include "autograd/shape.h"
#include <memory>
#include <vector>
#include <unordered_set>
#include <functional>
#include <utility>
#include <cassert>
#include <stdexcept>

namespace ag {

struct Var;
using VarPtr = std::shared_ptr<Var>;

struct Var {
    Mat data;
    Mat grad;
    Shape shape_;
    Stride stride_;
    bool cuda_ = false;
    int cuda_device_ = 0;
    float* cuda_data_ = nullptr;
    float* cuda_grad_ = nullptr;

    // Inputs to this node (for topo traversal).
    std::vector<VarPtr> parents;

    // Captured by apply<Fn>. Reads this->grad, accumulates into parents.
    std::function<void()> back_fn;

    explicit Var(Mat d)
        : data(std::move(d)),
          grad(Mat::Zero(data.rows(), data.cols())),
          shape_(make_shape(data.rows(), data.cols())),
          stride_(contiguous_stride(shape_)) {}

    ~Var();

    static VarPtr make(Mat d) { return std::make_shared<Var>(std::move(d)); }

    static VarPtr make4d(Mat d, int64_t N, int64_t C, int64_t H, int64_t W) {
        assert(d.rows() == N && d.cols() == C * H * W);
        auto v = make(std::move(d));
        v->set_shape({N, C, H, W});
        return v;
    }

    const Shape& shape() const noexcept { return shape_; }
    const Stride& stride() const noexcept { return stride_; }
    int ndim() const noexcept { return shape_.ndim(); }
    int64_t numel() const noexcept { return shape_.numel(); }
    int64_t dim(int i) const { return shape_[i]; }
    bool is4d() const noexcept { return shape_.ndim() == 4; }
    bool is_cuda() const noexcept { return cuda_; }
    int cuda_device() const noexcept { return cuda_device_; }
    float* cuda_data() const noexcept { return cuda_data_; }
    float* cuda_grad() const noexcept { return cuda_grad_; }

    void set_shape(std::initializer_list<int64_t> dims) {
        set_shape(Shape(dims));
    }

    void set_shape(const Shape& s) {
        assert_same_numel(s, static_cast<int64_t>(data.rows()) * data.cols());
        shape_ = s;
        stride_ = contiguous_stride(shape_);
    }

    void view(std::initializer_list<int64_t> dims) {
        set_shape(dims);
    }

    VarPtr cuda(int device = 0) const;
    VarPtr cpu() const;

    void sync_data_from_cuda();
    void sync_grad_from_cuda();
    void sync_data_to_cuda();
    void sync_grad_to_cuda();
    void cuda_zero_grad();
    void cuda_grad_ones();

    // Call on a scalar (1x1) loss node. Builds topo order, then runs
    // back_fn in reverse. Safe to call once per forward pass.
    void backward();

    // Zeroes grad on every node reachable from this node.
    void zero_grad();

private:
    void _build_topo(std::vector<Var*>& order,
                     std::unordered_set<Var*>& visited);
};

} // namespace ag
