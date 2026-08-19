#include "detail/tensor_kernels.h"

#include "tensor_dispatch_internal.h"

namespace ag {
namespace detail {

Tensor tensor_sum(const Tensor& a) {
    return cpu_ops::tensor_sum(a);
}

Tensor tensor_broadcast_scalar(const Tensor& scalar, const Shape& target) {
    return cpu_ops::tensor_broadcast_scalar(scalar, target);
}

Tensor tensor_reshape_view(const Tensor& a, const Shape& target_shape) {
    validate_reshape(a, target_shape);
    return cpu_ops::tensor_reshape_view(a, target_shape);
}

Tensor tensor_flip_nd(const Tensor& a, int axis) {
    const int ax = normalize_axis(axis, static_cast<int>(a.shape().rank()),
                                  "flip");
    return tensor_flip_nd_cpu(a, ax);
}

Tensor tensor_concat_nd(const std::vector<Tensor>& inputs, int axis) {
    validate_concat(inputs, axis);
    return cpu_ops::tensor_concat_nd(inputs, axis);
}

Tensor tensor_slice_nd(const Tensor& a, int axis, int64_t start, int64_t len) {
    const int ax = normalize_axis(axis, static_cast<int>(a.shape().rank()), "slice");
    if (start < 0 || len <= 0 || start > a.shape()[ax] ||
        len > a.shape()[ax] - start) {
        std::ostringstream os;
        os << "slice: out of range (axis " << ax << " dim " << a.shape()[ax]
           << ", start " << start << ", len " << len << ")";
        throw std::invalid_argument(os.str());
    }
    return cpu_ops::tensor_slice_nd(a, axis, start, len);
}

Tensor tensor_slice_backward_nd(const Tensor& g, const Shape& input_shape,
                                int axis, int64_t start, int64_t len) {
    normalize_axis(axis, static_cast<int>(input_shape.rank()), "slice_backward");
    return cpu_ops::tensor_slice_backward_nd(g, input_shape, axis, start, len);
}

std::vector<Tensor> tensor_concat_backward_nd(
    const Tensor& g, const std::vector<int64_t>& along_per_input,
    const std::vector<Shape>& input_shapes, int axis) {
    if (input_shapes.empty()) {
        throw std::invalid_argument("concat_backward: empty inputs");
    }
    normalize_axis(axis, static_cast<int>(input_shapes[0].rank()),
                   "concat_backward");
    if (along_per_input.size() != input_shapes.size()) {
        throw std::invalid_argument(
            "concat_backward: along/size count mismatch");
    }
    return cpu_ops::tensor_concat_backward_nd(
        g, along_per_input, input_shapes, axis);
}

Tensor tensor_sum_axes_nd(const Tensor& a, const std::vector<int>& axes,
                          bool keep_dims) {
    normalize_axes(axes, static_cast<int>(a.shape().rank()), "sum");
    return cpu_ops::tensor_sum_axes_nd(a, axes, keep_dims);
}

Tensor tensor_sum_axes_backward_nd(const Tensor& g, const Shape& input_shape,
                                   const std::vector<int>& axes,
                                   bool keep_dims) {
    normalize_axes(axes, static_cast<int>(input_shape.rank()), "sum_backward");
    return cpu_ops::tensor_sum_axes_backward_nd(
        g, input_shape, axes, keep_dims);
}

Tensor tensor_broadcast_add_nd(const Tensor& a, const Tensor& b) {
    require_same_device("broadcast_add", a, b);
    const int rank_a = static_cast<int>(a.shape().rank());
    const int rank_b = static_cast<int>(b.shape().rank());
    const int out_rank = std::max(rank_a, rank_b);
    for (int d = 0; d < out_rank; ++d) {
        const int ai = d - (out_rank - rank_a);
        const int bi = d - (out_rank - rank_b);
        const int64_t ad = ai < 0 ? 1 : a.shape()[ai];
        const int64_t bd = bi < 0 ? 1 : b.shape()[bi];
        if (ad != bd && ad != 1 && bd != 1) {
            std::ostringstream os;
            os << "broadcast_add: dimension mismatch (" << a.shape()
               << " vs " << b.shape() << ")";
            throw std::invalid_argument(os.str());
        }
    }
    return cpu_ops::tensor_broadcast_add_nd(a, b);
}

Tensor tensor_broadcast_add_backward_nd(const Tensor& g,
                                        const Shape& input_shape) {
    if (input_shape.rank() > g.shape().rank()) {
        throw std::invalid_argument("broadcast_add_backward: rank mismatch");
    }
    return cpu_ops::tensor_broadcast_add_backward_nd(g, input_shape);
}

Tensor tensor_matmul_nd(const Tensor& a, const Tensor& b) {
    validate_matmul(a, b);
    return cpu_ops::tensor_matmul_nd(a, b);
}

Tensor tensor_matmul_backward_a_nd(const Tensor& g, const Tensor& b) {
    validate_matmul_backward_a(g, b);
    return cpu_ops::tensor_matmul_backward_a_nd(g, b);
}

Tensor tensor_matmul_backward_b_nd(const Tensor& a, const Tensor& g) {
    validate_matmul_backward_b(a, g);
    return cpu_ops::tensor_matmul_backward_b_nd(a, g);
}

}  // namespace detail
}  // namespace ag
