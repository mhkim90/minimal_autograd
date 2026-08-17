#include "detail/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ag {
namespace detail {
namespace cpu_ops {

namespace {

inline int64_t checked_dim_sum(const char* op, int64_t a, int64_t b) {
    if (b > std::numeric_limits<int64_t>::max() - a) {
        throw std::overflow_error(std::string(op) + ": dimension overflow");
    }
    return a + b;
}

inline std::vector<int64_t> contiguous_strides(const Shape& s) {
    return contiguous_stride(s).strides;
}

inline int64_t linear_offset(const std::vector<int64_t>& idx,
                             const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        off += idx[i] * strides[i];
    }
    return off;
}

inline int normalize_axis(int axis, int rank, const char* what) {
    if (rank <= 0) {
        std::ostringstream os;
        os << what << ": empty rank";
        throw std::invalid_argument(os.str());
    }
    int n = axis;
    if (n < 0) n += rank;
    if (n < 0 || n >= rank) {
        std::ostringstream os;
        os << what << ": axis " << axis << " out of range for rank "
           << rank;
        throw std::invalid_argument(os.str());
    }
    return n;
}

inline std::vector<int> normalize_axes(const std::vector<int>& axes,
                                       int rank,
                                       const char* what) {
    std::vector<int> out;
    out.reserve(axes.size());
    for (int axis : axes) {
        const int n = normalize_axis(axis, rank, what);
        for (int prev : out) {
            if (prev == n) {
                std::ostringstream os;
                os << what << ": duplicate axis " << n;
                throw std::invalid_argument(os.str());
            }
        }
        out.push_back(n);
    }
    return out;
}

inline bool increment_index(std::vector<int64_t>& idx, const Shape& s) {
    for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
        if (idx[i] + 1 < s.sizes[i]) {
            ++idx[i];
            return true;
        }
        idx[i] = 0;
    }
    return false;
}

}  // namespace

Tensor tensor_sum(const Tensor& a) {
    Tensor out = Tensor::empty(Shape{}, a.device());
    if (a.elements() == 0) {
        float z = 0.f;
        out.copy_from_host(&z, 1);
        return out;
    }
    std::vector<float> av(a.elements());
    a.copy_to_host(av.data(), av.size());
    float s = 0.f;
    for (float v : av) s += v;
    out.copy_from_host(&s, 1);
    return out;
}

Tensor tensor_broadcast_scalar(const Tensor& scalar, const Shape& target) {
    Tensor out = Tensor::empty(target, scalar.device());
    if (out.elements() == 0) return out;
    std::vector<float> seed(1);
    scalar.copy_to_host(seed.data(), 1);
    std::vector<float> buf(out.elements(), seed[0]);
    out.copy_from_host(buf.data(), buf.size());
    return out;
}

Tensor tensor_reshape_view(const Tensor& a, const Shape& target_shape) {
    return a.reshape(target_shape);
}

Tensor tensor_concat_nd(const std::vector<Tensor>& inputs, int axis) {
    if (inputs.empty()) {
        throw std::invalid_argument("concat: requires at least one input");
    }
    const Shape& first = inputs[0].shape();
    const int rank = static_cast<int>(first.rank());
    const int ax = normalize_axis(axis, rank, "concat");
    Dims out_sizes(first.sizes.begin(), first.sizes.end());
    int64_t total = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].device() != inputs[0].device()) {
            throw std::invalid_argument("concat: device mismatch");
        }
        if (inputs[i].shape().rank() != static_cast<std::size_t>(rank)) {
            std::ostringstream os;
            os << "concat: rank mismatch at input " << i
               << " (input rank " << inputs[i].shape().rank()
               << " vs expected " << rank << ")";
            throw std::invalid_argument(os.str());
        }
        for (int d = 0; d < rank; ++d) {
            if (d == ax) continue;
            if (inputs[i].shape()[d] != first[d]) {
                std::ostringstream os;
                os << "concat: dim " << d << " mismatch at input " << i
                   << " (have " << inputs[i].shape()[d]
                   << ", want " << first[d] << ")";
                throw std::invalid_argument(os.str());
            }
        }
        total = checked_dim_sum("concat", total, inputs[i].shape()[ax]);
    }
    out_sizes[ax] = total;
    const Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, inputs[0].device());
    if (out.elements() == 0) return out;

    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);
    std::vector<float> ov(out.elements());
    int64_t ax_offset = 0;
    for (const auto& t : inputs) {
        const int64_t along = t.shape()[ax];
        if (along == 0) continue;
        const std::vector<int64_t> in_strides = contiguous_strides(t.shape());
        std::vector<float> tv(t.elements());
        t.copy_to_host(tv.data(), tv.size());
        std::vector<int64_t> outer_shape;
        outer_shape.reserve(rank - 1);
        for (int d = 0; d < rank; ++d) {
            if (d != ax) outer_shape.push_back(out_sizes[d]);
        }
        int64_t outer_numel = 1;
        for (int64_t d : outer_shape) outer_numel *= d;
        std::vector<int64_t> outer(rank - 1, 0);
        for (int64_t o = 0; o < outer_numel; ++o) {
            int64_t in_base = 0;
            int64_t out_base = 0;
            int j = 0;
            for (int d = 0; d < rank; ++d) {
                if (d == ax) continue;
                in_base += outer[j] * in_strides[d];
                out_base += outer[j] * out_strides[d];
                ++j;
            }
            for (int64_t k = 0; k < along; ++k) {
                const int64_t in_off = in_base + k * in_strides[ax];
                const int64_t out_off =
                    out_base + (ax_offset + k) * out_strides[ax];
                ov[out_off] = tv[in_off];
            }
            for (int j = static_cast<int>(outer.size()) - 1; j >= 0; --j) {
                if (++outer[j] < outer_shape[j]) break;
                outer[j] = 0;
            }
        }
        ax_offset += along;
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_slice_nd(const Tensor& a, int axis, int64_t start, int64_t len) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "slice");
    if (start < 0 || len <= 0 || start > s[ax] || len > s[ax] - start) {
        std::ostringstream os;
        os << "slice: out of range (axis " << ax << " dim " << s[ax]
           << ", start " << start << ", len " << len << ")";
        throw std::invalid_argument(os.str());
    }
    Dims out_sizes(s.sizes.begin(), s.sizes.end());
    out_sizes[ax] = len;
    const Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, a.device());
    if (out.elements() == 0) return out;

    const std::vector<int64_t> in_strides = contiguous_strides(s);
    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);
    std::vector<float> av(a.elements()), ov(out.elements());
    a.copy_to_host(av.data(), av.size());
    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int d = 0; d < rank; ++d) {
        if (d != ax) outer_shape.push_back(s.sizes[d]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;
    std::vector<int64_t> outer(rank - 1, 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t in_base = 0;
        int64_t out_base = 0;
        int j = 0;
        for (int d = 0; d < rank; ++d) {
            if (d == ax) continue;
            in_base += outer[j] * in_strides[d];
            out_base += outer[j] * out_strides[d];
            ++j;
        }
        for (int64_t k = 0; k < len; ++k) {
            ov[out_base + k * out_strides[ax]] =
                av[in_base + (start + k) * in_strides[ax]];
        }
        for (int j = static_cast<int>(outer.size()) - 1; j >= 0; --j) {
            if (++outer[j] < outer_shape[j]) break;
            outer[j] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_slice_backward_nd(const Tensor& g,
                                const Shape& input_shape,
                                int axis, int64_t start, int64_t len) {
    const int rank = static_cast<int>(input_shape.rank());
    const int ax = normalize_axis(axis, rank, "slice_backward");
    Tensor out = Tensor::zeros(input_shape, g.device());
    if (out.elements() == 0) return out;
    const std::vector<int64_t> out_strides = contiguous_strides(input_shape);
    Dims gdims(input_shape.sizes.begin(), input_shape.sizes.end());
    gdims[ax] = len;
    const std::vector<int64_t> g_strides =
        contiguous_strides(Shape(gdims));
    std::vector<float> gv(g.elements()), ov(out.elements());
    g.copy_to_host(gv.data(), gv.size());
    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int d = 0; d < rank; ++d) {
        if (d != ax) outer_shape.push_back(input_shape.sizes[d]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;
    std::vector<int64_t> outer(rank - 1, 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t out_base = 0;
        int64_t g_base = 0;
        int j = 0;
        for (int d = 0; d < rank; ++d) {
            if (d == ax) continue;
            out_base += outer[j] * out_strides[d];
            g_base += outer[j] * g_strides[d];
            ++j;
        }
        for (int64_t k = 0; k < len; ++k) {
            ov[out_base + (start + k) * out_strides[ax]] =
                gv[g_base + k * g_strides[ax]];
        }
        for (int j = static_cast<int>(outer.size()) - 1; j >= 0; --j) {
            if (++outer[j] < outer_shape[j]) break;
            outer[j] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

std::vector<Tensor> tensor_concat_backward_nd(
    const Tensor& g, const std::vector<int64_t>& along_per_input,
    const std::vector<Shape>& input_shapes, int axis) {
    if (input_shapes.empty()) {
        throw std::invalid_argument("concat_backward: empty inputs");
    }
    const int rank = static_cast<int>(input_shapes[0].rank());
    const int ax = normalize_axis(axis, rank, "concat_backward");
    if (along_per_input.size() != input_shapes.size()) {
        throw std::invalid_argument(
            "concat_backward: along/size count mismatch");
    }
    std::vector<Tensor> out;
    out.reserve(input_shapes.size());
    const std::vector<int64_t> g_strides = contiguous_strides(g.shape());
    std::vector<float> gv(g.elements());
    g.copy_to_host(gv.data(), gv.size());
    int64_t ax_offset = 0;
    for (std::size_t i = 0; i < input_shapes.size(); ++i) {
        const int64_t along = along_per_input[i];
        Tensor piece = Tensor::empty(input_shapes[i], g.device());
        if (along > 0 && piece.elements() > 0) {
            const std::vector<int64_t> p_strides =
                contiguous_strides(input_shapes[i]);
            std::vector<float> pv(piece.elements());
            std::vector<int64_t> outer_shape;
            outer_shape.reserve(rank - 1);
            for (int d = 0; d < rank; ++d) {
                if (d != ax) outer_shape.push_back(input_shapes[i][d]);
            }
            int64_t outer_numel = 1;
            for (int64_t d : outer_shape) outer_numel *= d;
            std::vector<int64_t> outer(rank - 1, 0);
            for (int64_t o = 0; o < outer_numel; ++o) {
                int64_t g_base = 0;
                int64_t p_base = 0;
                int j = 0;
                for (int d = 0; d < rank; ++d) {
                    if (d == ax) continue;
                    g_base += outer[j] * g_strides[d];
                    p_base += outer[j] * p_strides[d];
                    ++j;
                }
                for (int64_t k = 0; k < along; ++k) {
                    pv[p_base + k * p_strides[ax]] =
                        gv[g_base + (ax_offset + k) * g_strides[ax]];
                }
                for (int j = static_cast<int>(outer.size()) - 1; j >= 0; --j) {
                    if (++outer[j] < outer_shape[j]) break;
                    outer[j] = 0;
                }
            }
            piece.copy_from_host(pv.data(), pv.size());
        }
        out.push_back(std::move(piece));
        ax_offset += along;
    }
    return out;
}

Tensor tensor_sum_axes_nd(const Tensor& a,
                          const std::vector<int>& axes_in,
                          bool keep_dims) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const std::vector<int> axes = normalize_axes(axes_in, rank, "sum");
    std::vector<bool> is_reduced(rank, false);
    for (int axis : axes) is_reduced[axis] = true;
    Dims out_sizes;
    out_sizes.reserve(keep_dims ? rank : rank - axes.size());
    for (int i = 0; i < rank; ++i) {
        if (!is_reduced[i]) {
            out_sizes.push_back(s.sizes[i]);
        } else if (keep_dims) {
            out_sizes.push_back(1);
        }
    }
    const Shape out_shape(out_sizes);
    Tensor out = Tensor::zeros(out_shape, a.device());
    if (out.elements() == 0 || a.elements() == 0) return out;
    const std::vector<int64_t> in_strides = contiguous_strides(s);
    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);
    std::vector<float> av(a.elements()), ov(out.elements(), 0.f);
    a.copy_to_host(av.data(), av.size());
    std::vector<int64_t> in_idx(rank, 0);
    for (int64_t count = 0; count < a.elements(); ++count) {
        int64_t out_off = 0;
        int out_axis = 0;
        for (int i = 0; i < rank; ++i) {
            if (is_reduced[i]) {
                if (keep_dims) ++out_axis;
            } else {
                out_off += in_idx[i] * out_strides[out_axis++];
            }
        }
        ov[out_off] += av[linear_offset(in_idx, in_strides)];
        increment_index(in_idx, s);
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_sum_axes_backward_nd(const Tensor& g,
                                   const Shape& input_shape,
                                   const std::vector<int>& axes_in,
                                   bool keep_dims) {
    const int rank = static_cast<int>(input_shape.rank());
    const std::vector<int> axes =
        normalize_axes(axes_in, rank, "sum_backward");
    std::vector<bool> is_reduced(rank, false);
    for (int axis : axes) is_reduced[axis] = true;
    Tensor out = Tensor::empty(input_shape, g.device());
    if (out.elements() == 0) return out;
    const std::vector<int64_t> out_strides = contiguous_strides(input_shape);
    const std::vector<int64_t> g_strides = contiguous_strides(g.shape());
    std::vector<float> gv(g.elements()), ov(out.elements());
    g.copy_to_host(gv.data(), gv.size());
    std::vector<int64_t> out_idx(rank, 0);
    for (int64_t count = 0; count < out.elements(); ++count) {
        int64_t g_off = 0;
        int g_axis = 0;
        for (int i = 0; i < rank; ++i) {
            if (is_reduced[i]) {
                if (keep_dims) ++g_axis;
            } else {
                g_off += out_idx[i] * g_strides[g_axis++];
            }
        }
        ov[linear_offset(out_idx, out_strides)] = gv[g_off];
        increment_index(out_idx, input_shape);
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_broadcast_add_nd(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::invalid_argument("broadcast_add: device mismatch");
    }
    const Shape& sa = a.shape();
    const Shape& sb = b.shape();
    const int rank_a = static_cast<int>(sa.rank());
    const int rank_b = static_cast<int>(sb.rank());
    const int out_rank = std::max(rank_a, rank_b);
    Dims out_sizes(out_rank, 1);
    for (int d = 0; d < out_rank; ++d) {
        const int ai = d - (out_rank - rank_a);
        const int bi = d - (out_rank - rank_b);
        const int64_t ad = ai < 0 ? 1 : sa[ai];
        const int64_t bd = bi < 0 ? 1 : sb[bi];
        if (ad != bd && ad != 1 && bd != 1) {
            std::ostringstream os;
            os << "broadcast_add: dimension mismatch (" << sa
               << " vs " << sb << ")";
            throw std::invalid_argument(os.str());
        }
        out_sizes[d] = ad == 1 ? bd : ad;
    }
    const Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, a.device());
    if (out.elements() == 0) return out;
    const std::vector<int64_t> a_strides = contiguous_strides(sa);
    const std::vector<int64_t> b_strides = contiguous_strides(sb);
    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);
    std::vector<float> av(a.elements()), bv(b.elements()), ov(out.elements());
    a.copy_to_host(av.data(), av.size());
    b.copy_to_host(bv.data(), bv.size());
    std::vector<int64_t> out_idx(out_rank, 0);
    for (std::size_t count = 0; count < out.elements(); ++count) {
        int64_t a_off = 0;
        int64_t b_off = 0;
        for (int d = 0; d < out_rank; ++d) {
            const int ai = d - (out_rank - rank_a);
            const int bi = d - (out_rank - rank_b);
            if (ai >= 0 && sa[ai] != 1) a_off += out_idx[d] * a_strides[ai];
            if (bi >= 0 && sb[bi] != 1) b_off += out_idx[d] * b_strides[bi];
        }
        ov[linear_offset(out_idx, out_strides)] = av[a_off] + bv[b_off];
        increment_index(out_idx, out_shape);
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_broadcast_add_backward_nd(const Tensor& g,
                                        const Shape& input_shape) {
    const Shape& output_shape = g.shape();
    const int out_rank = static_cast<int>(output_shape.rank());
    const int in_rank = static_cast<int>(input_shape.rank());
    if (in_rank > out_rank) {
        throw std::invalid_argument("broadcast_add_backward: rank mismatch");
    }
    Tensor out = Tensor::zeros(input_shape, g.device());
    if (out.elements() == 0 || g.elements() == 0) return out;
    const std::vector<int64_t> g_strides = contiguous_strides(output_shape);
    const std::vector<int64_t> out_strides = contiguous_strides(input_shape);
    std::vector<float> gv(g.elements()), ov(out.elements(), 0.f);
    g.copy_to_host(gv.data(), gv.size());
    std::vector<int64_t> g_idx(out_rank, 0);
    for (std::size_t count = 0; count < g.elements(); ++count) {
        int64_t in_off = 0;
        for (int d = 0; d < in_rank; ++d) {
            const int gd = out_rank - in_rank + d;
            if (input_shape[d] != 1) in_off += g_idx[gd] * out_strides[d];
        }
        ov[in_off] += gv[linear_offset(g_idx, g_strides)];
        increment_index(g_idx, output_shape);
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_matmul_nd(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::invalid_argument("matmul: device mismatch");
    }
    const Shape& sa = a.shape();
    const Shape& sb = b.shape();
    const int rank_a = static_cast<int>(sa.rank());
    const int rank_b = static_cast<int>(sb.rank());
    if (rank_a < 2 || rank_b < 2) {
        std::ostringstream os;
        os << "matmul: requires rank >= 2 (got " << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    if (rank_a != rank_b) {
        std::ostringstream os;
        os << "matmul: rank mismatch (" << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    const int64_t M = sa[rank_a - 2];
    const int64_t K = sa[rank_a - 1];
    const int64_t K2 = sb[rank_b - 2];
    const int64_t N = sb[rank_b - 1];
    if (K != K2) {
        std::ostringstream os;
        os << "matmul: inner dimensions mismatch (" << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    for (int d = 0; d < rank_a - 2; ++d) {
        if (sa[d] != sb[d]) {
            std::ostringstream os;
            os << "matmul: batch dim " << d << " mismatch ("
               << sa[d] << " vs " << sb[d] << ")";
            throw std::invalid_argument(os.str());
        }
    }
    Dims out_sizes(sa.sizes.begin(), sa.sizes.end());
    out_sizes[rank_a - 2] = M;
    out_sizes[rank_a - 1] = N;
    const Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, a.device());
    if (out.elements() == 0) return out;
    int64_t batch = 1;
    for (int d = 0; d < rank_a - 2; ++d) batch *= sa[d];
    const std::vector<int64_t> a_strides = contiguous_strides(sa);
    const std::vector<int64_t> b_strides = contiguous_strides(sb);
    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);
    std::vector<float> av(a.elements()), bv(b.elements()), ov(out.elements());
    a.copy_to_host(av.data(), av.size());
    b.copy_to_host(bv.data(), bv.size());
    std::vector<int64_t> batch_idx(rank_a - 2, 0);
    for (int64_t batch_i = 0; batch_i < batch; ++batch_i) {
        int64_t a_base = 0, b_base = 0, out_base = 0;
        for (int d = 0; d < rank_a - 2; ++d) {
            a_base += batch_idx[d] * a_strides[d];
            b_base += batch_idx[d] * b_strides[d];
            out_base += batch_idx[d] * out_strides[d];
        }
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float sum = 0.f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += av[a_base + m * a_strides[rank_a - 2] +
                              k * a_strides[rank_a - 1]] *
                           bv[b_base + k * b_strides[rank_b - 2] +
                              n * b_strides[rank_b - 1]];
                }
                ov[out_base + m * out_strides[rank_a - 2] +
                   n * out_strides[rank_a - 1]] = sum;
            }
        }
        for (int d = rank_a - 3; d >= 0; --d) {
            if (++batch_idx[d] < sa[d]) break;
            batch_idx[d] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_matmul_backward_a_nd(const Tensor& g, const Tensor& b) {
    const Shape& sg = g.shape();
    const Shape& sb = b.shape();
    const int rank_g = static_cast<int>(sg.rank());
    const int rank_b = static_cast<int>(sb.rank());
    if (rank_g < 2 || rank_b < 2) {
        throw std::invalid_argument(
            "matmul_backward_a: requires rank >= 2");
    }
    if (rank_g != rank_b) {
        throw std::invalid_argument("matmul_backward_a: rank mismatch");
    }
    const int64_t M = sg[rank_g - 2];
    const int64_t N = sg[rank_g - 1];
    const int64_t K = sb[rank_b - 2];
    if (N != sb[rank_b - 1]) {
        throw std::invalid_argument(
            "matmul_backward_a: g/b inner mismatch");
    }
    for (int d = 0; d < rank_g - 2; ++d) {
        if (sg[d] != sb[d]) {
            throw std::invalid_argument("matmul_backward_a: batch mismatch");
        }
    }
    Dims out_sizes(sg.sizes.begin(), sg.sizes.end());
    out_sizes[rank_g - 1] = K;
    const Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, g.device());
    if (out.elements() == 0) return out;
    int64_t batch = 1;
    for (int d = 0; d < rank_g - 2; ++d) batch *= sg[d];
    const std::vector<int64_t> g_strides = contiguous_strides(sg);
    const std::vector<int64_t> b_strides = contiguous_strides(sb);
    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);
    std::vector<float> gv(g.elements()), bv(b.elements()), ov(out.elements());
    g.copy_to_host(gv.data(), gv.size());
    b.copy_to_host(bv.data(), bv.size());
    std::vector<int64_t> batch_idx(rank_g - 2, 0);
    for (int64_t batch_i = 0; batch_i < batch; ++batch_i) {
        int64_t g_base = 0, b_base = 0, out_base = 0;
        for (int d = 0; d < rank_g - 2; ++d) {
            g_base += batch_idx[d] * g_strides[d];
            b_base += batch_idx[d] * b_strides[d];
            out_base += batch_idx[d] * out_strides[d];
        }
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                float sum = 0.f;
                for (int64_t n = 0; n < N; ++n) {
                    sum += gv[g_base + m * g_strides[rank_g - 2] +
                              n * g_strides[rank_g - 1]] *
                           bv[b_base + k * b_strides[rank_b - 2] +
                              n * b_strides[rank_b - 1]];
                }
                ov[out_base + m * out_strides[rank_g - 2] +
                   k * out_strides[rank_g - 1]] = sum;
            }
        }
        for (int d = rank_g - 3; d >= 0; --d) {
            if (++batch_idx[d] < sg[d]) break;
            batch_idx[d] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_matmul_backward_b_nd(const Tensor& a, const Tensor& g) {
    const Shape& sa = a.shape();
    const Shape& sg = g.shape();
    const int rank_a = static_cast<int>(sa.rank());
    const int rank_g = static_cast<int>(sg.rank());
    if (rank_a < 2 || rank_g < 2) {
        throw std::invalid_argument(
            "matmul_backward_b: requires rank >= 2");
    }
    if (rank_a != rank_g) {
        throw std::invalid_argument("matmul_backward_b: rank mismatch");
    }
    const int64_t M = sa[rank_a - 2];
    const int64_t K = sa[rank_a - 1];
    const int64_t N = sg[rank_g - 1];
    if (sg[rank_g - 2] != M) {
        throw std::invalid_argument(
            "matmul_backward_b: a/g inner mismatch");
    }
    for (int d = 0; d < rank_a - 2; ++d) {
        if (sa[d] != sg[d]) {
            throw std::invalid_argument("matmul_backward_b: batch mismatch");
        }
    }
    Dims out_sizes(sa.sizes.begin(), sa.sizes.end());
    out_sizes[rank_a - 2] = K;
    out_sizes[rank_a - 1] = N;
    const Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, g.device());
    if (out.elements() == 0) return out;
    int64_t batch = 1;
    for (int d = 0; d < rank_a - 2; ++d) batch *= sa[d];
    const std::vector<int64_t> a_strides = contiguous_strides(sa);
    const std::vector<int64_t> g_strides = contiguous_strides(sg);
    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);
    std::vector<float> av(a.elements()), gv(g.elements()), ov(out.elements());
    a.copy_to_host(av.data(), av.size());
    g.copy_to_host(gv.data(), gv.size());
    std::vector<int64_t> batch_idx(rank_a - 2, 0);
    for (int64_t batch_i = 0; batch_i < batch; ++batch_i) {
        int64_t a_base = 0, g_base = 0, out_base = 0;
        for (int d = 0; d < rank_a - 2; ++d) {
            a_base += batch_idx[d] * a_strides[d];
            g_base += batch_idx[d] * g_strides[d];
            out_base += batch_idx[d] * out_strides[d];
        }
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t n = 0; n < N; ++n) {
                float sum = 0.f;
                for (int64_t m = 0; m < M; ++m) {
                    sum += av[a_base + m * a_strides[rank_a - 2] +
                              k * a_strides[rank_a - 1]] *
                           gv[g_base + m * g_strides[rank_g - 2] +
                              n * g_strides[rank_g - 1]];
                }
                ov[out_base + k * out_strides[rank_a - 2] +
                   n * out_strides[rank_a - 1]] = sum;
            }
        }
        for (int d = rank_a - 3; d >= 0; --d) {
            if (++batch_idx[d] < sa[d]) break;
            batch_idx[d] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

Tensor tensor_add(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), bv(n), ov(n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] + bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_mul(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), bv(n), ov(n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] * bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_scale(const Tensor& a, float scalar) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), ov(n);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] * scalar;
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_relu(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = av[i] > 0.f ? av[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_relu_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = sv[i] > 0.f ? gv[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sigmoid(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = 1.f / (1.f + std::exp(-av[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sigmoid_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * sv[i] * (1.f - sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_tanh(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::tanh(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_tanh_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (1.f - sv[i] * sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_exp(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::exp(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_exp_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * sv[i];
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_log(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::log(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_log_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / sv[i];
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sqrt(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::sqrt(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sqrt_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / (2.f * sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    sigmoid_out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), so(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        so[i] = 1.f / (1.f + std::exp(-av[i]));
        ov[i] = av[i] * so[i];
    }
    out.copy_from_host(ov.data(), n);
    sigmoid_out.copy_from_host(so.data(), n);
    return out;
}

Tensor tensor_silu_backward(const Tensor& g,
                            const Tensor& x,
                            const Tensor& sig) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    sig.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (sv[i] + xv[i] * sv[i] * (1.f - sv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_softplus(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = av[i];
        const float ax = std::fabs(x);
        ov[i] = std::max(x, 0.f) + std::log(1.f + std::exp(-ax));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_softplus_backward(const Tensor& g, const Tensor& x) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / (1.f + std::exp(-xv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sub(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] - bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sub_backward_a(const Tensor& g) {
    return g.clone();
}

Tensor tensor_sub_backward_b(const Tensor& g) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = -gv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_div(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] / bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_div_backward_a(const Tensor& g, const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = gv[i] / bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_div_backward_b(const Tensor& g,
                             const Tensor& a,
                             const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = -(gv[i] * av[i]) / (bv[i] * bv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

}  // namespace cpu_ops
}  // namespace detail
}  // namespace ag
