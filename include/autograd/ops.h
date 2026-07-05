#pragma once
// ops.h — built-in differentiable operations.
//
// Each public op is a free function returning a VarPtr. The corresponding
// Function subclass lives alongside it; instantiate via apply<Fn>(inputs).
//
// All ops are out-of-place: they allocate a new node, never mutate inputs.

#include "autograd/function.h"
#include "autograd/tensor.h"
#include "autograd/cuda_core.h"
#include <stdexcept>
#include <utility>
#include <vector>

namespace ag {

// --- Phase 1: keep these on the new apply<Fn> design. ---

struct AddFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0] + in[1];
    }
    Mats backward(const Mat& g) override {
        return {g, g};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};
VarPtr add(VarPtr a, VarPtr b);

struct MulFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0].cwiseProduct(in[1]);
    }
    Mats backward(const Mat& g) override {
        // d/da (a*b) = b, d/db (a*b) = a
        return {g.cwiseProduct(saved[1]), g.cwiseProduct(saved[0])};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};
VarPtr mul(VarPtr a, VarPtr b);

struct MatMulFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0] * in[1];
    }
    Mats backward(const Mat& g) override {
        // d/da (a*b) = g * b^T,  d/db (a*b) = a^T * g
        return {g * saved[1].transpose(),
                saved[0].transpose() * g};
    }
};
VarPtr matmul(VarPtr a, VarPtr b);

struct ReLUFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0].cwiseMax(0.f);
    }
    Mats backward(const Mat& g) override {
        Mat mask = (saved[0].array() > 0.f).cast<float>();
        return {g.cwiseProduct(mask)};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};
VarPtr relu(VarPtr a);

struct SumFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        Mat s(1, 1);
        s(0, 0) = in[0].sum();
        return s;
    }
    Mats backward(const Mat& g) override {
        // broadcast scalar g to original shape
        return {Mat::Constant(saved[0].rows(), saved[0].cols(), g(0, 0))};
    }
};
VarPtr sum(VarPtr a);

// --- Phase 2c: bias add with proper backward. ---
//
// Forward: a (N, D) + b (1, D)   — same row is added to every row of a.
// Backward: db must be a (1, D) row-vector — sum over the batch axis.
struct BroadcastAddFn : Function {
    Mat forward(const Mats& in) override {
        // in[0]: (N, D), in[1]: (1, D)
        saved = in;
        return in[0].rowwise() + Eigen::RowVectorXf(in[1].row(0));
    }
    Mats backward(const Mat& g) override {
        // g: (N, D). Pass through for in[0], reduce rows for in[1].
        Mat bias_grad(1, saved[1].cols());
        bias_grad.row(0) = g.colwise().sum();
        return {g, bias_grad};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};
VarPtr broadcast_add(VarPtr a, VarPtr b);

// Scalar multiplication of a Var by a constant.
struct ScaleFn : Function {
    float s;
    explicit ScaleFn(float sc) : s(sc) {}
    Mat forward(const Mats& in) override {
        saved = in;
        return s * in[0];
    }
    Mats backward(const Mat& g) override {
        return {s * g};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};
VarPtr scale(VarPtr a, float s);

// --- Phase 3 ops. ---

struct SoftmaxFn : Function {
    Mat forward(const Mats& in) override {
        // Per-row softmax: exp(x - max) / sum(exp(x - max))
        Mat x = in[0];
        Mat m = x.rowwise().maxCoeff().replicate(1, x.cols());
        Mat e = (x.array() - m.array()).exp();
        Mat denom = e.rowwise().sum().replicate(1, x.cols());
        Mat s = e.cwiseQuotient(denom);
        saved = {s};
        return s;
    }
    Mats backward(const Mat& g) override {
        const Mat& s = saved[0];
        Mat gs = (g.cwiseProduct(s)).rowwise().sum().replicate(1, s.cols());
        return {s.cwiseProduct(g - gs)};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};
VarPtr softmax(VarPtr a);

struct LogSoftmaxFn : Function {
    Mat forward(const Mats& in) override {
        Mat x = in[0];
        Mat m = x.rowwise().maxCoeff().replicate(1, x.cols());
        Mat shifted = x.array() - m.array();
        Mat logsum  = shifted.array().exp().matrix().rowwise().sum().array().log();
        Mat lsm     = shifted.array() - logsum.replicate(1, x.cols()).array();
        saved = {x, lsm};
        return lsm;
    }
    Mats backward(const Mat& g) override {
        // Jacobian: d lsm_i / d x_j = delta_ij - softmax(x)_j
        // So grad_x = g - softmax(x) * sum_j g_j
        const Mat& x = saved[0];
        const Mat& lsm = saved[1];
        Mat sm = lsm.array().exp().matrix();
        Mat gs = g.rowwise().sum().replicate(1, g.cols());
        return {g - sm.cwiseProduct(gs)};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};
VarPtr log_softmax(VarPtr a);

struct TransposeFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0].transpose();
    }
    Mats backward(const Mat& g) override {
        return {g.transpose()};
    }
};
VarPtr transpose(VarPtr a);

struct ReshapeFn : Function {
    int rows, cols;
    ReshapeFn(int r, int c) : rows(r), cols(c) {}
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0].reshaped(rows, cols);
    }
    Mats backward(const Mat& g) override {
        // Reshape g back to saved[0] shape (must have same total size).
        return {g.reshaped(saved[0].rows(), saved[0].cols())};
    }
};
VarPtr reshape(VarPtr a, int rows, int cols);

struct ConcatFn : Function {
    Mat forward(const Mats& in) override {
        // Stack inputs vertically; all inputs must have the same number of columns.
        int total = 0;
        for (auto& m : in) total += m.rows();
        int cols = in.empty() ? 0 : in[0].cols();
        Mat out(total, cols);
        int off = 0;
        for (auto& m : in) {
            out.middleRows(off, m.rows()) = m;
            off += m.rows();
        }
        saved = in;
        return out;
    }
    Mats backward(const Mat& g) override {
        // Slice g back into one grad per input row block.
        Mats grads;
        grads.reserve(saved.size());
        int off = 0;
        for (size_t i = 0; i < saved.size(); ++i) {
            int r = saved[i].rows();
            grads.push_back(g.middleRows(off, r));
            off += r;
        }
        return grads;
    }
};

// --- Activation and arithmetic ops ---

struct SigmoidFn : Function {
    Mat forward(const Mats& in) override {
        Mat s = (1.f / (1.f + (-in[0].array()).exp())).matrix();
        saved = {s};
        return s;
    }
    Mats backward(const Mat& g) override {
        const auto& s = saved[0].array();
        return {(g.array() * s * (1.f - s)).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct TanhFn : Function {
    Mat forward(const Mats& in) override {
        Mat t = in[0].array().tanh().matrix();
        saved = {t};
        return t;
    }
    Mats backward(const Mat& g) override {
        return {(g.array() * (1.f - saved[0].array().square())).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct ExpFn : Function {
    Mat forward(const Mats& in) override {
        Mat e = in[0].array().exp().matrix();
        saved = {e};
        return e;
    }
    Mats backward(const Mat& g) override {
        return {g.cwiseProduct(saved[0])};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct LogFn : Function {
    Mat forward(const Mats& in) override {
        saved = {in[0]};
        return in[0].array().log().matrix();
    }
    Mats backward(const Mat& g) override {
        return {(g.array() / saved[0].array()).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct SqrtFn : Function {
    Mat forward(const Mats& in) override {
        Mat s = in[0].array().sqrt().matrix();
        saved = {s};
        return s;
    }
    Mats backward(const Mat& g) override {
        return {(g.array() / (2.f * saved[0].array())).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct SiLUFn : Function {
    Mat forward(const Mats& in) override {
        Mat s = (1.f / (1.f + (-in[0].array()).exp())).matrix();
        saved = {in[0], s};
        return (in[0].array() * s.array()).matrix();
    }
    Mats backward(const Mat& g) override {
        const auto& x = saved[0].array();
        const auto& s = saved[1].array();
        return {(g.array() * (s + x * s * (1.f - s))).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct SoftplusFn : Function {
    Mat forward(const Mats& in) override {
        saved = {in[0]};
        auto x = in[0].array();
        return (x.max(0.f) + (1.f + (-x.abs()).exp()).log()).matrix();
    }
    Mats backward(const Mat& g) override {
        const auto& x = saved[0].array();
        return {(g.array() / (1.f + (-x).exp())).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct SubFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0] - in[1];
    }
    Mats backward(const Mat& g) override { return {g, -g}; }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct DivFn : Function {
    Mat forward(const Mats& in) override {
        saved = in;
        return in[0].cwiseQuotient(in[1]);
    }
    Mats backward(const Mat& g) override {
        Mat ga = g.cwiseQuotient(saved[1]);
        Mat gb = -(g.cwiseProduct(saved[0])).cwiseQuotient(
                    saved[1].cwiseProduct(saved[1]));
        return {ga, gb};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

// Cumulative sum along axis (0=rows, 1=cols). Backward is suffix sum of g.
struct CumsumFn : Function {
    int axis;
    explicit CumsumFn(int ax) : axis(ax) {}
    Mat forward(const Mats& in) override {
        Mat out = in[0];
        if (axis == 1) {
            for (int c = 1; c < out.cols(); ++c) out.col(c) += out.col(c - 1);
        } else {
            for (int r = 1; r < out.rows(); ++r) out.row(r) += out.row(r - 1);
        }
        return out;
    }
    Mats backward(const Mat& g) override {
        Mat grad = g;
        if (axis == 1) {
            for (int c = grad.cols() - 2; c >= 0; --c) grad.col(c) += grad.col(c + 1);
        } else {
            for (int r = grad.rows() - 2; r >= 0; --r) grad.row(r) += grad.row(r + 1);
        }
        return {grad};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

// Flip along axis (0=flip rows, 1=flip cols).
struct FlipFn : Function {
    int axis;
    explicit FlipFn(int ax) : axis(ax) {}
    Mat forward(const Mats& in) override {
        if (axis == 1) return in[0].rowwise().reverse();
        return in[0].colwise().reverse();
    }
    Mats backward(const Mat& g) override {
        if (axis == 1) return {g.rowwise().reverse()};
        return {g.colwise().reverse()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct SinFn : Function {
    Mat forward(const Mats& in) override {
        saved = {in[0]};
        return in[0].array().sin().matrix();
    }
    Mats backward(const Mat& g) override {
        return {(g.array() * saved[0].array().cos()).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

struct CosFn : Function {
    Mat forward(const Mats& in) override {
        saved = {in[0]};
        return in[0].array().cos().matrix();
    }
    Mats backward(const Mat& g) override {
        return {(g.array() * (-saved[0].array().sin())).matrix()};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

// clamp(x, lo, hi) — element-wise. Backward is 0 outside [lo,hi], 1 inside.
struct ClampFn : Function {
    float lo, hi;
    ClampFn(float lo, float hi) : lo(lo), hi(hi) {}
    Mat forward(const Mats& in) override {
        saved = {in[0]};
        return in[0].array().max(lo).min(hi).matrix();
    }
    Mats backward(const Mat& g) override {
        Mat mask = ((saved[0].array() >= lo) && (saved[0].array() <= hi))
                       .cast<float>().matrix();
        return {g.cwiseProduct(mask)};
    }
    bool preserves_input_shape() const noexcept override { return true; }
};

// ColSliceFn — extract columns [start, start+len).
// Used to implement split/chunk: call twice with start=0 and start=half.
struct ColSliceFn : Function {
    int start, len;
    ColSliceFn(int start, int len) : start(start), len(len) {}
    Mat forward(const Mats& in) override {
        saved = {in[0]};
        return in[0].middleCols(start, len);
    }
    Mats backward(const Mat& g) override {
        Mat grad = Mat::Zero(saved[0].rows(), saved[0].cols());
        grad.middleCols(start, len) = g;
        return {grad};
    }
};
VarPtr col_slice(VarPtr a, int start, int len);

// split(x) — splits columns evenly in two. Returns {left, right}.
// x must have an even number of columns.
inline std::pair<VarPtr, VarPtr> split(VarPtr a) {
    int half = a->data.cols() / 2;
    return {col_slice(a, 0, half), col_slice(a, half, half)};
}

// Horizontal concatenation — stacks inputs column-wise.
// All inputs must have the same number of rows (same N).
// Mirrors torch.cat(tensors, dim=-1) / torch.cat(tensors, dim=1) in 2D layout.
struct HCatFn : Function {
    Mat forward(const Mats& in) override {
        int rows = in[0].rows();
        int total_cols = 0;
        for (auto& m : in) { assert(m.rows() == rows); total_cols += m.cols(); }
        Mat out(rows, total_cols);
        int off = 0;
        for (auto& m : in) {
            out.middleCols(off, m.cols()) = m;
            off += m.cols();
        }
        saved = in;
        return out;
    }
    Mats backward(const Mat& g) override {
        Mats grads;
        grads.reserve(saved.size());
        int off = 0;
        for (auto& m : saved) {
            grads.push_back(g.middleCols(off, m.cols()));
            off += m.cols();
        }
        return grads;
    }
};

// --- Free function API: thin wrappers over apply<Fn>. ---

inline VarPtr add(VarPtr a, VarPtr b)            {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda() || b->is_cuda()) return cuda_add_op(a, b);
#endif
    return apply<AddFn>({a, b});
}
inline VarPtr mul(VarPtr a, VarPtr b)            {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda() || b->is_cuda()) return cuda_mul_op(a, b);
#endif
    return apply<MulFn>({a, b});
}
inline VarPtr matmul(VarPtr a, VarPtr b)         {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda() || b->is_cuda()) return cuda_matmul_op(a, b);
#endif
    return apply<MatMulFn>({a, b});
}
inline VarPtr relu(VarPtr a)                     {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_relu_op(a);
#endif
    return apply<ReLUFn>({a});
}
inline VarPtr sum(VarPtr a)                      {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_sum_op(a);
#endif
    return apply<SumFn>({a});
}
inline VarPtr broadcast_add(VarPtr a, VarPtr b)  {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda() || b->is_cuda()) return cuda_broadcast_add_op(a, b);
#endif
    return apply<BroadcastAddFn>({a, b});
}
inline VarPtr scale(VarPtr a, float s)           {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_scale_op(a, s);
#endif
    return apply<ScaleFn>({a}, s);
}
inline VarPtr softmax(VarPtr a)                  {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_softmax_op(a);
#endif
    return apply<SoftmaxFn>({a});
}
inline VarPtr log_softmax(VarPtr a)              {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_log_softmax_op(a);
#endif
    return apply<LogSoftmaxFn>({a});
}
inline VarPtr transpose(VarPtr a)                { return apply<TransposeFn>({a}); }
inline VarPtr reshape(VarPtr a, int r, int c)    { return apply<ReshapeFn>({a}, r, c); }
inline VarPtr concat(std::vector<VarPtr> inputs) { return apply<ConcatFn>(inputs); }
inline VarPtr hcat(std::vector<VarPtr> inputs)   { return apply<HCatFn>(inputs); }

inline VarPtr sigmoid(VarPtr a)                  {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_sigmoid_op(a);
#endif
    return apply<SigmoidFn>({a});
}
inline VarPtr tanh_op(VarPtr a)                  {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_tanh_op(a);
#endif
    return apply<TanhFn>({a});
}
inline VarPtr exp_op(VarPtr a)                   {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_exp_op(a);
#endif
    return apply<ExpFn>({a});
}
inline VarPtr log_op(VarPtr a)                   {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_log_op(a);
#endif
    return apply<LogFn>({a});
}
inline VarPtr sqrt_op(VarPtr a)                  {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_sqrt_op(a);
#endif
    return apply<SqrtFn>({a});
}
inline VarPtr silu(VarPtr a)                     {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_silu_op(a);
#endif
    return apply<SiLUFn>({a});
}
inline VarPtr softplus(VarPtr a)                 {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda()) return cuda_softplus_op(a);
#endif
    return apply<SoftplusFn>({a});
}
inline VarPtr sub(VarPtr a, VarPtr b)            {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda() || b->is_cuda()) return cuda_sub_op(a, b);
#endif
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols()) {
        throw std::runtime_error("sub: shape mismatch");
    }
    return apply<SubFn>({a, b});
}
inline VarPtr div_op(VarPtr a, VarPtr b)         {
#ifdef AUTOGRAD_USE_CUDA
    if (a->is_cuda() || b->is_cuda()) return cuda_div_op(a, b);
#endif
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols()) {
        throw std::runtime_error("div_op: shape mismatch");
    }
    return apply<DivFn>({a, b});
}
inline VarPtr cumsum(VarPtr a, int axis = 1)     { return apply<CumsumFn>({a}, axis); }
inline VarPtr flip(VarPtr a, int axis = 1)       { return apply<FlipFn>({a}, axis); }

inline VarPtr sin_op(VarPtr a)                            { return apply<SinFn>({a}); }
inline VarPtr cos_op(VarPtr a)                            { return apply<CosFn>({a}); }
inline VarPtr clamp(VarPtr a, float lo, float hi)         { return apply<ClampFn>({a}, lo, hi); }
inline VarPtr col_slice(VarPtr a, int start, int len)     { return apply<ColSliceFn>({a}, start, len); }

} // namespace ag
