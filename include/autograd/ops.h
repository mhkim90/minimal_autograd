#pragma once
// ops.h — built-in differentiable operations.
//
// Each public op is a free function returning a VarPtr. The corresponding
// Function subclass lives alongside it; instantiate via apply<Fn>(inputs).
//
// All ops are out-of-place: they allocate a new node, never mutate inputs.

#include "autograd/function.h"
#include "autograd/tensor.h"
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

// --- Free function API: thin wrappers over apply<Fn>. ---

inline VarPtr add(VarPtr a, VarPtr b)            { return apply<AddFn>({a, b}); }
inline VarPtr mul(VarPtr a, VarPtr b)            { return apply<MulFn>({a, b}); }
inline VarPtr matmul(VarPtr a, VarPtr b)         { return apply<MatMulFn>({a, b}); }
inline VarPtr relu(VarPtr a)                     { return apply<ReLUFn>({a}); }
inline VarPtr sum(VarPtr a)                      { return apply<SumFn>({a}); }
inline VarPtr broadcast_add(VarPtr a, VarPtr b)  { return apply<BroadcastAddFn>({a, b}); }
inline VarPtr scale(VarPtr a, float s)           { return apply<ScaleFn>({a}, s); }
inline VarPtr softmax(VarPtr a)                  { return apply<SoftmaxFn>({a}); }
inline VarPtr log_softmax(VarPtr a)              { return apply<LogSoftmaxFn>({a}); }
inline VarPtr transpose(VarPtr a)                { return apply<TransposeFn>({a}); }
inline VarPtr reshape(VarPtr a, int r, int c)    { return apply<ReshapeFn>({a}, r, c); }

inline VarPtr concat(std::vector<VarPtr> inputs) { return apply<ConcatFn>(inputs); }

} // namespace ag
