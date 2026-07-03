// src/conv.cpp — im2col / col2im, Conv2dFn / MaxPool2dFn, factories.

#include "autograd/conv.h"

#include <stdexcept>

namespace ag {

// =====================================================================
// im2col
// =====================================================================
//
// input:  (N, C*H*W)   row n: input[n, c*H*W + h*W + w] = pixel
// output: (C*kH*kW, N*oH*oW)
//          column n*oH*oW + oh*oW + ow = one receptive field.
Mat im2col(const Mat& input,
           int N, int C, int H, int W,
           int kH, int kW,
           int stride, int pad) {
    int oH = (H + 2 * pad - kH) / stride + 1;
    int oW = (W + 2 * pad - kW) / stride + 1;
    assert((H + 2 * pad - kH) % stride == 0);
    assert((W + 2 * pad - kW) % stride == 0);

    Mat col = Mat::Zero(C * kH * kW, N * oH * oW);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int row_idx = (c * kH + kh) * kW + kw;
                    for (int oh = 0; oh < oH; ++oh) {
                        int ih = oh * stride + kh - pad;
                        for (int ow = 0; ow < oW; ++ow) {
                            int iw = ow * stride + kw - pad;
                            int col_idx = n * oH * oW + oh * oW + ow;
                            float v = 0.f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                v = input(n, c * H * W + ih * W + iw);
                            col(row_idx, col_idx) = v;
                        }
                    }
                }
            }
        }
    }
    return col;
}

// =====================================================================
// col2im
// =====================================================================
//
// col:    (C*kH*kW, N*oH*oW)
// output: (N, C*H*W)
//
// ⚠️ CRITICAL: overlapping patches ACCUMULATE (+=), not assign.
//   A single input pixel is hit by multiple output positions, so its
//   gradient must be the sum of all contributions.
Mat col2im(const Mat& col,
           int N, int C, int H, int W,
           int kH, int kW,
           int stride, int pad) {
    int oH = (H + 2 * pad - kH) / stride + 1;
    int oW = (W + 2 * pad - kW) / stride + 1;

    Mat out = Mat::Zero(N, C * H * W);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int row_idx = (c * kH + kh) * kW + kw;
                    for (int oh = 0; oh < oH; ++oh) {
                        int ih = oh * stride + kh - pad;
                        for (int ow = 0; ow < oW; ++ow) {
                            int iw = ow * stride + kw - pad;
                            int col_idx = n * oH * oW + oh * oW + ow;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                out(n, c * H * W + ih * W + iw) +=
                                    col(row_idx, col_idx);
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

// =====================================================================
// Conv2dFn
// =====================================================================

Mat Conv2dFn::forward(const Mats& in) {
    oH = (H + 2 * pad - kH) / stride + 1;
    oW = (W + 2 * pad - kW) / stride + 1;
    assert((H + 2 * pad - kH) % stride == 0);
    assert((W + 2 * pad - kW) % stride == 0);

    Mat col = im2col(in[0], N, C, H, W, kH, kW, stride, pad);
    Mat out = in[1] * col;                       // (out_ch, N*oH*oW)
    out.colwise() += in[2].row(0).transpose();   // bias broadcast

    saved = {col, in[1]};
    return out.reshaped(N, out_ch * oH * oW);
}

Mats Conv2dFn::backward(const Mat& grad) {
    Mat g_mat = grad.reshaped(out_ch, N * oH * oW);
    const Mat& col  = saved[0];
    const Mat& Wmat = saved[1];

    Mat grad_W = g_mat * col.transpose();
    Mat grad_col = Wmat.transpose() * g_mat;
    Mat grad_input = col2im(grad_col, N, C, H, W, kH, kW, stride, pad);
    Mat grad_bias(1, out_ch);
    grad_bias.row(0) = g_mat.rowwise().sum();

    return {grad_input, grad_W, grad_bias};
}

VarPtr conv2d_op(VarPtr input, VarPtr weight, VarPtr bias,
                 int N, int C, int H, int W,
                 int kH, int kW, int stride, int pad) {
    if (input->is_cuda() || weight->is_cuda() || bias->is_cuda()) {
        throw std::runtime_error("conv2d_op: CUDA inputs are not supported yet");
    }
    auto fn = std::make_shared<Conv2dFn>();
    fn->N = N; fn->C = C; fn->H = H; fn->W = W;
    fn->out_ch = weight->data.rows();
    fn->kH = kH; fn->kW = kW;
    fn->stride = stride; fn->pad = pad;
    fn->oH = (H + 2 * pad - kH) / stride + 1;
    fn->oW = (W + 2 * pad - kW) / stride + 1;

    Mats in_data = {input->data, weight->data, bias->data};
    auto out = Var::make(fn->forward(in_data));
    out->set_shape({N, fn->out_ch, fn->oH, fn->oW});
    out->parents = {input, weight, bias};
    out->back_fn = [fn, ins = std::vector<VarPtr>{input, weight, bias},
                    wp = std::weak_ptr<Var>(out)]() {
        auto self = wp.lock();
        auto gs = fn->backward(self->grad);
        for (size_t i = 0; i < ins.size(); ++i) ins[i]->grad += gs[i];
    };
    return out;
}

// =====================================================================
// MaxPool2dFn
// =====================================================================

Mat MaxPool2dFn::forward(const Mats& in) {
    oH = (H - kH) / stride + 1;
    oW = (W - kW) / stride + 1;
    assert((H - kH) % stride == 0);
    assert((W - kW) % stride == 0);

    Mat col = im2col(in[0], N, C, H, W, kH, kW, /*stride=*/stride, /*pad=*/0);
    Mat out = Mat::Zero(C, N * oH * oW);
    Mat mask = Mat::Zero(C * kH * kW, N * oH * oW);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    int col_idx = n * oH * oW + oh * oW + ow;
                    int best_k = 0;
                    float best_v = col(c * kH * kW, col_idx);
                    for (int k = 1; k < kH * kW; ++k) {
                        float v = col(c * kH * kW + k, col_idx);
                        if (v > best_v) { best_v = v; best_k = k; }
                    }
                    out(c, col_idx) = best_v;
                    mask(c * kH * kW + best_k, col_idx) = 1.f;
                }
            }
        }
    }

    saved = {mask};
    return out.reshaped(N, C * oH * oW);
}

Mats MaxPool2dFn::backward(const Mat& grad) {
    Mat g_mat = grad.reshaped(C, N * oH * oW);
    const Mat& mask = saved[0];
    Mat grad_col = Mat::Zero(C * kH * kW, N * oH * oW);
    for (int c = 0; c < C; ++c) {
        for (int k = 0; k < kH * kW; ++k) {
            grad_col.row(c * kH * kW + k) =
                mask.row(c * kH * kW + k).cwiseProduct(g_mat.row(c));
        }
    }
    Mat grad_input = col2im(grad_col, N, C, H, W, kH, kW, stride, /*pad=*/0);
    return {grad_input};
}

VarPtr maxpool2d_op(VarPtr input,
                    int N, int C, int H, int W,
                    int kH, int kW, int stride) {
    if (input->is_cuda()) {
        throw std::runtime_error("maxpool2d_op: CUDA inputs are not supported yet");
    }
    auto fn = std::make_shared<MaxPool2dFn>();
    fn->N = N; fn->C = C; fn->H = H; fn->W = W;
    fn->kH = kH; fn->kW = kW; fn->stride = stride;
    fn->oH = (H - kH) / stride + 1;
    fn->oW = (W - kW) / stride + 1;

    Mats in_data = {input->data};
    auto out = Var::make(fn->forward(in_data));
    out->set_shape({N, C, fn->oH, fn->oW});
    out->parents = {input};
    out->back_fn = [fn, ins = std::vector<VarPtr>{input},
                    wp = std::weak_ptr<Var>(out)]() {
        auto self = wp.lock();
        auto gs = fn->backward(self->grad);
        for (size_t i = 0; i < ins.size(); ++i) ins[i]->grad += gs[i];
    };
    return out;
}

// =====================================================================
// Module constructors
// =====================================================================

Conv2d::Conv2d(int in_ch_, int out_ch_, int kH_, int kW_,
               int stride_, int pad_)
    : in_ch(in_ch_), out_ch(out_ch_), kH(kH_), kW(kW_),
      stride(stride_), pad(pad_) {
    // Kaiming uniform init.
    float fan_in = static_cast<float>(in_ch * kH * kW);
    float bound  = std::sqrt(1.f / fan_in);
    W = Var::make(Mat::Random(out_ch, in_ch * kH * kW) * bound);
    b = Var::make(Mat::Zero(1, out_ch));
}

VarPtr Conv2d::forward(VarPtr x, int H, int W_) {
    int N = x->data.rows();
    return conv2d_op(x, W, b, N, in_ch, H, W_, kH, kW, stride, pad);
}

VarPtr Conv2d::forward(VarPtr x) {
    assert(x->is4d() && "Conv2d::forward(x) requires x shape (N,C,H,W)");
    assert(x->dim(1) == in_ch && "Conv2d::forward: input channels mismatch");
    return forward(x, static_cast<int>(x->dim(2)), static_cast<int>(x->dim(3)));
}

VarPtr MaxPool2d::forward(VarPtr x, int H, int W_) {
    int N = x->data.rows();
    int C = x->data.cols() / (H * W_);
    return maxpool2d_op(x, N, C, H, W_, kH, kW, stride);
}

VarPtr MaxPool2d::forward(VarPtr x) {
    assert(x->is4d() && "MaxPool2d::forward(x) requires x shape (N,C,H,W)");
    return forward(x, static_cast<int>(x->dim(2)), static_cast<int>(x->dim(3)));
}

// =====================================================================
// AvgPool2dFn
// =====================================================================

Mat AvgPool2dFn::forward(const Mats& in) {
    oH = (H - kH) / stride + 1;
    oW = (W - kW) / stride + 1;
    assert((H - kH) % stride == 0);
    assert((W - kW) % stride == 0);

    Mat col = im2col(in[0], N, C, H, W, kH, kW, stride, 0);
    int ksz = kH * kW;
    Mat out(C, N * oH * oW);
    for (int c = 0; c < C; ++c) {
        out.row(c) = col.middleRows(c * ksz, ksz).colwise().mean();
    }
    saved = {col};
    return out.reshaped(N, C * oH * oW);
}

Mats AvgPool2dFn::backward(const Mat& grad) {
    Mat g_mat = grad.reshaped(C, N * oH * oW);
    int ksz = kH * kW;
    float inv_k = 1.f / static_cast<float>(ksz);
    Mat grad_col = Mat::Zero(C * ksz, N * oH * oW);
    for (int c = 0; c < C; ++c) {
        grad_col.middleRows(c * ksz, ksz) = g_mat.row(c).replicate(ksz, 1) * inv_k;
    }
    return {col2im(grad_col, N, C, H, W, kH, kW, stride, 0)};
}

VarPtr avgpool2d_op(VarPtr input,
                    int N, int C, int H, int W,
                    int kH, int kW, int stride) {
    if (input->is_cuda()) {
        throw std::runtime_error("avgpool2d_op: CUDA inputs are not supported yet");
    }
    auto fn = std::make_shared<AvgPool2dFn>();
    fn->N = N; fn->C = C; fn->H = H; fn->W = W;
    fn->kH = kH; fn->kW = kW; fn->stride = stride;
    fn->oH = (H - kH) / stride + 1;
    fn->oW = (W - kW) / stride + 1;

    Mats in_data = {input->data};
    auto out = Var::make(fn->forward(in_data));
    out->set_shape({N, C, fn->oH, fn->oW});
    out->parents = {input};
    out->back_fn = [fn, ins = std::vector<VarPtr>{input},
                    wp = std::weak_ptr<Var>(out)]() {
        auto self = wp.lock();
        auto gs = fn->backward(self->grad);
        for (size_t i = 0; i < ins.size(); ++i) ins[i]->grad += gs[i];
    };
    return out;
}

VarPtr AvgPool2d::forward(VarPtr x, int H, int W_) {
    int N = x->data.rows();
    int C = x->data.cols() / (H * W_);
    return avgpool2d_op(x, N, C, H, W_, kH, kW, stride);
}

VarPtr AvgPool2d::forward(VarPtr x) {
    assert(x->is4d() && "AvgPool2d::forward(x) requires x shape (N,C,H,W)");
    return forward(x, static_cast<int>(x->dim(2)), static_cast<int>(x->dim(3)));
}

// =====================================================================
// DepthwiseConv2dFn
// =====================================================================

Mat DepthwiseConv2dFn::forward(const Mats& in) {
    oH = (H + 2 * pad - kH) / stride + 1;
    oW = (W + 2 * pad - kW) / stride + 1;
    assert((H + 2 * pad - kH) % stride == 0);
    assert((W + 2 * pad - kW) % stride == 0);

    Mat col = im2col(in[0], N, C, H, W, kH, kW, stride, pad);
    int ksz = kH * kW;
    Mat out(C, N * oH * oW);
    for (int c = 0; c < C; ++c) {
        out.row(c) = in[1].row(c) * col.middleRows(c * ksz, ksz);
    }
    out.colwise() += in[2].row(0).transpose();

    saved = {col, in[1]};
    return out.reshaped(N, C * oH * oW);
}

Mats DepthwiseConv2dFn::backward(const Mat& grad) {
    Mat g_mat = grad.reshaped(C, N * oH * oW);
    const Mat& col  = saved[0];
    const Mat& Wmat = saved[1];

    int ksz = kH * kW;
    Mat grad_W(C, ksz);
    Mat grad_col = Mat::Zero(C * ksz, N * oH * oW);
    for (int c = 0; c < C; ++c) {
        grad_W.row(c) = g_mat.row(c) * col.middleRows(c * ksz, ksz).transpose();
        grad_col.middleRows(c * ksz, ksz) =
            Wmat.row(c).transpose() * g_mat.row(c);
    }
    Mat grad_bias(1, C);
    grad_bias.row(0) = g_mat.rowwise().sum();
    Mat grad_input = col2im(grad_col, N, C, H, W, kH, kW, stride, pad);

    return {grad_input, grad_W, grad_bias};
}

VarPtr depthwise_conv2d_op(VarPtr input, VarPtr weight, VarPtr bias,
                            int N, int C, int H, int W,
                            int kH, int kW, int stride, int pad) {
    if (input->is_cuda() || weight->is_cuda() || bias->is_cuda()) {
        throw std::runtime_error("depthwise_conv2d_op: CUDA inputs are not supported yet");
    }
    auto fn = std::make_shared<DepthwiseConv2dFn>();
    fn->N = N; fn->C = C; fn->H = H; fn->W = W;
    fn->kH = kH; fn->kW = kW; fn->stride = stride; fn->pad = pad;
    fn->oH = (H + 2 * pad - kH) / stride + 1;
    fn->oW = (W + 2 * pad - kW) / stride + 1;

    Mats in_data = {input->data, weight->data, bias->data};
    auto out = Var::make(fn->forward(in_data));
    out->set_shape({N, C, fn->oH, fn->oW});
    out->parents = {input, weight, bias};
    out->back_fn = [fn, ins = std::vector<VarPtr>{input, weight, bias},
                    wp = std::weak_ptr<Var>(out)]() {
        auto self = wp.lock();
        auto gs = fn->backward(self->grad);
        for (size_t i = 0; i < ins.size(); ++i) ins[i]->grad += gs[i];
    };
    return out;
}

DepthwiseConv2d::DepthwiseConv2d(int channels_, int kH_, int kW_,
                                   int stride_, int pad_)
    : channels(channels_), kH(kH_), kW(kW_), stride(stride_), pad(pad_) {
    float fan_in = static_cast<float>(kH * kW);
    float bound  = std::sqrt(1.f / fan_in);
    W = Var::make(Mat::Random(channels, kH * kW) * bound);
    b = Var::make(Mat::Zero(1, channels));
}

VarPtr DepthwiseConv2d::forward(VarPtr x, int H, int W_) {
    int N = x->data.rows();
    return depthwise_conv2d_op(x, W, b, N, channels, H, W_, kH, kW, stride, pad);
}

VarPtr DepthwiseConv2d::forward(VarPtr x) {
    assert(x->is4d() && "DepthwiseConv2d::forward(x) requires x shape (N,C,H,W)");
    assert(x->dim(1) == channels && "DepthwiseConv2d::forward: channel mismatch");
    return forward(x, static_cast<int>(x->dim(2)), static_cast<int>(x->dim(3)));
}

// =====================================================================
// NearestUpsample2dFn
// =====================================================================

Mat NearestUpsample2dFn::forward(const Mats& in) {
    int oH = H * scale, oW = W * scale;
    Mat out = Mat::Zero(N, C * oH * oW);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    float v = in[0](n, c * H * W + h * W + w);
                    for (int sh = 0; sh < scale; ++sh)
                        for (int sw = 0; sw < scale; ++sw)
                            out(n, c * oH * oW + (h*scale+sh) * oW + (w*scale+sw)) = v;
                }
    return out;
}

Mats NearestUpsample2dFn::backward(const Mat& grad) {
    int oH = H * scale, oW = W * scale;
    Mat grad_in = Mat::Zero(N, C * H * W);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    float s = 0.f;
                    for (int sh = 0; sh < scale; ++sh)
                        for (int sw = 0; sw < scale; ++sw)
                            s += grad(n, c * oH * oW + (h*scale+sh) * oW + (w*scale+sw));
                    grad_in(n, c * H * W + h * W + w) = s;
                }
    return {grad_in};
}

VarPtr nearest_upsample2d_op(VarPtr input,
                              int N, int C, int H, int W, int scale) {
    if (input->is_cuda()) {
        throw std::runtime_error("nearest_upsample2d_op: CUDA inputs are not supported yet");
    }
    auto fn = std::make_shared<NearestUpsample2dFn>();
    fn->N = N; fn->C = C; fn->H = H; fn->W = W; fn->scale = scale;

    Mats in_data = {input->data};
    auto out = Var::make(fn->forward(in_data));
    out->set_shape({N, C, H * scale, W * scale});
    out->parents = {input};
    out->back_fn = [fn, ins = std::vector<VarPtr>{input},
                    wp = std::weak_ptr<Var>(out)]() {
        auto self = wp.lock();
        auto gs = fn->backward(self->grad);
        for (size_t i = 0; i < ins.size(); ++i) ins[i]->grad += gs[i];
    };
    return out;
}

VarPtr NearestUpsample2d::forward(VarPtr x, int H, int W_) {
    int N = x->data.rows();
    int C = x->data.cols() / (H * W_);
    return nearest_upsample2d_op(x, N, C, H, W_, scale);
}

VarPtr NearestUpsample2d::forward(VarPtr x) {
    assert(x->is4d() && "NearestUpsample2d::forward(x) requires x shape (N,C,H,W)");
    return forward(x, static_cast<int>(x->dim(2)), static_cast<int>(x->dim(3)));
}

} // namespace ag
