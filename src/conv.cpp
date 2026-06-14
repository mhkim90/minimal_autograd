// src/conv.cpp — im2col / col2im, Conv2dFn / MaxPool2dFn, factories.

#include "autograd/conv.h"

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
    auto fn = std::make_shared<Conv2dFn>();
    fn->N = N; fn->C = C; fn->H = H; fn->W = W;
    fn->out_ch = weight->data.rows();
    fn->kH = kH; fn->kW = kW;
    fn->stride = stride; fn->pad = pad;
    fn->oH = (H + 2 * pad - kH) / stride + 1;
    fn->oW = (W + 2 * pad - kW) / stride + 1;

    Mats in_data = {input->data, weight->data, bias->data};
    auto out = Var::make(fn->forward(in_data));
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
    auto fn = std::make_shared<MaxPool2dFn>();
    fn->N = N; fn->C = C; fn->H = H; fn->W = W;
    fn->kH = kH; fn->kW = kW; fn->stride = stride;
    fn->oH = (H - kH) / stride + 1;
    fn->oW = (W - kW) / stride + 1;

    Mats in_data = {input->data};
    auto out = Var::make(fn->forward(in_data));
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

VarPtr MaxPool2d::forward(VarPtr x, int H, int W_) {
    int N = x->data.rows();
    int C = x->data.cols() / (H * W_);
    return maxpool2d_op(x, N, C, H, W_, kH, kW, stride);
}

} // namespace ag
