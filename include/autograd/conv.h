#pragma once
// conv.h — im2col, col2im, Conv2d, MaxPool2d.
//
// Layout convention (matches the plan):
//
//   Var::data is always a 2D Eigen::MatrixXf. For conv layers, a 4D tensor
//   (N, C, H, W) is flattened to Mat(N, C*H*W), where each row holds one
//   image with channels stored contiguously. Shape metadata (C, H, W) is
//   passed explicitly into the op/module and not encoded in the Var.
//
// im2col: input (N, C*H*W)   → columns (C*kH*kW, N*oH*oW).
//         each column is one receptive field.
//
// col2im: the inverse, with + (not =) when patches overlap.

#include "autograd/ops.h"
#include "autograd/module.h"
#include "autograd/tensor.h"
#include <cmath>
#include <vector>
#include <cassert>

namespace ag {

// im2col — pure function.
//
// input:  (N, C*H*W)        N images, channels contiguous.
// output: (C*kH*kW, N*oH*oW)
Mat im2col(const Mat& input,
           int N, int C, int H, int W,
           int kH, int kW,
           int stride, int pad);

// col2im — pure function.
//
// col:    (C*kH*kW, N*oH*oW)
// output: (N, C*H*W) — overlapping patches ACCUMULATE (use +=).
Mat col2im(const Mat& col,
           int N, int C, int H, int W,
           int kH, int kW,
           int stride, int pad);

// --- differentiable conv op ---

struct Conv2dFn : Function {
    int N = 0, C = 0, H = 0, W = 0;
    int out_ch = 0, kH = 0, kW = 0;
    int stride = 0, pad = 0;
    int oH = 0, oW = 0;

    Mat forward(const Mats& in) override;
    Mats backward(const Mat& grad) override;
};

// Factory: build the conv op directly (it needs constructor args).
// in[0]: input  (N, C*H*W)
// in[1]: weight (out_ch, C*kH*kW)
// in[2]: bias   (1, out_ch)
VarPtr conv2d_op(VarPtr input, VarPtr weight, VarPtr bias,
                 int N, int C, int H, int W,
                 int kH, int kW, int stride, int pad);

struct MaxPool2dFn : Function {
    int N = 0, C = 0, H = 0, W = 0, kH = 0, kW = 0, stride = 0;
    int oH = 0, oW = 0;

    Mat forward(const Mats& in) override;
    Mats backward(const Mat& grad) override;
};

VarPtr maxpool2d_op(VarPtr input,
                    int N, int C, int H, int W,
                    int kH, int kW, int stride);

struct Conv2d : Module {
    VarPtr W;   // (out_ch, in_ch * kH * kW)
    VarPtr b;   // (1, out_ch)

    int in_ch, out_ch, kH, kW, stride, pad;

    Conv2d(int in_ch, int out_ch, int kH, int kW,
           int stride = 1, int pad = 0);

    // x: (N, in_ch * H * W). Caller passes H, W.
    VarPtr forward(VarPtr x, int H, int W);

    // Uses x logical shape (N, C, H, W), if present.
    VarPtr forward(VarPtr x) override;

    std::vector<VarPtr> parameters() override { return {W, b}; }
};

struct MaxPool2d : Module {
    int kH, kW, stride;

    // stride < 0 → stride = kH (default non-overlapping pool).
    MaxPool2d(int kH, int kW, int stride = -1)
        : kH(kH), kW(kW), stride(stride < 0 ? kH : stride) {}

    VarPtr forward(VarPtr x, int H, int W);

    VarPtr forward(VarPtr x) override;

    std::vector<VarPtr> parameters() override { return {}; }
};

// --- AvgPool2d ---

struct AvgPool2dFn : Function {
    int N = 0, C = 0, H = 0, W = 0, kH = 0, kW = 0, stride = 0;
    int oH = 0, oW = 0;

    Mat forward(const Mats& in) override;
    Mats backward(const Mat& grad) override;
};

VarPtr avgpool2d_op(VarPtr input,
                    int N, int C, int H, int W,
                    int kH, int kW, int stride);

struct AvgPool2d : Module {
    int kH, kW, stride;

    AvgPool2d(int kH, int kW, int stride = -1)
        : kH(kH), kW(kW), stride(stride < 0 ? kH : stride) {}

    VarPtr forward(VarPtr x, int H, int W);

    VarPtr forward(VarPtr x) override;

    std::vector<VarPtr> parameters() override { return {}; }
};

// --- NearestUpsample2d ---

struct NearestUpsample2dFn : Function {
    int N = 0, C = 0, H = 0, W = 0, scale = 0;

    Mat forward(const Mats& in) override;
    Mats backward(const Mat& grad) override;
};

VarPtr nearest_upsample2d_op(VarPtr input,
                              int N, int C, int H, int W, int scale);

struct NearestUpsample2d : Module {
    int scale;
    explicit NearestUpsample2d(int scale) : scale(scale) {}

    VarPtr forward(VarPtr x, int H, int W);

    VarPtr forward(VarPtr x) override;

    std::vector<VarPtr> parameters() override { return {}; }
};

// --- DepthwiseConv2d (groups = in_channels = out_channels) ---

struct DepthwiseConv2dFn : Function {
    int N = 0, C = 0, H = 0, W = 0;
    int kH = 0, kW = 0, stride = 0, pad = 0;
    int oH = 0, oW = 0;

    Mat forward(const Mats& in) override;
    Mats backward(const Mat& grad) override;
};

// in[0]: input  (N, C*H*W)
// in[1]: weight (C, kH*kW)    — one filter per input channel
// in[2]: bias   (1, C)
VarPtr depthwise_conv2d_op(VarPtr input, VarPtr weight, VarPtr bias,
                            int N, int C, int H, int W,
                            int kH, int kW, int stride, int pad);

struct DepthwiseConv2d : Module {
    VarPtr W;   // (C, kH*kW)
    VarPtr b;   // (1, C)

    int channels, kH, kW, stride, pad;

    DepthwiseConv2d(int channels, int kH, int kW,
                    int stride = 1, int pad = 0);

    VarPtr forward(VarPtr x, int H, int W);

    VarPtr forward(VarPtr x) override;

    std::vector<VarPtr> parameters() override { return {W, b}; }
};

} // namespace ag
