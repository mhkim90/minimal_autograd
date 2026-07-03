// test/test_conv.cpp — im2col/col2im, Conv2dFn, MaxPool2dFn, end-to-end.

#include "autograd.h"
#include "grad_check.h"

#include <cstdio>
#include <cassert>
#include <cmath>

using namespace ag;

static int passed = 0, failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++passed; std::printf("  ok   %s\n", msg); } \
    else      { ++failed; std::printf("  FAIL %s\n", msg); } \
} while (0)

int main() {
    std::printf("== test_conv ==\n");

    // -----------------------------------------------------------------
    // conv2d forward matches naive nested-loop reference
    // -----------------------------------------------------------------
    {
        const int N = 1, C = 2, H = 3, W = 3, K = 2;
        const int out_ch = 1, stride = 1, pad = 0;
        const int oH = H - K + 1, oW = W - K + 1;

        Mat input  = Mat::Random(N, C * H * W);
        Mat weight = Mat::Random(out_ch, C * K * K);
        Mat bias   = Mat::Random(1, out_ch);

        // Our op
        auto y = conv2d_op(Var::make(input), Var::make(weight), Var::make(bias),
                           N, C, H, W, K, K, stride, pad);

        // Naive reference
        Mat ref = Mat::Zero(N, out_ch * oH * oW);
        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < out_ch; ++oc) {
                for (int oh = 0; oh < oH; ++oh) {
                    for (int ow = 0; ow < oW; ++ow) {
                        float s = bias(0, oc);
                        for (int c = 0; c < C; ++c) {
                            for (int kh = 0; kh < K; ++kh) {
                                for (int kw = 0; kw < K; ++kw) {
                                    int ih = oh + kh;
                                    int iw = ow + kw;
                                    s += weight(oc, c * K * K + kh * K + kw)
                                       * input(n, c * H * W + ih * W + iw);
                                }
                            }
                        }
                        ref(n, oc * oH * oW + oh * oW + ow) = s;
                    }
                }
            }
        }
        float diff = (y->data - ref).cwiseAbs().maxCoeff();
        CHECK(diff < 1e-4f, "Conv2d forward matches naive reference");
    }

    // -----------------------------------------------------------------
    // im2col/col2im round-trip (non-overlapping patches)
    // -----------------------------------------------------------------
    {
        Mat input(1, 16);
        for (int i = 0; i < 16; ++i) input(0, i) = static_cast<float>(i);
        Mat col = im2col(input, 1, 1, 4, 4, 2, 2, 2, 0);
        CHECK(col.rows() == 4 && col.cols() == 4, "im2col shape: stride=kH");

        Mat recovered = col2im(col, 1, 1, 4, 4, 2, 2, 2, 0);
        float diff = (recovered - input).cwiseAbs().maxCoeff();
        CHECK(diff < 1e-5f, "im2col/col2im round-trip (non-overlapping)");
    }

    // -----------------------------------------------------------------
    // col2im accumulates overlapping patches
    // -----------------------------------------------------------------
    {
        // 1 image, 1 channel, 3x3 input, 2x2 kernel, stride 1.
        // Each input pixel is hit by multiple patches; col2im must use +=.
        Mat col = Mat::Ones(4, 4);  // 4 patches x 4 output positions... wait,
                                    // C*kH*kW=4, N*oH*oW=4 for N=1, oH=oW=2.
        Mat out = col2im(col, 1, 1, 3, 3, 2, 2, 1, 0);

        // Hits: corners=1, edges=2, center=4.
        Mat expected = (Mat(1, 9) <<
            1, 2, 1,
            2, 4, 2,
            1, 2, 1).finished();
        float diff = (out - expected).cwiseAbs().maxCoeff();
        CHECK(diff < 1e-5f, "col2im accumulates overlapping patches");
    }

    // -----------------------------------------------------------------
    // grad_check on Conv2dFn
    // -----------------------------------------------------------------
    {
        const int N = 1, C = 1, H = 4, W = 4, K = 2;
        auto x = Var::make(Mat::Random(N, C * H * W));
        auto w = Var::make(Mat::Random(2, C * K * K));
        auto b = Var::make(Mat::Random(1, 2));

        CHECK(grad_check([&](VarPtr v) {
                  return sum(conv2d_op(v, w, b, N, C, H, W, K, K, 1, 0));
              }, x, 1e-3f, 5e-2f),
              "grad_check: Conv2dFn dInput");
        CHECK(grad_check([&](VarPtr v) {
                  return sum(conv2d_op(x, v, b, N, C, H, W, K, K, 1, 0));
              }, w, 1e-3f, 5e-2f),
              "grad_check: Conv2dFn dWeight");
        CHECK(grad_check([&](VarPtr v) {
                  return sum(conv2d_op(x, w, v, N, C, H, W, K, K, 1, 0));
              }, b, 1e-3f, 5e-2f),
              "grad_check: Conv2dFn dBias");
    }

    // -----------------------------------------------------------------
    // grad_check on MaxPool2dFn
    // -----------------------------------------------------------------
    {
        const int N = 1, C = 2, H = 4, W = 4, K = 2;
        auto x = Var::make(Mat::Random(N, C * H * W));
        CHECK(grad_check([&](VarPtr v) {
                  return sum(maxpool2d_op(v, N, C, H, W, K, K, 2));
              }, x, 1e-3f, 5e-2f),
              "grad_check: MaxPool2dFn");
    }

    // -----------------------------------------------------------------
    // End-to-end: Conv -> MaxPool -> Reshape -> Linear forward + backward
    // -----------------------------------------------------------------
    {
        const int N = 2, C = 1, H = 8, W = 8;
        Conv2d conv(C, 4, 3, 3, 1, 0);   // -> 4 x 6 x 6
        MaxPool2d pool(2, 2, 2);          // -> 4 x 3 x 3
        Linear lin(4 * 3 * 3, 2);

        Mat x = Mat::Random(N, C * H * W);
        auto xv = Var::make(x);
        auto h1 = conv.forward(xv, H, W);
        auto h2 = pool.forward(h1, 6, 6);
        // h2 is already (N, 4*3*3); reshape confirms the API.
        auto flat = reshape(h2, N, 4 * 3 * 3);
        auto out = lin.forward(flat);

        auto loss = sum(out);
        for (auto& p : conv.parameters()) p->grad.setZero();
        for (auto& p : lin.parameters())  p->grad.setZero();
        loss->backward();

        bool all_ok =
            conv.W->grad.cwiseAbs().maxCoeff() > 0.f &&
            conv.b->grad.cwiseAbs().maxCoeff() > 0.f &&
            lin.W->grad.cwiseAbs().maxCoeff()  > 0.f &&
            lin.b->grad.cwiseAbs().maxCoeff()  > 0.f;
        CHECK(all_ok, "Conv+Pool+Linear: every parameter got a non-zero gradient");
    }

    // -----------------------------------------------------------------
    // 4D logical shape: Conv/Pool infer H,W and preserve output metadata
    // -----------------------------------------------------------------
    {
        const int N = 2, C = 1, H = 8, W = 8;
        Conv2d conv(C, 4, 3, 3, 1, 0);   // -> 4 x 6 x 6
        MaxPool2d pool(2, 2, 2);          // -> 4 x 3 x 3

        auto x = Var::make4d(Mat::Random(N, C * H * W), N, C, H, W);
        auto y = conv.forward(x);
        CHECK(y->is4d(), "Conv2d inferred 4D input and returned 4D metadata");
        CHECK(y->dim(0) == N && y->dim(1) == 4 && y->dim(2) == 6 && y->dim(3) == 6,
              "Conv2d output shape metadata is N,OC,oH,oW");

        auto z = pool.forward(y);
        CHECK(z->is4d(), "MaxPool2d inferred 4D input and returned 4D metadata");
        CHECK(z->dim(0) == N && z->dim(1) == 4 && z->dim(2) == 3 && z->dim(3) == 3,
              "MaxPool2d output shape metadata is N,C,oH,oW");
    }

    std::printf("--\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
