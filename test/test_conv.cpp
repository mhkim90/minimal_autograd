// test_conv.cpp — public NCHW packing/overlap, convolution, pooling and composition.

#include "autograd/core/module.h"
#include "grad_check.h"

#include <cstdio>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace ag;
using namespace ag_test;

static int passed = 0, failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++passed; std::printf("  ok   %s\n", msg); } \
    else      { ++failed; std::printf("  FAIL %s\n", msg); } \
} while (0)

void test_conv_packing_and_overlap() {
    auto w = Variable(Tensor::ones(Shape{1,1,2,2}), true);
    auto b = Variable(Tensor::zeros(Shape{1}), true);
    auto nonoverlap = Variable(Tensor::ones(Shape{1,1,4,4}), true);
    auto packed = conv2d(nonoverlap, w, b, 2, 0);
    CHECK((packed.value().shape() == Shape{1,1,2,2}), "public conv packing shape, stride=kernel");
    sum(packed).backward();
    CHECK(max_diff(nonoverlap.grad(), Tensor::ones(Shape{1,1,4,4})) < 1e-5f,
          "non-overlapping patch coverage is exactly one");
    auto x = Variable(Tensor::ones(Shape{1,1,3,3}), true);
    auto y = conv2d(x, w, b, 1, 0);
    CHECK(max_diff(y.value(), filled(y.value().shape(), 4.f)) < 1e-5f,
          "unit 2x2 filter on 3x3 ones gives four in every output");
    sum(y).backward();
    CHECK(max_diff(x.grad(), tensor(Shape{1,1,3,3}, {1,2,1,2,4,2,1,2,1})) < 1e-5f,
          "overlapping patch gradients: corners1 edges2 center4");
}

int main() {
    std::printf("== test_conv ==\n");

    // -----------------------------------------------------------------
    // conv2d forward matches naive nested-loop reference
    // -----------------------------------------------------------------
    {
        const int N = 1, C = 2, H = 3, W = 3, K = 2;
        const int out_ch = 1, stride = 1, pad = 0;
        const int oH = H - K + 1, oW = W - K + 1;

        Tensor input  = random_tensor(Shape{N, C, H, W});
        Tensor weight = random_tensor(Shape{out_ch, C, K, K});
        Tensor bias   = random_tensor(Shape{out_ch});

        // Our op
        auto y = conv2d(Variable(input, true), Variable(weight, true), Variable(bias, true), stride, pad);

        // Naive reference
        std::vector<float> ref(N*out_ch*oH*oW, 0.f);
        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < out_ch; ++oc) {
                for (int oh = 0; oh < oH; ++oh) {
                    for (int ow = 0; ow < oW; ++ow) {
                        float s = at(bias, oc);
                        for (int c = 0; c < C; ++c) {
                            for (int kh = 0; kh < K; ++kh) {
                                for (int kw = 0; kw < K; ++kw) {
                                    int ih = oh + kh;
                                    int iw = ow + kw;
                                    s += at(weight, oc, c * K * K + kh * K + kw)
                                       * at(input, n, c * H * W + ih * W + iw);
                                }
                            }
                        }
                        ref[((n*out_ch+oc)*oH+oh)*oW+ow] = s;
                    }
                }
            }
        }
        float diff = max_diff(y.value(), tensor(y.value().shape(), ref));
        CHECK(diff < 1e-4f, "nn::Conv2d forward matches naive reference");
    }

    test_conv_packing_and_overlap();

    // -----------------------------------------------------------------
    // grad_check on Conv2dFn
    // -----------------------------------------------------------------
    {
        const int N = 1, C = 1, H = 4, W = 4, K = 2;
        auto x = Variable(random_tensor(Shape{N, C, H, W}), true);
        auto w = Variable(random_tensor(Shape{2, C, K, K}), true);
        auto b = Variable(random_tensor(Shape{2}), true);

        CHECK(grad_check([&](Variable v) {
                  return sum(conv2d(v, w, b, 1, 0));
              }, x, 1e-3f, 5e-2f),
              "grad_check: Conv2dFn dInput");
        CHECK(grad_check([&](Variable v) {
                  return sum(conv2d(x, v, b, 1, 0));
              }, w, 1e-3f, 5e-2f),
              "grad_check: Conv2dFn dWeight");
        CHECK(grad_check([&](Variable v) {
                  return sum(conv2d(x, w, v, 1, 0));
              }, b, 1e-3f, 5e-2f),
              "grad_check: Conv2dFn dBias");
    }

    // -----------------------------------------------------------------
    // grad_check on MaxPool2dFn
    // -----------------------------------------------------------------
    {
        const int N = 1, C = 2, H = 4, W = 4, K = 2;
        auto x = Variable(random_tensor(Shape{N, C, H, W}), true);
        CHECK(grad_check([&](Variable v) {
                  return sum(max_pool2d(v, K, K, 2));
              }, x, 1e-3f, 5e-2f),
              "grad_check: MaxPool2dFn");
    }

    // -----------------------------------------------------------------
    // End-to-end: Conv -> MaxPool -> Reshape -> nn::Linear forward + backward
    // -----------------------------------------------------------------
    {
        const int N = 2, C = 1, H = 8, W = 8;
        nn::Conv2d conv(C, 4, 3, 3, 1, 0);   // -> 4 x 6 x 6
        nn::MaxPool2d pool(2, 2, 2);          // -> 4 x 3 x 3
        nn::Linear lin(4 * 3 * 3, 2);

        CHECK(max_abs(conv.weight().value()) <= std::sqrt(1.f/(C*3*3)), "Conv initializer Kaiming bound");
        CHECK(max_abs(conv.bias().value()) == 0.f, "Conv initializer zero bias");
        Tensor x = random_tensor(Shape{N, C, H, W});
        auto xv = Variable(x, true);
        auto h1 = conv.forward(xv);
        auto h2 = pool.forward(h1);
        // Explicitly flatten NCHW before the Linear boundary.
        auto flat = reshape(h2, N, 4 * 3 * 3);
        auto out = lin.forward(flat);

        auto loss = sum(out);
        for (auto& p : conv.parameters()) p.zero_grad();
        for (auto& p : lin.parameters())  p.zero_grad();
        loss.backward();

        bool all_ok =
            max_abs(conv.weight().grad()) > 0.f &&
            max_abs(conv.bias().grad()) > 0.f &&
            max_abs(lin.weight().grad())  > 0.f &&
            max_abs(lin.bias().grad())  > 0.f;
        CHECK(all_ok, "Conv+Pool+nn::Linear: every parameter got a non-zero gradient");
    }

    // -----------------------------------------------------------------
    // 4D logical shape: Conv/Pool infer H,W and preserve output metadata
    // -----------------------------------------------------------------
    {
        const int N = 2, C = 1, H = 8, W = 8;
        nn::Conv2d conv(C, 4, 3, 3, 1, 0);   // -> 4 x 6 x 6
        nn::MaxPool2d pool(2, 2, 2);          // -> 4 x 3 x 3

        auto x = Variable(random_tensor(Shape{N, C, H, W}), true);
        auto y = conv.forward(x);
        CHECK(y.value().shape().rank() == 4, "nn::Conv2d inferred 4D input and returned 4D metadata");
        CHECK(y.value().shape()[0] == N && y.value().shape()[1] == 4 && y.value().shape()[2] == 6 && y.value().shape()[3] == 6,
              "nn::Conv2d output shape metadata is N,OC,oH,oW");

        auto z = pool.forward(y);
        CHECK(z.value().shape().rank() == 4, "nn::MaxPool2d inferred 4D input and returned 4D metadata");
        CHECK(z.value().shape()[0] == N && z.value().shape()[1] == 4 && z.value().shape()[2] == 3 && z.value().shape()[3] == 3,
              "nn::MaxPool2d output shape metadata is N,C,oH,oW");
    }

    // -----------------------------------------------------------------
    // Release builds: Conv modules validate channel/shape mismatches
    // -----------------------------------------------------------------
    {
        bool conv_threw = false;
        try {
            nn::Conv2d conv(2, 4, 3, 3, 1, 0);
            (void)conv.forward(Variable(random_tensor(Shape{1,1,5,5}), true));
        } catch (const std::invalid_argument&) {
            conv_threw = true;
        }
        CHECK(conv_threw, "nn::Conv2d rejects 4D input channel mismatch without assert");

        bool depthwise_threw = false;
        try {
            nn::DepthwiseConv2d conv(2, 3, 3, 1, 0);
            (void)conv.forward(Variable(random_tensor(Shape{1,1,5,5}), true));
        } catch (const std::invalid_argument&) {
            depthwise_threw = true;
        }
        CHECK(depthwise_threw,
              "nn::DepthwiseConv2d rejects 4D input channel mismatch without assert");

        bool explicit_threw = false;
        try {
            nn::Conv2d conv(2, 4, 3, 3, 1, 0);
            (void)conv.forward(Variable(random_tensor(Shape{1, 25}), true));
        } catch (const std::invalid_argument&) {
            explicit_threw = true;
        }
        CHECK(explicit_threw, "nn::Conv2d rejects non-NCHW input");

        bool depthwise_explicit_threw = false;
        try {
            nn::DepthwiseConv2d conv(2, 3, 3, 1, 0);
            (void)conv.forward(Variable(random_tensor(Shape{1, 25}), true));
        } catch (const std::invalid_argument&) {
            depthwise_explicit_threw = true;
        }
        CHECK(depthwise_explicit_threw,
              "nn::DepthwiseConv2d rejects non-NCHW input");

        bool maxpool_shape_threw = false;
        try {
            nn::MaxPool2d pool(2, 2, 2);
            (void)pool.forward(Variable(random_tensor(Shape{1, 25}), true));
        } catch (const std::invalid_argument&) {
            maxpool_shape_threw = true;
        }
        CHECK(maxpool_shape_threw, "nn::MaxPool2d rejects non-NCHW input");

        bool avgpool_shape_threw = false;
        try {
            nn::AvgPool2d pool(2, 2, 2);
            (void)pool.forward(Variable(random_tensor(Shape{1, 25}), true));
        } catch (const std::invalid_argument&) {
            avgpool_shape_threw = true;
        }
        CHECK(avgpool_shape_threw, "nn::AvgPool2d rejects non-NCHW input");

        bool upsample_shape_threw = false;
        try {
            nn::NearestUpsample2d upsample(2);
            (void)upsample.forward(Variable(random_tensor(Shape{1, 25}), true));
        } catch (const std::invalid_argument&) {
            upsample_shape_threw = true;
        }
        CHECK(upsample_shape_threw,
              "nn::NearestUpsample2d rejects non-NCHW input");

        bool conv_geometry_threw = false;
        try {
            nn::Conv2d conv(1, 1, 2, 2, 2, 0);
            (void)conv.forward(Variable(random_tensor(Shape{1,1,5,5}), true));
        } catch (const std::invalid_argument&) {
            conv_geometry_threw = true;
        }
        CHECK(conv_geometry_threw, "nn::Conv2d rejects invalid kernel/stride geometry");
    }

    std::printf("--\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
