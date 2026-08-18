#include "detail/tensor_ops.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ag {
namespace detail {
namespace cpu_ops {

namespace {

inline int64_t nchw_output_extent(int64_t input, int64_t kernel,
                                  int64_t stride, int64_t pad,
                                  const char* what) {
    const int64_t span = input + 2 * pad - kernel;
    if (span < 0) {
        std::ostringstream os;
        os << what << ": input plus 2*pad smaller than kernel ("
           << input << " + 2*" << pad << " < " << kernel << ")";
        throw std::invalid_argument(os.str());
    }
    if (span % stride != 0) {
        std::ostringstream os;
        os << what << ": output extent non-integer (span " << span
           << " not divisible by stride " << stride << ")";
        throw std::invalid_argument(os.str());
    }
    return span / stride + 1;
}

inline Tensor tensor_im2col_nchw(const Tensor& input,
                                 int kH, int kW,
                                 int stride, int pad) {
    const Shape& s = input.shape();
    const int N = static_cast<int>(s[0]);
    const int C = static_cast<int>(s[1]);
    const int H = static_cast<int>(s[2]);
    const int W = static_cast<int>(s[3]);
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "im2col"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "im2col"));
    const int K_flat = C * kH * kW;
    const int P_flat = oH * oW;

    Tensor col = Tensor::empty(Shape({N, K_flat, P_flat}), input.device());
    if (col.elements() == 0) return col;

    std::vector<float> in_data(input.elements()), col_data(col.elements());
    input.copy_to_host(in_data.data(), in_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int col_stride_n = K_flat * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    const int k = c * kH * kW + kh * kW + kw;
                    for (int oh = 0; oh < oH; ++oh) {
                        const int ih = oh * stride + kh - pad;
                        for (int ow = 0; ow < oW; ++ow) {
                            const int iw = ow * stride + kw - pad;
                            const int p = oh * oW + ow;
                            float v = 0.f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                v = in_data[n * in_stride_n
                                          + c * in_stride_c
                                          + ih * in_stride_h
                                          + iw * in_stride_w];
                            }
                            col_data[n * col_stride_n
                                   + k * col_stride_k
                                   + p * col_stride_p] = v;
                        }
                    }
                }
            }
        }
    }
    col.copy_from_host(col_data.data(), col_data.size());
    return col;
}

inline Tensor tensor_col2im_nchw(const Tensor& col,
                                 int N, int C, int H, int W,
                                 int kH, int kW,
                                 int stride, int pad) {
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "col2im"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "col2im"));
    const int K_flat = C * kH * kW;
    const int P_flat = oH * oW;

    Tensor out = Tensor::zeros(Shape({N, C, H, W}), col.device());
    if (out.elements() == 0) return out;

    std::vector<float> col_data(col.elements()), out_data(out.elements());
    col.copy_to_host(col_data.data(), col_data.size());
    out.copy_to_host(out_data.data(), out_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int col_stride_n = K_flat * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    const int k = c * kH * kW + kh * kW + kw;
                    for (int oh = 0; oh < oH; ++oh) {
                        const int ih = oh * stride + kh - pad;
                        for (int ow = 0; ow < oW; ++ow) {
                            const int iw = ow * stride + kw - pad;
                            const int p = oh * oW + ow;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            const int in_off = n * in_stride_n
                                             + c * in_stride_c
                                             + ih * in_stride_h
                                             + iw * in_stride_w;
                            const int col_off = n * col_stride_n
                                              + k * col_stride_k
                                              + p * col_stride_p;
                            out_data[in_off] += col_data[col_off];
                        }
                    }
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    return out;
}

}  // namespace

Tensor tensor_conv2d_nchw_forward(const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& bias,
                                  int stride, int pad,
                                  Tensor& saved_col) {
    const Shape& in_s = input.shape();
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);
    const Shape& w_s = weight.shape();
    const int OC = static_cast<int>(w_s[0]);
    const int kH = static_cast<int>(w_s[2]);
    const int kW = static_cast<int>(w_s[3]);
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "conv2d"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "conv2d"));
    const int K_flat = C * kH * kW;
    const int P_flat = oH * oW;

    saved_col = tensor_im2col_nchw(input, kH, kW, stride, pad);
    Tensor out = Tensor::empty(Shape({N, OC, oH, oW}), input.device());
    if (out.elements() == 0) return out;

    std::vector<float> w_data(weight.elements());
    std::vector<float> b_data(bias.elements());
    std::vector<float> col_data(saved_col.elements());
    std::vector<float> out_data(out.elements());
    weight.copy_to_host(w_data.data(), w_data.size());
    bias.copy_to_host(b_data.data(), b_data.size());
    saved_col.copy_to_host(col_data.data(), col_data.size());

    const int w_stride_oc = C * kH * kW;
    const int w_stride_c  = kH * kW;
    const int w_stride_kh = kW;
    const int w_stride_kw = 1;
    const int col_stride_n = K_flat * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;
    const int out_stride_n = OC * oH * oW;
    const int out_stride_oc = oH * oW;
    const int out_stride_oh = oW;
    const int out_stride_ow = 1;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    const int p = oh * oW + ow;
                    float s = b_data[oc];
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                const int k = c * kH * kW + kh * kW + kw;
                                const int w_off = oc * w_stride_oc
                                                 + c * w_stride_c
                                                 + kh * w_stride_kh
                                                 + kw * w_stride_kw;
                                const int col_off = n * col_stride_n
                                                  + k * col_stride_k
                                                  + p * col_stride_p;
                                s += w_data[w_off] * col_data[col_off];
                            }
                        }
                    }
                    out_data[n * out_stride_n
                           + oc * out_stride_oc
                           + oh * out_stride_oh
                           + ow * out_stride_ow] = s;
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    return out;
}

Tensor tensor_conv2d_nchw_backward_input(
        const Tensor& g,
        const Tensor& weight,
        int N, int C, int H, int W,
        int kH, int kW, int stride, int pad) {
    const int OC = static_cast<int>(weight.shape()[0]);
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "conv2d_backward_input"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "conv2d_backward_input"));
    const int K_flat = C * kH * kW;
    const int P_flat = oH * oW;

    Tensor d_col = Tensor::zeros(Shape({N, K_flat, P_flat}), g.device());
    if (d_col.elements() == 0) {
        return Tensor::zeros(Shape({N, C, H, W}), g.device());
    }

    std::vector<float> g_data(g.elements());
    std::vector<float> w_data(weight.elements());
    std::vector<float> dc_data(d_col.elements());
    g.copy_to_host(g_data.data(), g_data.size());
    weight.copy_to_host(w_data.data(), w_data.size());

    const int g_stride_n = OC * oH * oW;
    const int g_stride_oc = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;
    const int w_stride_oc = C * kH * kW;
    const int w_stride_c  = kH * kW;
    const int w_stride_kh = kW;
    const int w_stride_kw = 1;
    const int dc_stride_n = K_flat * P_flat;
    const int dc_stride_k = P_flat;
    const int dc_stride_p = 1;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    const int p = oh * oW + ow;
                    const float g_val = g_data[n * g_stride_n
                                            + oc * g_stride_oc
                                            + oh * g_stride_oh
                                            + ow * g_stride_ow];
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                const int k = c * kH * kW + kh * kW + kw;
                                const int w_off = oc * w_stride_oc
                                                 + c * w_stride_c
                                                 + kh * w_stride_kh
                                                 + kw * w_stride_kw;
                                const int dc_off = n * dc_stride_n
                                                  + k * dc_stride_k
                                                  + p * dc_stride_p;
                                dc_data[dc_off] += g_val * w_data[w_off];
                            }
                        }
                    }
                }
            }
        }
    }
    d_col.copy_from_host(dc_data.data(), dc_data.size());
    return tensor_col2im_nchw(d_col, N, C, H, W, kH, kW, stride, pad);
}

Tensor tensor_conv2d_nchw_backward_weight(
        const Tensor& g, const Tensor& col, const Shape& w_shape) {
    const int OC = static_cast<int>(w_shape[0]);
    const int C  = static_cast<int>(w_shape[1]);
    const int kH = static_cast<int>(w_shape[2]);
    const int kW = static_cast<int>(w_shape[3]);
    const int N  = static_cast<int>(col.shape()[0]);
    const int P_flat = static_cast<int>(col.shape()[2]);
    const int oH = static_cast<int>(g.shape()[2]);
    const int oW = static_cast<int>(g.shape()[3]);

    Tensor d_w = Tensor::zeros(w_shape, g.device());
    if (d_w.elements() == 0) return d_w;

    std::vector<float> g_data(g.elements());
    std::vector<float> col_data(col.elements());
    std::vector<float> dw_data(d_w.elements());
    g.copy_to_host(g_data.data(), g_data.size());
    col.copy_to_host(col_data.data(), col_data.size());

    const int g_stride_n = OC * oH * oW;
    const int g_stride_oc = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;
    const int col_stride_n = (C * kH * kW) * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;
    const int dw_stride_oc = C * kH * kW;
    const int dw_stride_c  = kH * kW;
    const int dw_stride_kh = kW;
    const int dw_stride_kw = 1;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    const int p = oh * oW + ow;
                    const float g_val = g_data[n * g_stride_n
                                            + oc * g_stride_oc
                                            + oh * g_stride_oh
                                            + ow * g_stride_ow];
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                const int k = c * kH * kW + kh * kW + kw;
                                const int col_off = n * col_stride_n
                                                  + k * col_stride_k
                                                  + p * col_stride_p;
                                const int dw_off = oc * dw_stride_oc
                                                 + c * dw_stride_c
                                                 + kh * dw_stride_kh
                                                 + kw * dw_stride_kw;
                                dw_data[dw_off] += g_val * col_data[col_off];
                            }
                        }
                    }
                }
            }
        }
    }
    d_w.copy_from_host(dw_data.data(), dw_data.size());
    return d_w;
}

Tensor tensor_conv2d_nchw_backward_bias(const Tensor& g) {
    const Shape& gs = g.shape();
    const int N = static_cast<int>(gs[0]);
    const int OC = static_cast<int>(gs[1]);
    const int oH = static_cast<int>(gs[2]);
    const int oW = static_cast<int>(gs[3]);

    Tensor d_b = Tensor::zeros(Shape({OC}), g.device());
    if (d_b.elements() == 0) return d_b;

    std::vector<float> g_data(g.elements());
    std::vector<float> db_data(d_b.elements());
    g.copy_to_host(g_data.data(), g_data.size());

    const int g_stride_n = OC * oH * oW;
    const int g_stride_oc = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;

    for (int oc = 0; oc < OC; ++oc) {
        double s = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    s += static_cast<double>(
                        g_data[n * g_stride_n
                             + oc * g_stride_oc
                             + oh * g_stride_oh
                             + ow * g_stride_ow]);
                }
            }
        }
        db_data[oc] = static_cast<float>(s);
    }
    d_b.copy_from_host(db_data.data(), db_data.size());
    return d_b;
}

Tensor tensor_maxpool2d_nchw_forward(const Tensor& input,
                                     int kH, int kW,
                                     int stride,
                                     Tensor& saved_mask) {
    const Shape& s = input.shape();
    const int N = static_cast<int>(s[0]);
    const int C = static_cast<int>(s[1]);
    const int H = static_cast<int>(s[2]);
    const int W = static_cast<int>(s[3]);
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, 0, "max_pool2d"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, 0, "max_pool2d"));

    Tensor out = Tensor::empty(Shape({N, C, oH, oW}), input.device());
    saved_mask = Tensor::empty(Shape({N, C, oH, oW}), input.device());
    if (out.elements() == 0) return out;

    std::vector<float> in_data(input.elements());
    std::vector<float> out_data(out.elements());
    std::vector<float> mask_data(saved_mask.elements());
    input.copy_to_host(in_data.data(), in_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int out_stride_n = C * oH * oW;
    const int out_stride_c = oH * oW;
    const int out_stride_oh = oW;
    const int out_stride_ow = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float best_v = -std::numeric_limits<float>::infinity();
                    int best_k = 0;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = oh * stride + kh;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = ow * stride + kw;
                            const float v =
                                in_data[n * in_stride_n
                                      + c * in_stride_c
                                      + ih * in_stride_h
                                      + iw * in_stride_w];
                            if (v > best_v) { best_v = v; best_k = kh * kW + kw; }
                        }
                    }
                    const int out_off = n * out_stride_n
                                      + c * out_stride_c
                                      + oh * out_stride_oh
                                      + ow * out_stride_ow;
                    out_data[out_off] = best_v;
                    mask_data[out_off] = static_cast<float>(best_k);
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    saved_mask.copy_from_host(mask_data.data(), mask_data.size());
    return out;
}

Tensor tensor_maxpool2d_nchw_backward(const Tensor& g,
                                     const Tensor& mask,
                                     int N, int C, int H, int W,
                                     int kH, int kW, int stride) {
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, 0, "max_pool2d_backward"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, 0, "max_pool2d_backward"));
    const int K_flat = C * kH * kW;
    const int P_flat = oH * oW;

    Tensor d_col = Tensor::zeros(Shape({N, K_flat, P_flat}), g.device());
    if (d_col.elements() == 0) {
        return Tensor::zeros(Shape({N, C, H, W}), g.device());
    }

    std::vector<float> g_data(g.elements());
    std::vector<float> mask_data(mask.elements());
    std::vector<float> dc_data(d_col.elements());
    g.copy_to_host(g_data.data(), g_data.size());
    mask.copy_to_host(mask_data.data(), mask_data.size());

    const int g_stride_n = C * oH * oW;
    const int g_stride_c = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;
    const int dc_stride_n = K_flat * P_flat;
    const int dc_stride_k = P_flat;
    const int dc_stride_p = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    const int p = oh * oW + ow;
                    const float g_val = g_data[n * g_stride_n
                                            + c * g_stride_c
                                            + oh * g_stride_oh
                                            + ow * g_stride_ow];
                    const int k = static_cast<int>(
                        mask_data[n * g_stride_n
                                + c * g_stride_c
                                + oh * g_stride_oh
                                + ow * g_stride_ow] + 0.5f);
                    dc_data[n * dc_stride_n
                          + (c * kH * kW + k) * dc_stride_k
                          + p * dc_stride_p] += g_val;
                }
            }
        }
    }
    d_col.copy_from_host(dc_data.data(), dc_data.size());
    return tensor_col2im_nchw(d_col, N, C, H, W, kH, kW, stride, 0);
}

}  // namespace cpu_ops
}  // namespace detail
}  // namespace ag
