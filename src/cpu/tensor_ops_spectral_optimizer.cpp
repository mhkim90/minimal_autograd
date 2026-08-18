#include "detail/tensor_ops.h"

#include "detail/constants.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ag {
namespace detail {
namespace cpu_ops {

TensorDFT2Result tensor_dft2_last2(const Tensor& real_in,
                                   const Tensor& imag_in,
                                   bool inverse, bool scale_output) {
    const Shape& s = real_in.shape();
    const int rank = static_cast<int>(s.rank());
    const int H = static_cast<int>(s[rank - 2]);
    const int W = static_cast<int>(s[rank - 1]);
    const int64_t plane = static_cast<int64_t>(H) * W;
    const int64_t batch_count = s.numel() / plane;

    TensorDFT2Result out;
    out.real = Tensor::empty(s, real_in.device());
    out.imag = Tensor::empty(imag_in.shape(), imag_in.device());
    if (real_in.elements() == 0) return out;

    std::vector<float> r_in(real_in.elements());
    std::vector<float> i_in(imag_in.elements());
    std::vector<float> r_out(real_in.elements(), 0.f);
    std::vector<float> i_out(real_in.elements(), 0.f);
    real_in.copy_to_host(r_in.data(), r_in.size());
    imag_in.copy_to_host(i_in.data(), i_in.size());

    const float norm = scale_output
        ? 1.f / static_cast<float>(H * W) : 1.f;

    for (int64_t b = 0; b < batch_count; ++b) {
        const float* rp = r_in.data() + b * plane;
        const float* ip = i_in.data() + b * plane;
        float* ro = r_out.data() + b * plane;
        float* io = i_out.data() + b * plane;
        for (int kr = 0; kr < H; ++kr) {
            for (int kc = 0; kc < W; ++kc) {
                float sum_r = 0.f;
                float sum_i = 0.f;
                for (int r = 0; r < H; ++r) {
                    for (int c = 0; c < W; ++c) {
                        const float angle = 2.f * kPi *
                            (static_cast<float>(kr * r) /
                                 static_cast<float>(H) +
                             static_cast<float>(kc * c) /
                                 static_cast<float>(W));
                        const float wr = std::cos(angle);
                        const float wi = inverse
                            ? std::sin(angle) : -std::sin(angle);
                        const float xr = rp[r * W + c];
                        const float xi = ip[r * W + c];
                        sum_r += xr * wr - xi * wi;
                        sum_i += xr * wi + xi * wr;
                    }
                }
                ro[kr * W + kc] = norm * sum_r;
                io[kr * W + kc] = norm * sum_i;
            }
        }
    }
    out.real.copy_from_host(r_out.data(), r_out.size());
    out.imag.copy_from_host(i_out.data(), i_out.size());
    return out;
}

}  // namespace cpu_ops
}  // namespace detail
}  // namespace ag
