// src/norm.cpp — GroupNorm implementation.

#include "autograd/norm.h"
#include <cmath>
#include <cassert>

namespace ag {

GroupNorm::GroupNorm(int num_groups_, int channels_, float eps_)
    : num_groups(num_groups_), channels(channels_), eps(eps_) {
    assert(channels % num_groups == 0);
    gamma = Var::make(Mat::Ones(1, channels));
    beta  = Var::make(Mat::Zero(1, channels));
}

VarPtr GroupNorm::forward(VarPtr x, int C, int HW) {
    assert(C == channels);
    assert(C % num_groups == 0);

    int N = x->data.rows();
    int G = num_groups;
    int ch_per_group = C / G;
    int group_elems  = ch_per_group * HW;

    const Mat& d = x->data;
    Mat out(N, C * HW);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            int ch0 = g * ch_per_group;

            float mean = 0.f;
            for (int c = ch0; c < ch0 + ch_per_group; ++c)
                for (int p = 0; p < HW; ++p) mean += d(n, c * HW + p);
            mean /= static_cast<float>(group_elems);

            float var = 0.f;
            for (int c = ch0; c < ch0 + ch_per_group; ++c)
                for (int p = 0; p < HW; ++p) {
                    float diff = d(n, c * HW + p) - mean;
                    var += diff * diff;
                }
            var /= static_cast<float>(group_elems);
            float inv_std = 1.f / std::sqrt(var + eps);

            for (int c = ch0; c < ch0 + ch_per_group; ++c) {
                float gc = gamma->data(0, c);
                float bc = beta->data(0, c);
                for (int p = 0; p < HW; ++p) {
                    float xn = (d(n, c * HW + p) - mean) * inv_std;
                    out(n, c * HW + p) = gc * xn + bc;
                }
            }
        }
    }

    // Return as a new leaf (no gradient path — inference only).
    return Var::make(out);
}

} // namespace ag
