#pragma once
// cuda_core.h — minimal CUDA helpers for core autograd ops.

#include "autograd/variable.h"
#include <cstddef>

namespace ag {

#ifdef AUTOGRAD_USE_CUDA

void cuda_alloc_floats(float** p, std::size_t n, int device);
void cuda_free_floats(float* p);
void cuda_copy_h2d(float* dst, const float* src, std::size_t n, int device);
void cuda_copy_d2h(float* dst, const float* src, std::size_t n, int device);
void cuda_zero(float* dst, std::size_t n, int device);
void cuda_fill(float* dst, float value, std::size_t n, int device);

VarPtr cuda_add_op(VarPtr a, VarPtr b);
VarPtr cuda_mul_op(VarPtr a, VarPtr b);
VarPtr cuda_matmul_op(VarPtr a, VarPtr b);
VarPtr cuda_broadcast_add_op(VarPtr a, VarPtr b);
VarPtr cuda_scale_op(VarPtr a, float s);
VarPtr cuda_relu_op(VarPtr a);
VarPtr cuda_sum_op(VarPtr a);
VarPtr cuda_softmax_op(VarPtr a);
VarPtr cuda_log_softmax_op(VarPtr a);
VarPtr cuda_conv2d_op(VarPtr input, VarPtr weight, VarPtr bias,
                      int N, int C, int H, int W,
                      int kH, int kW, int stride, int pad);
VarPtr cuda_maxpool2d_op(VarPtr input,
                         int N, int C, int H, int W,
                         int kH, int kW, int stride);
void cuda_sgd_step(Var& p, float lr);

#endif

} // namespace ag
