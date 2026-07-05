#pragma once
// cuda_core.h — minimal CUDA helpers for core autograd ops.

#include "autograd/variable.h"
#include <cstddef>
#include <string>

namespace ag {

#ifdef AUTOGRAD_USE_CUDA

struct CudaDeviceInfo {
    int device = -1;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_global_mem = 0;
    int multiprocessor_count = 0;
};

struct CudaRuntimeInfo {
    int device_count = 0;
    int current_device = -1;
    int driver_version = 0;
    int runtime_version = 0;
    std::string status;
    CudaDeviceInfo device;

    bool has_device() const noexcept { return device_count > 0 && current_device >= 0; }
};

// Probes CUDA runtime state and selects preferred_device when one is available.
CudaRuntimeInfo cuda_runtime_info(int preferred_device = 0);
std::string cuda_runtime_summary(const CudaRuntimeInfo& info);

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
VarPtr cuda_sigmoid_op(VarPtr a);
VarPtr cuda_tanh_op(VarPtr a);
VarPtr cuda_exp_op(VarPtr a);
VarPtr cuda_log_op(VarPtr a);
VarPtr cuda_sqrt_op(VarPtr a);
VarPtr cuda_silu_op(VarPtr a);
VarPtr cuda_softplus_op(VarPtr a);
VarPtr cuda_sub_op(VarPtr a, VarPtr b);
VarPtr cuda_div_op(VarPtr a, VarPtr b);
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
