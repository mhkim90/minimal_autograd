// Consumer smoke check — exercises the public autograd API surface
// through the `autograd::autograd` target. Deterministic compile/link/run
// check (no random init, no training loop): runs a fixed forward pass
// (Var, scale, sum), checks the exact scalar result, then backward() and
// checks the exact analytic gradient. Also exercises the new Phase 2
// Shape + Device value types end-to-end through the public umbrella
// header, so the installed `find_package` consumer is proof that those
// types are reachable from external code. Does not include any
// internal, Eigen, or CUDA headers; only `#include "autograd.h"`.
//
// This file is identical across the `find_package` and `add_subdirectory`
// consumer CMakeLists.txt variants; it must compile and run verbatim
// against either consumption mode.

#include "autograd.h"

#include <cmath>
#include <cstdio>

using namespace ag;

namespace {

bool run_forward_backward_smoke() {
    // x is a fixed 2x3 matrix of 4s (sum = 24). y = scale(x, 2.5f) yields a
    // 2x3 matrix of 10s (sum = 60). z = sum(y) is a 1x1 scalar = 60.0f.
    const int rows = 2;
    const int cols = 3;
    const float x_val = 4.0f;
    const float s = 2.5f;

    Mat x_mat = Mat::Constant(rows, cols, x_val);
    auto x = Var::make(x_mat);
    auto y = scale(x, s);
    auto z = sum(y);

    // Forward: z must equal s * rows * cols * x_val = 60.0f.
    if (std::fabs(z->data(0, 0) - s * rows * cols * x_val) > 1e-5f) {
        std::fprintf(stderr,
                     "FAIL: forward z=%g, expected %g\n",
                     z->data(0, 0), s * rows * cols * x_val);
        return false;
    }

    // Backward: d z / d x = s everywhere (since z = s * sum(x)).
    z->backward();
    Mat expected_grad = Mat::Constant(rows, cols, s);
    const float max_diff = (x->grad - expected_grad).cwiseAbs().maxCoeff();
    if (max_diff > 1e-5f) {
        std::fprintf(stderr,
                     "FAIL: backward grad max |diff|=%g (expected %g)\n",
                     max_diff, s);
        return false;
    }
    return true;
}

bool run_shape_device_smoke() {
    // Shape: rank/numel contract.
    Shape s = make_shape({2, 3, 4});
    if (s.rank() != 3) return false;
    if (s.ndim() != 3) return false;
    if (s.numel() != 24) return false;
    if (s[0] != 2 || s[2] != 4) return false;

    Stride st = contiguous_stride(s);
    if (st[0] != 1 || st[1] != 2 || st[2] != 6) return false;

    Shape scalar;
    if (scalar.rank() != 0) return false;
    if (scalar.numel() != 1) return false;

    // Equality/inequality preserved.
    if (!(Shape{2, 3} == Shape{2, 3})) return false;
    if (!(Shape{2, 3} != Shape{2, 4})) return false;
    (void)Shape{}.operator==(Shape{});

    // Device: CPU/CUDA descriptor semantics.
    Device cpu = Device::cpu();
    Device gpu0 = Device::cuda();
    Device gpu1 = Device::cuda(1);

    if (!cpu.is_cpu() || cpu.is_cuda()) return false;
    if (!gpu0.is_cuda() || gpu0.is_cpu()) return false;
    if (cpu.type() != DeviceType::Cpu) return false;
    if (gpu1.type() != DeviceType::Cuda) return false;
    if (gpu1.index() != 1) return false;

    if (cpu != Device::cpu()) return false;
    if (gpu0 == gpu1) return false;
    if (cpu == gpu0) return false;

    if (cpu.to_string() != "cpu") return false;
    if (gpu1.to_string() != "cuda:1") return false;

    return true;
}

}  // namespace

int main() {
    if (!run_forward_backward_smoke()) {
        return 1;
    }
    if (!run_shape_device_smoke()) {
        std::fprintf(stderr, "FAIL: Shape/Device smoke\n");
        return 1;
    }
    std::printf("OK: forward=60, grad=2.5, shape=2x3x4, device=cpu/cuda:1\n");
    return 0;
}
