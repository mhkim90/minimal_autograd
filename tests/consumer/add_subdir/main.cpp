// Consumer smoke check — see tests/consumer/README.md. Identical content
// to tests/consumer/find_package/main.cpp, kept here so each consumption
// mode has a self-contained directory.

#include "autograd.h"

#include <cmath>
#include <cstdio>

using namespace ag;

namespace {

bool run_smoke() {
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

}  // namespace

int main() {
    if (!run_smoke()) {
        return 1;
    }
    std::printf("OK: forward=60, grad=2.5\n");
    return 0;
}
