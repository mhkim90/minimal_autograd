#include "autograd.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace ag;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define CHECK_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (std::fabs(_a - _b) > (tol)) { \
        std::cerr << "FAIL: " << #a << " (" << _a << ") vs " << #b << " (" << _b \
                  << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

int main() {
    Mat m(2, 3);
    m << 1.f, -2.f, 3.f,
         4.f, -5.f, 6.f;

    auto x = Var::make(m)->cuda();
    CHECK(x->is_cuda());

    auto y = sum(relu(add(x, x)));
    CHECK(y->is_cuda());
    y->backward();

    auto y_cpu = y->cpu();
    CHECK_NEAR(y_cpu->data(0, 0), 28.f, 1e-4f);

    auto x_cpu = x->cpu();
    CHECK_NEAR(x_cpu->grad(0, 0), 2.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(0, 1), 0.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(0, 2), 2.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(1, 0), 2.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(1, 1), 0.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(1, 2), 2.f, 1e-4f);

    auto a = Var::make(Mat::Constant(2, 2, 3.f))->cuda();
    auto b = Var::make(Mat::Constant(2, 2, 4.f))->cuda();
    auto z = sum(scale(mul(a, b), 0.5f));
    z->backward();
    CHECK_NEAR(z->cpu()->data(0, 0), 24.f, 1e-4f);
    CHECK_NEAR(a->cpu()->grad(0, 0), 2.f, 1e-4f);
    CHECK_NEAR(b->cpu()->grad(0, 0), 1.5f, 1e-4f);

    bool threw = false;
    try {
        (void)matmul(a, b);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    std::cout << "ALL CUDA CORE TESTS PASSED\n";
    return 0;
}
