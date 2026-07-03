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
        (void)transpose(a);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    Mat A(2, 3);
    A << 1.f, 2.f, 3.f,
         4.f, 5.f, 6.f;
    Mat B(3, 2);
    B << 1.f, 2.f,
         3.f, 4.f,
         5.f, 6.f;
    auto ma = Var::make(A)->cuda();
    auto mb = Var::make(B)->cuda();
    auto mm = sum(matmul(ma, mb));
    mm->backward();
    CHECK_NEAR(mm->cpu()->data(0, 0), (A * B).sum(), 1e-4f);
    Mat expected_ma_grad = Mat::Ones(2, 2) * B.transpose();
    Mat expected_mb_grad = A.transpose() * Mat::Ones(2, 2);
    CHECK_NEAR(ma->cpu()->grad(1, 2), expected_ma_grad(1, 2), 1e-4f);
    CHECK_NEAR(mb->cpu()->grad(2, 1), expected_mb_grad(2, 1), 1e-4f);

    auto bx = Var::make(Mat::Constant(4, 3, 1.f))->cuda();
    auto bb = Var::make(Mat::Constant(1, 3, 2.f))->cuda();
    auto by = sum(broadcast_add(bx, bb));
    by->backward();
    CHECK_NEAR(by->cpu()->data(0, 0), 36.f, 1e-4f);
    CHECK_NEAR(bx->cpu()->grad(3, 2), 1.f, 1e-4f);
    CHECK_NEAR(bb->cpu()->grad(0, 2), 4.f, 1e-4f);

    auto p = Var::make(Mat::Constant(2, 2, 1.f))->cuda();
    auto loss = sum(scale(p, 2.f));
    loss->backward();
    SGD opt({p}, 0.1f);
    opt.step();
    CHECK_NEAR(p->cpu()->data(0, 0), 0.8f, 1e-4f);

    Linear lin(3, 2);
    lin.W = lin.W->cuda();
    lin.b = lin.b->cuda();
    auto lx = Var::make(Mat::Constant(4, 3, 1.f))->cuda();
    auto lloss = sum(lin.forward(lx));
    lloss->backward();
    SGD lin_opt(lin.parameters(), 0.01f);
    Mat w_before = lin.W->cpu()->data;
    lin_opt.step();
    CHECK(lloss->is_cuda());
    CHECK(lin.W->is_cuda());
    CHECK(lin.W->cpu()->grad.cwiseAbs().maxCoeff() > 0.f);
    CHECK((lin.W->cpu()->data - w_before).cwiseAbs().maxCoeff() > 0.f);

    std::cout << "ALL CUDA CORE TESTS PASSED\n";
    return 0;
}
