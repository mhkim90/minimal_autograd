// test_nn.cpp — end-to-end neural net training: XOR with 2-layer MLP + Adam.

#include "autograd.h"
#include <cassert>
#include <cmath>
#include <iostream>

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

void test_linear_forward() {
    Linear lin(4, 3);
    auto x = Var::make(Mat::Random(5, 4));   // batch of 5
    auto y = lin.forward(x);
    CHECK(y->data.rows() == 5);
    CHECK(y->data.cols() == 3);
    std::cout << "[ok] Linear forward shape (5,3)\n";
}

void test_sequential() {
    auto net = std::make_shared<Sequential>();
    net->add(std::make_shared<Linear>(4, 8));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<Linear>(8, 2));

    auto x = Var::make(Mat::Random(3, 4));
    auto y = net->forward(x);
    CHECK(y->data.rows() == 3);
    CHECK(y->data.cols() == 2);
    auto params = net->parameters();
    CHECK(params.size() == 4);   // 2 weights + 2 biases
    std::cout << "[ok] Sequential(Linear, ReLU, Linear) forward shape (3,2)\n";
}

// XOR — should converge in <500 epochs.
void test_xor() {
    Mat X(4, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         1, 1;
    Mat Y(4, 1);
    Y << 0,
         1,
         1,
         0;

    auto net = std::make_shared<Sequential>();
    net->add(std::make_shared<Linear>(2, 8));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<Linear>(8, 1));

    auto opt = Adam(net->parameters(), 0.05f);

    int target_epochs = 200;
    for (int epoch = 0; epoch < target_epochs; ++epoch) {
        opt.zero_grad();
        auto x = Var::make(X);
        auto y_pred = net->forward(x);
        auto loss = mse_loss(y_pred, Y);
        loss->backward();
        opt.step();
        if (epoch % 50 == 0)
            std::cout << "  epoch " << epoch
                      << "  loss=" << loss->data(0, 0) << "\n";
    }

    auto final_pred = net->forward(Var::make(X));
    float err = 0.f;
    for (int i = 0; i < 4; ++i) err += std::fabs(final_pred->data(i, 0) - Y(i, 0));
    std::cout << "  final err=" << err << "\n";
    CHECK(err < 0.3f);
    std::cout << "[ok] XOR learned in <" << target_epochs
              << " epochs (err<0.3)\n";
}

void test_sgd_step() {
    auto p = Var::make(Mat::Constant(1, 1, 1.f));
    p->grad = Mat::Constant(1, 1, 0.5f);
    SGD sgd({p}, 0.1f);
    sgd.step();
    CHECK_NEAR(p->data(0, 0), 0.95f, 1e-5f);
    std::cout << "[ok] SGD step: 1 - 0.1*0.5 = 0.95\n";
}

void test_adam_step() {
    auto p = Var::make(Mat::Constant(1, 1, 1.f));
    p->grad = Mat::Constant(1, 1, 1.f);
    Adam adam({p}, 0.1f);
    adam.step();
    // m = 0.1, v = 0.001, m_hat = 0.1/0.1 = 1, v_hat = 0.001/0.001 = 1
    // update = 0.1 * 1 / (1 + 1e-8) ≈ 0.1
    CHECK(std::fabs(p->data(0, 0) - 0.9f) < 1e-3f);
    std::cout << "[ok] Adam step: m/v/m_hat/v_hat update ok\n";
}

void test_cross_entropy() {
    Mat pred(2, 3);
    pred << 1.0f, 2.0f, 0.5f,
            0.1f, 0.2f, 0.7f;
    Mat target(2, 3);
    target << 1, 0, 0,
              0, 0, 1;

    auto logits = Var::make(pred);
    auto loss = cross_entropy(logits, target);
    CHECK(loss->data(0, 0) > 0.f);  // NLL is non-negative
    std::cout << "[ok] cross_entropy forward: loss=" << loss->data(0, 0) << "\n";
}

int main() {
    test_linear_forward();
    test_sequential();
    test_sgd_step();
    test_adam_step();
    test_cross_entropy();
    test_xor();
    std::cout << "\nALL NN TESTS PASSED\n";
    return 0;
}
