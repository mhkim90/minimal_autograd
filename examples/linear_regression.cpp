// linear_regression.cpp — build with AUTOGRAD_BUILD_EXAMPLES=ON.

#include "autograd/core/module.h"
#include "autograd/core/ops.h"
#include "autograd/core/loss.h"
#include "autograd/core/optim.h"
#include <stdexcept>
#include <vector>
#include <iostream>
using namespace ag;

int main() {
    // Tiny dataset: y = 2*x1 + 3*x2
    const float xv[] = {1,1,1,0,0,1,0,0}, yv[] = {5,2,3,0};
    Tensor X = Tensor::from_host(xv, Shape{4,2});
    Tensor Y = Tensor::from_host(yv, Shape{4,1});

    auto net = std::make_shared<nn::Sequential>();
    net->add(std::make_shared<nn::Linear>(2, 8));
    net->add(std::make_shared<nn::ReLU>());
    net->add(std::make_shared<nn::Linear>(8, 1));
    optim::Adam opt(net->parameters(), 0.05f);

    for (int epoch = 0; epoch < 200; ++epoch) {
        opt.zero_grad();
        auto pred = net->forward(Variable(X));
        auto loss = mse_loss(pred, Y);
        loss.backward();
        if (!all_finite(loss.value())) throw std::runtime_error("nonfinite loss");
        for (const auto& p : net->parameters())
            if (!p.has_grad() || !all_finite(p.grad())) throw std::runtime_error("nonfinite gradient");
        opt.step();
        for (const auto& p : net->parameters())
            if (!all_finite(p.value())) throw std::runtime_error("nonfinite update");
        float loss_value = 0.f; loss.value().copy_to_host(&loss_value, 1);
        if (epoch % 50 == 0)
            std::cout << "epoch " << epoch << "  loss=" << loss_value << "\n";
    }
    auto final = net->forward(Variable(X));
    if (!all_finite(final.value())) throw std::runtime_error("nonfinite evaluation");
    std::vector<float> values(final.value().elements());
    final.value().copy_to_host(values.data(), values.size());
    std::cout << "final pred:\n";
    for (float v : values) std::cout << v << "\n";
    return 0;
}
