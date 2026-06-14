// example.cpp — README quick-start sample.
//
// Build via:
//   g++ -std=c++17 -I include -I /usr/include/eigen3 -fopenmp \
//       test/example.cpp build/libautograd.a -o example
//   ./example

#include "autograd.h"
#include <iostream>
using namespace ag;

int main() {
    // Tiny dataset: y = 2*x1 + 3*x2
    Mat X(4, 2); X << 1, 1,  1, 0,  0, 1,  0, 0;
    Mat Y(4, 1); Y << 5,  2,  3,  0;

    auto net = std::make_shared<Sequential>();
    net->add(std::make_shared<Linear>(2, 8));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<Linear>(8, 1));
    Adam opt(net->parameters(), 0.05f);

    for (int epoch = 0; epoch < 200; ++epoch) {
        opt.zero_grad();
        auto pred = net->forward(Var::make(X));
        auto loss = mse_loss(pred, Y);
        loss->backward();
        opt.step();
        if (epoch % 50 == 0)
            std::cout << "epoch " << epoch << "  loss=" << loss->data(0, 0) << "\n";
    }
    std::cout << "final pred:\n" << net->forward(Var::make(X))->data << "\n";
    return 0;
}
