// mnist_classify_cpu.cpp — MNIST digit classification on CPU.
//
// Loads a slice of MNIST from data/MNIST/raw/, builds a small CNN
// (Conv → ReLU → Pool → Conv → ReLU → Pool → FC → ReLU → FC), trains
// it with cross-entropy + Adam, and reports test-set accuracy.
//
// Network architecture:
//   Conv2d(1,8,3,1,1) → ReLU → MaxPool2d(2,2)
//   Conv2d(8,16,3,1,1) → ReLU → MaxPool2d(2,2)
//   Linear(16*7*7, 128) → ReLU → Linear(128, 10)
//   cross_entropy loss + Adam optimizer, manual mini-batching.

#include "autograd/core/module.h"
#include "autograd/core/ops.h"
#include "autograd/core/loss.h"
#include "autograd/core/optim.h"
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "mnist_data.h"

#include <chrono>
#include <iostream>
#include <memory>

using namespace ag;

// Spatial tensors remain NCHW until this explicit fully-connected boundary.
class Flatten : public nn::Module {
public:
    Variable forward(const Variable& x) override {
        return reshape(x, x.value().shape()[0], 16*7*7);
    }
};

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1 && argv[1][0] != '\0')
        ? argv[1]
        : std::string(DATA_DIR_DEFAULT) + "/MNIST/raw";

    const int train_n = (argc > 2) ? std::atoi(argv[2]) : 5000;
    const int test_n  = (argc > 3) ? std::atoi(argv[3]) : 1000;
    const int epochs  = (argc > 4) ? std::atoi(argv[4]) : 10;
    const int batch   = (argc > 5) ? std::atoi(argv[5]) : 64;
    const float lr    = 3e-3f;

    std::cout << "loading MNIST from " << data_dir << "...\n";
    MnistData mnist = load_mnist(data_dir);

    const int N_tr_full = static_cast<int>(mnist.train_images.shape()[0]);
    const int N_te_full = static_cast<int>(mnist.test_images.shape()[0]);
    const int N_tr = std::min(train_n, N_tr_full);
    const int N_te = std::min(test_n,  N_te_full);

    Tensor Xtr_full = row_slice(Variable(mnist.train_images), 0, N_tr).value();
    Tensor Ytr_full = row_slice(Variable(mnist.train_labels), 0, N_tr).value();
    Tensor Xte = row_slice(Variable(mnist.test_images), 0, N_te).value();
    Tensor Yte = row_slice(Variable(mnist.test_labels), 0, N_te).value();
    std::cout << "using " << N_tr << " train / " << N_te << " test images\n";

    // Model: conv → relu → pool → conv → relu → pool → fc → relu → fc
    auto net = std::make_shared<nn::Sequential>();
    net->add(std::make_shared<nn::Conv2d>(1,  8, 3, 3, 1, 1));
    net->add(std::make_shared<nn::ReLU>());
    net->add(std::make_shared<nn::MaxPool2d>(2, 2));
    net->add(std::make_shared<nn::Conv2d>(8, 16, 3, 3, 1, 1));
    net->add(std::make_shared<nn::ReLU>());
    net->add(std::make_shared<nn::MaxPool2d>(2, 2));
    net->add(std::make_shared<Flatten>());
    net->add(std::make_shared<nn::Linear>(16 * 7 * 7, 128));
    net->add(std::make_shared<nn::ReLU>());
    net->add(std::make_shared<nn::Linear>(128, 10));

    optim::Adam opt(net->parameters(), lr);

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        int    epoch_batches = 0;

        for (int off = 0; off < N_tr; off += batch) {
            const int bs = std::min(batch, N_tr - off);
            Tensor xb = row_slice(Variable(Xtr_full), off, bs).value();
            Tensor yb = row_slice(Variable(Ytr_full), off, bs).value();

            opt.zero_grad();

            auto xv = Variable(xb.reshape(Shape{bs,1,28,28}));
            auto pred = net->forward(xv);  // (bs, 10) on CPU
            auto loss = cross_entropy(pred, yb);

            loss.backward();
            if (!all_finite(loss.value())) throw std::runtime_error("nonfinite loss");
            const auto parameters = net->parameters();
            for (const auto& p : parameters)
                if (!p.has_grad() || !all_finite(p.grad())) throw std::runtime_error("nonfinite gradient");
            opt.step();
            for (const auto& p : parameters)
                if (!all_finite(p.value())) throw std::runtime_error("nonfinite update");
            float loss_value = 0.f; loss.value().copy_to_host(&loss_value, 1);

            epoch_loss    += loss_value;
            ++epoch_batches;
        }
        const double avg = epoch_loss / std::max(1, epoch_batches);
        std::cout << "epoch " << epoch
                  << "  loss=" << avg << "  finite forward/backward/update OK\n";
    }

    // Test-set inference in batches (same model, no grad).
    int correct = 0;
    const int eval_batch = 128;
    for (int off = 0; off < N_te; off += eval_batch) {
        const int bs = std::min(eval_batch, N_te - off);
        Tensor xb = row_slice(Variable(Xte), off, bs).value();
        Tensor yb = row_slice(Variable(Yte), off, bs).value();

        auto xv = Variable(xb.reshape(Shape{bs,1,28,28}));
        auto logits = net->forward(xv);  // (bs, 10)
        if (!all_finite(logits.value())) throw std::runtime_error("nonfinite evaluation");
        std::vector<float> sm(logits.value().elements());
        logits.value().copy_to_host(sm.data(), sm.size());

        std::vector<float> labels(yb.elements());
        yb.copy_to_host(labels.data(), labels.size());
        for (int i = 0; i < bs; ++i) {
            int pred_class = 0;
            float best = sm[i*10];
            for (int c = 1; c < 10; ++c) {
                if (sm[i*10+c] > best) { best = sm[i*10+c]; pred_class = c; }
            }
            int true_class = 0;
            for (int c = 0; c < 10; ++c) if (labels[i*10+c] > 0.5f) { true_class = c; break; }
            if (pred_class == true_class) ++correct;
        }
    }
    const float test_acc = static_cast<float>(correct) /
                            static_cast<float>(N_te);
    auto t_end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "final test accuracy: " << test_acc
              << "  (" << correct << "/" << N_te << ")\n";
    std::cout << "training time: " << elapsed << " s\n";
    return 0;
}
