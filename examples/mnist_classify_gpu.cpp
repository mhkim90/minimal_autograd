// mnist_classify_gpu.cpp — MNIST digit classification on CUDA.
//
// Same architecture and CLI as the CPU example. Parameters are constructed
// directly on CUDA and registered in this example-local functional model.
// Keep SGD here to preserve the example's training configuration.

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

class MnistModel : public nn::Module {
public:
    MnistModel() {
        nn::Conv2d c1(1,8,3,3,1,1), c2(8,16,3,3,1,1);
        nn::Linear f1(16*7*7,128), f2(128,10);
        auto parameter = [&](const char* name, const Variable& source) {
            Variable result(source.value().to(Device::cuda()), true);
            register_parameter(name, result);
            return result;
        };
        cw1 = parameter("conv1_weight", c1.weight()); cb1 = parameter("conv1_bias", c1.bias());
        cw2 = parameter("conv2_weight", c2.weight()); cb2 = parameter("conv2_bias", c2.bias());
        fw1 = parameter("fc1_weight", f1.weight()); fb1 = parameter("fc1_bias", f1.bias());
        fw2 = parameter("fc2_weight", f2.weight()); fb2 = parameter("fc2_bias", f2.bias());
    }
    Variable forward(const Variable& x) override {
        auto h = max_pool2d(relu(conv2d(x, cw1, cb1, 1, 1)), 2, 2, 2);
        h = max_pool2d(relu(conv2d(h, cw2, cb2, 1, 1)), 2, 2, 2);
        h = reshape(h, h.value().shape()[0], 16*7*7);
        h = relu(broadcast_add(matmul(h, fw1), fb1));
        return broadcast_add(matmul(h, fw2), fb2);
    }
private:
    Variable cw1, cb1, cw2, cb2, fw1, fb1, fw2, fb2;
};

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1 && argv[1][0] != '\0')
        ? argv[1]
        : std::string(DATA_DIR_DEFAULT) + "/MNIST/raw";

    const int train_n = (argc > 2) ? std::atoi(argv[2]) : 5000;
    const int test_n  = (argc > 3) ? std::atoi(argv[3]) : 1000;
    const int epochs  = (argc > 4) ? std::atoi(argv[4]) : 5;
    const int batch   = (argc > 5) ? std::atoi(argv[5]) : 64;
    const float lr    = 0.05f;  // SGD lr chosen for this slice.

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

    auto net = std::make_shared<MnistModel>();
    optim::SGD opt(net->parameters(), lr);

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        int    epoch_batches = 0;

        for (int off = 0; off < N_tr; off += batch) {
            const int bs = std::min(batch, N_tr - off);
            Tensor xb = row_slice(Variable(Xtr_full), off, bs).value();
            Tensor yb = row_slice(Variable(Ytr_full), off, bs).value();

            opt.zero_grad();

            auto xv = Variable(xb.reshape(Shape{bs,1,28,28}).to(Device::cuda()));
            auto pred = net->forward(xv);  // (bs, 10) on CUDA
            auto loss = cross_entropy(pred, yb.to(Device::cuda()));  // CUDA scalar (1x1)

            loss.backward();
            if (!all_finite(loss.value())) throw std::runtime_error("nonfinite loss");
            const auto parameters = net->parameters();
            for (const auto& p : parameters)
                if (!p.has_grad() || !all_finite(p.grad())) throw std::runtime_error("nonfinite gradient");
            opt.step();
            for (const auto& p : parameters)
                if (!all_finite(p.value())) throw std::runtime_error("nonfinite update");
            float loss_value = 0.f; loss.value().copy_to_host(&loss_value, 1);

            // cross_entropy stays on CUDA; pull the scalar back to host
            // only to print/log.
            epoch_loss    += loss_value;
            ++epoch_batches;
        }
        const double avg = epoch_loss / std::max(1, epoch_batches);
        std::cout << "epoch " << epoch
                  << "  loss=" << avg << "  finite forward/backward/update OK\n";
    }

    // Test-set inference. We pull logits back to CPU once per batch
    // for argmax; no gradient is needed.
    int correct = 0;
    const int eval_batch = 128;
    for (int off = 0; off < N_te; off += eval_batch) {
        const int bs = std::min(eval_batch, N_te - off);
        Tensor xb = row_slice(Variable(Xte), off, bs).value();
        Tensor yb = row_slice(Variable(Yte), off, bs).value();

        auto xv = Variable(xb.reshape(Shape{bs,1,28,28}).to(Device::cuda()));
        auto logits = net->forward(xv);  // (bs, 10) on CUDA
        if (!logits.device().is_cuda() || !all_finite(logits.value())) throw std::runtime_error("invalid CUDA evaluation");
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
