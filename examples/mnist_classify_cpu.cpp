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

#include "autograd.h"
#include "mnist_data.h"

#include <chrono>
#include <iostream>
#include <memory>

using namespace ag;

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

    const int N_tr_full = static_cast<int>(mnist.train_images.rows());
    const int N_te_full = static_cast<int>(mnist.test_images.rows());
    const int N_tr = std::min(train_n, N_tr_full);
    const int N_te = std::min(test_n,  N_te_full);

    Mat Xtr_full = mnist.train_images.topRows(N_tr);
    Mat Ytr_full = mnist.train_labels.topRows(N_tr);
    Mat Xte      = mnist.test_images.topRows(N_te);
    Mat Yte      = mnist.test_labels.topRows(N_te);
    std::cout << "using " << N_tr << " train / " << N_te << " test images\n";

    // Model: conv → relu → pool → conv → relu → pool → fc → relu → fc
    auto net = std::make_shared<Sequential>();
    net->add(std::make_shared<Conv2d>(1,  8, 3, 3, 1, 1));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<MaxPool2d>(2, 2));
    net->add(std::make_shared<Conv2d>(8, 16, 3, 3, 1, 1));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<MaxPool2d>(2, 2));
    net->add(std::make_shared<Linear>(16 * 7 * 7, 128));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<Linear>(128, 10));

    Adam opt(net->parameters(), lr);

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        int    epoch_batches = 0;

        for (int off = 0; off < N_tr; off += batch) {
            const int bs = std::min(batch, N_tr - off);
            Mat xb = Xtr_full.middleRows(off, bs);
            Mat yb = Ytr_full.middleRows(off, bs);

            opt.zero_grad();

            auto xv = Var::make4d(xb, bs, 1, 28, 28);
            auto pred = net->forward(xv);  // (bs, 10) on CPU
            auto loss = cross_entropy(pred, yb);

            loss->backward();
            opt.step();

            epoch_loss    += loss->data(0, 0);
            ++epoch_batches;
        }
        const double avg = epoch_loss / std::max(1, epoch_batches);
        std::cout << "epoch " << epoch
                  << "  loss=" << avg << "\n";
    }

    // Test-set inference in batches (same model, no grad).
    int correct = 0;
    const int eval_batch = 128;
    for (int off = 0; off < N_te; off += eval_batch) {
        const int bs = std::min(eval_batch, N_te - off);
        Mat xb = Xte.middleRows(off, bs);
        Mat yb = Yte.middleRows(off, bs);

        auto xv = Var::make4d(xb, bs, 1, 28, 28);
        auto logits = net->forward(xv);  // (bs, 10)
        const Mat& sm = logits->data;

        for (int i = 0; i < bs; ++i) {
            int pred_class = 0;
            float best = sm(i, 0);
            for (int c = 1; c < 10; ++c) {
                if (sm(i, c) > best) { best = sm(i, c); pred_class = c; }
            }
            int true_class = 0;
            for (int c = 0; c < 10; ++c) if (yb(i, c) > 0.5f) { true_class = c; break; }
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
