// mnist_classify_gpu.cpp — MNIST digit classification on CUDA.
//
// Identical network and training loop to mnist_classify_cpu.cpp, but
// every parameter and every per-batch input is moved to the GPU via
// `.cuda()` so the full forward/backward runs on device.
//
// One deviation from the CPU version: the optimizer is SGD instead of
// Adam because Adam::step() throws when called on CUDA parameters in
// the current library. With the lr below both reach comparable
// test-set accuracy on the same data slice.

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
    const int epochs  = (argc > 4) ? std::atoi(argv[4]) : 5;
    const int batch   = (argc > 5) ? std::atoi(argv[5]) : 64;
    const float lr    = 0.05f;  // SGD lr chosen for this slice.

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

    auto conv1 = std::make_shared<Conv2d>(1,  8, 3, 3, 1, 1);
    auto conv2 = std::make_shared<Conv2d>(8, 16, 3, 3, 1, 1);
    auto fc1   = std::make_shared<Linear>(16 * 7 * 7, 128);
    auto fc2   = std::make_shared<Linear>(128, 10);

    // Move each layer's own W/b to the GPU in place. Module::parameters()
    // returns copies of the VarPtrs, so reassigning through that vector
    // (e.g. `for (auto& p : net->parameters()) p = p->cuda();`) would not
    // reach back into these layers' W/b members.
    conv1->W = conv1->W->cuda(); conv1->b = conv1->b->cuda();
    conv2->W = conv2->W->cuda(); conv2->b = conv2->b->cuda();
    fc1->W   = fc1->W->cuda();   fc1->b   = fc1->b->cuda();
    fc2->W   = fc2->W->cuda();   fc2->b   = fc2->b->cuda();

    auto net = std::make_shared<Sequential>();
    net->add(conv1);
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<MaxPool2d>(2, 2));
    net->add(conv2);
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<MaxPool2d>(2, 2));
    net->add(fc1);
    net->add(std::make_shared<ReLUModule>());
    net->add(fc2);

    SGD opt(net->parameters(), lr);

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        int    epoch_batches = 0;

        for (int off = 0; off < N_tr; off += batch) {
            const int bs = std::min(batch, N_tr - off);
            Mat xb = Xtr_full.middleRows(off, bs);
            Mat yb = Ytr_full.middleRows(off, bs);

            opt.zero_grad();

            auto xv = Var::make4d(xb, bs, 1, 28, 28)->cuda();
            auto pred = net->forward(xv);  // (bs, 10) on CUDA
            auto loss = cross_entropy(pred, yb);  // CUDA scalar (1x1)

            loss->backward();
            opt.step();

            // cross_entropy stays on CUDA; pull the scalar back to host
            // only to print/log.
            epoch_loss    += loss->cpu()->data(0, 0);
            ++epoch_batches;
        }
        const double avg = epoch_loss / std::max(1, epoch_batches);
        std::cout << "epoch " << epoch
                  << "  loss=" << avg << "\n";
    }

    // Test-set inference. We pull logits back to CPU once per batch
    // for argmax; no gradient is needed.
    int correct = 0;
    const int eval_batch = 128;
    for (int off = 0; off < N_te; off += eval_batch) {
        const int bs = std::min(eval_batch, N_te - off);
        Mat xb = Xte.middleRows(off, bs);
        Mat yb = Yte.middleRows(off, bs);

        auto xv = Var::make4d(xb, bs, 1, 28, 28)->cuda();
        auto logits = net->forward(xv);  // (bs, 10) on CUDA
        const Mat& sm = logits->cpu()->data;

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
