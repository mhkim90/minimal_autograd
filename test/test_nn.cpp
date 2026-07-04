// test_nn.cpp — end-to-end neural net training: XOR with 2-layer MLP + Adam.

#include "autograd.h"
#include "../examples/mnist_data.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

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

void write_be_u32(std::ofstream& f, uint32_t v) {
    unsigned char bytes[4] = {
        static_cast<unsigned char>((v >> 24) & 0xffu),
        static_cast<unsigned char>((v >> 16) & 0xffu),
        static_cast<unsigned char>((v >> 8) & 0xffu),
        static_cast<unsigned char>(v & 0xffu)
    };
    f.write(reinterpret_cast<const char*>(bytes), 4);
}

void write_idx_images(const std::string& path,
                      int n, int h, int w,
                      const std::vector<unsigned char>& pixels) {
    std::ofstream f(path, std::ios::binary);
    write_be_u32(f, 0x00000803u);
    write_be_u32(f, static_cast<uint32_t>(n));
    write_be_u32(f, static_cast<uint32_t>(h));
    write_be_u32(f, static_cast<uint32_t>(w));
    f.write(reinterpret_cast<const char*>(pixels.data()),
            static_cast<std::streamsize>(pixels.size()));
}

void write_idx_labels(const std::string& path,
                      const std::vector<unsigned char>& labels) {
    std::ofstream f(path, std::ios::binary);
    write_be_u32(f, 0x00000801u);
    write_be_u32(f, static_cast<uint32_t>(labels.size()));
    f.write(reinterpret_cast<const char*>(labels.data()),
            static_cast<std::streamsize>(labels.size()));
}

float classification_accuracy(std::shared_ptr<Sequential> net,
                              const Mat& X,
                              const Mat& Y) {
    auto out = net->forward(Var::make(X));
    int correct = 0;
    for (int i = 0; i < X.rows(); ++i) {
        int pred = 0;
        int truth = 0;
        out->data.row(i).maxCoeff(&pred);
        Y.row(i).maxCoeff(&truth);
        if (pred == truth) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(X.rows());
}

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

void test_multibatch_cross_entropy_adam() {
    std::srand(1);
    const int N_train = 300;
    const int N_test = 150;
    const int D = 12;
    const int C = 3;
    Mat centers = Mat::Zero(C, D);
    centers(0, 0) = 3.f;  centers(0, 1) = 1.f;
    centers(1, 0) = -3.f; centers(1, 2) = 1.f;
    centers(2, 1) = -3.f; centers(2, 3) = 1.f;

    auto make_split = [&](int n, unsigned seed, Mat& X, Mat& Y) {
        X.resize(n, D);
        Y = Mat::Zero(n, C);
        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0.f, 0.35f);
        for (int i = 0; i < n; ++i) {
            int cls = i % C;
            for (int d = 0; d < D; ++d) X(i, d) = centers(cls, d) + noise(rng);
            Y(i, cls) = 1.f;
        }
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        Mat Xs(n, D);
        Mat Ys(n, C);
        for (int i = 0; i < n; ++i) {
            Xs.row(i) = X.row(idx[i]);
            Ys.row(i) = Y.row(idx[i]);
        }
        X = Xs;
        Y = Ys;
    };

    Mat Xtr, Ytr, Xte, Yte;
    make_split(N_train, 123, Xtr, Ytr);
    make_split(N_test, 456, Xte, Yte);

    auto net = std::make_shared<Sequential>();
    net->add(std::make_shared<Linear>(D, 16));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<Linear>(16, C));

    Adam opt(net->parameters(), 0.01f);
    const int batch = 25;
    for (int epoch = 0; epoch < 10; ++epoch) {
        for (int off = 0; off < N_train; off += batch) {
            const int bs = std::min(batch, N_train - off);
            Mat xb = Xtr.middleRows(off, bs);
            Mat yb = Ytr.middleRows(off, bs);
            opt.zero_grad();
            auto pred = net->forward(Var::make(xb));
            auto loss = cross_entropy(pred, yb);
            loss->backward();
            opt.step();
        }
    }

    float train_acc = classification_accuracy(net, Xtr, Ytr);
    float test_acc = classification_accuracy(net, Xte, Yte);
    CHECK(train_acc > 0.98f);
    CHECK(test_acc > 0.98f);
    std::cout << "[ok] multi-batch cross_entropy+Adam classification: train="
              << train_acc << " test=" << test_acc << "\n";
}

void test_mnist_loader_row_major_images() {
    const std::string dir =
        "/tmp/autograd_mnist_loader_test_" +
        std::to_string(static_cast<long long>(::getpid()));
    ::mkdir(dir.c_str(), 0700);

    const std::vector<unsigned char> pixels = {
        1, 2, 3, 4, 5, 6,
        11, 12, 13, 14, 15, 16
    };
    const std::vector<unsigned char> labels = {2, 7};
    write_idx_images(dir + "/train-images-idx3-ubyte", 2, 2, 3, pixels);
    write_idx_labels(dir + "/train-labels-idx1-ubyte", labels);
    write_idx_images(dir + "/t10k-images-idx3-ubyte", 2, 2, 3, pixels);
    write_idx_labels(dir + "/t10k-labels-idx1-ubyte", labels);

    MnistData d = load_mnist(dir);
    CHECK(d.train_images.rows() == 2);
    CHECK(d.train_images.cols() == 6);
    for (int p = 0; p < 6; ++p) {
        CHECK_NEAR(d.train_images(0, p),
                   static_cast<float>(pixels[p]) / 255.f, 1e-7f);
        CHECK_NEAR(d.train_images(1, p),
                   static_cast<float>(pixels[6 + p]) / 255.f, 1e-7f);
    }
    CHECK(d.train_labels(0, 2) == 1.f);
    CHECK(d.train_labels(1, 7) == 1.f);
    std::cout << "[ok] MNIST IDX image loader preserves row-major image order\n";
}

int main() {
    test_linear_forward();
    test_sequential();
    test_sgd_step();
    test_adam_step();
    test_cross_entropy();
    test_multibatch_cross_entropy_adam();
    test_mnist_loader_row_major_images();
    test_xor();
    std::cout << "\nALL NN TESTS PASSED\n";
    return 0;
}
