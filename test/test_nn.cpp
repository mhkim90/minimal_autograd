// test_nn.cpp — end-to-end neural net training: XOR with 2-layer MLP + optim::Adam.

#include "autograd/core/module.h"
#include "autograd/core/optim.h"
#include "autograd/core/loss.h"
#include "grad_check.h"
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
using namespace ag_test;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define CHECK_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (!std::isfinite(_a) || !std::isfinite(_b) || std::fabs(_a - _b) > (tol)) { \
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

float classification_accuracy(std::shared_ptr<nn::Sequential> net,
                              const Tensor& X, const Tensor& Y) {
    return accuracy_argmax(net->forward(Variable(X)).value(), Y);
}

void test_linear_forward() {
    nn::Linear lin(4, 3);
    nn::Linear same(4, 3);
    CHECK((lin.weight().value().shape() == Shape{4, 3}));
    CHECK((lin.bias().value().shape() == Shape{1, 3}));
    CHECK(max_abs(lin.weight().value()) <= std::sqrt(6.f / 7.f));
    CHECK(max_abs(lin.bias().value()) == 0.f);
    CHECK(max_diff(lin.weight().value(), same.weight().value()) == 0.f);
    auto x = Variable(random_tensor(Shape{5, 4}), true);   // batch of 5
    auto y = lin.forward(x);
    CHECK(y.value().shape()[0] == 5);
    CHECK(y.value().shape()[1] == 3);
    std::cout << "[ok] nn::Linear forward shape (5,3)\n";
}

void test_sequential() {
    auto net = std::make_shared<nn::Sequential>();
    net->add(std::make_shared<nn::Linear>(4, 8));
    net->add(std::make_shared<nn::ReLU>());
    net->add(std::make_shared<nn::Linear>(8, 2));

    auto x = Variable(random_tensor(Shape{3, 4}), true);
    auto y = net->forward(x);
    CHECK(y.value().shape()[0] == 3);
    CHECK(y.value().shape()[1] == 2);
    auto params = net->parameters();
    CHECK(params.size() == 4);   // 2 weights + 2 biases
    std::cout << "[ok] nn::Sequential(nn::Linear, ReLU, nn::Linear) forward shape (3,2)\n";
}

// XOR — should converge in <500 epochs.
void test_xor() {
    Tensor X = tensor(Shape{4, 2}, {0, 0,
         0, 1,
         1, 0,
         1, 1});
    Tensor Y = tensor(Shape{4, 1}, {0,
         1,
         1,
         0});

    auto net = std::make_shared<nn::Sequential>();
    net->add(std::make_shared<nn::Linear>(2, 8));
    net->add(std::make_shared<nn::ReLU>());
    net->add(std::make_shared<nn::Linear>(8, 1));

    auto opt = optim::Adam(net->parameters(), 0.05f);

    int target_epochs = 200;
    for (int epoch = 0; epoch < target_epochs; ++epoch) {
        opt.zero_grad();
        auto x = Variable(X, true);
        auto y_pred = net->forward(x);
        auto loss = mse_loss(y_pred, Y);
        loss.backward();
        opt.step();
        if (epoch % 50 == 0)
            std::cout << "  epoch " << epoch
                      << "  loss=" << at(loss.value(), 0, 0) << "\n";
    }

    auto final_pred = net->forward(Variable(X, true));
    float err = 0.f;
    for (int i = 0; i < 4; ++i) err += std::fabs(at(final_pred.value(), i, 0) - at(Y, i, 0));
    std::cout << "  final err=" << err << "\n";
    CHECK(err < 0.3f);
    std::cout << "[ok] XOR learned in <" << target_epochs
              << " epochs (err<0.3)\n";
}

void test_sgd_step() {
    auto p = Variable(filled(Shape{1, 1}, 1.f), true);
    seed_grad(p, filled(Shape{1, 1}, 0.5f));
    optim::SGD sgd({p}, 0.1f);
    sgd.step();
    CHECK_NEAR(at(p.value(), 0, 0), 0.95f, 1e-5f);
    std::cout << "[ok] optim::SGD step: 1 - 0.1*0.5 = 0.95\n";
}

void test_adam_step() {
    auto p = Variable(filled(Shape{1, 1}, 1.f), true);
    seed_grad(p, filled(Shape{1, 1}, 1.f));
    optim::Adam adam({p}, 0.1f);
    const auto initial = adam.state();
    CHECK(initial.t == 0);
    CHECK(max_abs(initial.first_moments[0]) == 0.f);
    CHECK(max_abs(initial.second_moments[0]) == 0.f);
    adam.step();
    // m = 0.1, v = 0.001, m_hat = 0.1/0.1 = 1, v_hat = 0.001/0.001 = 1
    // update = 0.1 * 1 / (1 + 1e-8) ≈ 0.1
    CHECK(std::fabs(at(p.value(), 0, 0) - 0.9f) < 1e-3f);
    std::cout << "[ok] optim::Adam step: m/v/m_hat/v_hat update ok\n";
}

void test_cross_entropy() {
    Tensor pred = tensor(Shape{2, 3}, {1.0f, 2.0f, 0.5f,
            0.1f, 0.2f, 0.7f});
    Tensor target = tensor(Shape{2, 3}, {1, 0, 0,
              0, 0, 1});

    auto logits = Variable(pred, true);
    auto loss = cross_entropy(logits, target);
    CHECK(at(loss.value(), 0, 0) > 0.f);  // NLL is non-negative
    std::cout << "[ok] cross_entropy forward: loss=" << at(loss.value(), 0, 0) << "\n";
}

void test_multibatch_cross_entropy_adam() {
    std::srand(1);
    const int N_train = 300;
    const int N_test = 150;
    const int D = 12;
    const int C = 3;
    std::vector<float> centers(C * D, 0.f);
    centers[0] = 3.f; centers[1] = 1.f;
    centers[D] = -3.f; centers[D + 2] = 1.f;
    centers[2*D + 1] = -3.f; centers[2*D + 3] = 1.f;

    auto make_split = [&](int n, unsigned seed, Tensor& X, Tensor& Y) {
        std::vector<float> x(n * D), y(n * C, 0.f);
        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0.f, 0.35f);
        for (int i = 0; i < n; ++i) {
            const int cls = i % C;
            for (int d = 0; d < D; ++d) x[i*D+d] = centers[cls*D+d] + noise(rng);
            y[i*C+cls] = 1.f;
        }
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<float> shuffled_x(n*D), shuffled_y(n*C);
        for (int i = 0; i < n; ++i) {
            std::copy_n(x.begin()+idx[i]*D, D, shuffled_x.begin()+i*D);
            std::copy_n(y.begin()+idx[i]*C, C, shuffled_y.begin()+i*C);
        }
        X = tensor(Shape{n, D}, shuffled_x);
        Y = tensor(Shape{n, C}, shuffled_y);
    };
    Tensor Xtr, Ytr, Xte, Yte;
    make_split(N_train, 123, Xtr, Ytr);
    make_split(N_test, 456, Xte, Yte);

    auto net = std::make_shared<nn::Sequential>();
    net->add(std::make_shared<nn::Linear>(D, 16));
    net->add(std::make_shared<nn::ReLU>());
    net->add(std::make_shared<nn::Linear>(16, C));

    optim::Adam opt(net->parameters(), 0.01f);
    const int batch = 25;
    for (int epoch = 0; epoch < 10; ++epoch) {
        for (int off = 0; off < N_train; off += batch) {
            const int bs = std::min(batch, N_train - off);
            Tensor xb = row_slice(Variable(Xtr), off, bs).value();
            Tensor yb = row_slice(Variable(Ytr), off, bs).value();
            opt.zero_grad();
            auto pred = net->forward(Variable(xb, true));
            auto loss = cross_entropy(pred, yb);
            loss.backward();
            opt.step();
        }
    }

    float train_acc = classification_accuracy(net, Xtr, Ytr);
    float test_acc = classification_accuracy(net, Xte, Yte);
    CHECK(train_acc > 0.98f);
    CHECK(test_acc > 0.98f);
    std::cout << "[ok] multi-batch cross_entropy+optim::Adam classification: train="
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
    CHECK(d.train_images.shape()[0] == 2);
    CHECK(d.train_images.shape()[1] == 6);
    for (int p = 0; p < 6; ++p) {
        CHECK_NEAR(at(d.train_images, 0, p),
                   static_cast<float>(pixels[p]) / 255.f, 1e-7f);
        CHECK_NEAR(at(d.train_images, 1, p),
                   static_cast<float>(pixels[6 + p]) / 255.f, 1e-7f);
    }
    CHECK((d.test_images.shape() == Shape{2, 6}));
    CHECK((d.train_labels.shape() == Shape{2, 10}));
    CHECK((d.test_labels.shape() == Shape{2, 10}));
    CHECK(max_diff(d.train_images, d.test_images) == 0.f);
    CHECK(max_diff(d.train_labels, d.test_labels) == 0.f);
    CHECK(row_sum(d.train_labels, 0) == 1.f && row_sum(d.train_labels, 1) == 1.f);
    CHECK(at(d.train_labels, 0, 2) == 1.f);
    CHECK(at(d.train_labels, 1, 7) == 1.f);
    std::cout << "[ok] MNIST IDX image loader preserves row-major image order\n";
}

void test_mnist_loader_validation() {
    const std::string dir = "/tmp/autograd_mnist_validation_" +
        std::to_string(static_cast<long long>(::getpid()));
    ::mkdir(dir.c_str(), 0700);
    auto rejects = [&](const std::function<void()>& f) {
        bool threw = false;
        try { f(); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    };
    Tensor out; int n = 0;
    rejects([&] { read_idx_images(dir+"/missing", out, n); });
    rejects([&] { read_idx_labels(dir+"/missing", out); });
    const auto image = dir+"/image", label = dir+"/label";
    { std::ofstream f(image, std::ios::binary); f.put(0); }
    rejects([&] { read_idx_images(image, out, n); });
    { std::ofstream f(label, std::ios::binary); f.put(0); }
    rejects([&] { read_idx_labels(label, out); });
    { std::ofstream f(image, std::ios::binary);
      write_be_u32(f, 0); write_be_u32(f, 1); write_be_u32(f, 2); write_be_u32(f, 3); }
    rejects([&] { read_idx_images(image, out, n); });
    { std::ofstream f(label, std::ios::binary); write_be_u32(f, 0); write_be_u32(f, 1); }
    rejects([&] { read_idx_labels(label, out); });
    write_idx_images(image, 1, 2, 3, {1, 2});
    rejects([&] { read_idx_images(image, out, n); });
    { std::ofstream f(label, std::ios::binary); write_be_u32(f, 0x801); write_be_u32(f, 2); f.put(1); }
    rejects([&] { read_idx_labels(label, out); });
    write_idx_labels(label, {255});
    read_idx_labels(label, out);
    CHECK(max_abs(out) == 0.f);  // Historical out-of-class labels give allzero rows.
    write_idx_images(dir+"/train-images-idx3-ubyte", 1, 2, 3, {1,2,3,4,5,6});
    write_idx_labels(dir+"/train-labels-idx1-ubyte", {2,7});
    write_idx_images(dir+"/t10k-images-idx3-ubyte", 1, 2, 3, {1,2,3,4,5,6});
    write_idx_labels(dir+"/t10k-labels-idx1-ubyte", {2});
    rejects([&] { (void)load_mnist(dir); });
    std::cout << "[ok] MNIST IDX malformed headers/payloads/counts and allzero labels\n";
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
    test_mnist_loader_validation();
    std::cout << "\nALL NN TESTS PASSED\n";
    return 0;
}
