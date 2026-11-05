#include "autograd/core/module.h"
#include "autograd/core/optim.h"
#include "autograd/core/loss.h"
#include "grad_check.h"

#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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

static void check_mat_near(const Tensor& a, const Tensor& b, float tol, const char* what) {
    if (a.shape() != b.shape()) {
        std::cerr << "FAIL: " << what << " shape mismatch\n";
        std::exit(1);
    }
    float diff = max_diff(a, b);
    if (!std::isfinite(diff) || diff > tol) {
        std::cerr << "FAIL: " << what << " max diff " << diff << "\n";
        std::exit(1);
    }
}

static void check_unary_op(const char* name, const Tensor& input,
                           Variable (*op)(const Variable&), float tol = 1e-4f) {
    auto cpu_x = Variable(input, true);
    auto cpu_y = op(cpu_x);
    sum(cpu_y).backward();

    auto cuda_x = Variable(input, true).to(Device::cuda());
    auto cuda_y = op(cuda_x);
    CHECK(cuda_y.device().is_cuda());
    sum(cuda_y).backward();

    check_mat_near(cuda_y.value(), cpu_y.value(), tol, name);
    check_mat_near(cuda_x.grad(), cpu_x.grad(), tol, name);
}

static void check_binary_op(const char* name, const Tensor& lhs, const Tensor& rhs,
                            Variable (*op)(const Variable&, const Variable&), float tol = 1e-4f) {
    auto cpu_a = Variable(lhs, true);
    auto cpu_b = Variable(rhs, true);
    auto cpu_y = op(cpu_a, cpu_b);
    sum(cpu_y).backward();

    auto cuda_a = Variable(lhs, true).to(Device::cuda());
    auto cuda_b = Variable(rhs, true).to(Device::cuda());
    auto cuda_y = op(cuda_a, cuda_b);
    CHECK(cuda_y.device().is_cuda());
    sum(cuda_y).backward();

    check_mat_near(cuda_y.value(), cpu_y.value(), tol, name);
    check_mat_near(cuda_a.grad(), cpu_a.grad(), tol, name);
    check_mat_near(cuda_b.grad(), cpu_b.grad(), tol, name);
}

class DeviceLinear : public nn::Module {
public:
    DeviceLinear(int in, int out, Device device) {
        nn::Linear initial(in, out);
        weight_ = Variable(initial.weight().value().to(device).clone(), true);
        bias_ = Variable(initial.bias().value().to(device).clone(), true);
        register_parameter("weight", weight_); register_parameter("bias", bias_);
    }
    Variable forward(const Variable& x) override { return broadcast_add(matmul(x, weight_), bias_); }
    const Variable& weight() const { return weight_; }
    const Variable& bias() const { return bias_; }
private:
    Variable weight_, bias_;
};
class FixtureConv : public nn::Module {
public:
    FixtureConv(Tensor weight, Tensor bias, Device device, int stride, int pad)
        : weight_(weight.to(device).clone(), true), bias_(bias.to(device).clone(), true),
          stride_(stride), pad_(pad) {
        register_parameter("weight", weight_); register_parameter("bias", bias_);
    }
    Variable forward(const Variable& x) override { return conv2d(x, weight_, bias_, stride_, pad_); }
    const Variable& weight() const { return weight_; }
    const Variable& bias() const { return bias_; }
private:
    Variable weight_, bias_;
    int stride_, pad_;
};

int main() {
    int device_count = 0, current_device = -1;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess);
    CHECK(device_count > 0);
    CHECK(cudaGetDevice(&current_device) == cudaSuccess);
    CHECK(current_device >= 0);
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, current_device) == cudaSuccess);
    CHECK(properties.name[0] != '\0');
    CHECK(properties.major > 0);
    std::cout << "CUDA device " << current_device << ": " << properties.name
              << " compute " << properties.major << "." << properties.minor << "\n";

    Tensor m = tensor(Shape{2, 3}, {1.f, -2.f, 3.f,
         4.f, -5.f, 6.f});

    auto x = Variable(m, true).to(Device::cuda());
    CHECK(x.device().is_cuda());

    auto y = sum(relu(add(x, x)));
    CHECK(y.device().is_cuda());
    y.backward();

    auto y_cpu = y.to(Device::cpu());
    CHECK_NEAR(at(y_cpu.value(), 0, 0), 28.f, 1e-4f);

    const Tensor x_cpu_grad = x.grad().to(Device::cpu());
    CHECK_NEAR(at(x_cpu_grad, 0, 0), 2.f, 1e-4f);
    CHECK_NEAR(at(x_cpu_grad, 0, 1), 0.f, 1e-4f);
    CHECK_NEAR(at(x_cpu_grad, 0, 2), 2.f, 1e-4f);
    CHECK_NEAR(at(x_cpu_grad, 1, 0), 2.f, 1e-4f);
    CHECK_NEAR(at(x_cpu_grad, 1, 1), 0.f, 1e-4f);
    CHECK_NEAR(at(x_cpu_grad, 1, 2), 2.f, 1e-4f);

    {
        Tensor slice_data = Tensor::zeros(Shape{4,5});
        for (int i = 0; i < slice_data.elements(); ++i) {
            put(slice_data, i, 0.1f * static_cast<float>(i - 7));
        }

        auto cpu_mean_x = Variable(slice_data, true);
        auto cpu_mean = mean(cpu_mean_x);
        cpu_mean.backward();
        auto cuda_mean_x = Variable(slice_data, true).to(Device::cuda());
        auto cuda_mean = mean(cuda_mean_x);
        CHECK(cuda_mean.device().is_cuda());
        cuda_mean.backward();
        CHECK_NEAR(at(cuda_mean.value(), 0, 0), at(cpu_mean.value(), 0, 0), 1e-4f);
        check_mat_near(cuda_mean_x.grad(), cpu_mean_x.grad(), 1e-4f,
                       "cuda mean grad");

        auto cpu_col_x = Variable(slice_data, true);
        auto cpu_col = col_slice(cpu_col_x, 1, 3);
        sum(cpu_col).backward();
        auto cuda_col_x = Variable(slice_data, true).to(Device::cuda());
        auto cuda_col = col_slice(cuda_col_x, 1, 3);
        CHECK(cuda_col.device().is_cuda());
        CHECK(cuda_col.value().shape()[0] == 4 && cuda_col.value().shape()[1] == 3);
        sum(cuda_col).backward();
        check_mat_near(cuda_col.value(), cpu_col.value(), 1e-4f,
                       "cuda col_slice forward");
        check_mat_near(cuda_col_x.grad(), cpu_col_x.grad(), 1e-4f,
                       "cuda col_slice grad");

        auto cpu_row_x = Variable(slice_data, true);
        auto cpu_row = row_slice(cpu_row_x, 1, 2);
        sum(cpu_row).backward();
        auto cuda_row_x = Variable(slice_data, true).to(Device::cuda());
        auto cuda_row = row_slice(cuda_row_x, 1, 2);
        CHECK(cuda_row.device().is_cuda());
        CHECK(cuda_row.value().shape()[0] == 2 && cuda_row.value().shape()[1] == 5);
        sum(cuda_row).backward();
        check_mat_near(cuda_row.value(), cpu_row.value(), 1e-4f,
                       "cuda row_slice forward");
        check_mat_near(cuda_row_x.grad(), cpu_row_x.grad(), 1e-4f,
                       "cuda row_slice grad");

        auto finite_difference_regularizer = [](Variable v) {
            auto dx = sub(col_slice(v, 1, v.value().shape()[1] - 1),
                          col_slice(v, 0, v.value().shape()[1] - 1));
            auto dy = sub(row_slice(v, 1, v.value().shape()[0] - 1),
                          row_slice(v, 0, v.value().shape()[0] - 1));
            auto eps_x = Variable(filled(dx.value().shape(), 1e-3f).to(dx.device()));
            auto eps_y = Variable(filled(dy.value().shape(), 1e-3f).to(dy.device()));
            auto penalty_x = mean(sqrt_op(add(mul(dx, dx), eps_x)));
            auto penalty_y = mean(sqrt_op(add(mul(dy, dy), eps_y)));
            return add(penalty_x, penalty_y);
        };

        auto cpu_reg_x = Variable(slice_data, true);
        auto cpu_reg = finite_difference_regularizer(cpu_reg_x);
        cpu_reg.backward();
        auto cuda_reg_x = Variable(slice_data, true).to(Device::cuda());
        auto cuda_reg = finite_difference_regularizer(cuda_reg_x);
        CHECK(cuda_reg.device().is_cuda());
        cuda_reg.backward();
        CHECK_NEAR(at(cuda_reg.value(), 0, 0), at(cpu_reg.value(), 0, 0), 1e-4f);
        check_mat_near(cuda_reg_x.grad(), cpu_reg_x.grad(), 1e-4f,
                       "cuda finite difference regularizer grad");

        bool range_threw = false;
        try {
            (void)col_slice(Variable(slice_data, true).to(Device::cuda()), 1, 0);
        } catch (const std::invalid_argument&) {
            range_threw = true;
        }
        CHECK(range_threw);

        range_threw = false;
        try {
            (void)row_slice(Variable(slice_data, true).to(Device::cuda()), 3, 2);
        } catch (const std::invalid_argument&) {
            range_threw = true;
        }
        CHECK(range_threw);
    }

    {
        Tensor shaped_data = tensor(Shape{2, 6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
        auto shaped_cpu = Variable(shaped_data.reshape(Shape{2,3,1,2}), true);
        auto shaped_cuda = shaped_cpu.to(Device::cuda());
        CHECK(shaped_cuda.device().is_cuda());
        CHECK(shaped_cuda.value().shape().rank() == 4);
        CHECK(shaped_cuda.value().shape()[0] == 2 && shaped_cuda.value().shape()[1] == 3);
        CHECK(shaped_cuda.value().shape()[2] == 1 && shaped_cuda.value().shape()[3] == 2);
        auto shaped_roundtrip = shaped_cuda.to(Device::cpu());
        CHECK(!shaped_roundtrip.device().is_cuda());
        CHECK(shaped_roundtrip.value().shape().rank() == 4);
        CHECK(shaped_roundtrip.value().shape()[0] == 2 && shaped_roundtrip.value().shape()[1] == 3);
        CHECK(shaped_roundtrip.value().shape()[2] == 1 && shaped_roundtrip.value().shape()[3] == 2);
        check_mat_near(shaped_roundtrip.value(), shaped_data.reshape(Shape{2,3,1,2}), 1e-4f, "cuda/cpu shape roundtrip data");

        Tensor owned = shaped_cuda.value();
        auto nines = host(filled(owned.shape(), 9.f));
        owned.copy_from_host(nines.data(), nines.size());
        CHECK(shaped_cuda.value().shape().rank() == 4);
        auto transferred = owned.to(Device::cpu());
        CHECK(transferred.shape().rank() == 4);
        CHECK_NEAR(at(transferred, 1, 5), 9.f, 1e-4f);
        seed_grad(shaped_cuda, filled(owned.shape(), 4.f));
        CHECK(shaped_cuda.grad().shape().rank() == 4);
        auto host_gradient = shaped_cuda.grad().to(Device::cpu());
        CHECK(host_gradient.shape().rank() == 4);
        CHECK_NEAR(at(host_gradient, 0, 3), 4.f, 1e-4f);
        shaped_cuda.zero_grad();
        CHECK(!shaped_cuda.has_grad());
        CHECK(!shaped_cuda.to(Device::cpu()).has_grad()); // transfer is a fresh leaf
        auto sync_loss = sum(scale(shaped_cuda, 3.f));
        sync_loss.backward();
        auto shaped_after_backward = shaped_cuda.grad().to(Device::cpu());
        CHECK(shaped_after_backward.shape().rank() == 4);
        CHECK_NEAR(at(shaped_after_backward, 1, 4), 3.f, 1e-4f);
        shaped_cuda.zero_grad();
        CHECK(!shaped_cuda.has_grad());
        CHECK(!shaped_cuda.to(Device::cpu()).has_grad());
    }

    auto a = Variable(filled(Shape{2, 2}, 3.f), true).to(Device::cuda());
    auto b = Variable(filled(Shape{2, 2}, 4.f), true).to(Device::cuda());
    auto z = sum(scale(mul(a, b), 0.5f));
    z.backward();
    CHECK_NEAR(at(z.value(), 0, 0), 24.f, 1e-4f);
    CHECK_NEAR(at(a.grad(), 0, 0), 2.f, 1e-4f);
    CHECK_NEAR(at(b.grad(), 0, 0), 1.5f, 1e-4f);

    bool threw = false;
    try {
        (void)transpose(a);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    bool shape_threw = false;
    try {
        (void)add(a, Variable(filled(Shape{1, 2}, 1.f), true).to(Device::cuda()));
    } catch (const std::invalid_argument&) {
        shape_threw = true;
    }
    CHECK(shape_threw);

    Tensor A = tensor(Shape{2, 3}, {1.f, 2.f, 3.f,
         4.f, 5.f, 6.f});
    Tensor B = tensor(Shape{3, 2}, {1.f, 2.f,
         3.f, 4.f,
         5.f, 6.f});
    auto ma = Variable(A, true).to(Device::cuda());
    auto mb = Variable(B, true).to(Device::cuda());
    auto mm = sum(matmul(ma, mb));
    mm.backward();
    float expected_mm = 0.f;
    std::vector<float> expected_ma_grad(6, 0.f), expected_mb_grad(6, 0.f);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 3; ++k) {
                expected_mm += at(A, i, k)*at(B, k, j);
                expected_ma_grad[i*3+k] += at(B, k, j);
                expected_mb_grad[k*2+j] += at(A, i, k);
            }
    CHECK_NEAR(at(mm.value()), expected_mm, 1e-4f);
    CHECK_NEAR(at(ma.grad(), 1, 2), expected_ma_grad[5], 1e-4f);
    CHECK_NEAR(at(mb.grad(), 2, 1), expected_mb_grad[5], 1e-4f);

    auto bx = Variable(filled(Shape{4, 3}, 1.f), true).to(Device::cuda());
    auto bb = Variable(filled(Shape{1, 3}, 2.f), true).to(Device::cuda());
    auto by = sum(broadcast_add(bx, bb));
    by.backward();
    CHECK_NEAR(at(by.value(), 0, 0), 36.f, 1e-4f);
    CHECK_NEAR(at(bx.grad(), 3, 2), 1.f, 1e-4f);
    CHECK_NEAR(at(bb.grad(), 0, 2), 4.f, 1e-4f);

    auto p = Variable(filled(Shape{2, 2}, 1.f), true).to(Device::cuda());
    auto loss = sum(scale(p, 2.f));
    loss.backward();
    optim::SGD opt({p}, 0.1f);
    opt.step();
    CHECK_NEAR(at(p.value(), 0, 0), 0.8f, 1e-4f);
    opt.zero_grad();
    CHECK(!p.has_grad());
    auto loss2 = sum(scale(p, 2.f));
    loss2.backward();
    CHECK_NEAR(at(p.grad(), 0, 0), 2.f, 1e-4f);

    DeviceLinear lin(3, 2, Device::cuda());
    auto lx = Variable(filled(Shape{4, 3}, 1.f), true).to(Device::cuda());
    auto lloss = sum(lin.forward(lx));
    lloss.backward();
    optim::SGD lin_opt(lin.parameters(), 0.01f);
    Tensor w_before = lin.weight().value().clone();
    lin_opt.step();
    CHECK(lloss.device().is_cuda());
    CHECK(lin.weight().device().is_cuda());
    CHECK(max_abs(lin.weight().grad()) > 0.f);
    CHECK(max_diff(lin.weight().value(), w_before) > 0.f);
    lin.zero_grad();
    CHECK(!lin.weight().has_grad());

    Tensor logits = tensor(Shape{2, 3}, {1.f, 2.f, 3.f,
              1.f, 0.f, -1.f});
    Tensor target = Tensor::zeros(Shape{2, 3});
    put(target, 2, 1.f);
    put(target, 3, 1.f);
    auto lg = Variable(logits, true).to(Device::cuda());
    auto sm = softmax(lg).to(Device::cpu());
    CHECK_NEAR(row_sum(sm.value(), 0), 1.f, 1e-4f);
    CHECK_NEAR(row_sum(sm.value(), 1), 1.f, 1e-4f);

    auto ce = cross_entropy(lg, target.to(Device::cuda()));
    CHECK(ce.device().is_cuda());
    ce.backward();
    auto ce_cpu = ce.to(Device::cpu());
    auto lg_cpu_grad = lg.grad().to(Device::cpu());
    Tensor expected_sm = softmax(Variable(logits, true)).value();
    float expected_ce = -std::log(at(expected_sm, 0, 2)) / 2.f
                        -std::log(at(expected_sm, 1, 0)) / 2.f;
    CHECK_NEAR(at(ce_cpu.value(), 0, 0), expected_ce, 1e-4f);
    CHECK_NEAR(at(lg_cpu_grad, 0, 2), (at(expected_sm, 0, 2) - 1.f) / 2.f, 1e-4f);
    CHECK_NEAR(at(lg_cpu_grad, 1, 0), (at(expected_sm, 1, 0) - 1.f) / 2.f, 1e-4f);

    {
        Tensor act = tensor(Shape{2, 5}, {-8.f, -2.f, -0.25f, 0.5f, 8.f,
                3.f, -4.f, 0.f, 1.5f, -1.f});
        Tensor small = scale(Variable(act), .25f).value();
        Tensor pos = tensor(Shape{2, 5}, {0.1f, 0.5f, 1.f, 4.f, 16.f,
               2.f, 0.25f, 9.f, 0.75f, 6.f});
        check_unary_op("cuda sigmoid", act, sigmoid);
        check_unary_op("cuda tanh_op", act, tanh_op);
        check_unary_op("cuda exp_op", small, exp_op);
        check_unary_op("cuda log_op", pos, log_op);
        check_unary_op("cuda sqrt_op", pos, sqrt_op);
        check_unary_op("cuda silu", act, silu);
        check_unary_op("cuda softplus", act, softplus);

        Tensor lhs = tensor(Shape{2, 5}, {1.f, -2.f, 3.f, -4.f, 5.f,
               -1.5f, 2.5f, -3.5f, 4.5f, -5.5f});
        Tensor rhs = tensor(Shape{2, 5}, {0.5f, 2.f, -1.5f, 4.f, -2.f,
               3.f, -2.5f, 1.25f, -4.5f, 5.5f});
        Tensor div_rhs = tensor(Shape{2, 5}, {0.75f, 1.5f, 2.5f, 4.f, 8.f,
                   1.25f, 3.f, 5.f, 6.f, 9.f});
        check_binary_op("cuda sub", lhs, rhs, sub);
        check_binary_op("cuda div_op", lhs, div_rhs, div_op);

        auto shaped_a = Variable(filled(Shape{2,3,2,2}, 1.f), true).to(Device::cuda());
        auto shaped_b = Variable(filled(Shape{2,3,2,2}, .5f), true).to(Device::cuda());
        auto shaped_unary = sigmoid(shaped_a);
        auto shaped_binary = sub(shaped_a, shaped_b);
        CHECK(shaped_unary.value().shape().rank() == 4);
        CHECK(shaped_binary.value().shape().rank() == 4);
        CHECK(shaped_unary.value().shape()[0] == 2 && shaped_unary.value().shape()[1] == 3);
        CHECK(shaped_unary.value().shape()[2] == 2 && shaped_unary.value().shape()[3] == 2);
        CHECK(shaped_binary.value().shape()[0] == 2 && shaped_binary.value().shape()[1] == 3);
        CHECK(shaped_binary.value().shape()[2] == 2 && shaped_binary.value().shape()[3] == 2);

        bool mixed_threw = false;
        try {
            (void)sub(Variable(lhs, true).to(Device::cuda()), Variable(rhs, true));
        } catch (const std::invalid_argument&) {
            mixed_threw = true;
        }
        CHECK(mixed_threw);

        mixed_threw = false;
        try {
            (void)div_op(Variable(lhs, true), Variable(div_rhs, true).to(Device::cuda()));
        } catch (const std::invalid_argument&) {
            mixed_threw = true;
        }
        CHECK(mixed_threw);
    }

    {
        Tensor param0 = tensor(Shape{2, 3}, {1.f, -2.f, 3.f,
                  -4.f, 5.f, -6.f});
        Tensor grad1 = tensor(Shape{2, 3}, {0.1f, -0.2f, 0.3f,
                 -0.4f, 0.5f, -0.6f});
        Tensor grad2 = tensor(Shape{2, 3}, {-0.3f, 0.25f, -0.2f,
                  0.15f, -0.1f, 0.05f});
        Tensor grad3 = tensor(Shape{2, 3}, {0.01f, 0.02f, -0.03f,
                 -0.04f, 0.05f, 0.06f});
        std::vector<Tensor> grads = {grad1, grad2, grad3};

        auto cpu_p = Variable(param0.clone(), true);
        auto cuda_p = Variable(param0.clone(), true).to(Device::cuda());
        optim::Adam cpu_adam({cpu_p}, 0.01f);
        optim::Adam cuda_adam({cuda_p}, 0.01f);

        for (const auto& g : grads) {
            seed_grad(cpu_p, g);
            seed_grad(cuda_p, g);
            cpu_adam.step();
            cuda_adam.step();
            check_mat_near(cuda_p.value(), cpu_p.value(), 1e-4f, "cuda optim::Adam data");
        }
        cuda_adam.zero_grad();
        CHECK(!cuda_p.has_grad());

        auto mixed_cpu = Variable(param0.clone(), true);
        auto mixed_cuda = Variable(param0.clone(), true).to(Device::cuda());
        auto ref_cpu = Variable(param0.clone(), true);
        auto ref_cuda = Variable(param0.clone(), true).to(Device::cuda());
        optim::Adam mixed_adam({mixed_cpu, mixed_cuda}, 0.01f);
        optim::Adam ref_cpu_adam({ref_cpu}, 0.01f);
        optim::Adam ref_cuda_adam({ref_cuda}, 0.01f);
        seed_grad(mixed_cpu, grad1);
        seed_grad(ref_cpu, grad1);
        seed_grad(mixed_cuda, grad2);
        seed_grad(ref_cuda, grad2);
        mixed_adam.step();
        ref_cpu_adam.step();
        ref_cuda_adam.step();
        check_mat_near(mixed_cpu.value(), ref_cpu.value(), 1e-4f, "mixed optim::Adam CPU param");
        check_mat_near(mixed_cuda.value(), ref_cuda.value(), 1e-4f,
                       "mixed optim::Adam CUDA param");
    }

    {
        const int N = 2, C = 2, H = 4, W = 5;
        const int out_ch = 3, kH = 3, kW = 2, stride = 1, pad = 1;
        Tensor conv_x = Tensor::zeros(Shape{N,C,H,W});
        Tensor conv_w = Tensor::zeros(Shape{out_ch,C,kH,kW});
        Tensor conv_b = Tensor::zeros(Shape{out_ch});
        for (int i = 0; i < conv_x.elements(); ++i) {
            put(conv_x, i, 0.03f * static_cast<float>((i % 19) - 9));
        }
        for (int i = 0; i < conv_w.elements(); ++i) {
            put(conv_w, i, 0.02f * static_cast<float>((i % 17) - 8));
        }
        for (int i = 0; i < conv_b.elements(); ++i) {
            put(conv_b, i, 0.05f * static_cast<float>(i - 1));
        }

        FixtureConv conv_cpu(conv_w, conv_b, Device::cpu(), stride, pad);
        auto cx = Variable(conv_x.clone(), true);
        auto cy = conv_cpu.forward(cx);
        sum(cy).backward();

        FixtureConv conv_cuda(conv_w, conv_b, Device::cuda(), stride, pad);
        auto gx = Variable(conv_x.clone(), true).to(Device::cuda());
        auto gy = conv_cuda.forward(gx);
        CHECK(gy.device().is_cuda());
        CHECK(gy.value().shape().rank() == 4);
        sum(gy).backward();

        check_mat_near(gy.value(), cy.value(), 1e-4f, "cuda nn::Conv2d forward");
        check_mat_near(gx.grad(), cx.grad(), 1e-4f, "cuda nn::Conv2d grad input");
        check_mat_near(conv_cuda.weight().grad(), conv_cpu.weight().grad(), 1e-4f,
                       "cuda nn::Conv2d grad weight");
        check_mat_near(conv_cuda.bias().grad(), conv_cpu.bias().grad(), 1e-4f,
                       "cuda nn::Conv2d grad bias");
    }

    {
        const int N = 1, C = 2, H = 5, W = 7;
        const int out_ch = 2, kH = 3, kW = 3, stride = 2, pad = 1;
        Tensor conv_x = Tensor::zeros(Shape{N,C,H,W});
        Tensor conv_w = Tensor::zeros(Shape{out_ch,C,kH,kW});
        Tensor conv_b = Tensor::zeros(Shape{out_ch});
        for (int i = 0; i < conv_x.elements(); ++i) {
            put(conv_x, i, 0.01f * static_cast<float>((i % 23) - 11));
        }
        for (int i = 0; i < conv_w.elements(); ++i) {
            put(conv_w, i, 0.015f * static_cast<float>((i % 13) - 6));
        }
        put(conv_b, 0, -.03f); put(conv_b, 1, .04f);

        FixtureConv conv_cpu(conv_w, conv_b, Device::cpu(), stride, pad);
        auto cx = Variable(conv_x.clone(), true);
        auto cy = conv_cpu.forward(cx);
        CHECK(cy.value().shape()[2] == 3 && cy.value().shape()[3] == 4);
        sum(add(cy, cy)).backward();

        FixtureConv conv_cuda(conv_w, conv_b, Device::cuda(), stride, pad);
        auto gx = Variable(conv_x.clone(), true).to(Device::cuda());
        auto gy = conv_cuda.forward(gx);
        CHECK(gy.device().is_cuda());
        CHECK(gy.value().shape().rank() == 4);
        CHECK(gy.value().shape()[0] == N && gy.value().shape()[1] == out_ch);
        CHECK(gy.value().shape()[2] == 3 && gy.value().shape()[3] == 4);
        sum(add(gy, gy)).backward();

        check_mat_near(gy.value(), cy.value(), 1e-4f,
                       "cuda nn::Conv2d stride/pad forward");
        check_mat_near(gx.grad(), cx.grad(), 1e-4f,
                       "cuda nn::Conv2d stride/pad repeated grad input");
        check_mat_near(conv_cuda.weight().grad(), conv_cpu.weight().grad(), 1e-4f,
                       "cuda nn::Conv2d stride/pad repeated grad weight");
        check_mat_near(conv_cuda.bias().grad(), conv_cpu.bias().grad(), 1e-4f,
                       "cuda nn::Conv2d stride/pad repeated grad bias");
    }

    {
        const int N = 2, C = 2, H = 4, W = 4;
        Tensor pool_x = Tensor::zeros(Shape{N,C,H,W});
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        put(pool_x, ((n*C+c)*H+h)*W+w,
                            static_cast<float>(c*H*W+h*W+w) + .01f*n);
                    }
                }
            }
        }

        nn::MaxPool2d pool(2, 2, 1);
        auto px = Variable(pool_x.clone(), true);
        auto py = pool.forward(px);
        sum(py).backward();

        auto pgx = Variable(pool_x.clone(), true).to(Device::cuda());
        auto pgy = pool.forward(pgx);
        CHECK(pgy.device().is_cuda());
        CHECK(pgy.value().shape().rank() == 4);
        sum(pgy).backward();

        check_mat_near(pgy.value(), py.value(), 1e-4f, "cuda nn::MaxPool2d forward");
        check_mat_near(pgx.grad(), px.grad(), 1e-4f, "cuda nn::MaxPool2d grad input");
    }

    {
        const int N = 2, C = 2, H = 6, W = 7;
        Tensor pool_x = Tensor::zeros(Shape{N,C,H,W});
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        put(pool_x, ((n*C+c)*H+h)*W+w,
                            static_cast<float>(c*H*W+h*W+w) + .01f*n);
                    }
                }
            }
        }

        nn::MaxPool2d pool(2, 3, 2);
        auto px = Variable(pool_x.clone(), true);
        auto py = pool.forward(px);
        CHECK(py.value().shape()[2] == 3 && py.value().shape()[3] == 3);
        sum(add(py, py)).backward();

        auto pgx = Variable(pool_x.clone(), true).to(Device::cuda());
        auto pgy = pool.forward(pgx);
        CHECK(pgy.device().is_cuda());
        CHECK(pgy.value().shape().rank() == 4);
        CHECK(pgy.value().shape()[0] == N && pgy.value().shape()[1] == C);
        CHECK(pgy.value().shape()[2] == 3 && pgy.value().shape()[3] == 3);
        sum(add(pgy, pgy)).backward();

        check_mat_near(pgy.value(), py.value(), 1e-4f,
                       "cuda nn::MaxPool2d stride forward");
        check_mat_near(pgx.grad(), px.grad(), 1e-4f,
                       "cuda nn::MaxPool2d repeated grad input");
    }

    {
        auto unsupported_x = Variable(filled(Shape{1,2,4,4}, 1.f), true).to(Device::cuda());

        bool unsupported_threw = false;
        try {
            nn::AvgPool2d avg(2, 2);
            (void)avg.forward(unsupported_x);
        } catch (const std::runtime_error&) {
            unsupported_threw = true;
        }
        CHECK(unsupported_threw);

        unsupported_threw = false;
        try {
            (void)depthwise_conv2d(unsupported_x,
                Variable(Tensor::ones(Shape{2,3,3}, Device::cuda()), true),
                Variable(Tensor::zeros(Shape{2}, Device::cuda()), true), 1, 1);
        } catch (const std::runtime_error&) {
            unsupported_threw = true;
        }
        CHECK(unsupported_threw);

        unsupported_threw = false;
        try {
            nn::NearestUpsample2d upsample(2);
            (void)upsample.forward(unsupported_x);
        } catch (const std::runtime_error&) {
            unsupported_threw = true;
        }
        CHECK(unsupported_threw);
    }

    std::cout << "ALL CUDA CORE TESTS PASSED\n";
    return 0;
}
