#include "autograd/tensor.h"
#include "autograd/core/variable.h"
#include "autograd/core/ops.h"
#include "autograd/core/loss.h"
#include "autograd/core/module.h"
#include "autograd/core/optim.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "OOP consumer headers must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "OOP consumer headers must not include CUDA runtime headers"
#endif
#if defined(AUTOGRAD_USE_CUDA)
#error "normal OOP consumer must not receive AUTOGRAD_USE_CUDA"
#endif

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

bool near(float actual, float expected, float tolerance = 1e-4f) {
    return std::fabs(actual - expected) <= tolerance;
}

ag::Tensor tensor(const std::vector<float>& values, const ag::Shape& shape) {
    return ag::Tensor::from_host(values.data(), shape);
}

float scalar(const ag::Tensor& value) {
    float result = 0.f;
    value.copy_to_host(&result, 1);
    return result;
}

std::vector<float> values(const ag::Tensor& value) {
    std::vector<float> result(value.elements());
    value.copy_to_host(result.empty() ? nullptr : result.data(), result.size());
    return result;
}

ag::Variable loss_for(ag::nn::Sequential& net,
                      const ag::Variable& input,
                      const ag::Tensor& target) {
    return ag::mse_loss(net.forward(input), target);
}

bool fail(const char* message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return false;
}

}  // namespace

int main() {
    ag::nn::Sequential net;
    net.add(std::make_shared<ag::nn::Linear>(2, 4));
    net.add(std::make_shared<ag::nn::ReLU>());
    net.add(std::make_shared<ag::nn::Linear>(4, 1));

    const auto named = net.named_parameters();
    if (named.size() != 4) return fail("registered parameter count");
    const char* expected_names[] = {
        "0.weight", "0.bias", "2.weight", "2.bias"};
    const ag::Shape expected_shapes[] = {
        ag::Shape{2, 4}, ag::Shape{1, 4}, ag::Shape{4, 1}, ag::Shape{1, 1}};
    for (std::size_t i = 0; i < named.size(); ++i) {
        if (named[i].name != expected_names[i]) {
            return fail("registered parameter name traversal");
        }
        if (named[i].parameter.value().shape() != expected_shapes[i]) {
            return fail("registered parameter shape traversal");
        }
    }

    const std::vector<std::vector<float>> initial_parameters = {
        {0.1f, 0.2f, 0.3f, 0.4f,
         -0.1f, 0.2f, -0.3f, 0.4f},
        {0.f, 0.f, 0.f, 0.f},
        {0.5f, -0.5f, 0.25f, 0.75f},
        {0.1f},
    };
    for (std::size_t i = 0; i < named.size(); ++i) {
        ag::Tensor parameter = named[i].parameter.value();
        parameter.copy_from_host(initial_parameters[i].data(),
                                 initial_parameters[i].size());
    }

    const std::vector<float> input_values = {1.f, 2.f, 3.f, 4.f};
    const std::vector<float> target_values = {0.5f, -0.3f};
    ag::Variable input(tensor(input_values, ag::Shape{2, 2}));
    ag::Tensor target = tensor(target_values, ag::Shape{2, 1});

    ag::Variable first_loss = loss_for(net, input, target);
    const float initial_loss = scalar(first_loss.value());
    if (!std::isfinite(initial_loss)) return fail("finite initial loss");
    first_loss.backward();
    for (const auto& parameter : named) {
        if (!parameter.parameter.has_grad()) {
            return fail("registered parameter backward gradient");
        }
    }

    ag::optim::Adam adam(net.parameters(), 0.01f);
    adam.step();
    ag::Variable second_loss = loss_for(net, input, target);
    const float post_step_loss = scalar(second_loss.value());
    if (!(post_step_loss < initial_loss)) return fail("Adam step loss decrease");

    const auto post_first_parameters = net.parameters();
    std::vector<std::vector<float>> saved_values;
    saved_values.reserve(post_first_parameters.size());
    for (const auto& parameter : post_first_parameters) {
        saved_values.push_back(values(parameter.value()));
    }

    const ag::optim::AdamState snapshot = adam.state();
    if (snapshot.t != 1 || snapshot.lr != 0.01f ||
        snapshot.first_moments.size() != named.size() ||
        snapshot.second_moments.size() != named.size()) {
        return fail("AdamState snapshot");
    }

    adam.zero_grad();
    ag::Variable third_loss = loss_for(net, input, target);
    third_loss.backward();
    adam.step();
    const float continued_loss = scalar(loss_for(net, input, target).value());
    if (!(continued_loss < post_step_loss)) {
        return fail("Adam continuation loss decrease");
    }

    ag::optim::Adam replay(net.parameters(), 0.5f, 0.5f, 0.8f, 0.1f);
    for (std::size_t i = 0; i < post_first_parameters.size(); ++i) {
        ag::Tensor parameter = post_first_parameters[i].value();
        parameter.copy_from_host(saved_values[i].data(), saved_values[i].size());
    }
    replay.load_state(snapshot);
    if (replay.step_count() != 1 || replay.learning_rate() != 0.01f ||
        replay.beta1() != 0.9f || replay.beta2() != 0.999f ||
        replay.eps() != 1e-8f) {
        return fail("AdamState load_state hyperparameters");
    }
    replay.zero_grad();
    ag::Variable replay_loss = loss_for(net, input, target);
    replay_loss.backward();
    replay.step();
    if (!near(scalar(loss_for(net, input, target).value()), continued_loss)) {
        return fail("AdamState load_state continuation");
    }

    std::printf("OK: OOP forward/backward, named parameters, Adam, and state roundtrip\n");
    return 0;
}
