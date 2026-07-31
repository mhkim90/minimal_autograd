#include "autograd/tensor.h"
#include "autograd/core/ops.h"
#include "autograd/extension/custom_op.h"
#include "autograd/extension/eigen.h"

#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "Eigen custom consumer must not include CUDA runtime headers"
#endif

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-6f;
}

bool check_matrix(const Eigen::MatrixXf& actual,
                  const Eigen::MatrixXf& expected) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        return false;
    }
    for (Eigen::Index r = 0; r < actual.rows(); ++r) {
        for (Eigen::Index c = 0; c < actual.cols(); ++c) {
            if (!near(actual(r, c), expected(r, c))) return false;
        }
    }
    return true;
}

bool fail(const char* message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return false;
}

}  // namespace

int main() {
    Eigen::MatrixXf input_eigen(2, 3);
    input_eigen << 1.f, 2.f, 3.f,
                   4.f, 5.f, 6.f;

    ag::Tensor input_tensor = ag::tensor_from_eigen(input_eigen);
    if (input_tensor.shape() != ag::Shape{2, 3} ||
        !check_matrix(ag::tensor_to_eigen(input_tensor), input_eigen)) {
        return fail("Eigen/Tensor logical layout conversion");
    }

    input_eigen(0, 0) = 99.f;
    Eigen::MatrixXf copied_input = ag::tensor_to_eigen(input_tensor);
    if (!near(copied_input(0, 0), 1.f)) {
        return fail("Eigen/Tensor conversion must copy");
    }

    ag::Variable input(input_tensor, true);
    const Eigen::MatrixXf expected_output =
        2.f * copied_input + Eigen::MatrixXf::Constant(2, 3, 1.f);
    ag::Variable output = ag::make_custom_variable(
        ag::tensor_from_eigen(expected_output), {input},
        [](const ag::Tensor& output_gradient) {
            Eigen::MatrixXf gradient = ag::tensor_to_eigen(output_gradient);
            return std::vector<ag::Tensor>{
                ag::tensor_from_eigen(2.f * gradient)};
        });

    if (!check_matrix(ag::tensor_to_eigen(output.value()), expected_output)) {
        return fail("custom Eigen operation forward");
    }

    ag::sum(output).backward();
    const Eigen::MatrixXf expected_gradient =
        Eigen::MatrixXf::Constant(2, 3, 2.f);
    if (!input.has_grad() ||
        !check_matrix(ag::tensor_to_eigen(input.grad()), expected_gradient)) {
        return fail("custom Eigen operation backward");
    }

    bool rejected_rank = false;
    try {
        (void)ag::tensor_to_eigen(ag::Tensor::ones(ag::Shape{6}));
    } catch (const std::invalid_argument&) {
        rejected_rank = true;
    }
    if (!rejected_rank) return fail("tensor_to_eigen rank validation");

    std::printf("OK: Eigen custom op forward/backward with copied 2x3 layout\n");
    return 0;
}
