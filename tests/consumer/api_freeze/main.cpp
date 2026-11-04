#include "autograd/autograd.h"
#include "autograd.h"

#if defined(EIGEN_WORLD_VERSION) || defined(CUDART_VERSION) || defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "normal umbrella leaked an expert header"
#endif
#if defined(AUTOGRAD_USE_CUDA) || defined(AUTOGRAD_TEST_CUDA_ENABLED)
#error "private backend/test macro leaked"
#endif

#include <cstdio>
#include <type_traits>

static_assert(std::is_same<ag::Linear, ag::nn::Linear>::value, "identity alias");
static_assert(std::is_same<ag::AdamState, ag::optim::AdamState>::value, "identity state");

int main() {
    const float host[] = {2.f, 3.f};
    ag::Variable x(ag::Tensor::from_host(host, ag::Shape{1, 2}), true);
    auto y = ag::sum(ag::scale(x, 2.f));
    y.backward();
    float value = 0.f, gradient[2] = {};
    y.value().copy_to_host(&value, 1);
    x.grad().copy_to_host(gradient, 2);
    if (value != 10.f || gradient[0] != 2.f || gradient[1] != 2.f) return 1;
    std::puts("OK: installed frozen API forward10/backward2");
    return 0;
}
