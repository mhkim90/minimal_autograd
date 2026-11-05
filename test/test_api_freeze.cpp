#include "autograd.h"
#include "autograd/autograd.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || defined(EIGEN_MINOR_VERSION)
#error "normal umbrellas must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "normal umbrellas must not include CUDA headers"
#endif
#if defined(AUTOGRAD_USE_CUDA) || defined(AUTOGRAD_TEST_CUDA_ENABLED)
#error "normal consumers must not receive backend/test macros"
#endif

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <utility>

#define SAME(name, ns) static_assert(std::is_same<ag::name, ag::ns::name>::value, #name " must be an identity alias")
SAME(NamedParameter, nn);
SAME(Module, nn);
SAME(Linear, nn);
SAME(ReLU, nn);
SAME(Sequential, nn);
SAME(Conv2d, nn);
SAME(MaxPool2d, nn);
SAME(DepthwiseConv2d, nn);
SAME(AvgPool2d, nn);
SAME(NearestUpsample2d, nn);
SAME(GroupNorm, nn);
SAME(SGD, optim);
SAME(Adam, optim);
SAME(AdamState, optim);
#undef SAME

static_assert(&ag::randn == &ag::diffusion::randn, "same randn function");
static_assert(&ag::randn_like == &ag::diffusion::randn_like, "same randn_like function");
static_assert(&ag::sinusoidal_time_embedding == &ag::diffusion::sinusoidal_time_embedding, "same embedding function");
static_assert(&ag::q_sample == &ag::diffusion::q_sample, "same q_sample function");
static_assert(std::is_same<decltype(ag::randn(ag::Shape{2})), ag::Tensor>::value, "default seed preserved");
static_assert(std::is_same<decltype(ag::randn_like(std::declval<const ag::Tensor&>())), ag::Tensor>::value, "default seed preserved");
static_assert(std::is_same<decltype(ag::q_sample(std::declval<const ag::Variable&>(), 1.f, 0.f)), ag::Variable>::value, "default noise/seed preserved");
static_assert(std::is_same<decltype(std::declval<const ag::Variable&>().value()), const ag::Tensor&>::value, "const value boundary");
static_assert(std::is_same<decltype(std::declval<const ag::Variable&>().grad()), const ag::Tensor&>::value, "const gradient boundary");

#define NO_FIELD(field, type) \
    template<class T, class = void> struct has_##field : std::false_type {}; \
    template<class T> struct has_##field<T, std::void_t<decltype(std::declval<T&>().field)>> : std::true_type {}; \
    static_assert(!has_##field<ag::type>::value, #type " must not expose " #field)
NO_FIELD(data, Variable);
NO_FIELD(parents, Variable);
NO_FIELD(back_fn, Variable);
NO_FIELD(node_, Variable);
NO_FIELD(cuda_data_, Variable);
NO_FIELD(cuda_grad_, Variable);
NO_FIELD(W, Linear);
NO_FIELD(b, Linear);
NO_FIELD(layers, Sequential);
NO_FIELD(m, Adam);
NO_FIELD(v, Adam);
#undef NO_FIELD

int main() {
    const float host[] = {2.f, 3.f};
    ag::Variable input(ag::Tensor::from_host(host, ag::Shape{1, 2}), true);
    auto output = ag::sum(ag::scale(input, 2.f));
    output.backward();
    float value = 0.f, gradient[2] = {};
    output.value().copy_to_host(&value, 1);
    input.grad().copy_to_host(gradient, 2);
    if (value != 10.f || gradient[0] != 2.f || gradient[1] != 2.f) return 1;
    ag::Linear layer(2, 1);
    ag::Adam optimizer(layer.parameters());
    if (optimizer.step_count() != 0 || layer.named_parameters().size() != 2) return 1;
    std::puts("ALL API FREEZE TESTS PASSED (aliases, boundaries, forward/backward)");
    return 0;
}
