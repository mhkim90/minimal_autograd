// test_tensor.cpp — thin CPU Tensor foundation tests.
//
// Exercises the additive Tensor type declared in autograd/tensor.h with
// implementation hidden in src/core/tensor.cpp. Storage and per-Tensor
// metadata are both private; the only public handle is a
// std::shared_ptr<detail::TensorImpl>.
//
// Header independence is proven statically: autograd/tensor.h is the
// FIRST include; if it pulled in Eigen or a CUDA runtime header, the
// preprocessor check at the top of this file would #error out.

#include "autograd/tensor.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "autograd/tensor.h must not pull in any Eigen header"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) ||      \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "autograd/tensor.h must not pull in any CUDA runtime header"
#endif

#include "autograd/extension/eigen.h"
#include "autograd/device.h"
#include "autograd/shape.h"
#include "autograd.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

using ag::Device;
using ag::DeviceType;
using ag::Shape;
using ag::Tensor;

namespace {

int passed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s at %s:%d\n", \
                     #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_THROWS_AS(ExType, ...) do { \
    bool _threw = false; \
    try { (void)(__VA_ARGS__); } catch (const ExType&) { _threw = true; } \
    if (!_threw) { \
        std::fprintf(stderr, "FAIL: expected %s at %s:%d\n", \
                     #ExType, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_THROWS_ANY(...) do { \
    bool _threw = false; \
    try { (void)(__VA_ARGS__); } catch (...) { _threw = true; } \
    if (!_threw) { \
        std::fprintf(stderr, "FAIL: expected any exception at %s:%d\n", \
                     __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

void report(const char* name) {
    std::printf("  [ok] %s\n", name);
    ++passed;
}

// Tiny helpers so the per-test bodies stay focused on the contract.
void read_host(const Tensor& t, float* dst) { t.copy_to_host(dst, t.elements()); }
void write_host(Tensor& t, const float* src) { t.copy_from_host(src, t.elements()); }

void test_default_state() {
    // Default Tensor is Shape{0} (rank-1 zero extent), CPU, zero
    // elements. shape().numel() == elements() == 0, so the default is
    // coherent: no "rank-0 numel=1 vs elements=0" split.
    Tensor t;
    CHECK(t.empty());
    CHECK(t.device().is_cpu());
    CHECK(t.device().type() == DeviceType::Cpu);
    CHECK(t.shape() == (Shape{0}));
    CHECK(t.shape().rank() == 1);
    CHECK(t.shape().numel() == 0);
    CHECK(t.elements() == 0);
    // Repeated queries are safe; no UB.
    for (int i = 0; i < 32; ++i) {
        (void)t.shape();
        (void)t.device();
        (void)t.elements();
        (void)t.empty();
    }
    report("Tensor: default state is Shape{0}, CPU, zero elements, safe to query");
}

void test_factories_rank0_scalar() {
    // Rank-0 (Shape{}) follows the Shape contract: numel == 1.
    Tensor e = Tensor::empty(Shape{});
    CHECK(!e.empty());
    CHECK(e.elements() == 1);
    CHECK(e.shape() == Shape{});

    Tensor z = Tensor::zeros(Shape{});
    CHECK(!z.empty());
    CHECK(z.elements() == 1);
    float v = -1.f;
    read_host(z, &v);
    CHECK(v == 0.f);

    Tensor o = Tensor::ones(Shape{});
    read_host(o, &v);
    CHECK(v == 1.f);

    float scalar = 42.5f;
    Tensor s = Tensor::from_host(&scalar, Shape{});
    read_host(s, &v);
    CHECK(v == 42.5f);
    report("Tensor: rank-0 factories follow Shape{} numel=1 (scalar)");
}

void test_factories_normal() {
    Tensor z = Tensor::zeros(Shape{2, 3});
    CHECK(!z.empty());
    CHECK(z.elements() == 6);
    CHECK(z.shape() == (Shape{2, 3}));
    std::vector<float> out(6);
    read_host(z, out.data());
    for (float x : out) CHECK(x == 0.f);

    Tensor o = Tensor::ones(Shape{4});
    CHECK(o.elements() == 4);
    read_host(o, out.data());
    for (int i = 0; i < 4; ++i) CHECK(out[i] == 1.f);

    Tensor e = Tensor::empty(Shape{3, 2, 1});
    CHECK(e.elements() == 6);
    CHECK(e.shape() == (Shape{3, 2, 1}));
    report("Tensor: zeros/ones/empty factories for normal shapes");
}

void test_factories_zero_element_shape() {
    Tensor z = Tensor::zeros(Shape{2, 0, 4});
    CHECK(z.empty());
    CHECK(z.elements() == 0);
    CHECK(z.shape() == (Shape{2, 0, 4}));

    Tensor o = Tensor::ones(Shape{0});
    CHECK(o.empty());
    CHECK(o.elements() == 0);

    Tensor e = Tensor::empty(Shape{0, 5});
    CHECK(e.empty());

    // Host round trip on zero-element storage: zero count is a no-op.
    float sink = 0.f;
    z.copy_to_host(&sink, 0);
    e.copy_from_host(nullptr, 0);
    report("Tensor: zero-element shapes are empty across all factories");
}

void test_host_round_trip() {
    const std::vector<float> in{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Tensor t = Tensor::from_host(in.data(), Shape{2, 3});
    CHECK(t.shape() == (Shape{2, 3}));
    CHECK(t.elements() == 6);

    std::vector<float> out(6, 0.f);
    read_host(t, out.data());
    CHECK(out == in);

    // Round-trip back into a new tensor.
    Tensor t2 = Tensor::empty(Shape{2, 3});
    write_host(t2, out.data());
    std::vector<float> back(6, 0.f);
    read_host(t2, back.data());
    CHECK(back == in);
    report("Tensor: host round trip preserves values exactly");
}

void test_host_count_validation() {
    Tensor t = Tensor::zeros(Shape{2, 3});
    std::vector<float> out(6, 0.f);

    // Exact count is fine.
    t.copy_to_host(out.data(), 6);
    t.copy_from_host(out.data(), 6);

    // Count mismatch throws.
    CHECK_THROWS_AS(std::runtime_error, t.copy_to_host(out.data(), 5));
    CHECK_THROWS_AS(std::runtime_error, t.copy_to_host(out.data(), 7));
    CHECK_THROWS_AS(std::runtime_error, t.copy_from_host(out.data(), 5));
    CHECK_THROWS_AS(std::runtime_error, t.copy_from_host(out.data(), 7));

    // Zero-element tensors accept zero counts and null pointers.
    Tensor z = Tensor::empty(Shape{2, 0, 4});
    z.copy_to_host(nullptr, 0);
    z.copy_from_host(nullptr, 0);

    // Non-zero tensor with null pointer is rejected.
    CHECK_THROWS_AS(std::runtime_error, t.copy_to_host(nullptr, 6));
    CHECK_THROWS_AS(std::runtime_error, t.copy_from_host(nullptr, 6));
    report("Tensor: copy counts must match elements; null rejected for nonzero");
}

void test_from_host_null_validation() {
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::from_host(nullptr, Shape{2, 3}));
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::from_host(nullptr, Shape{1}));

    // Zero-element shape accepts null.
    Tensor a = Tensor::from_host(nullptr, Shape{0});
    CHECK(a.empty());
    CHECK(a.elements() == 0);
    Tensor b = Tensor::from_host(nullptr, Shape{2, 0, 4});
    CHECK(b.empty());
    CHECK(b.shape() == (Shape{2, 0, 4}));
    report("Tensor: from_host null validation matches zero-count rules");
}

void test_copy_shares_storage() {
    // Ordinary copy shares storage: mutating one is visible to all
    // aliases. Proven via host mutation only (no impl pointer compare).
    const std::vector<float> expected{7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    Tensor a = Tensor::ones(Shape{2, 3});
    Tensor b = a;
    Tensor c = a;

    write_host(b, expected.data());

    std::vector<float> out(6, 0.f);
    read_host(a, out.data());
    CHECK(out == expected);
    read_host(c, out.data());
    CHECK(out == expected);
    report("Tensor: ordinary copy shares storage; mutation propagates");
}

void test_clone_independence() {
    // clone() is an independent deep copy: mutating the clone does not
    // affect the source. Proven via host mutation only.
    const std::vector<float> ones6(6, 1.f);
    const std::vector<float> new_data{-1.f, -2.f, -3.f, -4.f, -5.f, -6.f};
    Tensor a = Tensor::ones(Shape{2, 3});
    Tensor b = a.clone();

    write_host(b, new_data.data());

    std::vector<float> out(6, 0.f);
    read_host(a, out.data());
    CHECK(out == ones6);

    read_host(b, out.data());
    CHECK(out == new_data);
    report("Tensor: clone() produces an independent deep copy");
}

void test_reshape_shares_storage_keeps_independent_shape() {
    // a.reshape(new_shape) must NOT mutate a.shape(). It must share the
    // float32 buffer so host mutations are visible through a.
    const std::vector<float> data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Tensor a = Tensor::ones(Shape{2, 3});
    Tensor alias = a;  // shares the same TensorImpl
    Tensor b = a.reshape(Shape{3, 2});

    // Source and any alias of the source are unchanged.
    CHECK(a.shape() == (Shape{2, 3}));
    CHECK(alias.shape() == (Shape{2, 3}));

    // The view has the new shape.
    CHECK(b.shape() == (Shape{3, 2}));
    CHECK(b.elements() == 6);

    // Mutating the view is visible through a and its alias: data is shared.
    write_host(b, data.data());

    std::vector<float> out(6, 0.f);
    read_host(a, out.data());
    CHECK(out == data);
    read_host(alias, out.data());
    CHECK(out == data);
    report("Tensor: reshape shares storage; source and aliases keep original shape");
}

void test_reshape_numel_mismatch() {
    Tensor a = Tensor::ones(Shape{2, 3});
    CHECK_THROWS_AS(std::invalid_argument, a.reshape(Shape{3, 3}));
    CHECK_THROWS_AS(std::invalid_argument, a.reshape(Shape{7}));
    CHECK_THROWS_AS(std::invalid_argument, a.reshape(Shape{2, 4}));
    CHECK_THROWS_AS(std::invalid_argument, a.reshape(Shape{}));  // 1 vs 6

    // Zero-element reshape between zero-element shapes is allowed.
    Tensor z = Tensor::empty(Shape{2, 0, 4});
    Tensor zr = z.reshape(Shape{0, 8});
    CHECK(zr.empty());
    CHECK(zr.elements() == 0);
    CHECK(zr.shape() == (Shape{0, 8}));
    CHECK(z.shape() == (Shape{2, 0, 4}));  // source unchanged

    // Reshape to a non-zero numel from a zero-element shape is rejected.
    CHECK_THROWS_AS(std::invalid_argument, z.reshape(Shape{1}));
    report("Tensor: reshape rejects numel mismatch; zero-element reshapes allowed");
}

void test_device_cpu_default_and_to_cpu() {
    Tensor t = Tensor::zeros(Shape{2, 3});
    CHECK(t.device().is_cpu());

    Tensor b = t.to(Device::cpu());
    CHECK(b.shape() == t.shape());
    CHECK(b.elements() == t.elements());

    // Round-trip preserves data.
    const std::vector<float> data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    write_host(t, data.data());
    std::vector<float> out(6, 0.f);
    read_host(b, out.data());
    CHECK(out == data);
    report("Tensor: device default is CPU; to(cpu) returns a shallow share");
}

void test_device_cuda_rejected() {
    // Device::cuda() descriptor is still valid.
    CHECK(Device::cuda(0).is_cuda());
    CHECK(Device::cuda(0).index() == 0);

    // Tensor creation targeting CUDA fails at runtime.
    CHECK_THROWS_ANY(Tensor::empty(Shape{2, 3}, Device::cuda(0)));
    CHECK_THROWS_ANY(Tensor::zeros(Shape{2, 3}, Device::cuda(1)));
    CHECK_THROWS_ANY(Tensor::ones(Shape{2, 3}, Device::cuda()));
    CHECK_THROWS_ANY(Tensor::from_host(nullptr, Shape{2, 3}, Device::cuda(0)));

    // to(cuda) also fails.
    Tensor a = Tensor::ones(Shape{2, 3});
    CHECK_THROWS_ANY(a.to(Device::cuda(0)));
    CHECK_THROWS_ANY(a.to(Device::cuda()));
    report("Tensor: CUDA Tensor creation and to(cuda) fail at runtime");
}

void test_extension_eigen_legacy_aliases() {
    // extension/eigen.h is the opt-in path for the legacy Mat / Mats /
    // shape(Mat) / numel(Mat) aliases. It must still provide them
    // alongside the new Tensor type.
    ag::Mat m = ag::Mat::Constant(2, 3, 1.f);
    ag::Mats v;
    v.push_back(m);
    auto sh = ag::shape(m);
    CHECK(sh.size() == 2);
    CHECK(sh[0] == 2 && sh[1] == 3);
    CHECK(ag::numel(m) == 6);
    (void)v;
    report("Tensor: extension/eigen.h still exposes Mat/Mats/shape(Mat)/numel(Mat)");
}

void test_umbrella_exposes_tensor() {
    // The autograd umbrella header must re-export the new Tensor.
    Tensor t = Tensor::zeros(Shape{2, 3});
    CHECK(!t.empty());
    CHECK(t.elements() == 6);
    (void)ag::Mat{};
    report("Tensor: autograd.h umbrella re-exports ag::Tensor");
}

void test_standalone_header_hygiene_smoke() {
    // The preprocessor check at the top of this file is the strongest
    // form of the standalone hygiene proof. This test is a thin API
    // smoke on top of it.
    static_assert(std::is_copy_constructible<Tensor>::value, "");
    static_assert(std::is_copy_assignable<Tensor>::value, "");
    static_assert(!std::is_same<Tensor, ag::Mat>::value,
                  "Tensor must be a distinct type from ag::Mat");
    report("Tensor: header independence is a preprocessor check; ABI smoke");
}

}  // namespace

int main() {
    test_default_state();
    test_factories_rank0_scalar();
    test_factories_normal();
    test_factories_zero_element_shape();
    test_host_round_trip();
    test_host_count_validation();
    test_from_host_null_validation();
    test_copy_shares_storage();
    test_clone_independence();
    test_reshape_shares_storage_keeps_independent_shape();
    test_reshape_numel_mismatch();
    test_device_cpu_default_and_to_cpu();
    test_device_cuda_rejected();
    test_extension_eigen_legacy_aliases();
    test_umbrella_exposes_tensor();
    test_standalone_header_hygiene_smoke();

    std::printf("\nALL TENSOR TESTS PASSED (%d)\n", passed);
    return 0;
}
