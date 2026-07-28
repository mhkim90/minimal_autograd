// test_shape_device.cpp — Phase 2: Shape and Device value-type tests.
//
// This is the focused test binary for the first thin Phase 2 PR. It covers
// the additive Shape validation/overflow contract and the new Device value
// type. It deliberately does NOT touch Var / Tensor / modules / CUDA so
// that the Phase 0 behavior contract remains unaffected by this PR.
//
// Coverage:
//   * Shape: rank-0 scalar convention, normal dims, rank(), ndim(),
//     index, equality, to_string / operator<<, zero dims allowed,
//     negative dim rejection, out-of-range indexing, numel overflow,
//     queries re-validate the public `sizes` field (numel, operator[],
//     contiguous_stride, to_string). Equality remains a noexcept raw
//     field compare and is explicitly not part of the revalidation set.
//   * Stride: contiguous_stride normal/zero-dim/rank-0/overflow,
//     negative stride rejection, format, to_string re-validates.
//   * Device: default CPU, cpu()/cuda() factories, negative CUDA index
//     rejection, equality/inequality, to_string / operator<<.
//   * CPU-only header independence for autograd/device.h: include
//     autograd/device.h FIRST and statically assert that no Eigen or
//     CUDA-runtime header guard is introduced by it. The umbrella
//     include autograd.h is brought in afterwards for the rest of the
//     tests.
//
// The independence check is run on the translation unit before any
// other autograd header is processed, and it does not require
// AUTOGRAD_USE_CUDA to be OFF — the check passes whether or not the
// target is configured for CUDA.

#include "autograd/device.h"

// Statically assert that including only autograd/device.h does not
// introduce any Eigen or CUDA-runtime header guard. This is the
// standalone header-independence proof.
#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "autograd/device.h must not pull in any Eigen header"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) ||      \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "autograd/device.h must not pull in any CUDA runtime header"
#endif

#include "autograd.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

using ag::Device;
using ag::DeviceType;
using ag::Dims;
using ag::Shape;
using ag::Stride;
using ag::contiguous_stride;
using ag::make_shape;

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

void report(const char* name) {
    std::printf("  [ok] %s\n", name);
    ++passed;
}

// ── Shape ──────────────────────────────────────────────────────────────

void test_shape_rank_zero() {
    Shape s;
    CHECK(s.sizes.empty());
    CHECK(s.ndim() == 0);
    CHECK(s.rank() == 0);
    CHECK(s.numel() == 1);
    CHECK(s.to_string() == "Shape[]");
    std::ostringstream os;
    os << s;
    CHECK(os.str() == "Shape[]");
    report("Shape: default-constructed rank-0 has numel=1 (scalar)");
}

void test_shape_normal_dims() {
    Shape s{2, 3, 4};
    CHECK(s.ndim() == 3);
    CHECK(s.rank() == static_cast<std::size_t>(3));
    CHECK(s.numel() == 24);
    CHECK(s[0] == 2);
    CHECK(s[1] == 3);
    CHECK(s[2] == 4);
    CHECK(s.to_string() == "Shape[2, 3, 4]");

    Shape s1{7};
    CHECK(s1.ndim() == 1);
    CHECK(s1.numel() == 7);
    CHECK(s1[0] == 7);

    Shape s2(static_cast<int64_t>(2), static_cast<int64_t>(3));
    CHECK(s2.ndim() == 2);
    CHECK(s2.numel() == 6);
    report("Shape: normal dims / rank / index / format");
}

void test_shape_make_shape_overloads() {
    auto a = make_shape({2, 3});
    CHECK(a.ndim() == 2);
    CHECK(a.numel() == 6);
    auto b = make_shape(4, 5);
    CHECK(b.ndim() == 2);
    CHECK(b.numel() == 20);
    report("Shape: make_shape overloads preserved");
}

void test_shape_equality() {
    CHECK((Shape{2, 3} == Shape{2, 3}));
    CHECK(!(Shape{2, 3} != Shape{2, 3}));
    CHECK((Shape{2, 3} != Shape{2, 4}));
    CHECK((Shape{2, 3} != Shape{2, 3, 1}));
    CHECK(Shape() == Shape{});
    CHECK(!(Shape() != Shape{}));
    report("Shape: equality and inequality");
}

void test_shape_zero_dims_allowed() {
    Shape s{2, 0, 4};
    CHECK(s.ndim() == 3);
    CHECK(s.numel() == 0);
    CHECK(s[1] == 0);
    CHECK(s.to_string() == "Shape[2, 0, 4]");

    Shape empty(0, 0);
    CHECK(empty.numel() == 0);

    Shape via_dims{std::vector<int64_t>{0, 1, 0}};
    CHECK(via_dims.numel() == 0);
    report("Shape: zero dimensions are allowed (numel=0)");
}

void test_shape_negative_rejection() {
    CHECK_THROWS_AS(std::invalid_argument, Shape{2, -1, 3});
    CHECK_THROWS_AS(std::invalid_argument, make_shape(-1, 5));
    CHECK_THROWS_AS(std::invalid_argument, make_shape({2, -1}));
    {
        Dims d{2, -1};
        CHECK_THROWS_AS(std::invalid_argument, Shape(std::move(d)));
    }
    report("Shape: negative dimension rejected with std::invalid_argument");
}

void test_shape_out_of_range_index() {
    Shape s{2, 3};
    CHECK_THROWS_AS(std::out_of_range, s[5]);
    CHECK_THROWS_AS(std::out_of_range, s[-1]);
    CHECK_THROWS_AS(std::out_of_range, s[2]);
    report("Shape: out-of-range indexing throws std::out_of_range");
}

void test_shape_numel_overflow() {
    constexpr int64_t big = std::numeric_limits<int64_t>::max();
    Shape s{big, 2};
    CHECK_THROWS_AS(std::overflow_error, s.numel());

    Shape s2{1, big, big};
    CHECK_THROWS_AS(std::overflow_error, s2.numel());

    Shape s3{big / 2, 3};
    CHECK_THROWS_AS(std::overflow_error, s3.numel());

    // Sanity: a fitting product does not throw.
    Shape ok{big / 2, 1};
    CHECK(ok.numel() == big / 2);
    report("Shape: numel() detects int64 overflow with std::overflow_error");
}

void test_shape_numel_mismatch_is_runtime_error() {
    ag::assert_same_numel(Shape{2, 3}, 6);
    CHECK_THROWS_AS(std::invalid_argument,
                    ag::assert_same_numel(Shape{2, 3}, 5));
    report("Shape: numel mismatch throws at runtime");
}

void test_shape_queries_revalidate_public_field() {
    // The `sizes` field is public for source compatibility; the
    // revalidating queries (numel, operator[], contiguous_stride,
    // to_string, and operator<<) must not trust it blindly. A direct
    // mutation that introduces a negative must make the next call to
    // any of these throw std::invalid_argument. Equality (==, !=) is
    // a noexcept raw field compare and is explicitly NOT part of the
    // revalidation contract.
    Shape s{2, 3, 4};
    s.sizes.push_back(-1);
    CHECK_THROWS_AS(std::invalid_argument, s.numel());
    CHECK_THROWS_AS(std::invalid_argument, s[0]);
    CHECK_THROWS_AS(std::invalid_argument, contiguous_stride(s));
    CHECK_THROWS_AS(std::invalid_argument, s.to_string());
    {
        std::ostringstream os;
        CHECK_THROWS_AS(std::invalid_argument, (os << s));
    }
    // Equality stays noexcept and is allowed to operate on mutated state
    // without throwing.
    Shape t{2, 3, 4};
    t.sizes.push_back(-1);
    CHECK((s == t));  // equal by raw field compare
    report("Shape: queries re-validate mutated public sizes; equality does not");
}

// ── Stride ─────────────────────────────────────────────────────────────

void test_stride_normal() {
    // Canonical row-major / last-axis-contiguous (D0...D{n-1}):
    //   stride[n-1] = 1, stride[i] = stride[i+1] * shape[i+1]
    // For Shape{2,3,4}: outer stride = 3*4 = 12, next = 4, last = 1.
    Stride st = contiguous_stride(Shape{2, 3, 4});
    CHECK(st.strides.size() == static_cast<std::size_t>(3));
    CHECK(st[0] == 12);
    CHECK(st[1] == 4);
    CHECK(st[2] == 1);
    CHECK((st == Stride({12, 4, 1})));
    CHECK(st.to_string() == "Stride[12, 4, 1]");
    std::ostringstream os;
    os << st;
    CHECK(os.str() == "Stride[12, 4, 1]");
    report("Stride: contiguous_stride is last-axis contiguous (row-major)");
}

void test_stride_zero_dim() {
    // With a zero middle dim, the trailing dim still has stride 1 and
    // the leading dims collapse to zero (the running product).
    Stride st = contiguous_stride(Shape{2, 0, 4});
    CHECK(st.strides.size() == static_cast<std::size_t>(3));
    CHECK(st[0] == 0);
    CHECK(st[1] == 4);
    CHECK(st[2] == 1);
    CHECK((st == Stride({0, 4, 1})));
    report("Stride: contiguous_stride with zero dim yields 0 outer + 1 inner");
}

void test_stride_rank_zero() {
    Stride st = contiguous_stride(Shape{});
    CHECK(st.strides.empty());
    CHECK((st == Stride{}));
    report("Stride: contiguous_stride on rank-0 shape is empty");
}

void test_stride_overflow() {
    constexpr int64_t big = std::numeric_limits<int64_t>::max();
    // Row-major: stride[n-1]=1, stride[i]=stride[i+1]*shape[i+1].
    // For Shape{2, big, big}: outer stride = big*big → overflow.
    CHECK_THROWS_AS(std::overflow_error,
                    contiguous_stride(Shape{2, big, big}));
    // For Shape{big, 2, big}: stride[1] = 2 * big → overflow.
    CHECK_THROWS_AS(std::overflow_error,
                    contiguous_stride(Shape{big, 2, big}));
    report("Stride: contiguous_stride overflow throws std::overflow_error");
}

void test_stride_validation() {
    CHECK_THROWS_AS(std::invalid_argument, Stride({1, -2, 3}));
    {
        // Out-of-range indexing on a clean Stride throws std::out_of_range
        // (delegated to vector::at) without being shadowed by a negative
        // re-validation in the public strides field.
        Stride s({1, 2, 3});
        CHECK_THROWS_AS(std::out_of_range, s[5]);
        CHECK_THROWS_AS(std::out_of_range, s[-1]);
    }
    {
        // Queries re-validate current contents: a stride mutated to a
        // negative value triggers std::invalid_argument on operator[],
        // to_string, and operator<<. Equality remains a noexcept raw
        // field compare and is allowed to operate on mutated state.
        Stride s({1, 2, 3});
        s.strides.push_back(-7);
        CHECK_THROWS_AS(std::invalid_argument, s[0]);
        CHECK_THROWS_AS(std::invalid_argument, s.to_string());
        {
            std::ostringstream os;
            CHECK_THROWS_AS(std::invalid_argument, (os << s));
        }
        Stride t({1, 2, 3});
        t.strides.push_back(-7);
        CHECK((s == t));  // equal by raw field compare
    }
    report("Stride: negative stride rejected, out-of-range throws, queries re-validate");
}

// ── Device ─────────────────────────────────────────────────────────────

void test_device_default_is_cpu() {
    Device d;
    CHECK(d.type() == DeviceType::Cpu);
    CHECK(d.index() == 0);
    CHECK(d.is_cpu());
    CHECK(!d.is_cuda());
    report("Device: default-constructed is CPU");
}

void test_device_cpu_factory() {
    Device d = Device::cpu();
    CHECK(d.type() == DeviceType::Cpu);
    CHECK(d.index() == 0);
    CHECK(d.is_cpu());
    CHECK(!d.is_cuda());
    CHECK(d.to_string() == "cpu");
    std::ostringstream os;
    os << d;
    CHECK(os.str() == "cpu");
    report("Device: cpu() factory + format");
}

void test_device_cuda_factory() {
    Device d0 = Device::cuda();
    CHECK(d0.type() == DeviceType::Cuda);
    CHECK(d0.index() == 0);
    CHECK(d0.is_cuda());
    CHECK(!d0.is_cpu());

    Device d3 = Device::cuda(3);
    CHECK(d3.type() == DeviceType::Cuda);
    CHECK(d3.index() == 3);
    CHECK(d3.is_cuda());
    CHECK(d3.to_string() == "cuda:3");
    std::ostringstream os;
    os << d3;
    CHECK(os.str() == "cuda:3");
    report("Device: cuda() factory default index 0, multi-index format");
}

void test_device_equality() {
    CHECK(Device::cpu() == Device::cpu());
    CHECK(Device::cuda(0) == Device::cuda(0));
    CHECK(Device::cuda(1) != Device::cuda(2));
    CHECK(Device::cpu() != Device::cuda(0));
    CHECK(Device() == Device::cpu());
    report("Device: equality and inequality");
}

void test_device_negative_rejection() {
    CHECK_THROWS_AS(std::invalid_argument, Device::cuda(-1));
    CHECK_THROWS_AS(std::invalid_argument, Device::cuda(-100));
    report("Device: negative CUDA index rejected with std::invalid_argument");
}

void test_device_no_cuda_headers() {
    // The standalone-include header-independence proof is the
    // preprocessor check at the top of this file: autograd/device.h
    // is processed before any other autograd header, and any of
    // EIGEN_WORLD_VERSION / EIGEN_MAJOR_VERSION / EIGEN_MINOR_VERSION /
    // CUDART_VERSION / __CUDART_API_VERSION / CUDA_VERSION /
    // __CUDA_RUNTIME_H__ would abort the build. This runtime test
    // exists as a thin API smoke on top of that static proof.
    static_assert(std::is_trivially_copyable<Device>::value,
                  "Device must remain a value type");

    Device cpu = Device::cpu();
    Device gpu0 = Device::cuda(0);
    Device gpu1 = Device::cuda(1);
    CHECK(cpu != gpu0);
    CHECK(gpu0 != gpu1);
    CHECK(cpu.is_cpu());
    CHECK(gpu0.is_cuda());

    // DeviceType can be streamed.
    std::ostringstream os;
    os << DeviceType::Cpu << "/" << DeviceType::Cuda;
    CHECK(os.str() == "cpu/cuda");
    report("Device: runtime API smoke (header independence is a preprocessor check)");
}

// ── Public API smoke ───────────────────────────────────────────────────

void test_umbrella_header_exports() {
    // Including just "autograd.h" must give us Shape, Device, make_shape,
    // contiguous_stride without explicit sub-header includes, and those
    // symbols must be reachable from the ag namespace.
    Shape s = make_shape({2, 3});
    Stride st = contiguous_stride(s);
    Device d = Device::cpu();
    CHECK(s.numel() == 6);
    CHECK(st[0] == 3);
    CHECK(st[1] == 1);
    CHECK(d.is_cpu());
    report("Public API: autograd.h re-exports Shape/Device/Stride");
}

}  // namespace

int main() {
    test_shape_rank_zero();
    test_shape_normal_dims();
    test_shape_make_shape_overloads();
    test_shape_equality();
    test_shape_zero_dims_allowed();
    test_shape_negative_rejection();
    test_shape_out_of_range_index();
    test_shape_numel_overflow();
    test_shape_numel_mismatch_is_runtime_error();
    test_shape_queries_revalidate_public_field();

    test_stride_normal();
    test_stride_zero_dim();
    test_stride_rank_zero();
    test_stride_overflow();
    test_stride_validation();

    test_device_default_is_cpu();
    test_device_cpu_factory();
    test_device_cuda_factory();
    test_device_equality();
    test_device_negative_rejection();
    test_device_no_cuda_headers();

    test_umbrella_header_exports();

    std::printf("\nALL SHAPE/DEVICE TESTS PASSED (%d)\n", passed);
    return 0;
}
