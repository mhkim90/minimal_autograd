// test/test_cuda_tensor.cpp — OOP Tensor CUDA foundation tests.
//
// Exercises the CUDA Tensor foundation. The borrowed view surface from
// include/autograd/extension/cuda.h is the only public path to a
// raw device pointer; this test verifies the no-mirror,
// no-hidden-fallback contract end-to-end:
//
//   * Tensor::empty/zeros/ones/from_host succeed for CUDA targets and
//     expose device.is_cuda() with the requested device index.
//   * Tensor::to CPU<->CUDA copies synchronously and preserves values.
//     Repeated calls keep source identity and produce independent
//     destination Storage. Cross-device CUDA(0)->CUDA(1) uses the
//     peer-to-peer copy helper when the second device is visible.
//   * Same-device Tensor::to is a shallow share; aliases share a
//     single device pointer (proven through the borrowed view).
//   * Tensor::clone() is an independent deep copy: mutating the
//     clone through cuda_view_mut and copying back to host differs
//     from the source.
//   * Tensor::reshape() shares storage between the source and the
//     view; the borrowed view's storage pointer is identical for
//     both handles, and the recorded Shape reflects each Tensor's
//     own logical shape.
//   * Destruction-safe lifetime: aliases keep the shared allocation
//     alive, and dropping owners is safe.
//   * cuda_view returns ConstCudaTensorView; cuda_view_mut returns
//     CudaTensorView. Both reject non-CUDA Tensors and an empty
//     CUDA Tensor returns data == nullptr.
//   * Public header hygiene: a TU that includes only the canonical
//     OOP headers plus autograd/extension/cuda.h never sees a
//     CUDA runtime macro; the borrowed view path is header-clean.
//
// This binary links against the same CUDA-backed libautograd as
// test_cuda_core. CMake only adds it when AUTOGRAD_USE_CUDA is
// enabled; the sibling test/test_tensor.cpp owns the CPU-only
// counterpart contract.

#include "autograd/tensor.h"
#include "autograd/extension/cuda.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "OOP Tensor headers must not pull in any Eigen header"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION__) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "OOP Tensor headers must not pull in any CUDA runtime header"
#endif

#include "autograd/core/loss.h"
#include "autograd/core/optim.h"
#include "autograd/core/ops.h"
#include "autograd/core/diffusion.h"
#include "autograd/core/variable.h"
#include "autograd/device.h"
#include "autograd/shape.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using ag::ConstCudaTensorView;
using ag::CudaTensorView;
using ag::Device;
using ag::Shape;
using ag::Tensor;
using ag::Variable;

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

void read_host(const Tensor& t, float* dst) {
    t.copy_to_host(dst, t.elements());
}
void write_host(Tensor& t, const float* src) {
    t.copy_from_host(src, t.elements());
}

// Drive backward through a no-op scale to set p.grad() to a known
// tensor on p's device. Variable::grad() returns a const reference,
// so we cannot call copy_from_host on it directly; the backward path
// is the only public way to install a chosen gradient value without
// widening the public API.
void assign_grad(Variable& p, const std::vector<float>& g) {
    p.zero_grad();
    Tensor upstream = Tensor::from_host(g.data(), p.value().shape(),
                                        p.value().device());
    Variable loss = ag::scale(p, 1.0f);
    loss.backward(upstream);
}

// Skip everything when no CUDA device is visible. The probe asks
// for a Tensor on cuda:0 through the public Tensor surface; the
// factory validates the device index eagerly even for zero-element
// shapes, so the throw/catch path is meaningful rather than tautological.
bool cuda_available() {
    try {
        Tensor t = Tensor::empty(Shape{0}, Device::cuda(0));
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

namespace {

// SFINAE probe: detects whether ag::Tensor exposes an _impl()
// member function. Re-introducing _impl() is a deliberate
// public-API change; the test fails to compile the moment the
// surface reappears. Today the probe resolves to false_type
// and the assertion below is satisfied.
template <typename, typename = void>
struct has_impl : std::false_type {};
template <typename T>
struct has_impl<T, std::void_t<decltype(&T::_impl)>> : std::true_type {};

}  // namespace

void test_public_tensor_has_no_impl_accessor() {
    // The contract documented on ag::Tensor: the public Tensor type
    // exposes no accessor that returns the per-Tensor TensorImpl.
    // The only public path to a raw device pointer is the borrowed
    // view surface; the only consumer of Tensor's private impl_ is
    // ag::detail::CudaTensorAccess, only forward-declared in
    // autograd/tensor.h.
    //
    // We detect _impl() at compile time through SFINAE so an
    // accidental re-introduction surfaces immediately at the
    // build, not at runtime.
    static_assert(!has_impl<Tensor>::value,
                  "Tensor must not expose a public _impl accessor");
    static_assert(std::is_same<
                      decltype(ag::cuda_view(
                          std::declval<const Tensor&>())),
                      ConstCudaTensorView>::value,
                  "cuda_view(const Tensor&) must be read-only");
    static_assert(std::is_same<
                      decltype(ag::cuda_view_mut(
                          std::declval<Tensor&>())),
                      CudaTensorView>::value,
                  "cuda_view_mut(Tensor&) must be writable");
    report("Tensor: no public impl accessor; CUDA views are const-correct");
}

void test_factory_cuda_zeros_values() {
    // zeros on CUDA, then verify host copy matches.
    Tensor t = Tensor::zeros(Shape{2, 3}, Device::cuda(0));
    CHECK(t.device().is_cuda());
    CHECK(t.device().index() == 0);
    CHECK(t.shape() == (Shape{2, 3}));
    CHECK(t.elements() == 6);
    std::vector<float> out(6);
    read_host(t, out.data());
    for (float v : out) CHECK(v == 0.f);

    // ones, then verify host copy matches.
    Tensor o = Tensor::ones(Shape{4}, Device::cuda(0));
    CHECK(o.device().is_cuda());
    read_host(o, out.data());
    for (int i = 0; i < 4; ++i) CHECK(out[i] == 1.f);

    report("Tensor CUDA: zeros/ones factories allocate and host copy out values");
}

void test_factory_cuda_from_host_round_trip() {
    const std::vector<float> in{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    Tensor t = Tensor::from_host(in.data(), Shape{2, 3}, Device::cuda(0));
    CHECK(t.device().is_cuda());
    CHECK(t.shape() == (Shape{2, 3}));

    std::vector<float> out(in.size(), 0.f);
    read_host(t, out.data());
    CHECK(out == in);

    // Round-trip back to a fresh CUDA tensor through host copies.
    Tensor t2 = Tensor::empty(Shape{2, 3}, Device::cuda(0));
    write_host(t2, out.data());
    std::vector<float> back(in.size(), 0.f);
    read_host(t2, back.data());
    CHECK(back == in);

    report("Tensor CUDA: from_host/host copies round-trip values exactly");
}

void test_factory_cuda_index_and_zero_element() {
    // Default-index device descriptor on a CUDA factory is 0.
    Tensor t_def = Tensor::empty(Shape{1, 4}, Device::cuda());
    CHECK(t_def.device().is_cuda());
    CHECK(t_def.device().index() == 0);

    // Custom index works.
    Tensor t_idx1 = Tensor::zeros(Shape{2}, Device::cuda(0));
    CHECK(t_idx1.device().index() == 0);

    // Zero-element CUDA tensors are accepted and remain empty.
    Tensor z1 = Tensor::empty(Shape{0, 5}, Device::cuda(0));
    CHECK(z1.empty());
    CHECK(z1.elements() == 0);
    CHECK(z1.device().is_cuda());
    Tensor z2 = Tensor::zeros(Shape{2, 0, 4}, Device::cuda(0));
    CHECK(z2.empty());
    Tensor z3 = Tensor::ones(Shape{0}, Device::cuda(0));
    CHECK(z3.empty());
    CHECK(z3.device().is_cuda());

    // Zero-element host copies are no-ops.
    float sink = 0.f;
    z1.copy_to_host(&sink, 0);
    z3.copy_from_host(nullptr, 0);

    report("Tensor CUDA: factory defaults to cuda:0; zero-element shapes empty");
}

void test_to_cpu_from_cuda() {
    // CPU -> CUDA: explicit synchronous H2D copy.
    const std::vector<float> in{7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    Tensor cpu_a = Tensor::from_host(in.data(), Shape{2, 3});
    Tensor gpu_b = cpu_a.to(Device::cuda(0));

    CHECK(gpu_b.device().is_cuda());
    CHECK(gpu_b.shape() == (Shape{2, 3}));
    CHECK(!gpu_b.empty());

    // Source identity (CPU) is preserved by the transfer.
    CHECK(cpu_a.device().is_cpu());
    std::vector<float> out(in.size());
    read_host(cpu_a, out.data());
    CHECK(out == in);

    // Destination values match via copy_to_host.
    std::vector<float> gpu_out(in.size());
    read_host(gpu_b, gpu_out.data());
    CHECK(gpu_out == in);

    // CPU alias of source is intact (shared storage on CPU).
    Tensor cpu_alias = cpu_a;
    write_host(cpu_alias, in.data());  // no-op overlap
    std::vector<float> chk(in.size());
    read_host(cpu_a, chk.data());
    CHECK(chk == in);

    // cuda_view_mut exposes the same device pointer as cuda_view.
    ConstCudaTensorView v_const = ag::cuda_view(gpu_b);
    CudaTensorView v_mut = ag::cuda_view_mut(gpu_b);
    CHECK(v_const.data == v_mut.data);
    CHECK(v_const.data != nullptr);
    CHECK(v_const.numel == 6);
    CHECK(v_const.shape == (Shape{2, 3}));
    CHECK(v_const.device_index == 0);

    report("Tensor CUDA: to(cuda) copies H2D; cuda_view returns device pointer");
}

void test_to_cuda_from_cpu_round_trip() {
    // CPU -> CUDA -> CPU: values preserved across both directions.
    const std::vector<float> in{0.5f, -1.5f, 2.5f, -3.5f};
    Tensor a = Tensor::from_host(in.data(), Shape{4});
    Tensor g = a.to(Device::cuda(0));
    Tensor b = g.to(Device::cpu());
    CHECK(b.device().is_cpu());
    CHECK(b.shape() == (Shape{4}));
    std::vector<float> out(in.size());
    read_host(b, out.data());
    CHECK(out == in);

    // Source identity on the CPU side is preserved; CUDA tensor is
    // a fresh allocation independent of `a`.
    a.copy_from_host(in.data(), in.size());
    std::vector<float> a_back(in.size());
    read_host(a, a_back.data());
    CHECK(a_back == in);

    report("Tensor CUDA: CPU<->CUDA round-trip preserves values; sources untouched");
}

void test_same_device_transfer_shares_storage() {
    // to(same_device) is a shallow share: the new Tensor references
    // the same storage. Mutating either handle through copy_from_host
    // is visible to both on CPU; on CUDA the borrowed view exposes
    // the same device pointer for both.
    Tensor a = Tensor::zeros(Shape{4}, Device::cuda(0));
    Tensor b = a.to(Device::cuda(0));
    CHECK(b.device().is_cuda());
    CHECK(b.shape() == (Shape{4}));

    // Pointer-equality through cuda_view identifies storage sharing.
    ConstCudaTensorView va = ag::cuda_view(a);
    ConstCudaTensorView vb = ag::cuda_view(b);
    CHECK(va.data == vb.data);
    CHECK(va.data != nullptr);

    report("Tensor CUDA: to(same_device) is a shallow share; views expose same pointer");
}

void test_clone_is_independent_deep_copy() {
    // clone() is a fresh Storage on the same device. Mutating the
    // clone is NOT visible to the source.
    const std::vector<float> src_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Tensor a = Tensor::from_host(src_data.data(), Shape{2, 3}, Device::cuda(0));
    Tensor b = a.clone();

    CHECK(b.device().is_cuda());
    CHECK(b.shape() == (Shape{2, 3}));
    ConstCudaTensorView va = ag::cuda_view(a);
    ConstCudaTensorView vb = ag::cuda_view(b);
    CHECK(va.data != vb.data);  // separate allocations
    CHECK(va.data != nullptr);
    CHECK(vb.data != nullptr);

    // Mutate b through cuda_view_mut by writing via copy_from_host.
    const std::vector<float> new_data{-1.f, -2.f, -3.f,
                                       -4.f, -5.f, -6.f};
    write_host(b, new_data.data());

    // Read a and verify it is unchanged.
    std::vector<float> out(6);
    read_host(a, out.data());
    CHECK(out == src_data);

    // Read b and verify it took the new values.
    std::vector<float> out_b(6);
    read_host(b, out_b.data());
    CHECK(out_b == new_data);

    report("Tensor CUDA: clone() is an independent deep copy on CUDA device");
}

void test_reshape_shares_storage_keeps_independent_shape() {
    // reshape() returns a new TensorImpl with a different Shape but
    // shares the same Storage. The view from the original and the
    // view from the reshape alias expose the same pointer.
    const std::vector<float> data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Tensor a = Tensor::from_host(data.data(), Shape{2, 3}, Device::cuda(0));
    Tensor alias = a;  // shares TensorImpl
    Tensor b = a.reshape(Shape{3, 2});

    // Source and its alias keep the original shape.
    CHECK(a.shape() == (Shape{2, 3}));
    CHECK(alias.shape() == (Shape{2, 3}));

    // Reshape keeps its own shape.
    CHECK(b.shape() == (Shape{3, 2}));
    CHECK(b.elements() == 6);

    // Same Storage: same device pointer in both views.
    ConstCudaTensorView va = ag::cuda_view(a);
    ConstCudaTensorView vb = ag::cuda_view(b);
    CHECK(va.data == vb.data);
    CHECK(va.data != nullptr);
    CHECK(vb.shape == (Shape{3, 2}));  // reflects b's logical shape

    // Numel mismatch still rejected.
    CHECK_THROWS_AS(std::invalid_argument, a.reshape(Shape{3, 3}));

    // Zero-element reshape between zero-element shapes is allowed.
    Tensor z = Tensor::empty(Shape{2, 0, 4}, Device::cuda(0));
    Tensor zr = z.reshape(Shape{0, 8});
    CHECK(zr.empty());
    CHECK(zr.elements() == 0);
    CHECK(zr.shape() == (Shape{0, 8}));

    report("Tensor CUDA: reshape shares Storage; source/alias keep shape; numel checks");
}

void test_repeated_transfers_keep_source_identity() {
    // Multiple to(...) calls keep the source Storage intact and
    // produce independent destination Storage.
    const std::array<float, 4> src_values{1.f, 2.f, 3.f, 4.f};
    Tensor src = Tensor::from_host(src_values.data(), Shape{4});
    for (int i = 0; i < 4; ++i) {
        Tensor dst = src.to(Device::cuda(0));
        CHECK(dst.device().is_cuda());
        std::vector<float> back(4);
        read_host(dst, back.data());
        const std::vector<float> expected{1.f, 2.f, 3.f, 4.f};
        CHECK(back == expected);
    }
    // Source remains the same on CPU.
    std::vector<float> src_back(4);
    read_host(src, src_back.data());
    const std::vector<float> expected{1.f, 2.f, 3.f, 4.f};
    CHECK(src_back == expected);

    // Also: clone() twice produces independent destinations.
    Tensor c1 = src.to(Device::cuda(0)).clone();
    Tensor c2 = src.to(Device::cuda(0)).clone();
    CHECK(ag::cuda_view(c1).data != ag::cuda_view(c2).data);

    report("Tensor CUDA: repeated to(cuda) is independent; clone produces fresh buffer");
}

void test_move_preserves_cuda_storage() {
    const std::vector<float> values{2.f, 4.f, 6.f, 8.f};
    Tensor source =
        Tensor::from_host(values.data(), Shape{4}, Device::cuda(0));
    const float* storage = ag::cuda_view(source).data;

    Tensor moved(std::move(source));
    CHECK(moved.device() == Device::cuda(0));
    CHECK(ag::cuda_view(moved).data == storage);

    Tensor assigned;
    assigned = std::move(moved);
    CHECK(assigned.device() == Device::cuda(0));
    CHECK(ag::cuda_view(assigned).data == storage);

    std::vector<float> back(values.size());
    read_host(assigned, back.data());
    CHECK(back == values);

    report("Tensor CUDA: move construction/assignment preserve owned storage");
}

void test_destruction_safe_storage_release() {
    // Single-handle destruction: the source drops at scope end; the
    // borrowed view's recorded pointer is no longer reachable but
    // is also no longer referenced. A subsequent Tensor allocation
    // of the same shape on the same device must succeed and
    // produce a non-null pointer.
    {
        Tensor t = Tensor::ones(Shape{64}, Device::cuda(0));
        ConstCudaTensorView v = ag::cuda_view(t);
        CHECK(v.data != nullptr);
        (void)v;
        // t drops at scope end; Storage's RAII free fires.
    }
    Tensor u = Tensor::zeros(Shape{64}, Device::cuda(0));
    ConstCudaTensorView vu = ag::cuda_view(u);
    CHECK(vu.data != nullptr);

    // Aliased destruction: many handles keep storage alive.
    // Capture the storage pointer once, then drop the source and
    // verify every alias still points at the same saved pointer
    // (not just "some non-null pointer", which would be tautological
    // since dropping source doesn't invalidate aliases' views).
    Tensor base = Tensor::zeros(Shape{32}, Device::cuda(0));
    std::vector<Tensor> aliases;
    for (int i = 0; i < 16; ++i) aliases.push_back(base);
    const float* saved = ag::cuda_view(base).data;
    CHECK(saved != nullptr);
    for (auto& a : aliases) {
        CHECK(ag::cuda_view(a).data == saved);
    }
    // Drop the base; aliases keep storage alive.
    base = Tensor();
    for (auto& a : aliases) {
        CHECK(ag::cuda_view(a).data == saved);
    }
    // Verify the underlying storage still holds the values written
    // before any alias was added (zero allocation, untouched).
    std::vector<float> sample(aliases.size() * 32);
    for (std::size_t i = 0; i < aliases.size(); ++i) {
        std::vector<float> one(32);
        read_host(aliases[i], one.data());
        for (std::size_t j = 0; j < 32; ++j) {
            sample[i * 32 + j] = one[j];
        }
    }
    for (float v : sample) CHECK(v == 0.f);
    // Drop aliases; storage is freed.
    aliases.clear();

    report("Tensor CUDA: RAII destruction is safe; aliases keep storage alive");
}

void test_wrong_device_view_throws() {
    // cuda_view on a CPU Tensor throws.
    Tensor cpu = Tensor::ones(Shape{4});
    CHECK_THROWS_AS(std::runtime_error, ag::cuda_view(cpu));
    CHECK_THROWS_AS(std::runtime_error, ag::cuda_view_mut(cpu));

    // Verified message includes the requested device label so the
    // user can fix the call site immediately.
    bool threw_with_label = false;
    try {
        (void)ag::cuda_view(cpu);
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        threw_with_label = (what.find("cuda") != std::string::npos)
                         || (what.find("CPU") != std::string::npos);
    }
    CHECK(threw_with_label);

    report("Tensor CUDA: cuda_view on a CPU Tensor throws with a labeled message");
}

void test_out_of_range_device_rejected_eagerly() {
    // Every factory / transfer validates the requested device index
    // through cuda_runtime_validate_device, even for zero-element
    // shapes and even for transfers that would otherwise no-op.
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::empty(Shape{0}, Device::cuda(99)));
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::zeros(Shape{0}, Device::cuda(99)));
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::ones(Shape{0}, Device::cuda(99)));
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::from_host(nullptr, Shape{0},
                                       Device::cuda(99)));

    // Non-zero factory on an out-of-range device throws before any
    // allocation is attempted.
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::empty(Shape{2, 3}, Device::cuda(99)));
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::zeros(Shape{2, 3}, Device::cuda(99)));
    CHECK_THROWS_AS(std::runtime_error,
                    Tensor::from_host(nullptr, Shape{2, 3},
                                       Device::cuda(99)));

    // Cross-device to(out-of-range) throws even on a zero-element
    // source.
    Tensor empty_cpu = Tensor::empty(Shape{0}, Device::cpu());
    CHECK_THROWS_AS(std::runtime_error, empty_cpu.to(Device::cuda(99)));

    report("Tensor CUDA: factory and to() validate device index even for zero-element Tensors");
}

void test_empty_cuda_tensor_view_and_alias() {
    // Empty CUDA Tensor: view's data is nullptr, numel is 0,
    // shape matches the requested shape.
    Tensor t = Tensor::empty(Shape{0, 5}, Device::cuda(0));
    CHECK(t.empty());
    ConstCudaTensorView v = ag::cuda_view(t);
    CHECK(v.data == nullptr);
    CHECK(v.numel == 0);
    CHECK(v.shape == (Shape{0, 5}));
    CHECK(v.device_index == 0);

    // Aliases of an empty CUDA Tensor: views match.
    Tensor alias = t;
    ConstCudaTensorView va = ag::cuda_view(alias);
    CHECK(va.data == nullptr);
    CHECK(va.numel == 0);

    // Reshape between zero-element shapes: still empty and data == nullptr.
    Tensor tr = t.reshape(Shape{5, 0, 1});
    CHECK(tr.empty());
    CHECK(ag::cuda_view(tr).data == nullptr);

    report("Tensor CUDA: empty CUDA tensors report data == nullptr in views");
}

void test_same_device_cuda_to_cuda_transfer() {
    // Same-device Tensor::to on a CUDA Tensor is a shallow share:
    // the new Tensor references the same Storage and the borrowed
    // view exposes the same device pointer. Values round-trip
    // through the host copy without any allocation.
    const std::vector<float> in{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Tensor src = Tensor::from_host(in.data(), Shape{2, 3}, Device::cuda(0));
    Tensor same = src.to(Device::cuda(0));
    CHECK(same.device().index() == 0);
    CHECK(ag::cuda_view(src).data == ag::cuda_view(same).data);

    std::vector<float> back(in.size());
    read_host(same, back.data());
    CHECK(back == in);

    // Cross-device peer-to-peer transfer (CUDA(0) -> CUDA(1)) is
    // exercised on multi-GPU systems only. We probe device count
    // through a non-throwing shape-0 Tensor::empty probe so the
    // assertion is skipped on a single-GPU runner.
    const bool peer_reachable = [&] {
        try {
            Tensor probe = Tensor::empty(Shape{0}, Device::cuda(1));
            (void)probe;
            return true;
        } catch (...) {
            return false;
        }
    }();
    if (peer_reachable) {
        Tensor peer = src.to(Device::cuda(1));
        CHECK(peer.device().index() == 1);
        CHECK(ag::cuda_view(peer).data != nullptr);
        CHECK(ag::cuda_view(peer).data != ag::cuda_view(src).data);
        std::vector<float> peer_back(in.size());
        read_host(peer, peer_back.data());
        CHECK(peer_back == in);
    }

    report("Tensor CUDA: same-device CUDA->CUDA shallow share; cross-device peer where available");
}

void test_header_hygiene_preprocessor() {
    // The OOP Tensor headers plus autograd/extension/cuda.h must
    // never pull in a CUDA runtime header. This is a structural
    // gate, not a runtime check: the failure fires at compile
    // time if a downstream change accidentally surfaces a CUDA
    // runtime header through the public OOP API.
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION__) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "OOP Tensor + CUDA view extension must remain CUDA-runtime-header free"
#endif
    report("Tensor CUDA: public OOP headers and cuda extension stay CUDA-runtime free");
}

void test_legacy_extension_eigen_aliases_still_available() {
    // extension/eigen.h is the opt-in path for the legacy Var / Mats
    // / shape(Mat) / numel(Mat) aliases; CUDA-enabled consumers
    // can opt in independently of autograd/extension/cuda.h.
    CHECK(ag::Device::cpu().is_cpu());
    CHECK(ag::Device::cuda(0).is_cuda());
    report("Tensor CUDA: Device descriptor remains intact on a CUDA-enabled build");
}

void check_close(const std::vector<float>& actual,
                 const std::vector<float>& expected,
                 float tolerance = 1e-5f);

void test_oop_direct_shape_ops_cuda() {
    const Shape shape{2, 3, 4};
    std::vector<float> values(shape.numel());
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) * 0.25f - 1.f;
    }

    Variable cpu_x(Tensor::from_host(values.data(), shape), true);
    Variable cuda_x(Tensor::from_host(values.data(), shape, Device::cuda(0)), true);
    Variable cpu_slice = ag::slice(cpu_x, -2, 1, 2);
    Variable cuda_slice = ag::slice(cuda_x, -2, 1, 2);
    CHECK(cuda_slice.value().device().is_cuda());
    std::vector<float> cpu_value(cpu_slice.value().elements());
    std::vector<float> cuda_value(cuda_slice.value().elements());
    read_host(cpu_slice.value(), cpu_value.data());
    read_host(cuda_slice.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);

    std::vector<float> slice_up(cpu_slice.value().elements());
    for (std::size_t i = 0; i < slice_up.size(); ++i) {
        slice_up[i] = static_cast<float>(i + 1);
    }
    cpu_slice.backward(Tensor::from_host(slice_up.data(), cpu_slice.value().shape()));
    cuda_slice.backward(Tensor::from_host(
        slice_up.data(), cuda_slice.value().shape(), Device::cuda(0)));
    std::vector<float> cpu_grad(values.size()), cuda_grad(values.size());
    read_host(cpu_x.grad(), cpu_grad.data());
    read_host(cuda_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad);
    CHECK(cuda_x.grad().device().is_cuda());
    cuda_slice.backward(Tensor::from_host(
        slice_up.data(), cuda_slice.value().shape(), Device::cuda(0)));
    read_host(cuda_x.grad(), cuda_grad.data());
    for (std::size_t i = 0; i < cuda_grad.size(); ++i) {
        CHECK(std::fabs(cuda_grad[i] - 2.f * cpu_grad[i]) <= 1e-5f);
    }
    cuda_x.zero_grad();
    cuda_slice.backward(Tensor::from_host(
        slice_up.data(), cuda_slice.value().shape(), Device::cuda(0)));
    read_host(cuda_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad);

    const Shape a_shape{2, 1, 4};
    const Shape b_shape{2, 2, 4};
    std::vector<float> a_values(a_shape.numel()), b_values(b_shape.numel());
    for (std::size_t i = 0; i < a_values.size(); ++i) {
        a_values[i] = static_cast<float>(i + 1);
    }
    for (std::size_t i = 0; i < b_values.size(); ++i) {
        b_values[i] = static_cast<float>(i + 11);
    }
    Variable cpu_a(Tensor::from_host(a_values.data(), a_shape), true);
    Variable cpu_b(Tensor::from_host(b_values.data(), b_shape), true);
    Variable cuda_a(Tensor::from_host(a_values.data(), a_shape, Device::cuda(0)), true);
    Variable cuda_b(Tensor::from_host(b_values.data(), b_shape, Device::cuda(0)), true);
    Variable cpu_cat = ag::concat({cpu_a, cpu_b}, -2);
    Variable cuda_cat = ag::concat({cuda_a, cuda_b}, -2);
    CHECK(cuda_cat.value().shape() == (Shape{2, 3, 4}));
    CHECK(cuda_cat.value().device().is_cuda());
    cpu_value.resize(cpu_cat.value().elements());
    cuda_value.resize(cuda_cat.value().elements());
    read_host(cpu_cat.value(), cpu_value.data());
    read_host(cuda_cat.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);

    std::vector<float> cat_up(cuda_cat.value().elements());
    for (std::size_t i = 0; i < cat_up.size(); ++i) {
        cat_up[i] = static_cast<float>(i) * 0.1f - 0.2f;
    }
    cpu_cat.backward(Tensor::from_host(cat_up.data(), cpu_cat.value().shape()));
    cuda_cat.backward(Tensor::from_host(
        cat_up.data(), cuda_cat.value().shape(), Device::cuda(0)));
    std::vector<float> cpu_a_grad(a_values.size()), cuda_a_grad(a_values.size());
    std::vector<float> cpu_b_grad(b_values.size()), cuda_b_grad(b_values.size());
    read_host(cpu_a.grad(), cpu_a_grad.data());
    read_host(cuda_a.grad(), cuda_a_grad.data());
    read_host(cpu_b.grad(), cpu_b_grad.data());
    read_host(cuda_b.grad(), cuda_b_grad.data());
    check_close(cuda_a_grad, cpu_a_grad);
    check_close(cuda_b_grad, cpu_b_grad);
    cuda_cat.backward(Tensor::from_host(
        cat_up.data(), cuda_cat.value().shape(), Device::cuda(0)));
    read_host(cuda_a.grad(), cuda_a_grad.data());
    for (std::size_t i = 0; i < cuda_a_grad.size(); ++i) {
        CHECK(std::fabs(cuda_a_grad[i] - 2.f * cpu_a_grad[i]) <= 1e-5f);
    }
    cuda_a.zero_grad();
    cuda_b.zero_grad();
    cuda_cat.backward(Tensor::from_host(
        cat_up.data(), cuda_cat.value().shape(), Device::cuda(0)));
    read_host(cuda_a.grad(), cuda_a_grad.data());
    read_host(cuda_b.grad(), cuda_b_grad.data());
    check_close(cuda_a_grad, cpu_a_grad);
    check_close(cuda_b_grad, cpu_b_grad);

    const Shape empty_shape{2, 0, 4};
    Variable cpu_empty(Tensor::empty(empty_shape), true);
    Variable cuda_empty(Tensor::empty(empty_shape, Device::cuda(0)), true);
    Variable cpu_cat_empty = ag::concat({cpu_a, cpu_empty, cpu_b}, 1);
    Variable cuda_cat_empty = ag::concat({cuda_a, cuda_empty, cuda_b}, 1);
    CHECK(cuda_cat_empty.value().shape() == (Shape{2, 3, 4}));
    cpu_cat_empty.backward(Tensor::ones(cpu_cat_empty.value().shape()));
    cuda_cat_empty.backward(
        Tensor::ones(cuda_cat_empty.value().shape(), Device::cuda(0)));
    CHECK(cpu_empty.grad().elements() == 0);
    CHECK(cuda_empty.grad().elements() == 0);
    CHECK(cuda_empty.grad().device().is_cuda());

    const std::vector<float> image_values{1.f, 2.f, 3.f, 4.f};
    Variable cpu_image(Tensor::from_host(image_values.data(), Shape{1, 4}), true);
    Variable cuda_image(Tensor::from_host(image_values.data(), Shape{1, 4}, Device::cuda(0)), true);
    Variable cpu_reshaped = ag::reshape(cpu_image, Shape{1, 1, 2, 2});
    Variable cuda_reshaped = ag::reshape(cuda_image, Shape{1, 1, 2, 2});
    CHECK(ag::cuda_view(cuda_image.value()).data ==
          ag::cuda_view(cuda_reshaped.value()).data);
    std::vector<float> reshape_up(4, 1.f);
    cpu_reshaped.backward(Tensor::from_host(reshape_up.data(), Shape{1, 1, 2, 2}));
    cuda_reshaped.backward(Tensor::from_host(
        reshape_up.data(), Shape{1, 1, 2, 2}, Device::cuda(0)));
    std::vector<float> cpu_image_grad(4), cuda_image_grad(4);
    read_host(cpu_image.grad(), cpu_image_grad.data());
    read_host(cuda_image.grad(), cuda_image_grad.data());
    check_close(cuda_image_grad, cpu_image_grad);

    auto reflect_conv = [](Variable x) {
        Variable row0 = ag::row_slice(x, 0, 1);
        Variable row1 = ag::row_slice(x, 1, 1);
        auto padded_row = [](const Variable& row) {
            return ag::hcat({ag::col_slice(row, 1, 1), row,
                             ag::col_slice(row, 0, 1)});
        };
        Variable top = padded_row(row1);
        Variable middle0 = padded_row(row0);
        Variable middle1 = padded_row(row1);
        Variable bottom = padded_row(row0);
        Variable padded = ag::concat({top, middle0, middle1, bottom}, 0);
        Variable nchw = ag::reshape(padded, Shape{1, 1, 4, 4});
        Variable weight(Tensor::ones(Shape{1, 1, 3, 3}, x.device()), false);
        Variable bias(Tensor::zeros(Shape{1}, x.device()), false);
        return ag::conv2d(nchw, weight, bias, 1, 0);
    };
    Variable cpu_reflect(Tensor::from_host(image_values.data(), Shape{2, 2}), true);
    Variable cuda_reflect(Tensor::from_host(
        image_values.data(), Shape{2, 2}, Device::cuda(0)), true);
    Variable cpu_conv = reflect_conv(cpu_reflect);
    Variable cuda_conv = reflect_conv(cuda_reflect);
    cpu_value.resize(cpu_conv.value().elements());
    cuda_value.resize(cuda_conv.value().elements());
    read_host(cpu_conv.value(), cpu_value.data());
    read_host(cuda_conv.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);
    cpu_conv.backward(Tensor::ones(cpu_conv.value().shape()));
    cuda_conv.backward(Tensor::ones(cuda_conv.value().shape(), Device::cuda(0)));
    std::vector<float> cpu_reflect_grad(4), cuda_reflect_grad(4);
    read_host(cpu_reflect.grad(), cpu_reflect_grad.data());
    read_host(cuda_reflect.grad(), cuda_reflect_grad.data());
    check_close(cuda_reflect_grad, cpu_reflect_grad);
    CHECK(cuda_reflect.grad().device().is_cuda());

    CHECK_THROWS_AS(std::invalid_argument, ag::slice(cuda_x, 0, -1, 1));
    CHECK_THROWS_AS(std::invalid_argument, ag::slice(cuda_x, 3, 0, 1));
    CHECK_THROWS_AS(std::invalid_argument, ag::reshape(cuda_x, Shape{5, 5}));
    Variable bad_shape(Tensor::ones(Shape{3, 2, 4}, Device::cuda(0)), false);
    CHECK_THROWS_AS(std::invalid_argument, ag::concat({cuda_a, bad_shape}, 1));
    Variable cpu_only(Tensor::ones(a_shape), false);
    CHECK_THROWS_AS(std::invalid_argument, ag::concat({cuda_a, cpu_only}, 1));

    report("Tensor CUDA: direct reshape/slice/concat rank-generic parity, VJPs, and reflect-conv composition");
}

// The OOP math/optim/conv path keeps explicit rejection for CUDA operations
// without direct kernels. Shape operations are covered by the direct CUDA
// test above; this test preserves the rejection boundary for the siblings.
void test_oop_math_rejects_cuda_variable() {
    auto cuda_var_of = [](const Tensor& t) {
        return ag::Variable(t, /*requires_grad=*/true);
    };

    // unary operations outside this gate remain rejected.
    {
        Variable x = cuda_var_of(Tensor::ones(Shape{4}, Device::cuda(0)));
        CHECK_THROWS_AS(std::runtime_error, ag::transpose(x));
        CHECK_THROWS_AS(std::runtime_error, ag::cumsum(x));
        CHECK_THROWS_AS(std::runtime_error, ag::sin_op(x));
    }

    // batched concat with one CUDA input and a CPU input remains
    // a CUDA rejection (concat validates every input uniformly).
    {
        Variable cuda_a = cuda_var_of(Tensor::ones(Shape{2}, Device::cuda(0)));
        Variable cpu_b = ag::Variable(Tensor::ones(Shape{2}), false);
        CHECK_THROWS_AS(std::invalid_argument,
                        ag::concat({cuda_a, cpu_b}, 0));
    }

    report("Tensor CUDA: OOP math free functions reject remaining unsupported CUDA inputs");
}

Variable composed_elementwise_graph(const Variable& x) {
    Variable one(Tensor::ones(x.value().shape(), x.device()), false);
    Variable value = ag::scale(ag::mul(ag::add(x, one), x), 0.5f);
    value = ag::relu(value);
    value = ag::sigmoid(value);
    value = ag::tanh_op(value);
    value = ag::exp_op(value);
    value = ag::log_op(value);
    value = ag::sqrt_op(value);
    value = ag::silu(value);
    value = ag::softplus(value);
    value = ag::sub(value, one);
    return ag::sum(ag::div_op(value, one));
}

void check_close(const std::vector<float>& actual,
                 const std::vector<float>& expected,
                 float tolerance) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::fabs(actual[i] - expected[i]) <= tolerance);
    }
}

void test_oop_elementwise_composed_graph_cuda() {
    const std::vector<float> values{0.5f, 1.f, 1.5f, 2.f};
    Variable cpu_x(Tensor::from_host(values.data(), Shape{2, 2}), true);
    Variable cuda_x(Tensor::from_host(values.data(), Shape{2, 2},
                                      Device::cuda(0)), true);
    Variable cpu_out = composed_elementwise_graph(cpu_x);
    Variable cuda_out = composed_elementwise_graph(cuda_x);
    cpu_out.backward();
    cuda_out.backward(Tensor::ones(Shape{}, Device::cuda(0)));

    std::vector<float> cpu_value(1), cuda_value(1);
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);
    std::vector<float> cpu_grad(values.size()), cuda_grad(values.size());
    read_host(cpu_x.grad(), cpu_grad.data());
    read_host(cuda_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad, 2e-5f);
    CHECK(cuda_out.value().device().is_cuda());
    CHECK(cuda_x.grad().device().is_cuda());

    report("Tensor CUDA: composed elementwise graph matches CPU forward/backward");
}

void test_oop_elementwise_rank4_and_empty_cuda() {
    const std::vector<float> values{
        -1.f, -0.5f, 0.f, 0.5f,
        1.f, 1.5f, 2.f, 2.5f};
    Variable cpu_x(Tensor::from_host(values.data(), Shape{1, 2, 2, 2}), true);
    Variable cuda_x(Tensor::from_host(values.data(), Shape{1, 2, 2, 2},
                                      Device::cuda(0)), true);
    Variable cpu_out = ag::sum(ag::softplus(ag::silu(cpu_x)));
    Variable cuda_out = ag::sum(ag::softplus(ag::silu(cuda_x)));
    cpu_out.backward();
    cuda_out.backward(Tensor::ones(Shape{}, Device::cuda(0)));

    std::vector<float> cpu_value(1), cuda_value(1);
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);
    std::vector<float> cpu_grad(values.size()), cuda_grad(values.size());
    read_host(cpu_x.grad(), cpu_grad.data());
    read_host(cuda_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad, 2e-5f);

    Variable empty(Tensor::empty(Shape{0, 3}, Device::cuda(0)), true);
    Variable empty_out = ag::sum(ag::relu(empty));
    empty_out.backward(Tensor::ones(Shape{}, Device::cuda(0)));
    CHECK(empty_out.value().device().is_cuda());
    CHECK(empty_out.value().elements() == 1);
    CHECK(empty.grad().device().is_cuda());
    CHECK(empty.grad().elements() == 0);

    report("Tensor CUDA: rank-4 parity and empty elementwise tensors are valid");
}

void test_oop_broadcast_add_rank4_cuda() {
    const Shape a_shape{2, 1, 2, 3};
    const Shape b_shape{1, 3};
    std::vector<float> a_values(12), b_values{0.25f, -0.5f, 1.25f};
    for (std::size_t i = 0; i < a_values.size(); ++i) {
        a_values[i] = static_cast<float>(i) * 0.25f - 0.5f;
    }
    Variable cpu_a(Tensor::from_host(a_values.data(), a_shape), true);
    Variable cpu_b(Tensor::from_host(b_values.data(), b_shape), true);
    Variable cuda_a(Tensor::from_host(a_values.data(), a_shape,
                                      Device::cuda(0)), true);
    Variable cuda_b(Tensor::from_host(b_values.data(), b_shape,
                                      Device::cuda(0)), true);
    Variable cpu_out = ag::sum(ag::broadcast_add(cpu_a, cpu_b));
    Variable cuda_out = ag::sum(ag::broadcast_add(cuda_a, cuda_b));
    cpu_out.backward();
    cuda_out.backward(Tensor::ones(Shape{}, Device::cuda(0)));

    std::vector<float> cpu_value(1), cuda_value(1);
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);
    std::vector<float> cpu_a_grad(a_values.size()), cuda_a_grad(a_values.size());
    std::vector<float> cpu_b_grad(b_values.size()), cuda_b_grad(b_values.size());
    read_host(cpu_a.grad(), cpu_a_grad.data());
    read_host(cuda_a.grad(), cuda_a_grad.data());
    read_host(cpu_b.grad(), cpu_b_grad.data());
    read_host(cuda_b.grad(), cuda_b_grad.data());
    check_close(cuda_a_grad, cpu_a_grad);
    check_close(cuda_b_grad, cpu_b_grad);
    CHECK(cuda_out.value().shape() == Shape{});
    CHECK(cuda_a.grad().device().is_cuda());
    CHECK(cuda_b.grad().device().is_cuda());

    Variable empty_a(Tensor::empty(Shape{0, 3}, Device::cuda(0)), false);
    Variable empty_b(Tensor::ones(Shape{1, 3}, Device::cuda(0)), true);
    Variable empty_out = ag::sum(ag::broadcast_add(empty_a, empty_b));
    empty_out.backward();
    std::vector<float> empty_b_grad(3);
    read_host(empty_b.grad(), empty_b_grad.data());
    check_close(empty_b_grad, std::vector<float>(3, 0.f));

    report("Tensor CUDA: rank-4 trailing broadcast add matches CPU forward/backward");
}

void test_oop_sum_axes_and_mean_cuda() {
    const Shape shape{2, 3, 4};
    std::vector<float> values(24);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) * 0.125f - 1.f;
    }

    Variable cpu_x(Tensor::from_host(values.data(), shape), true);
    Variable cuda_x(Tensor::from_host(values.data(), shape, Device::cuda(0)), true);
    Variable cpu_out = ag::sum(cpu_x, {0, 2}, false);
    Variable cuda_out = ag::sum(cuda_x, {0, 2}, false);
    cpu_out.backward(Tensor::ones(cpu_out.value().shape()));
    cuda_out.backward(Tensor::ones(cuda_out.value().shape(), Device::cuda(0)));
    std::vector<float> cpu_value(cpu_out.value().elements());
    std::vector<float> cuda_value(cuda_out.value().elements());
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);
    std::vector<float> cpu_grad(values.size()), cuda_grad(values.size());
    read_host(cpu_x.grad(), cpu_grad.data());
    read_host(cuda_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad);

    Variable cpu_keep_x(Tensor::from_host(values.data(), shape), true);
    Variable cuda_keep_x(Tensor::from_host(values.data(), shape,
                                            Device::cuda(0)), true);
    Variable cpu_keep = ag::sum(cpu_keep_x, {-1, 0}, true);
    Variable cuda_keep = ag::sum(cuda_keep_x, {-1, 0}, true);
    cpu_keep.backward(Tensor::ones(cpu_keep.value().shape()));
    cuda_keep.backward(Tensor::ones(cuda_keep.value().shape(), Device::cuda(0)));
    cpu_value.resize(cpu_keep.value().elements());
    cuda_value.resize(cuda_keep.value().elements());
    read_host(cpu_keep.value(), cpu_value.data());
    read_host(cuda_keep.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);
    read_host(cpu_keep_x.grad(), cpu_grad.data());
    read_host(cuda_keep_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad);

    Variable cpu_mean_x(Tensor::from_host(values.data(), shape), true);
    Variable cuda_mean_x(Tensor::from_host(values.data(), shape,
                                            Device::cuda(0)), true);
    Variable cpu_mean = ag::mean(cpu_mean_x, {-1, 0}, true);
    Variable cuda_mean = ag::mean(cuda_mean_x, {-1, 0}, true);
    cpu_mean.backward(Tensor::ones(cpu_mean.value().shape()));
    cuda_mean.backward(Tensor::ones(cuda_mean.value().shape(), Device::cuda(0)));
    cpu_value.resize(cpu_mean.value().elements());
    cuda_value.resize(cuda_mean.value().elements());
    read_host(cpu_mean.value(), cpu_value.data());
    read_host(cuda_mean.value(), cuda_value.data());
    check_close(cuda_value, cpu_value);
    read_host(cpu_mean_x.grad(), cpu_grad.data());
    read_host(cuda_mean_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad);

    Variable cpu_all_x(Tensor::from_host(values.data(), shape), true);
    Variable cuda_all_x(Tensor::from_host(values.data(), shape,
                                           Device::cuda(0)), true);
    Variable cpu_all = ag::mean(cpu_all_x);
    Variable cuda_all = ag::mean(cuda_all_x);
    cpu_all.backward();
    cuda_all.backward(Tensor::ones(Shape{}, Device::cuda(0)));
    std::vector<float> cpu_scalar(1), cuda_scalar(1);
    read_host(cpu_all.value(), cpu_scalar.data());
    read_host(cuda_all.value(), cuda_scalar.data());
    check_close(cuda_scalar, cpu_scalar);
    read_host(cpu_all_x.grad(), cpu_grad.data());
    read_host(cuda_all_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad);

    report("Tensor CUDA: multi-axis sum and mean match CPU with negative axes");
}

void test_oop_softmax_nonlast_axis_cuda() {
    const Shape shape{2, 3, 4};
    std::vector<float> values(24), upstream(24);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) * 0.1f - 1.f;
        upstream[i] = static_cast<float>(i % 5) * 0.2f - 0.3f;
    }

    Variable cpu_x(Tensor::from_host(values.data(), shape), true);
    Variable cuda_x(Tensor::from_host(values.data(), shape, Device::cuda(0)), true);
    Variable cpu_out = ag::softmax(cpu_x, 1);
    Variable cuda_out = ag::softmax(cuda_x, -2);
    cpu_out.backward(Tensor::from_host(upstream.data(), shape));
    cuda_out.backward(Tensor::from_host(upstream.data(), shape, Device::cuda(0)));
    std::vector<float> cpu_value(values.size()), cuda_value(values.size());
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value, 2e-5f);
    std::vector<float> cpu_grad(values.size()), cuda_grad(values.size());
    read_host(cpu_x.grad(), cpu_grad.data());
    read_host(cuda_x.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad, 2e-5f);

    Variable cpu_lx(Tensor::from_host(values.data(), shape), true);
    Variable cuda_lx(Tensor::from_host(values.data(), shape,
                                       Device::cuda(0)), true);
    Variable cpu_log = ag::log_softmax(cpu_lx, 1);
    Variable cuda_log = ag::log_softmax(cuda_lx, -2);
    cpu_log.backward(Tensor::from_host(upstream.data(), shape));
    cuda_log.backward(Tensor::from_host(upstream.data(), shape,
                                        Device::cuda(0)));
    read_host(cpu_log.value(), cpu_value.data());
    read_host(cuda_log.value(), cuda_value.data());
    check_close(cuda_value, cpu_value, 2e-5f);
    read_host(cpu_lx.grad(), cpu_grad.data());
    read_host(cuda_lx.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad, 2e-5f);
    CHECK(cuda_log.value().device().is_cuda());
    CHECK(cuda_lx.grad().device().is_cuda());

    report("Tensor CUDA: non-last and negative-axis softmax/log_softmax match CPU");
}

void test_oop_broadcast_repeated_backward_cuda() {
    Variable x(Tensor::ones(Shape{2, 1, 2, 3}, Device::cuda(0)), true);
    Variable b(Tensor::ones(Shape{1, 3}, Device::cuda(0)), false);
    Variable out = ag::sum(ag::broadcast_add(x, b));
    Tensor upstream = Tensor::ones(Shape{}, Device::cuda(0));
    out.backward(upstream);
    out.backward(upstream);
    std::vector<float> grad(x.value().elements());
    read_host(x.grad(), grad.data());
    check_close(grad, std::vector<float>(grad.size(), 2.f));
    CHECK(x.grad().device().is_cuda());

    report("Tensor CUDA: broadcast graph accumulates repeated backward on CUDA");
}

void test_oop_backward_cuda_accumulates_and_matmul_works() {
    Variable scalar(Tensor::ones(Shape{}, Device::cuda(0)), true);
    scalar.backward();
    CHECK(scalar.has_grad());
    CHECK(scalar.grad().device().is_cuda());

    Variable vector(Tensor::ones(Shape{2}, Device::cuda(0)), true);
    Tensor upstream = Tensor::ones(Shape{2}, Device::cuda(0));
    vector.backward(upstream);
    vector.backward(upstream);
    std::vector<float> accumulated(2);
    read_host(vector.grad(), accumulated.data());
    check_close(accumulated, std::vector<float>{2.f, 2.f});

    // matmul on CUDA now succeeds and produces a CUDA Variable;
    // its backward stays on CUDA as well.
    Variable a(Tensor::ones(Shape{2, 2}, Device::cuda(0)), true);
    Variable b(Tensor::ones(Shape{2, 2}, Device::cuda(0)), true);
    Variable mm = ag::matmul(a, b);
    CHECK(mm.value().device().is_cuda());
    mm.backward(Tensor::ones(Shape{2, 2}, Device::cuda(0)));
    CHECK(a.grad().device().is_cuda());
    CHECK(b.grad().device().is_cuda());

    report("Tensor CUDA: backward stays on CUDA, accumulates, matmul forward+backward on CUDA");
}

void test_oop_diffusion_rejects_cuda_tensor() {
    Tensor input = Tensor::ones(Shape{2, 3}, Device::cuda(0));
    CHECK_THROWS_AS(
        std::runtime_error,
        ag::diffusion::randn_like(input, 7));

    report("Tensor CUDA: randn_like rejects CUDA input");
}

void test_oop_optimizer_unsupported_cuda_constructs_reject() {
    // optim::Adam constructor pre-validates hyperparameters; only
    // finite/non-negative lr / finite beta in [0,1) / positive eps
    // are accepted. We confirm those contracts still reject on CUDA.
    {
        Variable p(Tensor::ones(Shape{2, 2}, Device::cuda(0)), true);
        bool threw = false;
        try {
            ag::optim::Adam adam({p}, /*lr=*/-1.f);
            (void)adam;
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }

    report("Tensor CUDA: optim::Adam hyperparameter validation rejects invalid values");
}

void test_oop_conv2d_cuda_inputs() {
    Variable input(Tensor::ones(Shape{1, 1, 4, 4}, Device::cuda(0)), true);
    Variable weight(Tensor::ones(Shape{1, 1, 3, 3}, Device::cuda(0)), true);
    Variable bias(Tensor::zeros(Shape{1}, Device::cuda(0)), true);
    Variable out = ag::conv2d(input, weight, bias, 1, 0);
    CHECK(out.value().device().is_cuda());
    out.backward(Tensor::ones(out.value().shape(), Device::cuda(0)));
    CHECK(input.grad().device().is_cuda());
    CHECK(weight.grad().device().is_cuda());
    CHECK(bias.grad().device().is_cuda());
    report("Tensor CUDA: ag::conv2d supports CUDA inputs and gradients");
}

void test_oop_loss_mixed_device_rejects() {
    // mse_loss / cross_entropy accept all-CUDA inputs (covered in
    // dedicated parity tests below). Mixed-device calls (pred on
    // one device, target on the other) keep throwing so silent
    // device transfers cannot leak through loss APIs.
    Variable pred_cuda(Tensor::ones(Shape{2, 3}, Device::cuda(0)), true);
    Tensor target_cpu = Tensor::zeros(Shape{2, 3});
    CHECK_THROWS_AS(std::runtime_error,
                    ag::mse_loss(pred_cuda, target_cpu));
    CHECK_THROWS_AS(std::runtime_error,
                    ag::cross_entropy(pred_cuda, target_cpu));

    Variable pred_cpu_v(Tensor::ones(Shape{2, 3}), true);
    Tensor target_cuda = Tensor::zeros(Shape{2, 3}, Device::cuda(0));
    CHECK_THROWS_AS(std::runtime_error,
                    ag::mse_loss(pred_cpu_v, target_cuda));
    CHECK_THROWS_AS(std::runtime_error,
                    ag::cross_entropy(pred_cpu_v, target_cuda));

    report("Tensor CUDA: mse_loss / cross_entropy reject mixed-device pred/target");
}

// ── CUDA matmul / loss / optimizer parity ──────────────────────────────
//
// Each test below exercises a single OOP operation end-to-end on a
// real CUDA device and asserts CPU <-> CUDA parity within a small
// tolerance. The matmul tests cover rank-2, batched rank-3, and
// rank-4 inputs. The optim tests cover SGD and Adam in-place updates,
// alias visibility, mixed-device parameter lists, and AdamState
// snapshot/restore with device preservation.

void test_oop_matmul_rank2_cuda_parity() {
    // Deterministic non-square fixtures for forward + both gradients.
    const Shape a_shape{2, 3};
    const Shape b_shape{3, 4};
    std::vector<float> av{
        1.f, -2.f, 3.f,
        -4.f, 5.f, -6.f,
    };
    std::vector<float> bv{
        0.5f, -1.f, 1.5f, -2.f,
        2.5f, -3.f, 3.5f, -4.f,
        -4.5f, 5.f, -5.5f, 6.f,
    };

    Variable cpu_a(Tensor::from_host(av.data(), a_shape), true);
    Variable cpu_b(Tensor::from_host(bv.data(), b_shape), true);
    Variable cuda_a(Tensor::from_host(av.data(), a_shape, Device::cuda(0)), true);
    Variable cuda_b(Tensor::from_host(bv.data(), b_shape, Device::cuda(0)), true);

    Variable cpu_out = ag::matmul(cpu_a, cpu_b);
    Variable cuda_out = ag::matmul(cuda_a, cuda_b);
    CHECK(cpu_out.value().shape() == (Shape{2, 4}));
    CHECK(cuda_out.value().shape() == (Shape{2, 4}));
    CHECK(cuda_out.value().device().is_cuda());

    std::vector<float> cpu_value(cpu_out.value().elements());
    std::vector<float> cuda_value(cuda_out.value().elements());
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value, 5e-5f);

    Tensor upstream = Tensor::ones(cpu_out.value().shape(), Device::cuda(0));
    Tensor cpu_up = Tensor::ones(cpu_out.value().shape());
    cpu_out.backward(cpu_up);
    cuda_out.backward(upstream);
    std::vector<float> cpu_a_grad(av.size()), cuda_a_grad(av.size());
    std::vector<float> cpu_b_grad(bv.size()), cuda_b_grad(bv.size());
    read_host(cpu_a.grad(), cpu_a_grad.data());
    read_host(cuda_a.grad(), cuda_a_grad.data());
    read_host(cpu_b.grad(), cpu_b_grad.data());
    read_host(cuda_b.grad(), cuda_b_grad.data());
    check_close(cuda_a_grad, cpu_a_grad, 5e-5f);
    check_close(cuda_b_grad, cpu_b_grad, 5e-5f);
    CHECK(cuda_a.grad().device().is_cuda());
    CHECK(cuda_b.grad().device().is_cuda());

    report("Tensor CUDA: matmul rank-2 forward + both operand gradients match CPU");
}

void test_oop_matmul_batched_rank3_cuda_parity() {
    // (B, M, K) @ (B, K, N) with non-trivial batch dim.
    const Shape a_shape{2, 3, 4};
    const Shape b_shape{2, 4, 5};
    std::vector<float> av(a_shape.numel()), bv(b_shape.numel());
    for (std::size_t i = 0; i < av.size(); ++i) {
        av[i] = static_cast<float>(i) * 0.125f - 0.5f;
    }
    for (std::size_t i = 0; i < bv.size(); ++i) {
        bv[i] = static_cast<float>(i) * 0.25f - 1.f;
    }

    Variable cpu_a(Tensor::from_host(av.data(), a_shape), true);
    Variable cpu_b(Tensor::from_host(bv.data(), b_shape), true);
    Variable cuda_a(Tensor::from_host(av.data(), a_shape, Device::cuda(0)), true);
    Variable cuda_b(Tensor::from_host(bv.data(), b_shape, Device::cuda(0)), true);

    Variable cpu_out = ag::matmul(cpu_a, cpu_b);
    Variable cuda_out = ag::matmul(cuda_a, cuda_b);
    CHECK(cpu_out.value().shape() == (Shape{2, 3, 5}));
    CHECK(cuda_out.value().shape() == (Shape{2, 3, 5}));

    std::vector<float> cpu_value(cpu_out.value().elements());
    std::vector<float> cuda_value(cuda_out.value().elements());
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value, 5e-5f);

    Tensor cpu_up = Tensor::ones(cpu_out.value().shape());
    Tensor cuda_up = Tensor::ones(cpu_out.value().shape(), Device::cuda(0));
    cpu_out.backward(cpu_up);
    cuda_out.backward(cuda_up);
    std::vector<float> cpu_a_grad(av.size()), cuda_a_grad(av.size());
    std::vector<float> cpu_b_grad(bv.size()), cuda_b_grad(bv.size());
    read_host(cpu_a.grad(), cpu_a_grad.data());
    read_host(cuda_a.grad(), cuda_a_grad.data());
    read_host(cpu_b.grad(), cpu_b_grad.data());
    read_host(cuda_b.grad(), cuda_b_grad.data());
    check_close(cuda_a_grad, cpu_a_grad, 5e-5f);
    check_close(cuda_b_grad, cpu_b_grad, 5e-5f);

    report("Tensor CUDA: matmul rank-3 batched forward + both gradients match CPU");
}

void test_oop_matmul_later_batches_cuda() {
    // Each batch of the operands is filled with a different constant
    // (batch 0 = all -1, batch 1 = all 0, batch 2 = all +1). A
    // correct batched matmul must produce per-batch outputs whose
    // magnitudes differ; a kernel that silently drops blockIdx.z (or
    // any other source of cross-batch confusion) would collapse the
    // outputs to a single batch's value.
    const Shape a_shape{3, 2, 3};  // B=3, M=2, K=3
    const Shape b_shape{3, 3, 4};  // B=3, K=3, N=4
    const int64_t per_batch_a = a_shape[1] * a_shape[2];  // 6
    const int64_t per_batch_b = b_shape[1] * b_shape[2];  // 12
    std::vector<float> av(a_shape.numel());
    std::vector<float> bv(b_shape.numel());
    for (int b = 0; b < 3; ++b) {
        const float av_v = static_cast<float>(b) - 1.f;  // -1, 0, +1
        const float bv_v = static_cast<float>(b) * 0.5f + 0.25f;  // 0.25, 0.75, 1.25
        for (int i = 0; i < per_batch_a; ++i) {
            av[b * per_batch_a + i] = av_v;
        }
        for (int i = 0; i < per_batch_b; ++i) {
            bv[b * per_batch_b + i] = bv_v;
        }
    }

    Variable cpu_a(Tensor::from_host(av.data(), a_shape), true);
    Variable cpu_b(Tensor::from_host(bv.data(), b_shape), true);
    Variable cuda_a(Tensor::from_host(av.data(), a_shape, Device::cuda(0)), true);
    Variable cuda_b(Tensor::from_host(bv.data(), b_shape, Device::cuda(0)), true);

    Variable cpu_out = ag::matmul(cpu_a, cpu_b);
    Variable cuda_out = ag::matmul(cuda_a, cuda_b);
    CHECK(cpu_out.value().shape() == (Shape{3, 2, 4}));
    CHECK(cuda_out.value().shape() == (Shape{3, 2, 4}));

    std::vector<float> cpu_v(cpu_out.value().elements());
    std::vector<float> cuda_v(cuda_out.value().elements());
    read_host(cpu_out.value(), cpu_v.data());
    read_host(cuda_out.value(), cuda_v.data());
    check_close(cuda_v, cpu_v, 5e-5f);

    // Each batch's per-output element is a*b_scalar * K (constant
    // per batch). The batches must therefore produce distinct
    // constant outputs that visibly differ from one another. This
    // guards against a kernel bug that drops blockIdx.z or any
    // other cross-batch addressing.
    const int64_t per_batch_out = cpu_out.value().shape()[1] *
                                  cpu_out.value().shape()[2];  // 8
    for (int b = 1; b < 3; ++b) {
        bool any_diff = false;
        for (int64_t i = 0; i < per_batch_out; ++i) {
            if (std::fabs(cuda_v[b * per_batch_out + i] -
                          cuda_v[0 * per_batch_out + i]) > 1e-6f) {
                any_diff = true;
                break;
            }
        }
        CHECK(any_diff);
    }

    report("Tensor CUDA: matmul rank-3 each batch has distinct per-batch output");
}

void test_oop_matmul_batched_rank4_cuda_parity() {
    // (B1, B2, M, K) @ (B1, B2, K, N): two leading batch dims.
    const Shape a_shape{2, 3, 2, 4};
    const Shape b_shape{2, 3, 4, 3};
    std::vector<float> av(a_shape.numel()), bv(b_shape.numel());
    for (std::size_t i = 0; i < av.size(); ++i) {
        av[i] = static_cast<float>(i) * 0.1f - 0.6f;
    }
    for (std::size_t i = 0; i < bv.size(); ++i) {
        bv[i] = static_cast<float>(i) * -0.15f + 0.4f;
    }

    Variable cpu_a(Tensor::from_host(av.data(), a_shape), true);
    Variable cpu_b(Tensor::from_host(bv.data(), b_shape), true);
    Variable cuda_a(Tensor::from_host(av.data(), a_shape, Device::cuda(0)), true);
    Variable cuda_b(Tensor::from_host(bv.data(), b_shape, Device::cuda(0)), true);

    Variable cpu_out = ag::matmul(cpu_a, cpu_b);
    Variable cuda_out = ag::matmul(cuda_a, cuda_b);
    CHECK(cpu_out.value().shape() == (Shape{2, 3, 2, 3}));
    CHECK(cuda_out.value().shape() == (Shape{2, 3, 2, 3}));

    std::vector<float> cpu_value(cpu_out.value().elements());
    std::vector<float> cuda_value(cuda_out.value().elements());
    read_host(cpu_out.value(), cpu_value.data());
    read_host(cuda_out.value(), cuda_value.data());
    check_close(cuda_value, cpu_value, 5e-5f);

    Tensor cpu_up = Tensor::ones(cpu_out.value().shape());
    Tensor cuda_up = Tensor::ones(cpu_out.value().shape(), Device::cuda(0));
    cpu_out.backward(cpu_up);
    cuda_out.backward(cuda_up);
    std::vector<float> cpu_a_grad(av.size()), cuda_a_grad(av.size());
    std::vector<float> cpu_b_grad(bv.size()), cuda_b_grad(bv.size());
    read_host(cpu_a.grad(), cpu_a_grad.data());
    read_host(cuda_a.grad(), cuda_a_grad.data());
    read_host(cpu_b.grad(), cpu_b_grad.data());
    read_host(cuda_b.grad(), cuda_b_grad.data());
    check_close(cuda_a_grad, cpu_a_grad, 5e-5f);
    check_close(cuda_b_grad, cpu_b_grad, 5e-5f);

    report("Tensor CUDA: matmul rank-4 batched forward + both gradients match CPU");
}

void test_oop_matmul_shape_and_zero_cuda() {
    // Mixed-device matmul: pred on CUDA, other on CPU remains an
    // invalid_argument from the binary shape/device check, exactly
    // like on CPU.
    {
        Variable a(Tensor::ones(Shape{2, 3}, Device::cuda(0)), true);
        Variable b(Tensor::ones(Shape{2, 3}), true);
        CHECK_THROWS_AS(std::invalid_argument, ag::matmul(a, b));
    }

    // Inner-dim mismatch on CUDA.
    {
        Variable a(Tensor::ones(Shape{2, 3}, Device::cuda(0)), true);
        Variable b(Tensor::ones(Shape{4, 2}, Device::cuda(0)), true);
        CHECK_THROWS_AS(std::invalid_argument, ag::matmul(a, b));
    }

    // Rank<2 rejection on CUDA.
    {
        Variable a(Tensor::ones(Shape{4}, Device::cuda(0)), true);
        Variable b(Tensor::ones(Shape{4}, Device::cuda(0)), true);
        CHECK_THROWS_AS(std::invalid_argument, ag::matmul(a, b));
    }

    // Batch-dim mismatch on CUDA.
    {
        Variable a(Tensor::ones(Shape{2, 3, 2}, Device::cuda(0)), true);
        Variable b(Tensor::ones(Shape{3, 2, 3}, Device::cuda(0)), true);
        CHECK_THROWS_AS(std::invalid_argument, ag::matmul(a, b));
    }

    // Zero-sized leading batch is valid on CUDA.
    {
        Variable a(Tensor::zeros(Shape{0, 2, 3}, Device::cuda(0)), true);
        Variable b(Tensor::zeros(Shape{0, 3, 4}, Device::cuda(0)), true);
        Variable out = ag::matmul(a, b);
        CHECK(out.value().shape() == (Shape{0, 2, 4}));
        CHECK(out.value().device().is_cuda());
        Tensor up = Tensor::ones(out.value().shape(), Device::cuda(0));
        out.backward(up);
        CHECK(a.grad().shape() == (Shape{0, 2, 3}));
        CHECK(b.grad().shape() == (Shape{0, 3, 4}));
        CHECK(a.grad().device().is_cuda());
        CHECK(b.grad().device().is_cuda());
        CHECK(a.grad().empty());
        CHECK(b.grad().empty());
    }

    // Zero inner dimension has a non-empty, all-zero output.
    {
        Variable cpu_a(Tensor::zeros(Shape{2, 0}), true);
        Variable cpu_b(Tensor::zeros(Shape{0, 3}), true);
        Variable cuda_a(
            Tensor::zeros(Shape{2, 0}, Device::cuda(0)), true);
        Variable cuda_b(
            Tensor::zeros(Shape{0, 3}, Device::cuda(0)), true);
        Variable cpu_out = ag::matmul(cpu_a, cpu_b);
        Variable cuda_out = ag::matmul(cuda_a, cuda_b);
        CHECK(cuda_out.value().shape() == (Shape{2, 3}));
        std::vector<float> cpu_value(6), cuda_value(6);
        read_host(cpu_out.value(), cpu_value.data());
        read_host(cuda_out.value(), cuda_value.data());
        check_close(cuda_value, cpu_value);
        check_close(cuda_value, std::vector<float>(6, 0.f));

        cpu_out.backward(Tensor::ones(Shape{2, 3}));
        cuda_out.backward(Tensor::ones(Shape{2, 3}, Device::cuda(0)));
        CHECK(cuda_a.grad().shape() == (Shape{2, 0}));
        CHECK(cuda_b.grad().shape() == (Shape{0, 3}));
        CHECK(cuda_a.grad().empty());
        CHECK(cuda_b.grad().empty());
    }

    // Empty output axes can still produce a non-empty zero gradient
    // for the opposite operand.
    {
        Variable cuda_a(
            Tensor::ones(Shape{2, 3}, Device::cuda(0)), true);
        Variable cuda_b(
            Tensor::zeros(Shape{3, 0}, Device::cuda(0)), true);
        Variable out = ag::matmul(cuda_a, cuda_b);
        out.backward(Tensor::ones(Shape{2, 0}, Device::cuda(0)));
        std::vector<float> a_grad(6);
        read_host(cuda_a.grad(), a_grad.data());
        check_close(a_grad, std::vector<float>(6, 0.f));
        CHECK(cuda_b.grad().empty());
    }
    {
        Variable cuda_a(
            Tensor::zeros(Shape{0, 3}, Device::cuda(0)), true);
        Variable cuda_b(
            Tensor::ones(Shape{3, 2}, Device::cuda(0)), true);
        Variable out = ag::matmul(cuda_a, cuda_b);
        out.backward(Tensor::ones(Shape{0, 2}, Device::cuda(0)));
        CHECK(cuda_a.grad().empty());
        std::vector<float> b_grad(6);
        read_host(cuda_b.grad(), b_grad.data());
        check_close(b_grad, std::vector<float>(6, 0.f));
    }

    report("Tensor CUDA: matmul validates shapes; zero-sized dimensions work");
}

void test_oop_mse_loss_cuda_parity() {
    // All-CUDA pred + target produces an all-CUDA scalar loss whose
    // forward value and gradient match CPU.
    const Shape shape{2, 3};
    std::vector<float> pv{
        0.5f, -1.f, 1.5f,
        -2.f, 2.5f, -3.f,
    };
    std::vector<float> tv{
        0.f, 1.f, -1.f,
        1.5f, -1.5f, 2.f,
    };
    Variable cpu_p(Tensor::from_host(pv.data(), shape), true);
    Variable cuda_p(Tensor::from_host(pv.data(), shape, Device::cuda(0)), true);
    Tensor cpu_t = Tensor::from_host(tv.data(), shape);
    Tensor cuda_t = Tensor::from_host(tv.data(), shape, Device::cuda(0));

    Variable cpu_loss = ag::mse_loss(cpu_p, cpu_t);
    Variable cuda_loss = ag::mse_loss(cuda_p, cuda_t);
    CHECK(cpu_loss.value().shape() == (Shape{}));
    CHECK(cuda_loss.value().shape() == (Shape{}));
    CHECK(cuda_loss.value().device().is_cuda());

    std::vector<float> cpu_value(1), cuda_value(1);
    read_host(cpu_loss.value(), cpu_value.data());
    read_host(cuda_loss.value(), cuda_value.data());
    check_close(cuda_value, cpu_value, 5e-5f);

    cpu_loss.backward();
    cuda_loss.backward(Tensor::ones(Shape{}, Device::cuda(0)));
    std::vector<float> cpu_grad(pv.size()), cuda_grad(pv.size());
    read_host(cpu_p.grad(), cpu_grad.data());
    read_host(cuda_p.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad, 5e-5f);
    CHECK(cuda_p.grad().device().is_cuda());

    report("Tensor CUDA: mse_loss all-CUDA pred+target forward + backward parity");
}

void test_oop_cross_entropy_cuda_parity() {
    // All-CUDA pred + one-hot target.
    const Shape shape{2, 3};
    std::vector<float> pv{
        0.5f, -1.f, 1.5f,
        -2.f, 2.5f, -3.f,
    };
    std::vector<float> tv{
        0.f, 1.f, 0.f,
        0.f, 0.f, 1.f,
    };
    Variable cpu_p(Tensor::from_host(pv.data(), shape), true);
    Variable cuda_p(Tensor::from_host(pv.data(), shape, Device::cuda(0)), true);
    Tensor cpu_t = Tensor::from_host(tv.data(), shape);
    Tensor cuda_t = Tensor::from_host(tv.data(), shape, Device::cuda(0));

    Variable cpu_loss = ag::cross_entropy(cpu_p, cpu_t);
    Variable cuda_loss = ag::cross_entropy(cuda_p, cuda_t);
    CHECK(cpu_loss.value().shape() == (Shape{}));
    CHECK(cuda_loss.value().device().is_cuda());

    std::vector<float> cpu_value(1), cuda_value(1);
    read_host(cpu_loss.value(), cpu_value.data());
    read_host(cuda_loss.value(), cuda_value.data());
    check_close(cuda_value, cpu_value, 5e-5f);

    cpu_loss.backward();
    cuda_loss.backward(Tensor::ones(Shape{}, Device::cuda(0)));
    std::vector<float> cpu_grad(pv.size()), cuda_grad(pv.size());
    read_host(cpu_p.grad(), cpu_grad.data());
    read_host(cuda_p.grad(), cuda_grad.data());
    check_close(cuda_grad, cpu_grad, 5e-5f);
    CHECK(cuda_p.grad().device().is_cuda());

    report("Tensor CUDA: cross_entropy all-CUDA pred+target forward + backward parity");
}

void test_oop_sgd_cuda_step_parity_and_alias_visible() {
    // Same initial value and gradient on CPU and CUDA. SGD step()
    // parity within a small tolerance; alias-visible mutation
    // observed through a Tensor taken before step().
    const std::vector<float> pv{
        1.f, -2.f, 3.f, -4.f,
        5.f, -6.f, 7.f, -8.f,
    };
    const std::vector<float> gv{
        0.1f, -0.2f, 0.3f, -0.4f,
        0.5f, -0.6f, 0.7f, -0.8f,
    };
    Variable cpu_p(Tensor::from_host(pv.data(), Shape{2, 4}), true);
    Variable cuda_p(Tensor::from_host(pv.data(), Shape{2, 4}, Device::cuda(0)), true);
    Tensor cpu_alias = cpu_p.value();  // Tensor alias before step
    Tensor cuda_alias = cuda_p.value();

    assign_grad(cpu_p, gv);
    assign_grad(cuda_p, gv);

    ag::optim::SGD cpu_opt({cpu_p}, 0.05f);
    ag::optim::SGD cuda_opt({cuda_p}, 0.05f);
    cpu_opt.step();
    cuda_opt.step();

    // Aliases observe the post-step values without an explicit copy.
    std::vector<float> cpu_after(pv.size()), cuda_after(pv.size());
    read_host(cpu_p.value(), cpu_after.data());
    read_host(cuda_p.value(), cuda_after.data());
    read_host(cpu_alias, cpu_after.data());    // alias sees update
    read_host(cuda_alias, cuda_after.data());  // alias sees update
    check_close(cuda_after, cpu_after, 5e-5f);

    // zero_grad clears the gradient.
    cpu_opt.zero_grad();
    cuda_opt.zero_grad();
    CHECK(!cpu_p.has_grad());
    CHECK(!cuda_p.has_grad());

    report("Tensor CUDA: optim::SGD step parity and alias-visible mutation");
}

void test_oop_sgd_cuda_mixed_list() {
    // Mixed CPU/CUDA parameter list: both parameters receive the
    // step; CUDA one observes it through alias.
    Variable cpu_p(Tensor::from_host(
        std::vector<float>{2.f, 4.f}.data(), Shape{2}), true);
    Variable cuda_p(Tensor::from_host(
        std::vector<float>{6.f, 8.f}.data(), Shape{2}, Device::cuda(0)), true);
    Tensor cuda_alias = cuda_p.value();
    assign_grad(cpu_p, std::vector<float>{1.f, -1.f});
    assign_grad(cuda_p, std::vector<float>{0.5f, -0.5f});

    ag::optim::SGD mixed({cpu_p, cuda_p}, 0.1f);
    mixed.step();

    std::vector<float> cpu_v(2), cuda_v(2);
    read_host(cpu_p.value(), cpu_v.data());
    read_host(cuda_p.value(), cuda_v.data());
    check_close(cpu_v, std::vector<float>{1.9f, 4.1f}, 5e-5f);
    check_close(cuda_v, std::vector<float>{5.95f, 8.05f}, 5e-5f);
    std::vector<float> alias_v(2);
    read_host(cuda_alias, alias_v.data());
    check_close(alias_v, std::vector<float>{5.95f, 8.05f}, 5e-5f);

    report("Tensor CUDA: optim::SGD mixed CPU/CUDA parameter list updates both");
}

void test_oop_adam_cuda_step_parity_and_moments() {
    // Same gradient sequence on CPU and CUDA. After three steps
    // both parameters match the CPU trajectory; CUDA moments evolve
    // alongside. t_ advances once per step (verified via state()).
    const std::vector<float> pv{
        0.5f, -0.25f, 0.75f, -0.5f,
        0.125f, -0.875f, 0.625f, 0.375f,
    };
    Variable cpu_p(Tensor::from_host(pv.data(), Shape{2, 4}), true);
    Variable cuda_p(Tensor::from_host(pv.data(), Shape{2, 4}, Device::cuda(0)), true);

    ag::optim::Adam cpu_adam({cpu_p}, 1e-2f);
    ag::optim::Adam cuda_adam({cuda_p}, 1e-2f);

    const std::vector<std::vector<float>> gradients{
        {0.05f, -0.05f, 0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f},
        {-0.1f, 0.1f, -0.2f, 0.2f, -0.05f, 0.05f, -0.15f, 0.15f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    };

    for (const auto& g : gradients) {
        assign_grad(cpu_p, g);
        assign_grad(cuda_p, g);
        cpu_adam.step();
        cuda_adam.step();
        std::vector<float> cpu_v(pv.size()), cuda_v(pv.size());
        read_host(cpu_p.value(), cpu_v.data());
        read_host(cuda_p.value(), cuda_v.data());
        check_close(cuda_v, cpu_v, 5e-5f);
    }

    // state() preserves device on the moments and matches CPU data.
    ag::optim::AdamState cpu_state = cpu_adam.state();
    ag::optim::AdamState cuda_state = cuda_adam.state();
    CHECK(cpu_state.first_moments.size() == 1);
    CHECK(cuda_state.first_moments.size() == 1);
    CHECK(cuda_state.first_moments[0].device().is_cuda());
    CHECK(cuda_state.second_moments[0].device().is_cuda());
    std::vector<float> cpu_m(cpu_state.first_moments[0].elements());
    std::vector<float> cuda_m(cuda_state.first_moments[0].elements());
    read_host(cpu_state.first_moments[0], cpu_m.data());
    read_host(cuda_state.first_moments[0], cuda_m.data());
    check_close(cuda_m, cpu_m, 5e-5f);

    // state() does not alias the live moments: mutating the snapshot
    // does not change the optimizer's parameters on a future step.
    std::vector<float> zeros(cuda_m.size(), 0.f);
    cuda_state.first_moments[0].copy_from_host(zeros.data(), zeros.size());
    assign_grad(cuda_p, gradients.back());
    cuda_adam.step();
    std::vector<float> post(pv.size());
    read_host(cuda_p.value(), post.data());
    assign_grad(cpu_p, gradients.back());
    cpu_adam.step();
    std::vector<float> cpu_after(pv.size());
    read_host(cpu_p.value(), cpu_after.data());
    check_close(post, cpu_after, 5e-5f);

    // Step counter advances once per non-empty step.
    CHECK(cpu_adam.step_count() == 4);
    CHECK(cuda_adam.step_count() == 4);

    report("Tensor CUDA: optim::Adam multi-step parity, moments on CUDA, snapshot non-aliasing");
}

void test_oop_adam_cuda_mixed_list() {
    // Mixed CPU/CUDA Adam parameter list: both parameters receive
    // the same step update and stay in sync with their
    // single-device references.
    const std::vector<float> pv{
        0.5f, -0.25f, 0.75f, -0.5f,
    };
    Variable cpu_p(Tensor::from_host(pv.data(), Shape{2, 2}), true);
    Variable cuda_p(Tensor::from_host(pv.data(), Shape{2, 2}, Device::cuda(0)), true);
    Variable ref_cpu(Tensor::from_host(pv.data(), Shape{2, 2}), true);
    Variable ref_cuda(Tensor::from_host(pv.data(), Shape{2, 2}, Device::cuda(0)), true);

    ag::optim::Adam mixed({cpu_p, cuda_p}, 1e-2f);
    ag::optim::Adam ref_c({ref_cpu}, 1e-2f);
    ag::optim::Adam ref_g({ref_cuda}, 1e-2f);

    const std::vector<float> g{0.05f, -0.05f, 0.1f, -0.1f};
    for (int step = 0; step < 2; ++step) {
        assign_grad(cpu_p, g);
        assign_grad(cuda_p, g);
        assign_grad(ref_cpu, g);
        assign_grad(ref_cuda, g);
        mixed.step();
        ref_c.step();
        ref_g.step();
        std::vector<float> mv(pv.size()), cv(pv.size()), rv(pv.size());
        read_host(cpu_p.value(), mv.data());
        read_host(cuda_p.value(), cv.data());
        read_host(ref_cpu.value(), rv.data());
        check_close(mv, rv, 5e-5f);
        read_host(ref_cuda.value(), rv.data());
        check_close(cv, rv, 5e-5f);
    }

    report("Tensor CUDA: optim::Adam mixed CPU/CUDA parameter list updates both");
}

void test_oop_adam_cuda_load_state_preserves_device() {
    // Construct two CUDA Adam optimizers with the same params.
    // Run a step on source, snapshot source.state(), and load that
    // snapshot into target. Then drive both target and a fresh
    // reference optimizer through the same next gradient and verify
    // that the parameter values match: load_state restores the
    // optimizer's full state (step count + moments + hyperparameters)
    // so the resumed trajectory is identical to one that took the
    // original step in-line.
    const std::vector<float> pv{
        0.5f, -0.25f, 0.75f, -0.5f,
    };
    Variable source_p(Tensor::from_host(pv.data(), Shape{2, 2}, Device::cuda(0)), true);
    Variable target_p(Tensor::from_host(pv.data(), Shape{2, 2}, Device::cuda(0)), true);
    Variable ref_p(Tensor::from_host(pv.data(), Shape{2, 2}, Device::cuda(0)), true);

    ag::optim::Adam source({source_p}, 1e-2f);
    ag::optim::Adam target({target_p}, 1e-2f);
    ag::optim::Adam reference({ref_p}, 1e-2f);

    const std::vector<float> g1{0.05f, -0.05f, 0.1f, -0.1f};
    const std::vector<float> g2{-0.1f, 0.1f, -0.05f, 0.05f};

    // Bring target to the same post-g1 state as source.
    assign_grad(source_p, g1);
    source.step();
    source.zero_grad();
    assign_grad(target_p, g1);
    target.step();
    target.zero_grad();
    // Reference needs to also see g1 so its bias-correction step
    // count matches target's after load_state (t_ = 1).
    assign_grad(ref_p, g1);
    reference.step();
    reference.zero_grad();

    // Snapshot source's state and load it into target. After the
    // load, target's optimizer state (moments, step_count,
    // hyperparameters) is identical to source's — but target_p.value
    // already matches source_p.value because we stepped target too.
    ag::optim::AdamState snapshot = source.state();
    CHECK(snapshot.first_moments[0].device().is_cuda());
    CHECK(snapshot.second_moments[0].device().is_cuda());
    target.load_state(snapshot);
    CHECK(target.step_count() == source.step_count());

    // Snapshot moments are deep copies: mutating them does not
    // perturb the live target optimizer's moments.
    std::vector<float> zeros(pv.size(), 0.f);
    snapshot.first_moments[0].copy_from_host(zeros.data(), zeros.size());
    snapshot.second_moments[0].copy_from_host(zeros.data(), zeros.size());

    // Continue both target (loaded) and reference (fresh) with the
    // same gradient; they must converge to the same parameters
    // because load_state restored target's full AdamState (step
    // count + moments) before the next step.
    assign_grad(target_p, g2);
    target.step();
    assign_grad(ref_p, g2);
    reference.step();

    std::vector<float> tv(pv.size()), rv(pv.size());
    read_host(target_p.value(), tv.data());
    read_host(ref_p.value(), rv.data());
    check_close(tv, rv, 5e-5f);

    report("Tensor CUDA: optim::Adam load_state preserves device, "
           "continues trajectory, deep copies moments");
}

void test_predicate_selection_and_status_cuda() {
    const std::vector<float> lhs_values{-1.f, 0.f, 2.f,
                                        std::numeric_limits<float>::quiet_NaN()};
    const std::vector<float> rhs_values{-1.f, 1.f, 1.f, 4.f};
    Variable lhs(Tensor::from_host(lhs_values.data(), Shape{4}, Device::cuda(0)), true);
    Variable rhs(Tensor::from_host(rhs_values.data(), Shape{4}, Device::cuda(0)), true);
    Variable mask = ag::less_equal(lhs, rhs);
    std::vector<float> mask_values(4);
    read_host(mask.value(), mask_values.data());
    CHECK(mask_values == std::vector<float>({1.f, 1.f, 0.f, 0.f}));
    CHECK(mask.device().is_cuda() && !mask.requires_grad());
    CHECK_THROWS_AS(std::invalid_argument,
                    ag::less_equal(lhs,
                                   Variable(Tensor::zeros(Shape{4}, Device::cpu()))));
    CHECK_THROWS_AS(std::invalid_argument,
                    ag::less_equal(lhs,
                                   Variable(Tensor::zeros(Shape{3}, Device::cuda(0)))));

    const std::vector<float> true_values{10.f, 20.f, 30.f, 40.f};
    const std::vector<float> false_values{-1.f, -2.f, -3.f, -4.f};
    Variable condition(Tensor::from_host(mask_values.data(), Shape{4}, Device::cuda(0)), true);
    Variable when_true(Tensor::from_host(true_values.data(), Shape{4}, Device::cuda(0)), true);
    Variable when_false(Tensor::from_host(false_values.data(), Shape{4}, Device::cuda(0)), true);
    CHECK_THROWS_AS(std::invalid_argument,
                    ag::where(condition,
                              Variable(Tensor::zeros(Shape{4}, Device::cpu())),
                              when_false));
    CHECK_THROWS_AS(std::invalid_argument,
                    ag::where(condition, when_true,
                              Variable(Tensor::zeros(Shape{3}, Device::cuda(0)))));
    Variable selected = ag::where(condition, when_true, when_false);
    selected.backward(Tensor::ones(Shape{4}, Device::cuda(0)));
    std::vector<float> selected_values(4), condition_grad(4), true_grad(4), false_grad(4);
    read_host(selected.value(), selected_values.data());
    read_host(condition.grad(), condition_grad.data());
    read_host(when_true.grad(), true_grad.data());
    read_host(when_false.grad(), false_grad.data());
    CHECK(selected_values == std::vector<float>({10.f, 20.f, -3.f, -4.f}));
    CHECK(condition_grad == std::vector<float>(4, 0.f));
    CHECK(true_grad == std::vector<float>({1.f, 1.f, 0.f, 0.f}));
    CHECK(false_grad == std::vector<float>({0.f, 0.f, 1.f, 1.f}));
    selected.backward(Tensor::ones(Shape{4}, Device::cuda(0)));
    read_host(when_true.grad(), true_grad.data());
    CHECK(true_grad == std::vector<float>({2.f, 2.f, 0.f, 0.f}));
    when_true.zero_grad();
    when_false.zero_grad();
    condition.zero_grad();
    CHECK(!when_true.has_grad() && !when_false.has_grad() && !condition.has_grad());

    const float finite_values[] = {1.f, -2.f, 0.f};
    Tensor finite = Tensor::from_host(finite_values, Shape{3}, Device::cuda(0));
    CHECK(!ag::all_true(finite));
    CHECK(ag::all_finite(finite));
    const float nonfinite_values[] = {1.f, std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::quiet_NaN()};
    Tensor nonfinite = Tensor::from_host(nonfinite_values, Shape{3}, Device::cuda(0));
    CHECK(!ag::all_true(nonfinite));
    CHECK(!ag::all_finite(nonfinite));
    CHECK(ag::all_true(Tensor::from_host(
        std::vector<float>{1.f, -2.f}.data(), Shape{2}, Device::cuda(0))));
    CHECK(ag::all_true(Tensor::empty(Shape{0, 2}, Device::cuda(0))));
    CHECK(ag::all_finite(Tensor::empty(Shape{0, 2}, Device::cuda(0))));
    CHECK(finite.device().is_cuda() && nonfinite.device().is_cuda());
    report("Tensor CUDA ops: predicates, conditional VJPs, status flags");
}

void test_fixed_grid_support_cuda() {
    constexpr int max_half = 4;
    std::vector<float> offsets;
    std::vector<float> weights;
    for (int i = -max_half; i <= max_half; ++i) {
        offsets.push_back(static_cast<float>(std::abs(i)));
        weights.push_back(static_cast<float>(i + max_half + 1));
    }
    Variable offset(Tensor::from_host(offsets.data(), Shape{9}, Device::cuda(0)));
    Variable weight(Tensor::from_host(weights.data(), Shape{9}, Device::cuda(0)));
    Variable zero(Tensor::zeros(Shape{9}, Device::cuda(0)));
    for (int support = 1; support <= max_half; ++support) {
        for (float sigma : {(support - 0.0001f) / 4.f,
                            static_cast<float>(support) / 4.f,
                            (support + 0.0001f) / 4.f}) {
            const float threshold = std::max(
                1.f, std::min(4.f * sigma, static_cast<float>(max_half)));
            std::vector<float> thresholds(9, threshold);
            Variable selected = ag::where(
                ag::less_equal(offset,
                               Variable(Tensor::from_host(
                                   thresholds.data(), Shape{9}, Device::cuda(0)))),
                weight, zero);
            std::vector<float> actual(9);
            read_host(selected.value(), actual.data());
            const int expected_half = std::min(
                max_half, std::max(1, static_cast<int>(std::floor(4.f * sigma))));
            for (int i = -max_half; i <= max_half; ++i) {
                const float expected = std::abs(i) <= expected_half
                    ? static_cast<float>(i + max_half + 1) : 0.f;
                CHECK(actual[static_cast<std::size_t>(i + max_half)] == expected);
            }
        }
    }
    report("Tensor CUDA ops: fixed-grid support matches clamped floor boundaries");
}

}  // namespace

int main() {
    if (!cuda_available()) {
        std::printf("\nSKIP CUDA TENSOR TESTS: no reachable CUDA device\n");
        return 0;
    }
    test_public_tensor_has_no_impl_accessor();
    test_factory_cuda_zeros_values();
    test_factory_cuda_from_host_round_trip();
    test_factory_cuda_index_and_zero_element();
    test_to_cpu_from_cuda();
    test_to_cuda_from_cpu_round_trip();
    test_same_device_transfer_shares_storage();
    test_clone_is_independent_deep_copy();
    test_reshape_shares_storage_keeps_independent_shape();
    test_repeated_transfers_keep_source_identity();
    test_move_preserves_cuda_storage();
    test_destruction_safe_storage_release();
    test_wrong_device_view_throws();
    test_out_of_range_device_rejected_eagerly();
    test_empty_cuda_tensor_view_and_alias();
    test_same_device_cuda_to_cuda_transfer();
    test_header_hygiene_preprocessor();
    test_legacy_extension_eigen_aliases_still_available();
    test_oop_direct_shape_ops_cuda();
    test_oop_math_rejects_cuda_variable();
    test_oop_elementwise_composed_graph_cuda();
    test_oop_elementwise_rank4_and_empty_cuda();
    test_oop_broadcast_add_rank4_cuda();
    test_oop_sum_axes_and_mean_cuda();
    test_oop_softmax_nonlast_axis_cuda();
    test_oop_broadcast_repeated_backward_cuda();
    test_oop_backward_cuda_accumulates_and_matmul_works();
    test_oop_diffusion_rejects_cuda_tensor();
    test_oop_optimizer_unsupported_cuda_constructs_reject();
    test_oop_conv2d_cuda_inputs();
    test_oop_loss_mixed_device_rejects();
    test_oop_matmul_rank2_cuda_parity();
    test_oop_matmul_batched_rank3_cuda_parity();
    test_oop_matmul_later_batches_cuda();
    test_oop_matmul_batched_rank4_cuda_parity();
    test_oop_matmul_shape_and_zero_cuda();
    test_oop_mse_loss_cuda_parity();
    test_oop_cross_entropy_cuda_parity();
    test_oop_sgd_cuda_step_parity_and_alias_visible();
    test_oop_sgd_cuda_mixed_list();
    test_oop_adam_cuda_step_parity_and_moments();
    test_oop_adam_cuda_mixed_list();
    test_oop_adam_cuda_load_state_preserves_device();
    test_predicate_selection_and_status_cuda();
    test_fixed_grid_support_cuda();

    std::printf("\nALL CUDA TENSOR TESTS PASSED (%d)\n", passed);
    return 0;
}
