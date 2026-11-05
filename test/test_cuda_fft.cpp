#include "autograd/core/fft.h"
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
} while (0)

static void check_mat_near(const Tensor& a, const Tensor& b, float tol, const char* what) {
    if (a.shape() != b.shape()) {
        std::cerr << "FAIL: " << what << " shape mismatch\n";
        std::exit(1);
    }
    const float diff = max_diff(a, b);
    if (!std::isfinite(diff) || diff > tol) {
        std::cerr << "FAIL: " << what << " max diff " << diff << "\n";
        std::exit(1);
    }
}

static void test_cuda_complex_ops() {
    Tensor ar = tensor(Shape{2, 2}, {0.2f, -0.4f,
          0.6f, -0.8f});
    Tensor ai = tensor(Shape{2, 2}, {-0.1f, 0.3f,
          -0.5f, 0.7f});
    Tensor br = tensor(Shape{2, 2}, {1.0f, -0.5f,
          0.25f, 0.75f});
    Tensor bi = tensor(Shape{2, 2}, {-0.2f, 0.4f,
          0.6f, -0.8f});

    auto cpu_ar = Variable(ar, true);
    auto cpu_ai = Variable(ai, true);
    auto cpu_br = Variable(br, true);
    auto cpu_bi = Variable(bi, true);
    auto cpu_z = make_complex(cpu_ar, cpu_ai);
    auto cpu_w = make_complex(cpu_br, cpu_bi);
    auto cpu_y = sum(abs2(complex_mul(cpu_z, cpu_w)));
    cpu_y.backward();

    auto cuda_ar = Variable(ar, true).to(Device::cuda());
    auto cuda_ai = Variable(ai, true).to(Device::cuda());
    auto cuda_br = Variable(br, true).to(Device::cuda());
    auto cuda_bi = Variable(bi, true).to(Device::cuda());
    auto cuda_z = make_complex(cuda_ar, cuda_ai);
    auto cuda_w = make_complex(cuda_br, cuda_bi);
    auto cuda_prod = complex_mul(cuda_z, cuda_w);
    CHECK(cuda_prod.real.device().is_cuda());
    CHECK(cuda_prod.imag.device().is_cuda());

    auto cuda_conj = conj(cuda_z);
    CHECK(cuda_conj.real.device().is_cuda());
    CHECK(cuda_conj.imag.device().is_cuda());

    auto cuda_scaled = complex_scale(cuda_z, 0.5f);
    CHECK(cuda_scaled.real.device().is_cuda());
    CHECK(cuda_scaled.imag.device().is_cuda());

    auto cuda_y = sum(abs2(cuda_prod));
    CHECK(cuda_y.device().is_cuda());
    cuda_y.backward();

    check_mat_near(cuda_prod.real.value(), complex_mul(cpu_z, cpu_w).real.value(),
                   1e-4f, "complex_mul real");
    check_mat_near(cuda_prod.imag.value(), complex_mul(cpu_z, cpu_w).imag.value(),
                   1e-4f, "complex_mul imag");
    check_mat_near(cuda_conj.real.value(), conj(cpu_z).real.value(),
                   1e-4f, "conj real");
    check_mat_near(cuda_conj.imag.value(), conj(cpu_z).imag.value(),
                   1e-4f, "conj imag");
    check_mat_near(cuda_scaled.real.value(), complex_scale(cpu_z, 0.5f).real.value(),
                   1e-4f, "complex_scale real");
    check_mat_near(cuda_scaled.imag.value(), complex_scale(cpu_z, 0.5f).imag.value(),
                   1e-4f, "complex_scale imag");
    check_mat_near(cuda_y.value(), cpu_y.value(), 1e-4f, "abs2 loss");
    check_mat_near(cuda_ar.grad(), cpu_ar.grad(), 1e-4f, "grad real");
    check_mat_near(cuda_ai.grad(), cpu_ai.grad(), 1e-4f, "grad imag");

    std::cout << "[ok] CUDA complex ops match CPU forward/backward\n";
}

static void test_cuda_complex_validation() {
    bool mixed_threw = false;
    try {
        (void)make_complex(Variable(Tensor::zeros(Shape{1, 1}), true).to(Device::cuda()),
                           Variable(Tensor::zeros(Shape{1, 1}), true));
    } catch (const std::invalid_argument&) {
        mixed_threw = true;
    }
    CHECK(mixed_threw);

    bool shape_threw = false;
    try {
        (void)make_complex(Variable(Tensor::zeros(Shape{1, 2}), true).to(Device::cuda()),
                           Variable(Tensor::zeros(Shape{2, 1}), true).to(Device::cuda()));
    } catch (const std::invalid_argument&) {
        shape_threw = true;
    }
    CHECK(shape_threw);

    std::cout << "[ok] CUDA complex validation paths\n";
}

static void check_complex_near(const ComplexVariable& cuda_z,
                               const ComplexVariable& cpu_z,
                               float tol,
                               const char* what) {
    check_mat_near(cuda_z.real.value(), cpu_z.real.value(), tol, what);
    check_mat_near(cuda_z.imag.value(), cpu_z.imag.value(), tol, what);
}

static void test_cuda_fft_forward() {
    Tensor zr = Tensor::zeros(Shape{8, 8});
    Tensor zi = Tensor::zeros(Shape{8, 8});
    for (int r = 0; r < zr.shape()[0]; ++r) {
        for (int c = 0; c < zr.shape()[1]; ++c) {
            put(zr, r*zr.shape()[1]+c, 0.1f * static_cast<float>(r) - 0.03f * static_cast<float>(c));
            put(zi, r*zi.shape()[1]+c, -0.05f * static_cast<float>(r) + 0.07f * static_cast<float>(c));
        }
    }

    auto cpu_z = make_complex(Variable(zr, true), Variable(zi, true));
    auto cuda_z = make_complex(Variable(zr, true).to(Device::cuda()), Variable(zi, true).to(Device::cuda()));

    auto cpu_spectrum = fft2(cpu_z);
    auto cuda_spectrum = fft2(cuda_z);
    CHECK(cuda_spectrum.real.device().is_cuda());
    CHECK(cuda_spectrum.imag.device().is_cuda());
    check_complex_near(cuda_spectrum, cpu_spectrum, 2e-4f, "fft2 cuda/cpu parity");

    auto cpu_roundtrip = ifft2(cpu_spectrum);
    auto cuda_roundtrip = ifft2(cuda_spectrum);
    check_complex_near(cuda_roundtrip, cpu_roundtrip, 3e-4f, "ifft2 cuda/cpu parity");
    check_mat_near(cuda_roundtrip.real.value(), zr, 3e-4f, "fft roundtrip real");
    check_mat_near(cuda_roundtrip.imag.value(), zi, 3e-4f, "fft roundtrip imag");

    auto cuda_spectrum_again = fft2(cuda_z);
    check_complex_near(cuda_spectrum_again, cpu_spectrum, 2e-4f, "fft2 deterministic");

    std::cout << "[ok] CUDA fft2/ifft2 forward parity and round-trip\n";
}

static void test_cuda_fft_non_square_forward() {
    Tensor zr = Tensor::zeros(Shape{8, 16});
    for (int r = 0; r < zr.shape()[0]; ++r) {
        for (int c = 0; c < zr.shape()[1]; ++c) {
            put(zr, r*zr.shape()[1]+c, 0.01f * static_cast<float>(r * zr.shape()[1] + c));
        }
    }

    auto cpu_y = ifft2(fft2(real_to_complex(Variable(zr, true))));
    auto cuda_y = ifft2(fft2(real_to_complex(Variable(zr, true).to(Device::cuda()))));
    check_complex_near(cuda_y, cpu_y, 4e-4f, "non-square cuda fft roundtrip");
    check_mat_near(cuda_y.real.value(), zr, 4e-4f, "non-square roundtrip real");

    std::cout << "[ok] CUDA fft2 non-square power-of-two round-trip\n";
}

static void test_cuda_fft_rejections() {
    auto non_power_two = real_to_complex(Variable(Tensor::ones(Shape{7,8}, Device::cuda()), true));
    auto roundtrip = ifft2(fft2(non_power_two));
    check_mat_near(roundtrip.real.value(), non_power_two.real.value(), 4e-4f, "non-power-of-two supported");
    for (const auto& shape : {Shape{8}, Shape{0,8}}) {
        bool threw = false;
        try { (void)fft2(real_to_complex(Variable(Tensor::zeros(shape, Device::cuda()), true))); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    bool norm_threw = false;
    try { (void)fft2(non_power_two, static_cast<FftNorm>(42)); }
    catch (const std::runtime_error&) { norm_threw = true; }
    CHECK(norm_threw);
    std::cout << "[ok] CUDA FFT current shape/norm checks, arbitrary positive extents supported\n";
}

static void test_cuda_fft_backward_parity() {
    Tensor zr = Tensor::zeros(Shape{8, 8});
    Tensor zi = Tensor::zeros(Shape{8, 8});
    for (int r = 0; r < zr.shape()[0]; ++r) {
        for (int c = 0; c < zr.shape()[1]; ++c) {
            put(zr, r*zr.shape()[1]+c, 0.04f * static_cast<float>(r) - 0.02f * static_cast<float>(c));
            put(zi, r*zi.shape()[1]+c, -0.03f * static_cast<float>(r) + 0.05f * static_cast<float>(c));
        }
    }

    {
        auto cpu_r = Variable(zr, true);
        auto cpu_i = Variable(zi, true);
        auto cpu_loss = sum(abs2(fft2(make_complex(cpu_r, cpu_i))));
        cpu_loss.backward();

        auto cuda_r = Variable(zr, true).to(Device::cuda());
        auto cuda_i = Variable(zi, true).to(Device::cuda());
        auto cuda_loss = sum(abs2(fft2(make_complex(cuda_r, cuda_i))));
        cuda_loss.backward();

        check_mat_near(cuda_r.grad(), cpu_r.grad(), 1e-2f, "fft2 grad real");
        check_mat_near(cuda_i.grad(), cpu_i.grad(), 1e-2f, "fft2 grad imag");
    }

    {
        auto cpu_r = Variable(zr, true);
        auto cpu_i = Variable(zi, true);
        auto cpu_loss = sum(abs2(ifft2(make_complex(cpu_r, cpu_i))));
        cpu_loss.backward();

        auto cuda_r = Variable(zr, true).to(Device::cuda());
        auto cuda_i = Variable(zi, true).to(Device::cuda());
        auto cuda_loss = sum(abs2(ifft2(make_complex(cuda_r, cuda_i))));
        cuda_loss.backward();

        check_mat_near(cuda_r.grad(), cpu_r.grad(), 1e-4f, "ifft2 grad real");
        check_mat_near(cuda_i.grad(), cpu_i.grad(), 1e-4f, "ifft2 grad imag");
    }

    std::cout << "[ok] CUDA fft2/ifft2 backward parity\n";
}

static void test_cuda_fft_spectral_filter_backward() {
    Tensor input = Tensor::zeros(Shape{8, 8});
    Tensor fr = Tensor::zeros(Shape{8, 8});
    Tensor fi = Tensor::zeros(Shape{8, 8});
    for (int r = 0; r < input.shape()[0]; ++r) {
        for (int c = 0; c < input.shape()[1]; ++c) {
            put(input, r*input.shape()[1]+c, 0.02f * static_cast<float>(r * input.shape()[1] + c) - 0.5f);
            put(fr, r*fr.shape()[1]+c, (r + c) % 3 == 0 ? 1.f : 0.5f);
            put(fi, r*fi.shape()[1]+c, 0.01f * static_cast<float>(r - c));
        }
    }

    auto cpu_input = Variable(input, true);
    auto cpu_filter = make_complex(Variable(fr, true), Variable(fi, true));
    auto cpu_field = ifft2(complex_mul(cpu_filter, fft2(real_to_complex(cpu_input))));
    auto cpu_loss = sum(abs2(cpu_field));
    cpu_loss.backward();

    auto cuda_input = Variable(input, true).to(Device::cuda());
    auto cuda_filter = make_complex(Variable(fr, true).to(Device::cuda()), Variable(fi, true).to(Device::cuda()));
    auto cuda_field = ifft2(complex_mul(cuda_filter, fft2(real_to_complex(cuda_input))));
    auto cuda_loss = sum(abs2(cuda_field));
    cuda_loss.backward();

    check_mat_near(cuda_input.grad(), cpu_input.grad(), 2e-3f,
                   "spectral filter input grad");
    std::cout << "[ok] CUDA spectral filter backward parity\n";
}

static void test_cuda_fft_repeated_branch_backward() {
    Tensor zr = filled(Shape{8, 8}, 0.125f);
    auto cpu_x = Variable(zr, true);
    auto cpu_y = fft2(real_to_complex(cpu_x));
    sum(add(abs2(cpu_y), abs2(cpu_y))).backward();

    auto cuda_x = Variable(zr, true).to(Device::cuda());
    auto cuda_y = fft2(real_to_complex(cuda_x));
    sum(add(abs2(cuda_y), abs2(cuda_y))).backward();

    check_mat_near(cuda_x.grad(), cpu_x.grad(), 1e-2f, "repeated fft branch grad");
    std::cout << "[ok] CUDA FFT repeated-branch backward parity\n";
}

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

    test_cuda_complex_ops();
    test_cuda_complex_validation();
    test_cuda_fft_forward();
    test_cuda_fft_non_square_forward();
    test_cuda_fft_rejections();
    test_cuda_fft_backward_parity();
    test_cuda_fft_spectral_filter_backward();
    test_cuda_fft_repeated_branch_backward();
    std::cout << "ALL CUDA FFT TESTS PASSED\n";
    return 0;
}
