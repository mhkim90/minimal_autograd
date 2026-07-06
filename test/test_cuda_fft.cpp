#include "autograd.h"
#include "autograd/cuda_core.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace ag;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while (0)

static void check_mat_near(const Mat& a, const Mat& b, float tol, const char* what) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        std::cerr << "FAIL: " << what << " shape mismatch\n";
        std::exit(1);
    }
    const float diff = (a - b).cwiseAbs().maxCoeff();
    if (diff > tol) {
        std::cerr << "FAIL: " << what << " max diff " << diff << "\n";
        std::exit(1);
    }
}

static void test_cuda_complex_ops() {
    Mat ar(2, 2);
    ar << 0.2f, -0.4f,
          0.6f, -0.8f;
    Mat ai(2, 2);
    ai << -0.1f, 0.3f,
          -0.5f, 0.7f;
    Mat br(2, 2);
    br << 1.0f, -0.5f,
          0.25f, 0.75f;
    Mat bi(2, 2);
    bi << -0.2f, 0.4f,
          0.6f, -0.8f;

    auto cpu_ar = Var::make(ar);
    auto cpu_ai = Var::make(ai);
    auto cpu_br = Var::make(br);
    auto cpu_bi = Var::make(bi);
    auto cpu_z = make_complex(cpu_ar, cpu_ai);
    auto cpu_w = make_complex(cpu_br, cpu_bi);
    auto cpu_y = sum(abs2(complex_mul(cpu_z, cpu_w)));
    cpu_y->backward();

    auto cuda_ar = Var::make(ar)->cuda();
    auto cuda_ai = Var::make(ai)->cuda();
    auto cuda_br = Var::make(br)->cuda();
    auto cuda_bi = Var::make(bi)->cuda();
    auto cuda_z = make_complex(cuda_ar, cuda_ai);
    auto cuda_w = make_complex(cuda_br, cuda_bi);
    auto cuda_prod = complex_mul(cuda_z, cuda_w);
    CHECK(cuda_prod.real->is_cuda());
    CHECK(cuda_prod.imag->is_cuda());

    auto cuda_conj = conj(cuda_z);
    CHECK(cuda_conj.real->is_cuda());
    CHECK(cuda_conj.imag->is_cuda());

    auto cuda_scaled = complex_scale(cuda_z, 0.5f);
    CHECK(cuda_scaled.real->is_cuda());
    CHECK(cuda_scaled.imag->is_cuda());

    auto cuda_y = sum(abs2(cuda_prod));
    CHECK(cuda_y->is_cuda());
    cuda_y->backward();

    check_mat_near(cuda_prod.real->cpu()->data, complex_mul(cpu_z, cpu_w).real->data,
                   1e-4f, "complex_mul real");
    check_mat_near(cuda_prod.imag->cpu()->data, complex_mul(cpu_z, cpu_w).imag->data,
                   1e-4f, "complex_mul imag");
    check_mat_near(cuda_conj.real->cpu()->data, conj(cpu_z).real->data,
                   1e-4f, "conj real");
    check_mat_near(cuda_conj.imag->cpu()->data, conj(cpu_z).imag->data,
                   1e-4f, "conj imag");
    check_mat_near(cuda_scaled.real->cpu()->data, complex_scale(cpu_z, 0.5f).real->data,
                   1e-4f, "complex_scale real");
    check_mat_near(cuda_scaled.imag->cpu()->data, complex_scale(cpu_z, 0.5f).imag->data,
                   1e-4f, "complex_scale imag");
    check_mat_near(cuda_y->cpu()->data, cpu_y->data, 1e-4f, "abs2 loss");
    check_mat_near(cuda_ar->cpu()->grad, cpu_ar->grad, 1e-4f, "grad real");
    check_mat_near(cuda_ai->cpu()->grad, cpu_ai->grad, 1e-4f, "grad imag");

    std::cout << "[ok] CUDA complex ops match CPU forward/backward\n";
}

static void test_cuda_complex_validation() {
    bool mixed_threw = false;
    try {
        (void)make_complex(Var::make(Mat::Zero(1, 1))->cuda(),
                           Var::make(Mat::Zero(1, 1)));
    } catch (const std::runtime_error&) {
        mixed_threw = true;
    }
    CHECK(mixed_threw);

    bool shape_threw = false;
    try {
        (void)make_complex(Var::make(Mat::Zero(1, 2))->cuda(),
                           Var::make(Mat::Zero(2, 1))->cuda());
    } catch (const std::runtime_error&) {
        shape_threw = true;
    }
    CHECK(shape_threw);

    std::cout << "[ok] CUDA complex validation paths\n";
}

static void check_complex_near(const ComplexVar& cuda_z,
                               const ComplexVar& cpu_z,
                               float tol,
                               const char* what) {
    check_mat_near(cuda_z.real->cpu()->data, cpu_z.real->data, tol, what);
    check_mat_near(cuda_z.imag->cpu()->data, cpu_z.imag->data, tol, what);
}

static void test_cuda_fft_forward() {
    Mat zr(8, 8);
    Mat zi(8, 8);
    for (int r = 0; r < zr.rows(); ++r) {
        for (int c = 0; c < zr.cols(); ++c) {
            zr(r, c) = 0.1f * static_cast<float>(r) - 0.03f * static_cast<float>(c);
            zi(r, c) = -0.05f * static_cast<float>(r) + 0.07f * static_cast<float>(c);
        }
    }

    auto cpu_z = make_complex(Var::make(zr), Var::make(zi));
    auto cuda_z = make_complex(Var::make(zr)->cuda(), Var::make(zi)->cuda());

    auto cpu_spectrum = fft2(cpu_z);
    auto cuda_spectrum = fft2(cuda_z);
    CHECK(cuda_spectrum.real->is_cuda());
    CHECK(cuda_spectrum.imag->is_cuda());
    check_complex_near(cuda_spectrum, cpu_spectrum, 2e-4f, "fft2 cuda/cpu parity");

    auto cpu_roundtrip = ifft2(cpu_spectrum);
    auto cuda_roundtrip = ifft2(cuda_spectrum);
    check_complex_near(cuda_roundtrip, cpu_roundtrip, 3e-4f, "ifft2 cuda/cpu parity");
    check_mat_near(cuda_roundtrip.real->cpu()->data, zr, 3e-4f, "fft roundtrip real");
    check_mat_near(cuda_roundtrip.imag->cpu()->data, zi, 3e-4f, "fft roundtrip imag");

    auto cuda_spectrum_again = fft2(cuda_z);
    check_complex_near(cuda_spectrum_again, cpu_spectrum, 2e-4f, "fft2 deterministic");

    std::cout << "[ok] CUDA fft2/ifft2 forward parity and round-trip\n";
}

static void test_cuda_fft_non_square_forward() {
    Mat zr(8, 16);
    for (int r = 0; r < zr.rows(); ++r) {
        for (int c = 0; c < zr.cols(); ++c) {
            zr(r, c) = 0.01f * static_cast<float>(r * zr.cols() + c);
        }
    }

    auto cpu_y = ifft2(fft2(real_to_complex(Var::make(zr))));
    auto cuda_y = ifft2(fft2(real_to_complex(Var::make(zr)->cuda())));
    check_complex_near(cuda_y, cpu_y, 4e-4f, "non-square cuda fft roundtrip");
    check_mat_near(cuda_y.real->cpu()->data, zr, 4e-4f, "non-square roundtrip real");

    std::cout << "[ok] CUDA fft2 non-square power-of-two round-trip\n";
}

static void test_cuda_fft_rejections() {
    bool shape_threw = false;
    try {
        (void)fft2(real_to_complex(Var::make(Mat::Zero(7, 8))->cuda()));
    } catch (const std::runtime_error&) {
        shape_threw = true;
    }
    CHECK(shape_threw);

    std::cout << "[ok] CUDA fft2 unsupported-shape rejection path\n";
}

static void test_cuda_fft_backward_parity() {
    Mat zr(8, 8);
    Mat zi(8, 8);
    for (int r = 0; r < zr.rows(); ++r) {
        for (int c = 0; c < zr.cols(); ++c) {
            zr(r, c) = 0.04f * static_cast<float>(r) - 0.02f * static_cast<float>(c);
            zi(r, c) = -0.03f * static_cast<float>(r) + 0.05f * static_cast<float>(c);
        }
    }

    {
        auto cpu_r = Var::make(zr);
        auto cpu_i = Var::make(zi);
        auto cpu_loss = sum(abs2(fft2(make_complex(cpu_r, cpu_i))));
        cpu_loss->backward();

        auto cuda_r = Var::make(zr)->cuda();
        auto cuda_i = Var::make(zi)->cuda();
        auto cuda_loss = sum(abs2(fft2(make_complex(cuda_r, cuda_i))));
        cuda_loss->backward();

        check_mat_near(cuda_r->cpu()->grad, cpu_r->grad, 1e-2f, "fft2 grad real");
        check_mat_near(cuda_i->cpu()->grad, cpu_i->grad, 1e-2f, "fft2 grad imag");
    }

    {
        auto cpu_r = Var::make(zr);
        auto cpu_i = Var::make(zi);
        auto cpu_loss = sum(abs2(ifft2(make_complex(cpu_r, cpu_i))));
        cpu_loss->backward();

        auto cuda_r = Var::make(zr)->cuda();
        auto cuda_i = Var::make(zi)->cuda();
        auto cuda_loss = sum(abs2(ifft2(make_complex(cuda_r, cuda_i))));
        cuda_loss->backward();

        check_mat_near(cuda_r->cpu()->grad, cpu_r->grad, 1e-4f, "ifft2 grad real");
        check_mat_near(cuda_i->cpu()->grad, cpu_i->grad, 1e-4f, "ifft2 grad imag");
    }

    std::cout << "[ok] CUDA fft2/ifft2 backward parity\n";
}

static void test_cuda_fft_spectral_filter_backward() {
    Mat input(8, 8);
    Mat fr(8, 8);
    Mat fi(8, 8);
    for (int r = 0; r < input.rows(); ++r) {
        for (int c = 0; c < input.cols(); ++c) {
            input(r, c) = 0.02f * static_cast<float>(r * input.cols() + c) - 0.5f;
            fr(r, c) = (r + c) % 3 == 0 ? 1.f : 0.5f;
            fi(r, c) = 0.01f * static_cast<float>(r - c);
        }
    }

    auto cpu_input = Var::make(input);
    auto cpu_filter = make_complex(Var::make(fr), Var::make(fi));
    auto cpu_field = ifft2(complex_mul(cpu_filter, fft2(real_to_complex(cpu_input))));
    auto cpu_loss = sum(abs2(cpu_field));
    cpu_loss->backward();

    auto cuda_input = Var::make(input)->cuda();
    auto cuda_filter = make_complex(Var::make(fr)->cuda(), Var::make(fi)->cuda());
    auto cuda_field = ifft2(complex_mul(cuda_filter, fft2(real_to_complex(cuda_input))));
    auto cuda_loss = sum(abs2(cuda_field));
    cuda_loss->backward();

    check_mat_near(cuda_input->cpu()->grad, cpu_input->grad, 2e-3f,
                   "spectral filter input grad");
    std::cout << "[ok] CUDA spectral filter backward parity\n";
}

static void test_cuda_fft_repeated_branch_backward() {
    Mat zr = Mat::Constant(8, 8, 0.125f);
    auto cpu_x = Var::make(zr);
    auto cpu_y = fft2(real_to_complex(cpu_x));
    sum(add(abs2(cpu_y), abs2(cpu_y)))->backward();

    auto cuda_x = Var::make(zr)->cuda();
    auto cuda_y = fft2(real_to_complex(cuda_x));
    sum(add(abs2(cuda_y), abs2(cuda_y)))->backward();

    check_mat_near(cuda_x->cpu()->grad, cpu_x->grad, 1e-2f, "repeated fft branch grad");
    std::cout << "[ok] CUDA FFT repeated-branch backward parity\n";
}

int main() {
    CudaRuntimeInfo cuda_info;
    try {
        cuda_info = cuda_runtime_info();
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA runtime probe failed: " << e.what() << "\n";
        return 1;
    }
    std::cout << cuda_runtime_summary(cuda_info) << "\n";
    if (!cuda_info.has_device()) {
        std::cout << "SKIP CUDA FFT TESTS: " << cuda_info.status << "\n";
        return 0;
    }

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
