#include "autograd.h"

#ifdef AUTOGRAD_USE_CUDA
#include "autograd/cuda_core.h"
#endif

#include <cmath>
#include <cstdio>

int main() {
    ag::Mat input_data(1, 2);
    input_data << 2.f, 3.f;
    ag::VarPtr input = ag::Var::make(input_data);

#ifdef AUTOGRAD_USE_CUDA
    const ag::CudaRuntimeInfo runtime = ag::cuda_runtime_info();
    if (!runtime.has_device()) {
        std::printf("SKIP: no CUDA device\n");
        return 0;
    }
    input = input->cuda();
#endif

    ag::VarPtr output = ag::sum(ag::scale(input, 2.f));
    output->backward();

    const ag::VarPtr host_output = output->cpu();
    const ag::VarPtr host_input = input->cpu();
    if (std::fabs(host_output->data(0, 0) - 10.f) > 1e-5f ||
        host_input->grad.rows() != 1 || host_input->grad.cols() != 2 ||
        std::fabs(host_input->grad(0, 0) - 2.f) > 1e-5f ||
        std::fabs(host_input->grad(0, 1) - 2.f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: legacy Mat/VarPtr forward/backward\n");
        return 1;
    }

    std::printf("OK: legacy Mat/VarPtr expert boundary\n");
    return 0;
}
