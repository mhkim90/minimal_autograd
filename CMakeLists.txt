cmake_minimum_required(VERSION 3.14)
project(autograd CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_library(autograd STATIC
    src/variable.cpp
    src/ops.cpp
    src/module.cpp
    src/loss.cpp
    src/optim.cpp
    src/conv.cpp
    src/norm.cpp
)
target_include_directories(autograd PUBLIC include)
target_link_libraries(autograd PUBLIC Eigen3::Eigen)

option(AUTOGRAD_USE_CUDA "Build minimal CUDA autograd core" OFF)
if(AUTOGRAD_USE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR "AUTOGRAD_USE_CUDA requires CMake 3.18+")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(autograd PUBLIC AUTOGRAD_USE_CUDA)
    target_sources(autograd PRIVATE src/cuda_core.cu)
    target_link_libraries(autograd PUBLIC CUDA::cudart)
    set_target_properties(autograd PROPERTIES CUDA_STANDARD 17)
endif()

# Optional: multi-threaded Eigen.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(autograd PUBLIC OpenMP::OpenMP_CXX)
endif()

# Tests
add_executable(test_core test/test_core.cpp)
target_link_libraries(test_core autograd)
target_include_directories(test_core PRIVATE test)

add_executable(test_nn test/test_nn.cpp)
target_link_libraries(test_nn autograd)

add_executable(test_conv test/test_conv.cpp)
target_link_libraries(test_conv autograd)
target_include_directories(test_conv PRIVATE test)

if(AUTOGRAD_USE_CUDA)
    add_executable(test_cuda_core test/test_cuda_core.cpp)
    target_link_libraries(test_cuda_core autograd)
    target_include_directories(test_cuda_core PRIVATE test)
endif()

option(AUTOGRAD_BUILD_ADVANCED_OPS "Build extended/advanced ops and tests" OFF)
if(AUTOGRAD_BUILD_ADVANCED_OPS)
    add_executable(test_extensions test/test_extensions.cpp)
    target_link_libraries(test_extensions autograd)
    target_include_directories(test_extensions PRIVATE test)

    add_executable(test_diffusion test/test_diffusion.cpp)
    target_link_libraries(test_diffusion autograd)
    target_include_directories(test_diffusion PRIVATE test)

    add_executable(test_smoke test/test_smoke.cpp)
    target_link_libraries(test_smoke autograd)
    target_include_directories(test_smoke PRIVATE test)
endif()

# Optional: build the README example.
option(AUTOGRAD_BUILD_EXAMPLES "Build README examples" OFF)
if(AUTOGRAD_BUILD_EXAMPLES)
    add_executable(linear_regression examples/linear_regression.cpp)
    target_link_libraries(linear_regression autograd)
endif()

# README quick-start example.
add_executable(example examples/linear_regression.cpp)
target_link_libraries(example autograd)
