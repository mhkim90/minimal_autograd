cmake_minimum_required(VERSION 3.14)

# Project version is intentionally a placeholder; the architecture refactor
# reserves semantic-version assignment for Phase 11. Consumers use
# `find_package(autograd)` unversioned until that phase.
project(autograd LANGUAGES CXX)

# C++17 is enforced per-target via target_compile_features below so we do not
# mutate the parent's global CMAKE_CXX_STANDARD when this project is added
# via add_subdirectory(...). When this IS the top-level CMake project we
# additionally pin the global standard so other targets compiled here pick
# up C++17 without per-target repetition.
set(_AUTOGRAD_CXX_STD 17)

# Top-level vs embedded detection. PROJECT_IS_TOP_LEVEL is CMake 3.21+, so we
# stay on the 3.14-compatible CMAKE_SOURCE_DIR / PROJECT_SOURCE_DIR comparison.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_AUTOGRAD_IS_TOP_LEVEL TRUE)
    set(CMAKE_CXX_STANDARD ${_AUTOGRAD_CXX_STD})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
    set(_AUTOGRAD_IS_TOP_LEVEL FALSE)
endif()

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
# Tests default to ON only when this project is the top-level CMake project.
# All tests/examples are OFF by default when consumed via add_subdirectory(...)
# so embedding projects do not inherit them and may opt in explicitly.
option(AUTOGRAD_BUILD_TESTS "Build the in-tree test suite" ${_AUTOGRAD_IS_TOP_LEVEL})
option(AUTOGRAD_BUILD_ADVANCED_OPS "Build advanced/extended ops and tests" OFF)
option(AUTOGRAD_BUILD_EXAMPLES "Build README/example executables" OFF)
option(AUTOGRAD_USE_CUDA "Build the optional CUDA backend" OFF)

# ----------------------------------------------------------------------------
# Required and optional dependencies
# ----------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
set(_AUTOGRAD_HAS_OPENMP FALSE)
if(OpenMP_CXX_FOUND)
    set(_AUTOGRAD_HAS_OPENMP TRUE)
endif()

if(AUTOGRAD_USE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR "AUTOGRAD_USE_CUDA requires CMake 3.18+")
    endif()
    set(AUTOGRAD_CUDA_ARCHITECTURES "" CACHE STRING
        "CUDA architectures for autograd; defaults to CMAKE_CUDA_ARCHITECTURES or 75")
    if(NOT AUTOGRAD_CUDA_ARCHITECTURES)
        if(CMAKE_CUDA_ARCHITECTURES)
            set(AUTOGRAD_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        else()
            set(AUTOGRAD_CUDA_ARCHITECTURES "75")
        endif()
        set(AUTOGRAD_CUDA_ARCHITECTURES "${AUTOGRAD_CUDA_ARCHITECTURES}" CACHE STRING
            "CUDA architectures for autograd; defaults to CMAKE_CUDA_ARCHITECTURES or 75"
            FORCE)
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${AUTOGRAD_CUDA_ARCHITECTURES}")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ----------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------
set(_AUTOGRAD_LIB_SOURCES
    src/variable.cpp
    src/ops.cpp
    src/module.cpp
    src/loss.cpp
    src/optim.cpp
    src/conv.cpp
    src/norm.cpp
    src/fft.cpp
    src/cpu/tensor_ops_elementwise.cpp
    src/cpu/tensor_ops_shape.cpp
    src/cpu/tensor_ops_nn.cpp
    src/cpu/tensor_ops_spectral_optimizer.cpp
    src/core/tensor.cpp
    src/core/cuda_view.cpp
    src/core/variable.cpp
    src/core/ops.cpp
    src/core/loss.cpp
    src/core/module.cpp
    src/core/optim.cpp
    src/core/conv.cpp
    src/core/norm.cpp
    src/core/diffusion.cpp
    src/core/fft.cpp
)

add_library(autograd STATIC ${_AUTOGRAD_LIB_SOURCES})
add_library(autograd::autograd ALIAS autograd)

set_target_properties(autograd PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD ${_AUTOGRAD_CXX_STD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

target_compile_features(autograd PUBLIC cxx_std_${_AUTOGRAD_CXX_STD})

target_include_directories(autograd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(autograd PUBLIC Eigen3::Eigen)

if(_AUTOGRAD_HAS_OPENMP)
    target_link_libraries(autograd PUBLIC OpenMP::OpenMP_CXX)
endif()

if(AUTOGRAD_USE_CUDA)
    target_compile_definitions(autograd PUBLIC AUTOGRAD_USE_CUDA)
    target_sources(autograd PRIVATE
        src/core/tensor_dispatch_cuda_elementwise.cpp
        src/core/tensor_dispatch_cuda_shape.cpp
        src/core/tensor_dispatch_cuda_nn.cpp
        src/core/tensor_dispatch_cuda_spectral_optimizer.cpp
        src/cuda_core.cu
        src/cuda_fft.cu
        src/cuda/tensor_ops_elementwise.cu
        src/cuda/tensor_ops_shape.cu
        src/cuda/tensor_ops_nn.cu
        src/cuda/tensor_ops_spectral_optimizer.cu
        src/cuda/tensor_storage_runtime.cu)
    target_link_libraries(autograd PUBLIC CUDA::cudart)
    set_target_properties(autograd PROPERTIES
        CUDA_STANDARD ${_AUTOGRAD_CXX_STD}
        CUDA_ARCHITECTURES "${AUTOGRAD_CUDA_ARCHITECTURES}")
else()
    target_sources(autograd PRIVATE
        src/core/tensor_dispatch_cpu_elementwise.cpp
        src/core/tensor_dispatch_cpu_shape.cpp
        src/core/tensor_dispatch_cpu_nn.cpp
        src/core/tensor_dispatch_cpu_spectral_optimizer.cpp
        src/core/cuda_runtime_stubs.cpp)
endif()

# ----------------------------------------------------------------------------
# Install rules + exported CMake targets
# ----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS autograd
    EXPORT autogradTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT autogradTargets
    FILE autogradTargets.cmake
    NAMESPACE autograd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/autograd)

# Build the @autograd_extra_deps@ substitution. The value is either empty
# (no extra find_dependency blocks) or one or more multi-line `set(...)`
# invocations that turn on the corresponding find_dependency branch in the
# installed autogradConfig.cmake. The flag values are evaluated at install
# time so consumers see exactly what was built.
set(autograd_extra_deps "")
if(OpenMP_CXX_FOUND)
    set(autograd_extra_deps
        "set(_AUTOGRAD_HAS_OPENMP TRUE)")
endif()
if(AUTOGRAD_USE_CUDA)
    set(autograd_extra_deps
        "${autograd_extra_deps}\nset(_AUTOGRAD_HAS_CUDA TRUE)")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/autogradConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/autogradConfig.cmake"
    @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/autogradConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/autograd)

# ----------------------------------------------------------------------------
# Tests (default ON at top-level only; OFF when embedded)
# ----------------------------------------------------------------------------
if(AUTOGRAD_BUILD_TESTS)
    enable_testing()

    add_executable(test_core test/test_core.cpp)
    target_link_libraries(test_core PRIVATE autograd::autograd)
    target_include_directories(test_core PRIVATE test)
    add_test(NAME test_core COMMAND test_core)

    add_executable(test_shape_device test/test_shape_device.cpp)
    target_link_libraries(test_shape_device PRIVATE autograd::autograd)
    target_include_directories(test_shape_device PRIVATE test)
    add_test(NAME test_shape_device COMMAND test_shape_device)

    add_executable(test_tensor test/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE autograd::autograd)
    target_include_directories(test_tensor PRIVATE test)
    add_test(NAME test_tensor COMMAND test_tensor)

    add_executable(test_autograd_core test/test_autograd_core.cpp)
    target_link_libraries(test_autograd_core PRIVATE autograd::autograd)
    target_include_directories(test_autograd_core PRIVATE test)
    add_test(NAME test_autograd_core COMMAND test_autograd_core)

    add_executable(test_cpu_ops test/test_cpu_ops.cpp)
    target_link_libraries(test_cpu_ops PRIVATE autograd::autograd)
    target_include_directories(test_cpu_ops PRIVATE test)
    add_test(NAME test_cpu_ops COMMAND test_cpu_ops)

    add_executable(test_nn test/test_nn.cpp)
    target_link_libraries(test_nn PRIVATE autograd::autograd)
    add_test(NAME test_nn COMMAND test_nn)

    add_executable(test_oop_training test/test_oop_training.cpp)
    target_link_libraries(test_oop_training PRIVATE autograd::autograd)
    target_include_directories(test_oop_training PRIVATE test)
    add_test(NAME test_oop_training COMMAND test_oop_training)

    add_executable(test_oop_conv test/test_oop_conv.cpp)
    target_link_libraries(test_oop_conv PRIVATE autograd::autograd)
    target_include_directories(test_oop_conv PRIVATE test)
    add_test(NAME test_oop_conv COMMAND test_oop_conv)

    add_executable(test_oop_norm test/test_oop_norm.cpp)
    target_link_libraries(test_oop_norm PRIVATE autograd::autograd)
    target_include_directories(test_oop_norm PRIVATE test)
    add_test(NAME test_oop_norm COMMAND test_oop_norm)

    add_executable(test_oop_diffusion test/test_oop_diffusion.cpp)
    target_link_libraries(test_oop_diffusion PRIVATE autograd::autograd)
    target_include_directories(test_oop_diffusion PRIVATE test)
    add_test(NAME test_oop_diffusion COMMAND test_oop_diffusion)

    add_executable(test_oop_fft test/test_oop_fft.cpp)
    target_link_libraries(test_oop_fft PRIVATE autograd::autograd)
    target_include_directories(test_oop_fft PRIVATE test)
    add_test(NAME test_oop_fft COMMAND test_oop_fft)

    add_executable(test_conv test/test_conv.cpp)
    target_link_libraries(test_conv PRIVATE autograd::autograd)
    target_include_directories(test_conv PRIVATE test)
    add_test(NAME test_conv COMMAND test_conv)

    add_executable(test_fft test/test_fft.cpp)
    target_link_libraries(test_fft PRIVATE autograd::autograd)
    target_include_directories(test_fft PRIVATE test)
    add_test(NAME test_fft COMMAND test_fft)

    add_executable(test_characterization test/test_characterization.cpp)
    target_link_libraries(test_characterization PRIVATE autograd::autograd)
    target_include_directories(test_characterization PRIVATE test)
    add_test(NAME test_characterization COMMAND test_characterization)

    if(AUTOGRAD_USE_CUDA)
        add_executable(test_cuda_core test/test_cuda_core.cpp)
        target_link_libraries(test_cuda_core PRIVATE autograd::autograd)
        target_include_directories(test_cuda_core PRIVATE test)
        add_test(NAME test_cuda_core COMMAND test_cuda_core)

        add_executable(test_cuda_fft test/test_cuda_fft.cpp)
        target_link_libraries(test_cuda_fft PRIVATE autograd::autograd)
        target_include_directories(test_cuda_fft PRIVATE test)
        add_test(NAME test_cuda_fft COMMAND test_cuda_fft)

        add_executable(test_cuda_tensor test/test_cuda_tensor.cpp)
        target_link_libraries(test_cuda_tensor PRIVATE autograd::autograd)
        target_include_directories(test_cuda_tensor PRIVATE test)
        add_test(NAME test_cuda_tensor COMMAND test_cuda_tensor)
    endif()

    if(AUTOGRAD_BUILD_ADVANCED_OPS)
        add_executable(test_extensions test/test_extensions.cpp)
        target_link_libraries(test_extensions PRIVATE autograd::autograd)
        target_include_directories(test_extensions PRIVATE test)
        add_test(NAME test_extensions COMMAND test_extensions)

        add_executable(test_diffusion test/test_diffusion.cpp)
        target_link_libraries(test_diffusion PRIVATE autograd::autograd)
        target_include_directories(test_diffusion PRIVATE test)
        add_test(NAME test_diffusion COMMAND test_diffusion)

        add_executable(test_smoke test/test_smoke.cpp)
        target_link_libraries(test_smoke PRIVATE autograd::autograd)
        target_include_directories(test_smoke PRIVATE test)
        add_test(NAME test_smoke COMMAND test_smoke)
    endif()
endif()

# ----------------------------------------------------------------------------
# Examples (default OFF; consumer projects do not inherit them)
# ----------------------------------------------------------------------------
if(AUTOGRAD_BUILD_EXAMPLES)
    add_executable(linear_regression examples/linear_regression.cpp)
    target_link_libraries(linear_regression PRIVATE autograd::autograd)

    add_executable(mnist_classify_cpu examples/mnist_classify_cpu.cpp)
    target_link_libraries(mnist_classify_cpu PRIVATE autograd::autograd)
    target_compile_definitions(mnist_classify_cpu PRIVATE
        DATA_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/data")

    if(AUTOGRAD_USE_CUDA)
        add_executable(mnist_classify_gpu examples/mnist_classify_gpu.cpp)
        target_link_libraries(mnist_classify_gpu PRIVATE autograd::autograd)
        target_compile_definitions(mnist_classify_gpu PRIVATE
            DATA_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/data")
    endif()
endif()
