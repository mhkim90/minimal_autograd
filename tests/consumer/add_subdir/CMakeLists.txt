cmake_minimum_required(VERSION 3.14)
project(autograd_addsub_consumer CXX)

# Standalone `add_subdirectory` consumer smoke project.
#
# `autograd` is consumed by adding the source tree of minimal_autograd at
# `${AUTOGRAD_SRC_DIR}` (passed at configure time, because this consumer
# cannot recurse into its own source tree). After this call the
# `autograd::autograd` target is available with all its usual transitive
# dependencies, and the library's in-tree tests/examples MUST NOT be
# inherited because the consumer never asked for them.
#
# Required: -DAUTOGRAD_SRC_DIR=<absolute path to minimal_autograd source>
if(NOT AUTOGRAD_SRC_DIR)
    message(FATAL_ERROR
        "AUTOGRAD_SRC_DIR must be set to the minimal_autograd source tree.")
endif()

add_subdirectory("${AUTOGRAD_SRC_DIR}" autograd_build)

add_executable(autograd_smoke main.cpp)
target_link_libraries(autograd_smoke PRIVATE autograd::autograd)

enable_testing()
add_test(NAME autograd_add_subdir_smoke COMMAND autograd_smoke)

get_target_property(_autograd_normal_definitions
    autograd::autograd INTERFACE_COMPILE_DEFINITIONS)
if(_autograd_normal_definitions MATCHES "AUTOGRAD_USE_CUDA")
    message(FATAL_ERROR
        "embedded normal autograd target leaks AUTOGRAD_USE_CUDA")
endif()

# --- Inheritance self-check -------------------------------------------------
# The package boundary contract is "embedded add_subdirectory
# consumers do not inherit autograd's in-tree tests/examples". The check
# below makes that contract executable: configure fails if any of the
# known autograd test/example targets are visible in this consumer's scope
# after the add_subdirectory() call.
#
# The only reason to disable this check is to deliberately opt back in to
# the leaked targets by also passing -DAUTOGRAD_BUILD_TESTS=ON or
# -DAUTOGRAD_BUILD_EXAMPLES=ON, which is reserved for in-tree validation.
# Leave AUTOGRAD_CONSUMER_ALLOW_TARGET_LEAK unset (or OFF) for normal
# embedded consumption.
option(AUTOGRAD_CONSUMER_ALLOW_TARGET_LEAK
    "Disable the no-inheritance self-check (only for explicit opt-in)."
    OFF)
mark_as_advanced(
    FORCE AUTOGRAD_CONSUMER_ALLOW_TARGET_LEAK)

if(NOT AUTOGRAD_CONSUMER_ALLOW_TARGET_LEAK)
    set(_autograd_consumer_forbidden_targets
        test_core
        test_nn
        test_conv
        test_fft
        test_characterization
        test_extensions
        test_diffusion
        test_smoke
        test_cuda_core
        test_cuda_fft
        test_cuda_tensor
        linear_regression
        mnist_classify_cpu
        mnist_classify_gpu
        example)
    set(_autograd_consumer_leak "")
    foreach(_t IN LISTS _autograd_consumer_forbidden_targets)
        if(TARGET ${_t})
            list(APPEND _autograd_consumer_leak "${_t}")
        endif()
    endforeach()
    if(_autograd_consumer_leak)
        list(JOIN _autograd_consumer_leak ", " _autograd_consumer_leak_str)
        message(FATAL_ERROR
            "autograd::autograd consumer inherited forbidden targets: "
            "${_autograd_consumer_leak_str}. Embedded consumption must "
            "default to no tests and no examples; explicit opt-in requires "
            "-DAUTOGRAD_BUILD_TESTS=ON / -DAUTOGRAD_BUILD_EXAMPLES=ON "
            "(and -DAUTOGRAD_CONSUMER_ALLOW_TARGET_LEAK=ON to silence this "
            "check).")
    endif()
endif()
