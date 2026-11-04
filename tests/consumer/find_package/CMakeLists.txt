cmake_minimum_required(VERSION 3.14)
project(autograd_findpkg_consumer CXX)

# Standalone `find_package(CONFIG)` consumer smoke project. Requires
# `-DCMAKE_PREFIX_PATH=<install-prefix>` (or equivalent) so that the
# installed `autogradConfig.cmake` is discoverable. Provides a single
# executable target `autograd_smoke` linking to `autograd::autograd`.
find_package(autograd CONFIG REQUIRED)

get_target_property(_autograd_cuda_enabled
    autograd::autograd AUTOGRAD_CUDA_ENABLED)
if(NOT "${_autograd_cuda_enabled}" STREQUAL "0" AND
   NOT "${_autograd_cuda_enabled}" STREQUAL "1")
    message(FATAL_ERROR "autograd AUTOGRAD_CUDA_ENABLED must be 0 or 1")
endif()
if(DEFINED AUTOGRAD_EXPECT_CUDA_ENABLED AND
   NOT "${_autograd_cuda_enabled}" STREQUAL "${AUTOGRAD_EXPECT_CUDA_ENABLED}")
    message(FATAL_ERROR "autograd AUTOGRAD_CUDA_ENABLED does not match expected value")
endif()

if(TARGET autograd::autograd_expert OR TARGET autograd_expert)
    message(FATAL_ERROR
        "component-free autograd package unexpectedly exposes the expert target")
endif()
get_target_property(_autograd_normal_links
    autograd::autograd INTERFACE_LINK_LIBRARIES)
if(_autograd_normal_links MATCHES "Eigen3::Eigen" OR
   _autograd_normal_links MATCHES "AUTOGRAD_USE_CUDA")
    message(FATAL_ERROR
        "normal autograd target leaks an expert dependency or CUDA macro")
endif()
get_target_property(_autograd_normal_definitions
    autograd::autograd INTERFACE_COMPILE_DEFINITIONS)
if(_autograd_normal_definitions MATCHES "AUTOGRAD_USE_CUDA")
    message(FATAL_ERROR
        "normal autograd target leaks AUTOGRAD_USE_CUDA")
endif()

add_executable(autograd_smoke main.cpp)
target_link_libraries(autograd_smoke PRIVATE autograd::autograd)

enable_testing()
add_test(NAME autograd_find_package_smoke COMMAND autograd_smoke)
