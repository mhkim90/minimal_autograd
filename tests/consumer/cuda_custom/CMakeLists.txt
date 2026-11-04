cmake_minimum_required(VERSION 3.18)
project(autograd_cuda_custom_consumer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(autograd CONFIG REQUIRED COMPONENTS expert)

if(NOT TARGET autograd::autograd_expert)
    message(FATAL_ERROR "expert component did not provide autograd::autograd_expert")
endif()
get_target_property(_autograd_cuda_enabled
    autograd::autograd AUTOGRAD_CUDA_ENABLED)
if(NOT "${_autograd_cuda_enabled}" STREQUAL "1")
    message(FATAL_ERROR "CUDA custom consumer requires AUTOGRAD_CUDA_ENABLED=1")
endif()

add_executable(autograd_cuda_custom main.cu)
target_link_libraries(autograd_cuda_custom PRIVATE autograd::autograd_expert)

enable_testing()
add_test(NAME autograd_cuda_custom_smoke COMMAND autograd_cuda_custom)
