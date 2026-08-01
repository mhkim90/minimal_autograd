cmake_minimum_required(VERSION 3.18)
project(autograd_cuda_custom_consumer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(autograd CONFIG REQUIRED)

add_executable(autograd_cuda_custom main.cu)
target_link_libraries(autograd_cuda_custom PRIVATE autograd::autograd)
