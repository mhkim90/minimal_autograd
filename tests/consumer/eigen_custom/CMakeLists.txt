cmake_minimum_required(VERSION 3.14)
project(autograd_eigen_custom_consumer LANGUAGES CXX)

find_package(autograd CONFIG REQUIRED)

add_executable(autograd_eigen_custom main.cpp)
target_link_libraries(autograd_eigen_custom PRIVATE autograd::autograd)
