cmake_minimum_required(VERSION 3.14)
project(autograd_eigen_custom_consumer LANGUAGES CXX)

find_package(autograd CONFIG REQUIRED COMPONENTS expert)

if(NOT TARGET autograd::autograd_expert)
    message(FATAL_ERROR "expert component did not provide autograd::autograd_expert")
endif()
get_target_property(_autograd_expert_links
    autograd::autograd_expert INTERFACE_LINK_LIBRARIES)
if(NOT _autograd_expert_links MATCHES "Eigen3::Eigen")
    message(FATAL_ERROR "expert target does not provide Eigen3::Eigen")
endif()

add_executable(autograd_eigen_custom main.cpp)
target_link_libraries(autograd_eigen_custom PRIVATE autograd::autograd_expert)

enable_testing()
add_test(NAME autograd_eigen_custom_smoke COMMAND autograd_eigen_custom)
