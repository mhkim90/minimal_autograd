cmake_minimum_required(VERSION 3.14)
project(autograd_legacy_expert_consumer LANGUAGES CXX)

# Source-preserving proof for the unmigrated legacy surface. The only changed
# consumer contract is the explicit expert component/target on this link line.
find_package(autograd CONFIG REQUIRED COMPONENTS expert)

if(NOT TARGET autograd::autograd_expert)
    message(FATAL_ERROR "expert component did not provide autograd::autograd_expert")
endif()

add_executable(autograd_legacy_expert main.cpp)
target_link_libraries(autograd_legacy_expert PRIVATE autograd::autograd_expert)

enable_testing()
add_test(NAME autograd_legacy_expert_smoke COMMAND autograd_legacy_expert)
