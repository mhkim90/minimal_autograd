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
)
target_include_directories(autograd PUBLIC include)
target_link_libraries(autograd PUBLIC Eigen3::Eigen)

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

# Optional: build the README example.
option(AUTOGRAD_BUILD_EXAMPLES "Build README examples" OFF)
if(AUTOGRAD_BUILD_EXAMPLES)
    add_executable(linear_regression examples/linear_regression.cpp)
    target_link_libraries(linear_regression autograd)
endif()

# README quick-start example.
add_executable(example examples/linear_regression.cpp)
target_link_libraries(example autograd)
