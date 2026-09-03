cmake_minimum_required(VERSION 3.14)
project(autograd_unknown_component_consumer LANGUAGES CXX)

# This fixture is expected to fail at configure time. It verifies that the
# package does not silently accept an unknown REQUIRED component.
find_package(autograd CONFIG REQUIRED COMPONENTS definitely_unknown)
