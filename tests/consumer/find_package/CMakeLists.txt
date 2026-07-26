cmake_minimum_required(VERSION 3.14)
project(autograd_findpkg_consumer CXX)

# Standalone `find_package(CONFIG)` consumer smoke project. Requires
# `-DCMAKE_PREFIX_PATH=<install-prefix>` (or equivalent) so that the
# installed `autogradConfig.cmake` is discoverable. Provides a single
# executable target `autograd_smoke` linking to `autograd::autograd`.
find_package(autograd CONFIG REQUIRED)

add_executable(autograd_smoke main.cpp)
target_link_libraries(autograd_smoke PRIVATE autograd::autograd)
