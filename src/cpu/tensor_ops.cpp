// Future CPU provider for the direct Tensor operation stack.
//
// Phase 1 deliberately leaves existing inline tensor_kernels.h symbols in
// place. This source is compiled in both build modes so later migrations can
// route CPU-resident tensors here without changing CMake source selection.

namespace ag {
namespace detail {
namespace cpu_ops {
}  // namespace cpu_ops
}  // namespace detail
}  // namespace ag
