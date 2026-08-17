// CUDA-build future dispatcher for direct Tensor operations.
//
// It intentionally owns no symbols in Phase 1: tensor_kernels.h remains the
// authoritative inline implementation until each operation family moves
// atomically in a later phase.
