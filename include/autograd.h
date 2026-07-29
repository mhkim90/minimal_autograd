// autograd.h — umbrella include for the entire library.

#pragma once

#include "autograd/tensor.h"
#include "autograd/shape.h"
#include "autograd/device.h"
#include "autograd/core/variable.h"
#include "autograd/core/ops.h"
#include "autograd/core/loss.h"
#include "autograd/core/module.h"
#include "autograd/core/optim.h"
#include "autograd/core/diffusion.h"
#include "autograd/extension/custom_op.h"
// The legacy `Mat` / `Mats` / `shape(Mat)` / `numel(Mat)` aliases
// used by the existing Var / Function / ops / modules surface live in
// the opt-in extension header. The umbrella still re-exports them so
// downstream code that includes "autograd.h" continues to work
// unchanged; the new public path is `autograd/tensor.h` alone.
#include "autograd/extension/eigen.h"
#include "autograd/variable.h"
#include "autograd/function.h"
#include "autograd/ops.h"
#include "autograd/module.h"
#include "autograd/loss.h"
#include "autograd/optim.h"
#include "autograd/conv.h"
#include "autograd/norm.h"
#include "autograd/diffusion.h"
#include "autograd/complex.h"
#include "autograd/fft.h"
// Core complex and FFT surface. The canonical ag::FftNorm enum is
// also reachable through the legacy autograd/fft.h include above.
#include "autograd/core/fft_norm.h"
#include "autograd/core/complex.h"
#include "autograd/core/fft.h"

#ifdef AUTOGRAD_USE_CUDA
#include "autograd/cuda_core.h"
#endif
