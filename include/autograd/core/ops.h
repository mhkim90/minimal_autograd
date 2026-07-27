#pragma once
// Tensor-based autograd operation vertical slice. Binary operations require
// identical shapes and devices; broadcasting is intentionally unsupported.

#include "autograd/core/variable.h"

namespace ag {

Variable add(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable scale(const Variable& a, float scalar);
Variable sum(const Variable& a);

}  // namespace ag
