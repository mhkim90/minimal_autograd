#pragma once
// Expert boundary for Tensor-based custom autograd operations.

#include "autograd/core/variable.h"

#include <functional>
#include <vector>

namespace ag {

using BackwardFunction =
    std::function<std::vector<Tensor>(const Tensor& output_grad)>;

// The backward function returns one gradient per input. Count, shape, and
// device are validated before any gradient is committed.
Variable make_custom_variable(Tensor output,
                              std::vector<Variable> inputs,
                              BackwardFunction backward);

}  // namespace ag
