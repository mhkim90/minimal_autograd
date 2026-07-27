#include "autograd/core/variable.h"

#include "autograd/extension/custom_op.h"
#include "detail/tensor_kernels.h"
#include "detail/variable_internal.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ag {
namespace {

using Node = detail::VariableNode;
using NodePtr = std::shared_ptr<Node>;
using GradMap = std::unordered_map<Node*, Tensor>;

void topo_visit(Node* node,
                std::unordered_set<Node*>& visited,
                std::vector<Node*>& order) {
    if (!visited.insert(node).second) return;
    for (const auto& parent : node->parents) {
        topo_visit(parent.get(), visited, order);
    }
    order.push_back(node);
}

void validate_gradient(const char* context,
                       std::size_t index,
                       const Tensor& gradient,
                       const Node& parent) {
    if (gradient.shape() != parent.value.shape()) {
        std::ostringstream os;
        os << context << ": gradient " << index << " shape mismatch ("
           << gradient.shape() << " vs " << parent.value.shape() << ")";
        throw std::invalid_argument(os.str());
    }
    if (gradient.device() != parent.value.device()) {
        std::ostringstream os;
        os << context << ": gradient " << index << " device mismatch ("
           << gradient.device() << " vs " << parent.value.device() << ")";
        throw std::invalid_argument(os.str());
    }
}

void accumulate(GradMap& grads, Node* node, const Tensor& gradient) {
    auto it = grads.find(node);
    if (it == grads.end()) {
        grads.emplace(node, gradient.clone());
    } else {
        it->second = detail::tensor_add(it->second, gradient);
    }
}

std::vector<Tensor> backward_step(Node& node, const Tensor& output_grad) {
    switch (node.kind) {
        case detail::OpKind::Leaf:
            return {};
        case detail::OpKind::Add:
            return {output_grad, output_grad};
        case detail::OpKind::Mul:
            return {
                detail::tensor_mul(output_grad, node.saved[1]),
                detail::tensor_mul(output_grad, node.saved[0]),
            };
        case detail::OpKind::Scale:
            return {detail::tensor_scale(output_grad, node.scalar)};
        case detail::OpKind::Sum:
            return {
                detail::tensor_broadcast_scalar(output_grad,
                                                node.parents[0]->value.shape()),
            };
        case detail::OpKind::MatMul:
            return {
                detail::tensor_matmul_backward_a(output_grad, node.saved[1]),
                detail::tensor_matmul_backward_b(node.saved[0], output_grad),
            };
        case detail::OpKind::ReLU:
            return {detail::tensor_relu_backward(output_grad, node.saved[0])};
        case detail::OpKind::BroadcastAdd:
            return {
                detail::tensor_broadcast_add_backward_a(output_grad),
                detail::tensor_broadcast_add_backward_b(output_grad),
            };
        case detail::OpKind::Softmax:
            return {
                detail::tensor_softmax_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::LogSoftmax:
            return {
                detail::tensor_log_softmax_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Transpose:
            return {detail::tensor_transpose(output_grad)};
        case detail::OpKind::Reshape:
            return {
                detail::tensor_reshape_view(output_grad,
                                            node.saved[0].shape()[0],
                                            node.saved[0].shape()[1]),
            };
        case detail::OpKind::Concat: {
            // saved[i] holds each parent's value; their rows indicate
            // the corresponding slice of the gradient.
            std::vector<int64_t> rows;
            rows.reserve(node.saved.size());
            for (const auto& t : node.saved) {
                rows.push_back(t.shape()[0]);
            }
            return detail::tensor_concat_backward(output_grad, rows);
        }
        case detail::OpKind::HCat: {
            std::vector<int64_t> cols;
            cols.reserve(node.saved.size());
            for (const auto& t : node.saved) {
                cols.push_back(t.shape()[1]);
            }
            return detail::tensor_hcat_backward(output_grad, cols);
        }
        case detail::OpKind::Sigmoid:
            return {
                detail::tensor_sigmoid_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Tanh:
            return {
                detail::tensor_tanh_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Exp:
            return {
                detail::tensor_exp_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Log:
            return {
                detail::tensor_log_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Sqrt:
            return {
                detail::tensor_sqrt_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::SiLU:
            return {
                detail::tensor_silu_backward(output_grad,
                                             node.saved[0], node.saved[1]),
            };
        case detail::OpKind::Softplus:
            return {
                detail::tensor_softplus_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Sub:
            return {
                detail::tensor_sub_backward_a(output_grad),
                detail::tensor_sub_backward_b(output_grad),
            };
        case detail::OpKind::Div:
            return {
                detail::tensor_div_backward_a(output_grad, node.saved[1]),
                detail::tensor_div_backward_b(output_grad,
                                               node.saved[0], node.saved[1]),
            };
        case detail::OpKind::Cumsum:
            return {
                detail::tensor_cumsum_backward(output_grad, node.axis),
            };
        case detail::OpKind::Flip:
            return {detail::tensor_flip(output_grad, node.axis)};
        case detail::OpKind::Sin:
            return {
                detail::tensor_sin_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Cos:
            return {
                detail::tensor_cos_backward(output_grad, node.saved[0]),
            };
        case detail::OpKind::Clamp:
            return {
                detail::tensor_clamp_backward(output_grad, node.saved[0],
                                              node.extra_f0, node.extra_f1),
            };
        case detail::OpKind::ColSlice:
            return {
                detail::tensor_col_slice_backward(
                    output_grad,
                    node.saved[0].shape()[0], node.saved[0].shape()[1],
                    node.extra_i0, node.extra_i1),
            };
        case detail::OpKind::RowSlice:
            return {
                detail::tensor_row_slice_backward(
                    output_grad,
                    node.saved[0].shape()[0], node.saved[0].shape()[1],
                    node.extra_i0, node.extra_i1),
            };
        case detail::OpKind::Custom:
            return node.custom_backward(output_grad);
    }
    throw std::logic_error("unknown autograd operation");
}

}  // namespace

Variable::Variable()
    : node_(std::make_shared<Node>(Tensor(), false)) {}

Variable::Variable(Tensor value, bool requires_grad)
    : node_(std::make_shared<Node>(std::move(value), requires_grad)) {}

Variable::Variable(std::shared_ptr<Node> node)
    : node_(std::move(node)) {}

const Tensor& Variable::value() const noexcept { return node_->value; }
bool Variable::requires_grad() const noexcept { return node_->requires_grad; }
bool Variable::has_grad() const noexcept { return node_->has_grad; }

const Tensor& Variable::grad() const {
    if (!node_->has_grad) {
        throw std::runtime_error("Variable::grad: gradient is not available");
    }
    return node_->grad;
}

Device Variable::device() const noexcept { return node_->value.device(); }

Variable Variable::to(Device target) const {
    if (target == device()) return *this;
    return Variable(node_->value.to(target), node_->requires_grad);
}

Variable Variable::detach() const {
    return Variable(node_->value, false);
}

void Variable::zero_grad() {
    node_->has_grad = false;
    node_->grad = Tensor();
}

void Variable::backward() {
    if (node_->value.elements() != 1) {
        throw std::invalid_argument(
            "Variable::backward: implicit gradient requires one element");
    }
    backward(Tensor::ones(node_->value.shape(), node_->value.device()));
}

void Variable::backward(const Tensor& upstream_gradient) {
    if (!node_->requires_grad) {
        throw std::runtime_error(
            "Variable::backward: variable does not require gradients");
    }
    if (upstream_gradient.shape() != node_->value.shape()) {
        throw std::invalid_argument(
            "Variable::backward: upstream gradient shape mismatch");
    }
    if (upstream_gradient.device() != node_->value.device()) {
        throw std::invalid_argument(
            "Variable::backward: upstream gradient device mismatch");
    }

    std::vector<Node*> topo;
    std::unordered_set<Node*> visited;
    topo_visit(node_.get(), visited, topo);

    GradMap pending;
    pending.emplace(node_.get(), upstream_gradient.clone());

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Node* current = *it;
        auto grad_it = pending.find(current);
        if (grad_it == pending.end() || current->kind == detail::OpKind::Leaf) {
            continue;
        }

        std::vector<Tensor> input_grads =
            backward_step(*current, grad_it->second);
        if (input_grads.size() != current->parents.size()) {
            std::ostringstream os;
            os << "custom backward returned " << input_grads.size()
               << " gradients; expected " << current->parents.size();
            throw std::invalid_argument(os.str());
        }

        for (std::size_t i = 0; i < current->parents.size(); ++i) {
            Node* parent = current->parents[i].get();
            validate_gradient("backward", i, input_grads[i], *parent);
            if (parent->requires_grad) {
                accumulate(pending, parent, input_grads[i]);
            }
        }
    }

    // Prepare every accumulated result before mutating any node.
    GradMap committed;
    for (auto& entry : pending) {
        Node* target = entry.first;
        if (!target->requires_grad) continue;
        if (target->has_grad) {
            committed.emplace(
                target, detail::tensor_add(target->grad, entry.second));
        } else {
            committed.emplace(target, entry.second.clone());
        }
    }

    for (auto& entry : committed) {
        entry.first->grad = std::move(entry.second);
        entry.first->has_grad = true;
    }
}

Variable make_custom_variable(Tensor output,
                              std::vector<Variable> inputs,
                              BackwardFunction backward) {
    if (!backward) {
        throw std::invalid_argument(
            "make_custom_variable: backward function is empty");
    }

    bool requires_grad = false;
    std::vector<NodePtr> parents;
    parents.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (input.device() != output.device()) {
            throw std::invalid_argument(
                "make_custom_variable: input/output device mismatch");
        }
        const auto& parent = detail::VariableAccess::node(input);
        requires_grad = requires_grad || parent->requires_grad;
        parents.push_back(parent);
    }

    auto node = std::make_shared<Node>(std::move(output), requires_grad);
    if (requires_grad) {
        node->kind = detail::OpKind::Custom;
        node->parents = std::move(parents);
        node->custom_backward = std::move(backward);
    }
    return detail::VariableAccess::make(std::move(node));
}

}  // namespace ag
