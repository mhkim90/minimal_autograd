// src/variable.cpp — Var methods.
//
// _build_topo / backward / zero_grad are out-of-line here so that the
// header (variable.h) can stay a clean data-only struct.

#include "autograd/variable.h"

namespace ag {

void Var::_build_topo(std::vector<Var*>& order,
                      std::unordered_set<Var*>& visited) {
    if (visited.count(this)) return;
    visited.insert(this);
    for (auto& p : parents) p->_build_topo(order, visited);
    order.push_back(this);
}

void Var::backward() {
    assert(data.rows() == 1 && data.cols() == 1 &&
           "backward() requires a scalar (1x1) loss");
    grad = Mat::Ones(1, 1);

    std::vector<Var*> topo;
    std::unordered_set<Var*> visited;
    _build_topo(topo, visited);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->back_fn) (*it)->back_fn();
    }
}

void Var::zero_grad() {
    std::vector<Var*> topo;
    std::unordered_set<Var*> visited;
    _build_topo(topo, visited);
    for (auto* v : topo) v->grad.setZero();
}

} // namespace ag
