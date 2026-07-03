// src/variable.cpp — Var methods.
//
// _build_topo / backward / zero_grad are out-of-line here so that the
// header (variable.h) can stay a clean data-only struct.

#include "autograd/variable.h"
#include "autograd/cuda_core.h"

namespace ag {

Var::~Var() {
#ifdef AUTOGRAD_USE_CUDA
    try {
        if (cuda_data_) cuda_free_floats(cuda_data_);
        if (cuda_grad_) cuda_free_floats(cuda_grad_);
    } catch (...) {}
#endif
}

VarPtr Var::cuda(int device) const {
#ifndef AUTOGRAD_USE_CUDA
    (void)device;
    throw std::runtime_error("Var::cuda(): built without AUTOGRAD_USE_CUDA");
#else
    auto out = Var::make(data);
    out->grad = grad;
    out->set_shape(shape_);
    out->cuda_ = true;
    out->cuda_device_ = device;
    const std::size_t n = static_cast<std::size_t>(out->data.size());
    cuda_alloc_floats(&out->cuda_data_, n, device);
    cuda_alloc_floats(&out->cuda_grad_, n, device);
    cuda_copy_h2d(out->cuda_data_, out->data.data(), n, device);
    cuda_copy_h2d(out->cuda_grad_, out->grad.data(), n, device);
    return out;
#endif
}

VarPtr Var::cpu() const {
    auto out = Var::make(data);
    out->grad = grad;
    out->set_shape(shape_);
#ifdef AUTOGRAD_USE_CUDA
    if (cuda_) {
        const std::size_t n = static_cast<std::size_t>(out->data.size());
        cuda_copy_d2h(out->data.data(), cuda_data_, n, cuda_device_);
        cuda_copy_d2h(out->grad.data(), cuda_grad_, n, cuda_device_);
    }
#endif
    return out;
}

void Var::sync_data_from_cuda() {
#ifdef AUTOGRAD_USE_CUDA
    if (cuda_) cuda_copy_d2h(data.data(), cuda_data_,
                            static_cast<std::size_t>(data.size()), cuda_device_);
#endif
}

void Var::sync_grad_from_cuda() {
#ifdef AUTOGRAD_USE_CUDA
    if (cuda_) cuda_copy_d2h(grad.data(), cuda_grad_,
                            static_cast<std::size_t>(grad.size()), cuda_device_);
#endif
}

void Var::sync_data_to_cuda() {
#ifdef AUTOGRAD_USE_CUDA
    if (cuda_) cuda_copy_h2d(cuda_data_, data.data(),
                            static_cast<std::size_t>(data.size()), cuda_device_);
#endif
}

void Var::sync_grad_to_cuda() {
#ifdef AUTOGRAD_USE_CUDA
    if (cuda_) cuda_copy_h2d(cuda_grad_, grad.data(),
                            static_cast<std::size_t>(grad.size()), cuda_device_);
#endif
}

void Var::cuda_zero_grad() {
#ifdef AUTOGRAD_USE_CUDA
    if (cuda_) cuda_zero(cuda_grad_, static_cast<std::size_t>(grad.size()), cuda_device_);
#endif
}

void Var::cuda_grad_ones() {
#ifdef AUTOGRAD_USE_CUDA
    if (cuda_) cuda_fill(cuda_grad_, 1.f, static_cast<std::size_t>(grad.size()), cuda_device_);
#endif
}

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
    cuda_grad_ones();

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
    for (auto* v : topo) {
        v->grad.setZero();
        v->cuda_zero_grad();
    }
}

} // namespace ag
