#pragma once
// function.h — Function base + apply<Fn>.
//
// apply<Fn> instantiates a Function, runs forward() to get the output, and
// captures a back_fn that calls the function's backward() and accumulates
// the resulting grads into each parent. All Function subclasses must
// implement forward() and backward().
//
// Two overloads:
//   apply<Fn>(ins)                          — Fn has a no-arg ctor.
//   apply<Fn>(ins, args...)                 — Fn has a ctor matching (args...).

#include "autograd/variable.h"
#include <memory>
#include <vector>
#include <utility>

namespace ag {

struct Function {
    // Inputs captured during forward, available to backward.
    Mats saved;

    virtual ~Function() = default;
    virtual Mat forward(const Mats& in) = 0;
    virtual Mats backward(const Mat& grad) = 0;
};

// apply<Fn>(ins): Fn must be default-constructible.
template<typename Fn>
VarPtr apply(std::vector<VarPtr> ins) {
    auto fn = std::make_shared<Fn>();
    Mats in_data;
    in_data.reserve(ins.size());
    for (auto& v : ins) in_data.push_back(v->data);

    auto out = Var::make(fn->forward(in_data));
    out->parents = ins;

    out->back_fn = [fn, ins, wp = std::weak_ptr<Var>(out)]() {
        auto self = wp.lock();
        auto gs = fn->backward(self->grad);
        for (size_t i = 0; i < ins.size() && i < gs.size(); ++i)
            ins[i]->grad += gs[i];
    };
    return out;
}

// apply<Fn>(ins, args...): Fn ctor takes (args...).
// Used when the function needs to know something about the computation
// (e.g. reshape target shape, concat split sizes) at construction time.
template<typename Fn, typename... Args>
VarPtr apply(std::vector<VarPtr> ins, Args... args) {
    auto fn = std::make_shared<Fn>(std::forward<Args>(args)...);
    Mats in_data;
    in_data.reserve(ins.size());
    for (auto& v : ins) in_data.push_back(v->data);

    auto out = Var::make(fn->forward(in_data));
    out->parents = ins;

    out->back_fn = [fn, ins, wp = std::weak_ptr<Var>(out)]() {
        auto self = wp.lock();
        auto gs = fn->backward(self->grad);
        for (size_t i = 0; i < ins.size() && i < gs.size(); ++i)
            ins[i]->grad += gs[i];
    };
    return out;
}

} // namespace ag
