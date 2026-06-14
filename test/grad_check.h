#pragma once
// grad_check.h — numerical gradient check.
//
// Compares the analytic gradient (from backward()) against a central
// finite-difference estimate. Use after writing a custom op.
//
// Each call rebuilds a fresh graph (the closure you pass creates a new
// VarPtr each invocation), so this is safe to call inside a loop.

#include "autograd.h"
#include <functional>
#include <cstdio>
#include <cmath>

inline bool grad_check(std::function<ag::VarPtr(ag::VarPtr)> f,
                       ag::VarPtr x,
                       float eps = 1e-3f,
                       float tol = 5e-2f) {
    x->grad.setZero();   // avoid accumulation from previous calls
    auto y = f(x);
    y->backward();
    ag::Mat analytic = x->grad;

    ag::Mat numeric = ag::Mat::Zero(x->data.rows(), x->data.cols());
    for (int i = 0; i < x->data.rows(); ++i) {
        for (int j = 0; j < x->data.cols(); ++j) {
            float orig = x->data(i, j);

            x->data(i, j) = orig + eps;
            float fwd = ag::sum(f(x))->data(0, 0);

            x->data(i, j) = orig - eps;
            float bwd = ag::sum(f(x))->data(0, 0);

            x->data(i, j) = orig;
            numeric(i, j) = (fwd - bwd) / (2.f * eps);
        }
    }
    float diff = (analytic - numeric).cwiseAbs().maxCoeff();
    if (diff > tol) {
        std::printf("    grad_check FAIL: max abs diff = %g (tol %g)\n", diff, tol);
        return false;
    }
    return true;
}

inline bool grad_check_multi(
        std::function<ag::VarPtr(std::vector<ag::VarPtr>)> f,
        std::vector<ag::VarPtr> xs,
        float eps = 1e-3f,
        float tol = 5e-2f) {
    for (auto& x : xs) x->grad.setZero();
    auto y = f(xs);
    y->backward();

    for (size_t vi = 0; vi < xs.size(); ++vi) {
        auto x = xs[vi];
        ag::Mat analytic = x->grad;
        ag::Mat numeric = ag::Mat::Zero(x->data.rows(), x->data.cols());
        for (int i = 0; i < x->data.rows(); ++i) {
            for (int j = 0; j < x->data.cols(); ++j) {
                float orig = x->data(i, j);
                x->data(i, j) = orig + eps;
                float fwd = ag::sum(f(xs))->data(0, 0);
                x->data(i, j) = orig - eps;
                float bwd = ag::sum(f(xs))->data(0, 0);
                x->data(i, j) = orig;
                numeric(i, j) = (fwd - bwd) / (2.f * eps);
            }
        }
        float diff = (analytic - numeric).cwiseAbs().maxCoeff();
        if (diff > tol) {
            std::printf("    grad_check_multi FAIL on input %zu: diff = %g\n", vi, diff);
            return false;
        }
    }
    return true;
}
