#pragma once
// Private CPU tensor arithmetic for the autograd vertical slice.
//
// All kernels act on Tensor-backed rank-2 storage laid out in the
// legacy Eigen column-major convention: flat index for shape (R, C) is
// `i + R * j`, where i is the row and j is the column of the logical
// matrix element. Host copies go through Tensor::copy_to_host /
// copy_from_host; shapes and devices are validated by callers.

#include "autograd/tensor.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ag {
namespace detail {

namespace {

inline int64_t rows_of(const Tensor& t) {
    return t.shape()[0];
}

inline int64_t cols_of(const Tensor& t) {
    return t.shape()[1];
}

inline void require_rank2(const char* op, const Tensor& t) {
    if (t.shape().rank() != 2) {
        std::ostringstream os;
        os << op << ": expected rank-2 tensor, got shape " << t.shape();
        throw std::invalid_argument(os.str());
    }
}

inline void require_same_shape(const char* op, const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        std::ostringstream os;
        os << op << ": shape mismatch (" << a.shape() << " vs "
           << b.shape() << ")";
        throw std::invalid_argument(os.str());
    }
}

inline void require_same_device(const char* op, const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        std::ostringstream os;
        os << op << ": device mismatch (" << a.device() << " vs "
           << b.device() << ")";
        throw std::invalid_argument(os.str());
    }
}

inline int64_t checked_dim_sum(const char* op, int64_t a, int64_t b) {
    if (b > std::numeric_limits<int64_t>::max() - a) {
        throw std::overflow_error(std::string(op) + ": dimension overflow");
    }
    return a + b;
}

}  // namespace

// ── Core arithmetic ────────────────────────────────────────────────────

inline Tensor tensor_add(const Tensor& a, const Tensor& b) {
    require_same_shape("add", a, b);
    require_same_device("add", a, b);
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), bv(n), ov(n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] + bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_mul(const Tensor& a, const Tensor& b) {
    require_same_shape("mul", a, b);
    require_same_device("mul", a, b);
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), bv(n), ov(n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] * bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_scale(const Tensor& a, float s) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), ov(n);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] * s;
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sum(const Tensor& a) {
    Tensor out = Tensor::empty(Shape{}, a.device());
    if (a.elements() == 0) {
        float z = 0.f;
        out.copy_from_host(&z, 1);
        return out;
    }
    std::vector<float> av(a.elements());
    a.copy_to_host(av.data(), av.size());
    float s = 0.f;
    for (float v : av) s += v;
    out.copy_from_host(&s, 1);
    return out;
}

inline Tensor tensor_broadcast_scalar(const Tensor& scalar,
                                      const Shape& target) {
    Tensor out = Tensor::empty(target, scalar.device());
    if (out.elements() == 0) return out;
    std::vector<float> seed(1);
    scalar.copy_to_host(seed.data(), 1);
    std::vector<float> buf(out.elements(), seed[0]);
    out.copy_from_host(buf.data(), buf.size());
    return out;
}

// ── Matmul, transpose, reshape, slice/concat/flip/cumsum ───────────────

inline Tensor tensor_matmul(const Tensor& a, const Tensor& b) {
    require_rank2("matmul", a);
    require_rank2("matmul", b);
    require_same_device("matmul", a, b);
    const int64_t M = rows_of(a);
    const int64_t K = cols_of(a);
    const int64_t K2 = rows_of(b);
    const int64_t N = cols_of(b);
    if (K != K2) {
        std::ostringstream os;
        os << "matmul: inner dimensions mismatch ("
           << a.shape() << " vs " << b.shape() << ")";
        throw std::invalid_argument(os.str());
    }
    Tensor out = Tensor::empty(Shape{M, N}, a.device());
    if (M == 0 || N == 0) return out;
    std::vector<float> av(M * K, 0.f), bv(K * N, 0.f), ov(M * N, 0.f);
    a.copy_to_host(av.data(), av.size());
    b.copy_to_host(bv.data(), bv.size());
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t c = 0; c < N; ++c) {
            float s = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                s += av[r + M * k] * bv[k + K * c];
            }
            ov[r + M * c] = s;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

// dA = dC * B^T for shapes (M, K) = (M, N) * (N, K).
inline Tensor tensor_matmul_backward_a(const Tensor& g, const Tensor& b) {
    require_rank2("matmul_backward_a", g);
    require_rank2("matmul_backward_a", b);
    require_same_device("matmul_backward_a", g, b);
    const int64_t M = rows_of(g);
    const int64_t N = cols_of(g);
    const int64_t K = rows_of(b);
    Tensor out = Tensor::empty(Shape{M, K}, g.device());
    if (M == 0 || K == 0) return out;
    std::vector<float> gv(M * N, 0.f), bv(K * N, 0.f), ov(M * K, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    b.copy_to_host(bv.data(), bv.size());
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t k = 0; k < K; ++k) {
            float s = 0.f;
            for (int64_t c = 0; c < N; ++c) {
                s += gv[r + M * c] * bv[k + K * c];
            }
            ov[r + M * k] = s;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

// dB = A^T * dC for shapes (K, N) = (K, M) * (M, N).
inline Tensor tensor_matmul_backward_b(const Tensor& a, const Tensor& g) {
    require_rank2("matmul_backward_b", a);
    require_rank2("matmul_backward_b", g);
    require_same_device("matmul_backward_b", a, g);
    const int64_t M = rows_of(a);
    const int64_t K = cols_of(a);
    const int64_t N = cols_of(g);
    Tensor out = Tensor::empty(Shape{K, N}, a.device());
    if (K == 0 || N == 0) return out;
    std::vector<float> av(M * K, 0.f), gv(M * N, 0.f), ov(K * N, 0.f);
    a.copy_to_host(av.data(), av.size());
    g.copy_to_host(gv.data(), gv.size());
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            float s = 0.f;
            for (int64_t m = 0; m < M; ++m) {
                s += av[m + M * k] * gv[m + M * n];
            }
            ov[k + K * n] = s;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_transpose(const Tensor& a) {
    require_rank2("transpose", a);
    const int64_t R = rows_of(a);
    const int64_t C = cols_of(a);
    Tensor out = Tensor::empty(Shape{C, R}, a.device());
    if (R == 0 || C == 0) return out;
    std::vector<float> av(R * C, 0.f), ov(R * C, 0.f);
    a.copy_to_host(av.data(), av.size());
    // out[c, r] = a[r, c]; out is shape (C, R).
    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < C; ++c) {
            ov[c + C * r] = av[r + R * c];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_reshape_view(const Tensor& a, int64_t rows, int64_t cols) {
    require_rank2("reshape", a);
    const int64_t expected = static_cast<int64_t>(a.elements());
    const Shape target_shape{rows, cols};
    const int64_t target = target_shape.numel();
    if (expected != target) {
        std::ostringstream os;
        os << "reshape: numel mismatch (have " << expected
           << ", want " << target << ")";
        throw std::invalid_argument(os.str());
    }
    return a.reshape(target_shape);
}

inline Tensor tensor_concat(const std::vector<Tensor>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("concat: requires at least one input");
    }
    int64_t total_rows = 0;
    int64_t cols = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        require_rank2("concat", inputs[i]);
        if (i == 0) {
            cols = cols_of(inputs[i]);
        } else if (cols_of(inputs[i]) != cols) {
            std::ostringstream os;
            os << "concat: column count mismatch at input " << i
               << " (have " << cols_of(inputs[i]) << ", want " << cols << ")";
            throw std::invalid_argument(os.str());
        }
        total_rows = checked_dim_sum(
            "concat", total_rows, rows_of(inputs[i]));
    }
    Tensor out = Tensor::empty(Shape{total_rows, cols}, inputs[0].device());
    if (total_rows == 0 || cols == 0) return out;
    std::vector<float> ov(total_rows * cols, 0.f);
    int64_t row_off = 0;
    for (const auto& t : inputs) {
        const int64_t R = rows_of(t);
        const int64_t C = cols_of(t);
        std::vector<float> tv(R * C, 0.f);
        t.copy_to_host(tv.data(), tv.size());
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t c = 0; c < C; ++c) {
                ov[(row_off + r) + total_rows * c] = tv[r + R * c];
            }
        }
        row_off += R;
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_hcat(const std::vector<Tensor>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("hcat: requires at least one input");
    }
    int64_t rows = 0;
    int64_t total_cols = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        require_rank2("hcat", inputs[i]);
        if (i == 0) {
            rows = rows_of(inputs[i]);
        } else if (rows_of(inputs[i]) != rows) {
            std::ostringstream os;
            os << "hcat: row count mismatch at input " << i
               << " (have " << rows_of(inputs[i]) << ", want " << rows << ")";
            throw std::invalid_argument(os.str());
        }
        total_cols = checked_dim_sum(
            "hcat", total_cols, cols_of(inputs[i]));
    }
    Tensor out = Tensor::empty(Shape{rows, total_cols}, inputs[0].device());
    if (rows == 0 || total_cols == 0) return out;
    std::vector<float> ov(rows * total_cols, 0.f);
    int64_t col_off = 0;
    for (const auto& t : inputs) {
        const int64_t R = rows_of(t);
        const int64_t C = cols_of(t);
        std::vector<float> tv(R * C, 0.f);
        t.copy_to_host(tv.data(), tv.size());
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t c = 0; c < C; ++c) {
                ov[r + rows * (col_off + c)] = tv[r + R * c];
            }
        }
        col_off += C;
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_col_slice(const Tensor& a, int64_t start, int64_t len) {
    require_rank2("col_slice", a);
    if (start < 0 || len <= 0 || start > cols_of(a) ||
        len > cols_of(a) - start) {
        std::ostringstream os;
        os << "col_slice: out of range (cols=" << cols_of(a)
           << ", start=" << start << ", len=" << len << ")";
        throw std::invalid_argument(os.str());
    }
    const int64_t R = rows_of(a);
    const int64_t C = cols_of(a);
    Tensor out = Tensor::empty(Shape{R, len}, a.device());
    if (R == 0 || len == 0) return out;
    std::vector<float> av(R * C, 0.f), ov(R * len, 0.f);
    a.copy_to_host(av.data(), av.size());
    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < len; ++c) {
            ov[r + R * c] = av[r + R * (start + c)];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_row_slice(const Tensor& a, int64_t start, int64_t len) {
    require_rank2("row_slice", a);
    if (start < 0 || len <= 0 || start > rows_of(a) ||
        len > rows_of(a) - start) {
        std::ostringstream os;
        os << "row_slice: out of range (rows=" << rows_of(a)
           << ", start=" << start << ", len=" << len << ")";
        throw std::invalid_argument(os.str());
    }
    const int64_t R = rows_of(a);
    const int64_t C = cols_of(a);
    Tensor out = Tensor::empty(Shape{len, C}, a.device());
    if (len == 0 || C == 0) return out;
    std::vector<float> av(R * C, 0.f), ov(len * C, 0.f);
    a.copy_to_host(av.data(), av.size());
    for (int64_t r = 0; r < len; ++r) {
        for (int64_t c = 0; c < C; ++c) {
            ov[r + len * c] = av[(start + r) + R * c];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

// Place g back into a zero tensor of shape (R, C) at cols [start, start+len).
inline Tensor tensor_col_slice_backward(const Tensor& g,
                                        int64_t rows,
                                        int64_t cols,
                                        int64_t start,
                                        int64_t len) {
    Tensor out = Tensor::zeros(Shape{rows, cols}, g.device());
    if (rows == 0 || cols == 0) return out;
    const int64_t Rg = rows_of(g);
    const int64_t Cg = cols_of(g);
    if (Rg != rows || Cg != len) {
        std::ostringstream os;
        os << "col_slice backward: gradient shape mismatch (got " << g.shape()
           << ", expected Shape{" << rows << ", " << len << "})";
        throw std::invalid_argument(os.str());
    }
    std::vector<float> gv(Rg * Cg, 0.f), ov(rows * cols, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    for (int64_t r = 0; r < Rg; ++r) {
        for (int64_t c = 0; c < Cg; ++c) {
            ov[r + rows * (start + c)] = gv[r + Rg * c];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

// Place g back into a zero tensor of shape (R, C) at rows [start, start+len).
inline Tensor tensor_row_slice_backward(const Tensor& g,
                                        int64_t rows,
                                        int64_t cols,
                                        int64_t start,
                                        int64_t len) {
    Tensor out = Tensor::zeros(Shape{rows, cols}, g.device());
    if (rows == 0 || cols == 0) return out;
    const int64_t Rg = rows_of(g);
    const int64_t Cg = cols_of(g);
    if (Rg != len || Cg != cols) {
        std::ostringstream os;
        os << "row_slice backward: gradient shape mismatch (got " << g.shape()
           << ", expected Shape{" << len << ", " << cols << "})";
        throw std::invalid_argument(os.str());
    }
    std::vector<float> gv(Rg * Cg, 0.f), ov(rows * cols, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    for (int64_t r = 0; r < Rg; ++r) {
        for (int64_t c = 0; c < Cg; ++c) {
            ov[(start + r) + rows * c] = gv[r + Rg * c];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

// Slice g (which has the same per-input row block size as concatenated)
// back into N tensors matching the parent shapes.
inline std::vector<Tensor> tensor_concat_backward(
        const Tensor& g, const std::vector<int64_t>& rows_per_input) {
    std::vector<Tensor> out;
    out.reserve(rows_per_input.size());
    int64_t row_off = 0;
    const int64_t total_rows = rows_of(g);
    const int64_t cols = cols_of(g);
    std::vector<float> gv(total_rows * cols, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    for (int64_t r : rows_per_input) {
        Tensor piece = Tensor::empty(Shape{r, cols}, g.device());
        if (r > 0 && cols > 0) {
            std::vector<float> pv(r * cols, 0.f);
            for (int64_t i = 0; i < r; ++i) {
                for (int64_t c = 0; c < cols; ++c) {
                    pv[i + r * c] = gv[(row_off + i) + total_rows * c];
                }
            }
            piece.copy_from_host(pv.data(), pv.size());
        }
        out.push_back(std::move(piece));
        row_off += r;
    }
    return out;
}

inline std::vector<Tensor> tensor_hcat_backward(
        const Tensor& g, const std::vector<int64_t>& cols_per_input) {
    std::vector<Tensor> out;
    out.reserve(cols_per_input.size());
    int64_t col_off = 0;
    const int64_t rows = rows_of(g);
    const int64_t total_cols = cols_of(g);
    std::vector<float> gv(rows * total_cols, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    for (int64_t c_count : cols_per_input) {
        Tensor piece = Tensor::empty(Shape{rows, c_count}, g.device());
        if (rows > 0 && c_count > 0) {
            std::vector<float> pv(rows * c_count, 0.f);
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 0; c < c_count; ++c) {
                    pv[r + rows * c] = gv[r + rows * (col_off + c)];
                }
            }
            piece.copy_from_host(pv.data(), pv.size());
        }
        out.push_back(std::move(piece));
        col_off += c_count;
    }
    return out;
}

inline Tensor tensor_cumsum(const Tensor& a, int axis) {
    require_rank2("cumsum", a);
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("cumsum: axis must be 0 or 1");
    }
    const int64_t R = rows_of(a);
    const int64_t C = cols_of(a);
    Tensor out = Tensor::empty(Shape{R, C}, a.device());
    if (R == 0 || C == 0) return out;
    std::vector<float> av(R * C, 0.f), ov(R * C, 0.f);
    a.copy_to_host(av.data(), av.size());
    if (axis == 1) {
        for (int64_t r = 0; r < R; ++r) {
            float s = av[r + R * 0];
            ov[r + R * 0] = s;
            for (int64_t c = 1; c < C; ++c) {
                s += av[r + R * c];
                ov[r + R * c] = s;
            }
        }
    } else {
        for (int64_t c = 0; c < C; ++c) {
            float s = av[0 + R * c];
            ov[0 + R * c] = s;
            for (int64_t r = 1; r < R; ++r) {
                s += av[r + R * c];
                ov[r + R * c] = s;
            }
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_cumsum_backward(const Tensor& g, int axis) {
    require_rank2("cumsum_backward", g);
    const int64_t R = rows_of(g);
    const int64_t C = cols_of(g);
    Tensor out = Tensor::empty(Shape{R, C}, g.device());
    if (R == 0 || C == 0) return out;
    std::vector<float> gv(R * C, 0.f), ov(R * C, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    if (axis == 1) {
        for (int64_t r = 0; r < R; ++r) {
            float s = gv[r + R * (C - 1)];
            ov[r + R * (C - 1)] = s;
            for (int64_t c = C - 2; c >= 0; --c) {
                s += gv[r + R * c];
                ov[r + R * c] = s;
            }
        }
    } else {
        for (int64_t c = 0; c < C; ++c) {
            float s = gv[(R - 1) + R * c];
            ov[(R - 1) + R * c] = s;
            for (int64_t r = R - 2; r >= 0; --r) {
                s += gv[r + R * c];
                ov[r + R * c] = s;
            }
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_flip(const Tensor& a, int axis) {
    require_rank2("flip", a);
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("flip: axis must be 0 or 1");
    }
    const int64_t R = rows_of(a);
    const int64_t C = cols_of(a);
    Tensor out = Tensor::empty(Shape{R, C}, a.device());
    if (R == 0 || C == 0) return out;
    std::vector<float> av(R * C, 0.f), ov(R * C, 0.f);
    a.copy_to_host(av.data(), av.size());
    if (axis == 1) {
        // Reverse the column index inside each row.
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t c = 0; c < C; ++c) {
                ov[r + R * c] = av[r + R * (C - 1 - c)];
            }
        }
    } else {
        // Reverse the row index inside each column.
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t c = 0; c < C; ++c) {
                ov[r + R * c] = av[(R - 1 - r) + R * c];
            }
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

// ── Activations and scalar functions ───────────────────────────────────

inline Tensor tensor_relu(const Tensor& a) {
    require_same_device("relu", a, a);
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = av[i] > 0.f ? av[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_relu_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("relu_backward", g, saved);
    require_same_device("relu_backward", g, saved);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = sv[i] > 0.f ? gv[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sigmoid(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = 1.f / (1.f + std::exp(-av[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sigmoid_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("sigmoid_backward", g, saved);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * sv[i] * (1.f - sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_tanh(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::tanh(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_tanh_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("tanh_backward", g, saved);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (1.f - sv[i] * sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_exp(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::exp(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_exp_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("exp_backward", g, saved);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * sv[i];
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_log(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::log(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_log_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("log_backward", g, saved);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / sv[i];
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sqrt(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::sqrt(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sqrt_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("sqrt_backward", g, saved);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / (2.f * sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    sigmoid_out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), so(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        so[i] = 1.f / (1.f + std::exp(-av[i]));
        ov[i] = av[i] * so[i];
    }
    out.copy_from_host(ov.data(), n);
    sigmoid_out.copy_from_host(so.data(), n);
    return out;
}

inline Tensor tensor_silu_backward(const Tensor& g,
                                    const Tensor& x,
                                    const Tensor& sig) {
    require_same_shape("silu_backward", g, x);
    require_same_shape("silu_backward", g, sig);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    sig.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (sv[i] + xv[i] * sv[i] * (1.f - sv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_softplus(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = av[i];
        const float ax = std::fabs(x);
        ov[i] = std::max(x, 0.f) + std::log(1.f + std::exp(-ax));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_softplus_backward(const Tensor& g,
                                       const Tensor& x) {
    require_same_shape("softplus_backward", g, x);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / (1.f + std::exp(-xv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sin(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::sin(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_cos(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::cos(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sin_backward(const Tensor& g, const Tensor& x) {
    require_same_shape("sin_backward", g, x);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * std::cos(xv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_cos_backward(const Tensor& g, const Tensor& x) {
    require_same_shape("cos_backward", g, x);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (-std::sin(xv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_clamp(const Tensor& a, float lo, float hi) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = av[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        ov[i] = v;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_clamp_backward(const Tensor& g,
                                    const Tensor& x,
                                    float lo,
                                    float hi) {
    require_same_shape("clamp_backward", g, x);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = (xv[i] >= lo && xv[i] <= hi) ? gv[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_broadcast_add(const Tensor& a, const Tensor& b) {
    require_rank2("broadcast_add", a);
    require_rank2("broadcast_add", b);
    require_same_device("broadcast_add", a, b);
    const int64_t R = rows_of(a);
    const int64_t D = cols_of(a);
    if (rows_of(b) != 1 || cols_of(b) != D) {
        std::ostringstream os;
        os << "broadcast_add: bias shape mismatch (got " << b.shape()
           << ", want Shape{1, " << D << "})";
        throw std::invalid_argument(os.str());
    }
    Tensor out = Tensor::empty(Shape{R, D}, a.device());
    if (R * D == 0) return out;
    std::vector<float> av(R * D, 0.f), bv(D, 0.f), ov(R * D, 0.f);
    a.copy_to_host(av.data(), av.size());
    b.copy_to_host(bv.data(), bv.size());
    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < D; ++c) {
            ov[r + R * c] = av[r + R * c] + bv[c];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_broadcast_add_backward_a(const Tensor& g) {
    require_rank2("broadcast_add_backward_a", g);
    return g.clone();
}

// Bias gradient: sum g over rows -> (1, D).
inline Tensor tensor_broadcast_add_backward_b(const Tensor& g) {
    require_rank2("broadcast_add_backward_b", g);
    const int64_t R = rows_of(g);
    const int64_t D = cols_of(g);
    Tensor out = Tensor::empty(Shape{1, D}, g.device());
    if (R * D == 0) return out;
    std::vector<float> gv(R * D, 0.f), ov(D, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    for (int64_t c = 0; c < D; ++c) {
        float s = 0.f;
        for (int64_t r = 0; r < R; ++r) {
            s += gv[r + R * c];
        }
        ov[c] = s;
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_sub(const Tensor& a, const Tensor& b) {
    require_same_shape("sub", a, b);
    require_same_device("sub", a, b);
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] - bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sub_backward_a(const Tensor& g) {
    return g.clone();
}

inline Tensor tensor_sub_backward_b(const Tensor& g) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = -gv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_div(const Tensor& a, const Tensor& b) {
    require_same_shape("div", a, b);
    require_same_device("div", a, b);
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] / bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_div_backward_a(const Tensor& g, const Tensor& b) {
    require_same_shape("div_backward_a", g, b);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = gv[i] / bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_div_backward_b(const Tensor& g,
                                    const Tensor& a,
                                    const Tensor& b) {
    require_same_shape("div_backward_b", g, a);
    require_same_shape("div_backward_b", g, b);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = -(gv[i] * av[i]) / (bv[i] * bv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

// ── Row reductions (softmax / log_softmax / mean bias) ────────────────

// Per-row softmax on a rank-2 input. The reduced-max-per-row trick is
// applied for numerical stability. Output Tensor has the same shape as
// a; saved_softmax holds the post-softmax matrix for backward.
inline Tensor tensor_softmax(const Tensor& a, Tensor& saved_softmax) {
    require_rank2("softmax", a);
    const int64_t R = rows_of(a);
    const int64_t C = cols_of(a);
    Tensor out = Tensor::empty(Shape{R, C}, a.device());
    saved_softmax = Tensor::empty(Shape{R, C}, a.device());
    if (R * C == 0) return out;
    std::vector<float> av(R * C, 0.f), sv(R * C, 0.f), ov(R * C, 0.f);
    a.copy_to_host(av.data(), av.size());

    std::vector<float> maxvals(R, 0.f);
    for (int64_t r = 0; r < R; ++r) {
        float m = av[r + R * 0];
        for (int64_t c = 1; c < C; ++c) {
            const float v = av[r + R * c];
            if (v > m) m = v;
        }
        maxvals[r] = m;
    }
    for (int64_t r = 0; r < R; ++r) {
        float denom = 0.f;
        for (int64_t c = 0; c < C; ++c) {
            const float e = std::exp(av[r + R * c] - maxvals[r]);
            sv[r + R * c] = e;
            denom += e;
        }
        const float inv_denom = 1.f / denom;
        for (int64_t c = 0; c < C; ++c) {
            ov[r + R * c] = sv[r + R * c] * inv_denom;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    saved_softmax.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_softmax_backward(const Tensor& g,
                                      const Tensor& saved_softmax) {
    require_same_shape("softmax_backward", g, saved_softmax);
    require_rank2("softmax_backward", g);
    const int64_t R = rows_of(g);
    const int64_t C = cols_of(g);
    Tensor out = Tensor::empty(Shape{R, C}, g.device());
    if (R * C == 0) return out;
    std::vector<float> gv(R * C, 0.f), sv(R * C, 0.f), ov(R * C, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    saved_softmax.copy_to_host(sv.data(), sv.size());

    std::vector<float> dot(R, 0.f);
    for (int64_t r = 0; r < R; ++r) {
        float s = 0.f;
        for (int64_t c = 0; c < C; ++c) {
            s += gv[r + R * c] * sv[r + R * c];
        }
        dot[r] = s;
    }
    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < C; ++c) {
            const float sm = sv[r + R * c];
            ov[r + R * c] = sm * (gv[r + R * c] - dot[r]);
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_log_softmax(const Tensor& a, Tensor& saved_lsm) {
    require_rank2("log_softmax", a);
    const int64_t R = rows_of(a);
    const int64_t C = cols_of(a);
    Tensor out = Tensor::empty(Shape{R, C}, a.device());
    saved_lsm = Tensor::empty(Shape{R, C}, a.device());
    if (R * C == 0) return out;
    std::vector<float> av(R * C, 0.f), lv(R * C, 0.f), ov(R * C, 0.f);
    a.copy_to_host(av.data(), av.size());

    for (int64_t r = 0; r < R; ++r) {
        float m = av[r + R * 0];
        for (int64_t c = 1; c < C; ++c) {
            const float v = av[r + R * c];
            if (v > m) m = v;
        }
        float denom = 0.f;
        for (int64_t c = 0; c < C; ++c) {
            denom += std::exp(av[r + R * c] - m);
        }
        const float log_sum = m + std::log(denom);
        for (int64_t c = 0; c < C; ++c) {
            lv[r + R * c] = av[r + R * c] - log_sum;
            ov[r + R * c] = lv[r + R * c];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    saved_lsm.copy_from_host(lv.data(), lv.size());
    return out;
}

inline Tensor tensor_log_softmax_backward(const Tensor& g,
                                          const Tensor& saved_lsm) {
    require_same_shape("log_softmax_backward", g, saved_lsm);
    require_rank2("log_softmax_backward", g);
    const int64_t R = rows_of(g);
    const int64_t C = cols_of(g);
    Tensor out = Tensor::empty(Shape{R, C}, g.device());
    if (R * C == 0) return out;
    std::vector<float> gv(R * C, 0.f), lv(R * C, 0.f), ov(R * C, 0.f);
    g.copy_to_host(gv.data(), gv.size());
    saved_lsm.copy_to_host(lv.data(), lv.size());

    std::vector<float> row_sum(R, 0.f);
    for (int64_t r = 0; r < R; ++r) {
        float s = 0.f;
        for (int64_t c = 0; c < C; ++c) {
            s += gv[r + R * c];
        }
        row_sum[r] = s;
    }
    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < C; ++c) {
            const float sm = std::exp(lv[r + R * c]);
            ov[r + R * c] = gv[r + R * c] - sm * row_sum[r];
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

}  // namespace detail
}  // namespace ag
