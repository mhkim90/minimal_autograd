#pragma once
// mnist_data.h — header-only IDX ubyte loader for MNIST.
//
// Reads MNIST `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
// `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` from the directory
// passed to `load_mnist()` and returns owned row-major Tensors.
//
// IDX layout (big endian):
//   offset 0..3   : magic   (0x00000803 for images, 0x00000801 for labels)
//   offset 4..7   : number of items N
//   offset 8..11+ : per-dim sizes (for images: rows, cols; for labels: none)
//   rest          : raw ubyte pixels, or ubyte labels
//
// Returns:
//   images : Tensor{N, 784}, pixel values in [0, 1] (uint8 / 255.f)
//   labels : Tensor{N, 10}, one-hot float

#include "autograd/tensor.h"
#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ag {

namespace {

inline uint32_t read_be_u32(const std::vector<unsigned char>& buf, std::size_t off) {
    return (uint32_t(buf[off])     << 24) |
           (uint32_t(buf[off + 1]) << 16) |
           (uint32_t(buf[off + 2]) <<  8) |
            uint32_t(buf[off + 3]);
}

inline void read_idx_images(const std::string& path,
                            Tensor& out_images, int& out_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
    if (buf.size() < 16) throw std::runtime_error("IDX header too small: " + path);

    uint32_t magic = read_be_u32(buf, 0);
    if (magic != 0x00000803u) {
        throw std::runtime_error("not an IDX image file: " + path);
    }
    const int N  = static_cast<int>(read_be_u32(buf, 4));
    const int H  = static_cast<int>(read_be_u32(buf, 8));
    const int W  = static_cast<int>(read_be_u32(buf, 12));
    const std::size_t expected = 16ull + static_cast<std::size_t>(N) * H * W;
    if (buf.size() < expected) {
        throw std::runtime_error("IDX image file truncated: " + path);
    }

    std::vector<float> values(static_cast<std::size_t>(N) * H * W);
    for (int n = 0; n < N; ++n) {
        for (int p = 0; p < H * W; ++p) {
            values[n * H * W + p] =
                static_cast<float>(buf[16 + n * H * W + p]) / 255.f;
        }
    }
    out_images = Tensor::from_host(values.data(), Shape{N, H * W});
    out_n = N;
}

inline void read_idx_labels(const std::string& path,
                            Tensor& out_labels_oh) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
    if (buf.size() < 8) throw std::runtime_error("IDX header too small: " + path);

    uint32_t magic = read_be_u32(buf, 0);
    if (magic != 0x00000801u) {
        throw std::runtime_error("not an IDX label file: " + path);
    }
    const int N = static_cast<int>(read_be_u32(buf, 4));
    if (buf.size() < 8ull + static_cast<std::size_t>(N)) {
        throw std::runtime_error("IDX label file truncated: " + path);
    }

    std::vector<float> values(static_cast<std::size_t>(N) * 10, 0.f);
    for (int i = 0; i < N; ++i) {
        unsigned char lbl = buf[8 + i];
        if (lbl < 10) values[i * 10 + lbl] = 1.f;
    }
    out_labels_oh = Tensor::from_host(values.data(), Shape{N, 10});
}

}  // namespace

// Convenience: read all four MNIST files from `root` (which is the
// directory containing the `train-*-idx?-ubyte` files). Returns N counts
// as a sanity check.
struct MnistData {
    Tensor train_images;
    Tensor train_labels;
    Tensor test_images;
    Tensor test_labels;
};

inline MnistData load_mnist(const std::string& root) {
    MnistData d;
    int n_tr_i = 0, n_te_i = 0;
    const std::string sep = root.empty() || root.back() == '/' ? "" : "/";
    read_idx_images(root + sep + "train-images-idx3-ubyte", d.train_images, n_tr_i);
    read_idx_labels(root + sep + "train-labels-idx1-ubyte", d.train_labels);
    read_idx_images(root + sep + "t10k-images-idx3-ubyte",  d.test_images,  n_te_i);
    read_idx_labels(root + sep + "t10k-labels-idx1-ubyte",  d.test_labels);

    if (d.train_images.shape()[0] != d.train_labels.shape()[0] ||
        d.test_images.shape()[0]  != d.test_labels.shape()[0]) {
        throw std::runtime_error("MNIST image/label row count mismatch");
    }
    return d;
}

// argmax accuracy: pred_cols is (N, 10) class scores, target_oh is (N, 10) one-hot.
inline float accuracy_argmax(const Tensor& pred_cols, const Tensor& target_oh) {
    int correct = 0;
    int N = pred_cols.shape()[0];
    std::vector<float> pred(pred_cols.elements()), target(target_oh.elements());
    pred_cols.copy_to_host(pred.data(), pred.size());
    target_oh.copy_to_host(target.data(), target.size());
    const int cols = pred_cols.shape()[1];
    for (int i = 0; i < N; ++i) {
        auto p = pred.begin() + i * cols;
        auto t = target.begin() + i * cols;
        if (std::max_element(p, p + cols) - p ==
            std::max_element(t, t + cols) - t) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(N);
}

}  // namespace ag
