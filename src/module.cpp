#include "autograd/module.h"

namespace ag {

Linear::Linear(int in_features, int out_features) {
    float scale = std::sqrt(2.f / static_cast<float>(in_features));
    W = Var::make(Mat::Random(in_features, out_features) * scale);
    b = Var::make(Mat::Zero(1, out_features));
}

} // namespace ag
