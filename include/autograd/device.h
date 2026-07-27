#pragma once
// device.h — minimal device descriptor value type.
//
// This header is the first half of the Phase 2 foundation (Shape + Device).
// It is intentionally small and standalone:
//   * No CUDA headers or runtime symbols are referenced. CPU-only builds
//     compile and use Device unchanged.
//   * No runtime singleton, stream, node, or distributed-ID concept is
//     introduced.
//   * No Tensor, Variable, or backend-interface integration is performed
//     in this phase; that work is reserved for the next Phase 2 PR.
//
// See ARCHITECTURE_REFACTOR_PLAN.md §5.2 for the target API.

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ag {

enum class DeviceType {
    Cpu,
    Cuda,
};

class Device {
public:
    Device() noexcept : type_(DeviceType::Cpu), index_(0) {}

    static Device cpu() noexcept { return Device(DeviceType::Cpu, 0); }

    static Device cuda(int index = 0) {
        if (index < 0) {
            std::ostringstream os;
            os << "Device::cuda: negative device index (" << index << ")";
            throw std::invalid_argument(os.str());
        }
        return Device(DeviceType::Cuda, index);
    }

    DeviceType type() const noexcept { return type_; }
    int index() const noexcept { return index_; }

    bool is_cpu() const noexcept { return type_ == DeviceType::Cpu; }
    bool is_cuda() const noexcept { return type_ == DeviceType::Cuda; }

    bool operator==(const Device& o) const noexcept {
        return type_ == o.type_ && index_ == o.index_;
    }
    bool operator!=(const Device& o) const noexcept { return !(*this == o); }

    std::string to_string() const {
        std::ostringstream os;
        if (type_ == DeviceType::Cpu) {
            os << "cpu";
        } else {
            os << "cuda:" << index_;
        }
        return os.str();
    }

private:
    DeviceType type_;
    int index_;

    Device(DeviceType t, int i) noexcept : type_(t), index_(i) {}
};

inline std::ostream& operator<<(std::ostream& os, const Device& d) {
    return os << d.to_string();
}

inline std::ostream& operator<<(std::ostream& os, DeviceType t) {
    switch (t) {
        case DeviceType::Cpu:  return os << "cpu";
        case DeviceType::Cuda: return os << "cuda";
    }
    return os << "DeviceType(?)";
}

} // namespace ag
