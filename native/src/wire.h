// The wire conventions CrystalNetBridge.java mirrors - the shapes are a contract,
// not an implementation detail. A version bump must land on both sides in one commit.
#pragma once

#include <cstddef>

namespace fc {
namespace wire {

constexpr int32_t kVersion = 1;

constexpr int32_t kChannels = 11;
constexpr int32_t kAuxDim = 14;
// The crop, datagen's TeacherLabeler.SY/SZ/SX.
constexpr int32_t kDimY = 7;
constexpr int32_t kDimZ = 13;
constexpr int32_t kDimX = 13;
constexpr int32_t kCells = kDimY * kDimZ * kDimX;  // 1183

constexpr size_t kXBytes = static_cast<size_t>(kChannels) * kCells * sizeof(float);
constexpr size_t kAuxBytes = static_cast<size_t>(kAuxDim) * sizeof(float);
constexpr size_t kMapBytes = static_cast<size_t>(kCells) * sizeof(float);

}  // namespace wire
}  // namespace fc
