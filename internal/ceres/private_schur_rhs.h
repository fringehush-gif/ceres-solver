// SPDX-FileCopyrightText: 2026 fringehush-gif
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CERES_INTERNAL_PRIVATE_SCHUR_RHS_H_
#define CERES_INTERNAL_PRIVATE_SCHUR_RHS_H_

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

#include "absl/log/check.h"
#include "ceres/internal/export.h"

namespace ceres::internal {

// One zero-initialized vector per ParallelFor worker for one Eliminate call.
// Init never writes the target; addresses are stable through the joined work.
class CERES_NO_EXPORT PrivateSchurRhs {
 public:
  static bool RequiredBytes(int scalars, int threads, size_t* bytes) {
    if (bytes == nullptr) return false;
    *bytes = 0;
    constexpr size_t kMaxBytes = 32 * 1024 * 1024;
    constexpr size_t kStateBytes = 64;
    if (scalars <= 0 || threads <= 1 ||
        size_t(scalars) >
            (kMaxBytes - kStateBytes) / sizeof(double) / size_t(threads)) {
      return false;
    }
    *bytes = size_t(scalars) * size_t(threads) * sizeof(double) + kStateBytes;
    return true;
  }

  bool Init(int scalars, int threads) {
    size_t bytes;
    if (values_ != nullptr || !RequiredBytes(scalars, threads, &bytes)) {
      return false;
    }
    // Publish state only after allocation succeeds. Failure leaves shared-RHS
    // locking selected, even if the private matrix was already allocated.
    try {
      auto values =
          std::make_unique<double[]>(size_t(scalars) * size_t(threads));
      values_ = std::move(values);
    } catch (const std::bad_alloc&) {
      return false;
    }
    scalars_ = scalars;
    threads_ = threads;
    return true;
  }

  double* Worker(int thread_id) {
    if (values_ == nullptr || thread_id < 0 || thread_id >= threads_) {
      return nullptr;
    }
    return values_.get() + size_t(thread_id) * size_t(scalars_);
  }

  // Call only after ParallelFor joins, before unchanged serial NoE RHS work.
  // Worker grouping changes floating-point addition order from shared locks;
  // thread-index reduction is fixed, but chunk-to-worker scheduling is not.
  void Reduce(double* target) const {
    CHECK(values_ != nullptr && target != nullptr);
    for (int thread = 0; thread < threads_; ++thread) {
      const double* source = values_.get() + size_t(thread) * size_t(scalars_);
      for (int scalar = 0; scalar < scalars_; ++scalar) {
        target[scalar] += source[scalar];
      }
    }
  }

 private:
  std::unique_ptr<double[]> values_;
  int scalars_ = 0;
  int threads_ = 0;
};
static_assert(sizeof(PrivateSchurRhs) <= 64);

}  // namespace ceres::internal
#endif
