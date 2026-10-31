// SPDX-FileCopyrightText: 2026 fringehush-gif
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CERES_INTERNAL_PRIVATE_SCHUR_ACCUMULATION_H_
#define CERES_INTERNAL_PRIVATE_SCHUR_ACCUMULATION_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

#include "ceres/block_random_access_matrix.h"
#include "ceres/block_structure.h"

namespace ceres::internal {

// Owns upper block-triangle worker values for a single Eliminate invocation.
// Schur callers request i <= j; full diagonal blocks are retained. Admission
// still accounts for every target cell, including lower cells, exactly as
// before. Cell addresses never escape that invocation. Source cells may be
// submatrices of a larger allocation (dense/diagonal implementations also use
// this API).
class CERES_NO_EXPORT PrivateSchurAccumulation {
 public:
  bool Init(BlockRandomAccessMatrix* target,
            const std::vector<Block>& blocks,
            int first_block,
            int num_threads
#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
            ,
            size_t reserved_bytes = 0
#endif
            ) try {
    constexpr size_t kMaxLookupSlots = 1024 * 1024;
#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
    constexpr size_t kAggregateMaxBytes = 256 * 1024 * 1024;
    if (reserved_bytes >= kAggregateMaxBytes) return false;
    const size_t kMaxBytes = kAggregateMaxBytes - reserved_bytes;
#else
    constexpr size_t kMaxBytes = 256 * 1024 * 1024;
#endif
    if (num_threads <= 1 || first_block < 0 ||
        static_cast<size_t>(first_block) >= blocks.size()) {
      return false;
    }
    num_blocks_ = blocks.size() - first_block;
    if (num_blocks_ > kMaxLookupSlots / num_blocks_) {
      return false;
    }
    lookup_.assign(num_blocks_ * num_blocks_, -1);
    const size_t lookup_bytes = lookup_.size() * sizeof(int);
    size_t admission_scalar_count = scalar_count_;
    size_t admission_cell_count = entries_.size();
    for (size_t i = 0; i < num_blocks_; ++i) {
      for (size_t j = 0; j < num_blocks_; ++j) {
        Entry entry;
        entry.target = target->GetCell(
            i, j, &entry.row, &entry.col, &entry.row_stride, &entry.col_stride);
        if (entry.target == nullptr) {
          continue;
        }
        entry.rows = blocks[first_block + i].size;
        entry.cols = blocks[first_block + j].size;
        if (entry.rows <= 0 || entry.cols <= 0 ||
            static_cast<size_t>(entry.rows) >
                kMaxBytes / sizeof(double) / entry.cols) {
          return false;
        }
        const size_t cell_scalars = size_t(entry.rows) * entry.cols;
        if (admission_scalar_count >
            kMaxBytes / sizeof(double) - cell_scalars) {
          return false;
        }
        entry.offset = scalar_count_;
        const size_t next_scalars = admission_scalar_count + cell_scalars;
        const size_t next_cells = admission_cell_count + 1;
        // Include twice the entry bytes for vector capacity. Bound all worker
        // values and mutex-bearing CellInfo objects, not only numeric payload.
        const size_t shared_bytes =
            lookup_bytes + 2 * next_cells * sizeof(Entry);
        const size_t worker_bytes =
            next_scalars * sizeof(double) + next_cells * sizeof(CellInfo);
        if (shared_bytes >= kMaxBytes ||
            size_t(num_threads) > (kMaxBytes - shared_bytes) / worker_bytes) {
          return false;
        }
        admission_scalar_count = next_scalars;
        admission_cell_count = next_cells;
        // Preserve full-cell admission while storing only the upper triangle.
        if (j < i) {
          continue;
        }
        lookup_[i * num_blocks_ + j] = entries_.size();
        entries_.push_back(entry);
        scalar_count_ += cell_scalars;
      }
    }
    if (admission_cell_count == 0) {
      return false;
    }
    num_threads_ = num_threads;
    values_ = std::make_unique<double[]>(scalar_count_ * num_threads_);
    cells_ = std::make_unique<CellInfo[]>(entries_.size() * num_threads_);
    for (int t = 0; t < num_threads_; ++t) {
      for (size_t k = 0; k < entries_.size(); ++k) {
        cells_[size_t(t) * entries_.size() + k].values =
            values_.get() + size_t(t) * scalar_count_ + entries_[k].offset;
      }
    }
    return true;
  } catch (const std::bad_alloc&) {
    // Init has not written target values. Drop partial allocations before
    // selecting ordinary locked accumulation.
    ResetCallMetadata();
    return false;
  }

  class CERES_NO_EXPORT WorkerMatrix final : public BlockRandomAccessMatrix {
   public:
    WorkerMatrix(PrivateSchurAccumulation* owner, int thread_id, int num_rows)
        : owner_(owner), thread_id_(thread_id), num_rows_(num_rows) {}
    CellInfo* GetCell(
        int i, int j, int* r, int* c, int* row_stride, int* col_stride) final {
      if (i < 0 || j < 0 || size_t(i) >= owner_->num_blocks_ ||
          size_t(j) >= owner_->num_blocks_) {
        return nullptr;
      }
      const int index = owner_->lookup_[size_t(i) * owner_->num_blocks_ + j];
      if (index < 0) {
        return nullptr;
      }
      const Entry& entry = owner_->entries_[index];
      *r = *c = 0;
      *row_stride = entry.rows;
      *col_stride = entry.cols;
      return &owner_->cells_[size_t(thread_id_) * owner_->entries_.size() +
                             index];
    }
    void SetZero() final {
      std::fill_n(
          owner_->values_.get() + size_t(thread_id_) * owner_->scalar_count_,
          owner_->scalar_count_,
          0.0);
    }
    int num_rows() const final { return num_rows_; }
    int num_cols() const final { return num_rows_; }

   private:
    PrivateSchurAccumulation* owner_;
    int thread_id_;
    int num_rows_;
  };

  // Called only after the existing ParallelFor joins. Preserve any diagonal
  // damping already in target. Reduction changes floating-point addition order.
  void Reduce() {
    for (const Entry& entry : entries_) {
      for (int t = 0; t < num_threads_; ++t) {
        const double* src =
            values_.get() + size_t(t) * scalar_count_ + entry.offset;
        for (int r = 0; r < entry.rows; ++r) {
          for (int c = 0; c < entry.cols; ++c) {
            entry.target->values[size_t(entry.row + r) * entry.col_stride +
                                 entry.col + c] +=
                src[size_t(r) * entry.cols + c];
          }
        }
      }
    }
  }

 private:
  void ResetCallMetadata() {
    // Drop partial allocations after an allocation failure.
    cells_.reset();
    values_.reset();
    std::vector<int>().swap(lookup_);
    std::vector<Entry>().swap(entries_);
    num_blocks_ = scalar_count_ = 0;
    num_threads_ = 0;
  }
  struct CERES_NO_EXPORT Entry {
    CellInfo* target = nullptr;
    int row = 0, col = 0, row_stride = 0, col_stride = 0;
    int rows = 0, cols = 0;
    size_t offset = 0;
  };
  size_t num_blocks_ = 0;
  size_t scalar_count_ = 0;
  int num_threads_ = 0;
  std::vector<int> lookup_;
  std::vector<Entry> entries_;
  std::unique_ptr<double[]> values_;
  std::unique_ptr<CellInfo[]> cells_;
};

}  // namespace ceres::internal
#endif
