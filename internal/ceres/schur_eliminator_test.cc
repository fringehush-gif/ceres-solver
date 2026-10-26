// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/schur_eliminator.h"

#include <algorithm>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>

#include "Eigen/Dense"
#include "ceres/block_random_access_dense_matrix.h"
#include "ceres/block_random_access_diagonal_matrix.h"
#include "ceres/block_random_access_sparse_matrix.h"
#include "ceres/private_schur_accumulation.h"
#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
#include <array>
#include <atomic>
#include <climits>

#include "ceres/parallel_for.h"
#include "ceres/private_schur_rhs.h"
#endif
#include "absl/container/btree_set.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/casts.h"
#include "ceres/context_impl.h"
#include "ceres/detect_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/test_util.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

// TODO(sameeragarwal): Reduce the size of these tests and redo the
// parameterization to be more efficient.

namespace ceres::internal {

class SchurEliminatorTest : public ::testing::Test {
 protected:
  void SetUpFromId(int id) {
    auto problem = CreateLinearLeastSquaresProblemFromId(id);
    ASSERT_TRUE(problem != nullptr);
    SetupHelper(problem.get());
  }

  void SetupHelper(LinearLeastSquaresProblem* problem) {
    A.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));
    b = std::move(problem->b);
    D = std::move(problem->D);

    num_eliminate_blocks = problem->num_eliminate_blocks;
    num_eliminate_cols = 0;
    const CompressedRowBlockStructure* bs = A->block_structure();

    for (int i = 0; i < num_eliminate_blocks; ++i) {
      num_eliminate_cols += bs->cols[i].size;
    }
  }

  // Compute the golden values for the reduced linear system and the
  // solution to the linear least squares problem using dense linear
  // algebra.
  void ComputeReferenceSolution(const Vector& D) {
    Matrix J;
    A->ToDenseMatrix(&J);
    VectorRef f(b.get(), J.rows());

    Matrix H = (D.cwiseProduct(D)).asDiagonal();
    H.noalias() += J.transpose() * J;

    const Vector g = J.transpose() * f;
    const int schur_size = J.cols() - num_eliminate_cols;

    lhs_expected.resize(schur_size, schur_size);
    lhs_expected.setZero();

    rhs_expected.resize(schur_size);
    rhs_expected.setZero();

    sol_expected.resize(J.cols());
    sol_expected.setZero();

    Matrix P = H.block(0, 0, num_eliminate_cols, num_eliminate_cols);
    Matrix Q = H.block(0, num_eliminate_cols, num_eliminate_cols, schur_size);
    Matrix R =
        H.block(num_eliminate_cols, num_eliminate_cols, schur_size, schur_size);
    int row = 0;
    const CompressedRowBlockStructure* bs = A->block_structure();
    for (int i = 0; i < num_eliminate_blocks; ++i) {
      const int block_size = bs->cols[i].size;
      P.block(row, row, block_size, block_size) =
          P.block(row, row, block_size, block_size)
              .llt()
              .solve(Matrix::Identity(block_size, block_size));
      row += block_size;
    }

    lhs_expected.triangularView<Eigen::Upper>() = R - Q.transpose() * P * Q;
    rhs_expected =
        g.tail(schur_size) - Q.transpose() * P * g.head(num_eliminate_cols);
    sol_expected = H.llt().solve(g);
  }

  void EliminateSolveAndCompare(const VectorRef& diagonal,
                                bool use_static_structure,
                                const double relative_tolerance) {
    const CompressedRowBlockStructure* bs = A->block_structure();
    const int num_col_blocks = bs->cols.size();
    auto blocks = Tail(bs->cols, num_col_blocks - num_eliminate_blocks);
    BlockRandomAccessDenseMatrix lhs(blocks, &context_, 1);

    const int num_cols = A->num_cols();
    const int schur_size = lhs.num_rows();

    Vector rhs(schur_size);

    LinearSolver::Options options;
    options.context = &context_;
    options.elimination_groups.push_back(num_eliminate_blocks);
    if (use_static_structure) {
      DetectStructure(*bs,
                      num_eliminate_blocks,
                      &options.row_block_size,
                      &options.e_block_size,
                      &options.f_block_size);
    }

    std::unique_ptr<SchurEliminatorBase> eliminator =
        SchurEliminatorBase::Create(options);
    const bool kFullRankETE = true;
    eliminator->Init(num_eliminate_blocks, kFullRankETE, A->block_structure());
    eliminator->Eliminate(
        BlockSparseMatrixData(*A), b.get(), diagonal.data(), &lhs, rhs.data());

    MatrixRef lhs_ref(lhs.mutable_values(), lhs.num_rows(), lhs.num_cols());
    Vector reduced_sol =
        lhs_ref.selfadjointView<Eigen::Upper>().llt().solve(rhs);

    // Solution to the linear least squares problem.
    Vector sol(num_cols);
    sol.setZero();
    sol.tail(schur_size) = reduced_sol;
    eliminator->BackSubstitute(BlockSparseMatrixData(*A),
                               b.get(),
                               diagonal.data(),
                               reduced_sol.data(),
                               sol.data());

    Matrix delta = (lhs_ref - lhs_expected).selfadjointView<Eigen::Upper>();
    double diff = delta.norm();
    EXPECT_NEAR(diff / lhs_expected.norm(), 0.0, relative_tolerance);
    EXPECT_NEAR((rhs - rhs_expected).norm() / rhs_expected.norm(),
                0.0,
                relative_tolerance);
    EXPECT_NEAR((sol - sol_expected).norm() / sol_expected.norm(),
                0.0,
                relative_tolerance);
  }

  ContextImpl context_;

  std::unique_ptr<BlockSparseMatrix> A;
  std::unique_ptr<double[]> b;
  std::unique_ptr<double[]> D;
  int num_eliminate_blocks;
  int num_eliminate_cols;

  Matrix lhs_expected;
  Vector rhs_expected;
  Vector sol_expected;
};

TEST_F(SchurEliminatorTest, ScalarProblemNoRegularization) {
  SetUpFromId(2);
  Vector zero(A->num_cols());
  zero.setZero();

  ComputeReferenceSolution(VectorRef(zero.data(), A->num_cols()));
  EliminateSolveAndCompare(VectorRef(zero.data(), A->num_cols()), true, 1e-14);
  EliminateSolveAndCompare(VectorRef(zero.data(), A->num_cols()), false, 1e-14);
}

TEST_F(SchurEliminatorTest, ScalarProblemWithRegularization) {
  SetUpFromId(2);
  ComputeReferenceSolution(VectorRef(D.get(), A->num_cols()));
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), true, 1e-14);
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), false, 1e-14);
}

TEST_F(SchurEliminatorTest, VaryingFBlockSizeWithStaticStructure) {
  SetUpFromId(4);
  ComputeReferenceSolution(VectorRef(D.get(), A->num_cols()));
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), true, 1e-14);
}

TEST_F(SchurEliminatorTest, VaryingFBlockSizeWithoutStaticStructure) {
  SetUpFromId(4);
  ComputeReferenceSolution(VectorRef(D.get(), A->num_cols()));
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), false, 1e-14);
}

// Compare parallel private accumulation across matrix representations.
TEST_F(SchurEliminatorTest, PrivateParallelMatrixRepresentations) {
  for (int kThreads : {1, 3}) {
    context_.EnsureMinimumThreads(kThreads);
    // Problem 2 already has an upstream unregularized reference test. Problem 4
    // has unequal F block sizes but is underdetermined without regularization.
    for (int problem_id : {2, 4}) {
      SetUpFromId(problem_id);
      const auto* bs = A->block_structure();
      const auto blocks =
          Tail(bs->cols, bs->cols.size() - num_eliminate_blocks);
      absl::btree_set<std::pair<int, int>> pairs;
      for (int i = 0; i < blocks.size(); ++i) {
        for (int j = i; j < blocks.size(); ++j) pairs.emplace(i, j);
      }
      for (bool regularized : {false, true}) {
        if (problem_id == 4 && !regularized) continue;
        Vector diagonal = regularized
                              ? Vector(VectorRef(D.get(), A->num_cols()))
                              : Vector::Zero(A->num_cols());
        ComputeReferenceSolution(diagonal);
        for (bool static_structure : {false, true}) {
          LinearSolver::Options options;
          options.context = &context_;
          options.num_threads = kThreads;
          options.elimination_groups.push_back(num_eliminate_blocks);
          if (static_structure) {
            DetectStructure(*bs,
                            num_eliminate_blocks,
                            &options.row_block_size,
                            &options.e_block_size,
                            &options.f_block_size);
          }
          auto eliminator = SchurEliminatorBase::Create(options);
          eliminator->Init(num_eliminate_blocks, true, bs);
          for (int representation = 0; representation < 3; ++representation) {
            std::unique_ptr<BlockRandomAccessMatrix> lhs;
            if (representation == 0) {
              lhs = std::make_unique<BlockRandomAccessDenseMatrix>(
                  blocks, &context_, kThreads);
            } else if (representation == 1) {
              lhs = std::make_unique<BlockRandomAccessSparseMatrix>(
                  blocks, pairs, &context_, kThreads);
            } else {
              lhs = std::make_unique<BlockRandomAccessDiagonalMatrix>(
                  blocks, &context_, kThreads);
            }
            // Prove the parallel shape is eligible and the single-thread case
            // takes the original accumulation path before checking both
            // oracles.
            {
              PrivateSchurAccumulation eligibility;
              ASSERT_EQ(
                  eligibility.Init(
                      lhs.get(), bs->cols, num_eliminate_blocks, kThreads),
                  kThreads > 1);
            }
            for (bool with_rhs : {false, true}) {
              // Repeat with the same eliminator and matrix to expose stale
              // values/pointers and missing zeroing between Eliminate calls.
              for (int repeat = 0; repeat < 2; ++repeat) {
                SCOPED_TRACE(::testing::Message()
                             << kThreads << ":" << problem_id << ":"
                             << regularized << ":" << static_structure << ":"
                             << representation << ":" << with_rhs << ":"
                             << repeat);
                Vector rhs = Vector::Constant(lhs->num_rows(), -123.0);
                eliminator->Eliminate(BlockSparseMatrixData(*A),
                                      with_rhs ? b.get() : nullptr,
                                      regularized ? diagonal.data() : nullptr,
                                      lhs.get(),
                                      with_rhs ? rhs.data() : nullptr);
                Matrix actual = Matrix::Zero(lhs->num_rows(), lhs->num_cols());
                Matrix expected = actual;
                int row_position = 0;
                for (int i = 0; i < blocks.size(); ++i) {
                  int col_position = 0;
                  for (int j = 0; j < blocks.size(); ++j) {
                    int r, c, rs, cs;
                    CellInfo* cell = lhs->GetCell(i, j, &r, &c, &rs, &cs);
                    if (cell && i <= j) {
                      actual.block(row_position,
                                   col_position,
                                   blocks[i].size,
                                   blocks[j].size) =
                          MatrixRef(cell->values, rs, cs)
                              .block(r, c, blocks[i].size, blocks[j].size);
                      expected.block(row_position,
                                     col_position,
                                     blocks[i].size,
                                     blocks[j].size) =
                          lhs_expected.block(row_position,
                                             col_position,
                                             blocks[i].size,
                                             blocks[j].size);
                    }
                    col_position += blocks[j].size;
                  }
                  row_position += blocks[i].size;
                }
                // Ceres exposes the global upper triangle as the reduced
                // symmetric matrix. Diagonal blocks may also contain lower
                // entries, while ComputeReferenceSolution leaves those zero.
                // Compare the same defined region as the upstream oracle.
                actual.triangularView<Eigen::StrictlyLower>().setZero();
                EXPECT_LE((actual - expected).norm() / expected.norm(), 1e-13);
                if (with_rhs) {
                  EXPECT_LE((rhs - rhs_expected).norm() / rhs_expected.norm(),
                            1e-13);
                  // A diagonal preconditioner is not the full reduced system.
                  if (representation != 2) {
                    Vector z =
                        actual.selfadjointView<Eigen::Upper>().llt().solve(rhs);
                    Vector solution = Vector::Zero(A->num_cols());
                    solution.tail(z.size()) = z;
                    eliminator->BackSubstitute(
                        BlockSparseMatrixData(*A),
                        b.get(),
                        regularized ? diagonal.data() : nullptr,
                        z.data(),
                        solution.data());
                    EXPECT_LE(
                        (solution - sol_expected).norm() / sol_expected.norm(),
                        1e-13);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Capacity rejection must not write the caller's target or allocate worker
// values.
TEST_F(SchurEliminatorTest, PrivateCapacityFallbackPreservesTarget) {
  class SentinelMatrix final : public BlockRandomAccessMatrix {
   public:
    explicit SentinelMatrix(bool present) : cell(&value), present(present) {}
    CellInfo* GetCell(
        int, int, int* row, int* col, int* rows, int* cols) final {
      ++calls;
      if (!present) return nullptr;
      *row = *col = 0;
      *rows = *cols = 1 << 30;
      return &cell;
    }
    void SetZero() final {
      ADD_FAILURE() << "eligibility must not zero target";
    }
    int num_rows() const final { return 1 << 30; }
    int num_cols() const final { return 1 << 30; }
    double value = 123.5;
    CellInfo cell;
    bool present;
    int calls = 0;
  };
  struct Case {
    int count, size, first, threads;
    bool present;
    int calls;
  };
  const Case cases[] = {
      {1, 1, 0, 1, true, 0},     // single-thread original path
      {1, 1, 0, 0, true, 0},     // invalid worker count
      {0, 1, 0, 2, true, 0},     // no remaining blocks
      {1, 1, -1, 2, true, 0},    // invalid elimination prefix
      {1, 1, 1, 2, true, 0},     // prefix consumes all blocks
      {1025, 1, 0, 2, true, 0},  // lookup slot ceiling before GetCell
      {1, 1, 0, 2, false, 1},    // target has no stored cells
      {1, 0, 0, 2, true, 1},     // invalid cell extent
      {1, -1, 0, 2, true, 1},
      {1, 1 << 30, 0, 2, true, 1},  // oversized cell, checked without
                                    // multiplication overflow
      {1, 256, 0, 1024, true, 1},  // worker values exceed budget on first entry
      {2, 128, 0, 1024, true, 2},  // budget exceeded after one accepted entry
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << c.count << ":" << c.size << ":" << c.first << ":"
                 << c.threads << ":" << c.present);
    std::vector<Block> blocks(c.count);
    for (auto& block : blocks) block.size = c.size;
    SentinelMatrix target(c.present);
    PrivateSchurAccumulation accumulation;
    EXPECT_FALSE(accumulation.Init(&target, blocks, c.first, c.threads));
    EXPECT_EQ(target.calls, c.calls);
    EXPECT_EQ(target.value, 123.5);
  }
}

TEST(SchurEliminatorForOneFBlock, MatchesSchurEliminator) {
  constexpr int kRowBlockSize = 2;
  constexpr int kEBlockSize = 3;
  constexpr int kFBlockSize = 6;
  constexpr int num_e_blocks = 5;

  ContextImpl context;

  auto* bs = new CompressedRowBlockStructure;
  bs->cols.resize(num_e_blocks + 1);
  int col_pos = 0;
  for (int i = 0; i < num_e_blocks; ++i) {
    bs->cols[i].position = col_pos;
    bs->cols[i].size = kEBlockSize;
    col_pos += kEBlockSize;
  }
  bs->cols.back().position = col_pos;
  bs->cols.back().size = kFBlockSize;

  bs->rows.resize(2 * num_e_blocks + 1);
  int row_pos = 0;
  int cell_pos = 0;
  for (int i = 0; i < num_e_blocks; ++i) {
    {
      auto& row = bs->rows[2 * i];
      row.block.position = row_pos;
      row.block.size = kRowBlockSize;
      row_pos += kRowBlockSize;
      auto& cells = row.cells;
      cells.resize(2);
      cells[0].block_id = i;
      cells[0].position = cell_pos;
      cell_pos += kRowBlockSize * kEBlockSize;
      cells[1].block_id = num_e_blocks;
      cells[1].position = cell_pos;
      cell_pos += kRowBlockSize * kFBlockSize;
    }
    {
      auto& row = bs->rows[2 * i + 1];
      row.block.position = row_pos;
      row.block.size = kRowBlockSize;
      row_pos += kRowBlockSize;
      auto& cells = row.cells;
      cells.resize(1);
      cells[0].block_id = i;
      cells[0].position = cell_pos;
      cell_pos += kRowBlockSize * kEBlockSize;
    }
  }

  {
    auto& row = bs->rows.back();
    row.block.position = row_pos;
    row.block.size = kEBlockSize;
    row_pos += kRowBlockSize;
    auto& cells = row.cells;
    cells.resize(1);
    cells[0].block_id = num_e_blocks;
    cells[0].position = cell_pos;
    cell_pos += kEBlockSize * kEBlockSize;
  }

  BlockSparseMatrix matrix(bs);
  double* values = matrix.mutable_values();
  std::mt19937 prng;
  std::normal_distribution<> standard_normal;
  std::generate_n(values, matrix.num_nonzeros(), [&prng, &standard_normal] {
    return standard_normal(prng);
  });

  Vector b(matrix.num_rows());
  b.setRandom();

  Vector diagonal(matrix.num_cols());
  diagonal.setOnes();

  std::vector<Block> blocks;
  blocks.emplace_back(kFBlockSize, 0);
  BlockRandomAccessDenseMatrix actual_lhs(blocks, &context, 1);
  BlockRandomAccessDenseMatrix expected_lhs(blocks, &context, 1);
  Vector actual_rhs(kFBlockSize);
  Vector expected_rhs(kFBlockSize);

  Vector f_sol(kFBlockSize);
  f_sol.setRandom();
  Vector actual_e_sol(num_e_blocks * kEBlockSize);
  actual_e_sol.setZero();
  Vector expected_e_sol(num_e_blocks * kEBlockSize);
  expected_e_sol.setZero();

  {
    LinearSolver::Options linear_solver_options;
    linear_solver_options.e_block_size = kEBlockSize;
    linear_solver_options.row_block_size = kRowBlockSize;
    linear_solver_options.f_block_size = kFBlockSize;
    linear_solver_options.context = &context;
    std::unique_ptr<SchurEliminatorBase> eliminator(
        SchurEliminatorBase::Create(linear_solver_options));
    eliminator->Init(num_e_blocks, true, matrix.block_structure());
    eliminator->Eliminate(BlockSparseMatrixData(matrix),
                          b.data(),
                          diagonal.data(),
                          &expected_lhs,
                          expected_rhs.data());
    eliminator->BackSubstitute(BlockSparseMatrixData(matrix),
                               b.data(),
                               diagonal.data(),
                               f_sol.data(),
                               actual_e_sol.data());
  }

  {
    SchurEliminatorForOneFBlock<2, 3, 6> eliminator;
    eliminator.Init(num_e_blocks, true, matrix.block_structure());
    eliminator.Eliminate(BlockSparseMatrixData(matrix),
                         b.data(),
                         diagonal.data(),
                         &actual_lhs,
                         actual_rhs.data());
    eliminator.BackSubstitute(BlockSparseMatrixData(matrix),
                              b.data(),
                              diagonal.data(),
                              f_sol.data(),
                              expected_e_sol.data());
  }
  ConstMatrixRef actual_lhsref(
      actual_lhs.values(), actual_lhs.num_cols(), actual_lhs.num_cols());
  ConstMatrixRef expected_lhsref(
      expected_lhs.values(), actual_lhs.num_cols(), actual_lhs.num_cols());

  EXPECT_NEAR((actual_lhsref - expected_lhsref).norm() / expected_lhsref.norm(),
              0.0,
              1e-12)
      << "expected: \n"
      << expected_lhsref << "\nactual: \n"
      << actual_lhsref;

  EXPECT_NEAR(
      (actual_rhs - expected_rhs).norm() / expected_rhs.norm(), 0.0, 1e-12)
      << "expected: \n"
      << expected_rhs << "\nactual: \n"
      << actual_rhs;

  EXPECT_NEAR((actual_e_sol - expected_e_sol).norm() / expected_e_sol.norm(),
              0.0,
              1e-12)
      << "expected: \n"
      << expected_e_sol << "\nactual: \n"
      << actual_e_sol;
}

#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
// Exercise RHS admission and reduction when the RHS path is enabled.
TEST(PrivateSchurRhs, BudgetOverflowAndTransaction) {
  size_t bytes = 999;
  for (const auto& shape :
       std::vector<std::pair<int, int>>{{0, 3},
                                        {-1, 3},
                                        {3, 0},
                                        {3, 1},
                                        {INT_MAX, INT_MAX},
                                        {INT_MAX, 2},
                                        {32 * 1024 * 1024 / 8, 2}}) {
    PrivateSchurRhs rhs;
    EXPECT_FALSE(
        PrivateSchurRhs::RequiredBytes(shape.first, shape.second, &bytes));
    EXPECT_EQ(bytes, 0);
    EXPECT_FALSE(rhs.Init(shape.first, shape.second));
    EXPECT_EQ(rhs.Worker(0), nullptr);
  }
  EXPECT_FALSE(PrivateSchurRhs::RequiredBytes(1, 3, nullptr));
  PrivateSchurRhs rhs;
  ASSERT_TRUE(rhs.Init(3, 3));
  double* first = rhs.Worker(0);
  first[0] = 7.0;
  EXPECT_FALSE(rhs.Init(INT_MAX, INT_MAX));
  EXPECT_EQ(rhs.Worker(0), first);
  EXPECT_EQ(first[0], 7.0);
  EXPECT_EQ(rhs.Worker(-1), nullptr);
  EXPECT_EQ(rhs.Worker(3), nullptr);
}

TEST(PrivateSchurRhs, WorkerOwnershipLifetimeAndFixedReduction) {
  ContextImpl context;
  context.EnsureMinimumThreads(3);
  PrivateSchurRhs rhs;
  ASSERT_TRUE(rhs.Init(2, 3));
  std::array<double*, 3> addresses;
  for (int worker = 0; worker < 3; ++worker) {
    addresses[worker] = rhs.Worker(worker);
    ASSERT_NE(addresses[worker], nullptr);
    EXPECT_EQ(addresses[worker][0], 0.0);
    EXPECT_EQ(addresses[worker][1], 0.0);
    for (int previous = 0; previous < worker; ++previous) {
      EXPECT_NE(addresses[worker], addresses[previous]);
      EXPECT_NE(addresses[worker], addresses[previous] + 1);
    }
  }
  std::array<std::atomic<bool>, 3> active{};
  for (auto& flag : active) flag.store(false);
  ParallelFor(&context, 0, 128, 3, [&](int worker, int) {
    ASSERT_GE(worker, 0);
    ASSERT_LT(worker, 3);
    EXPECT_FALSE(active[worker].exchange(true));
    double* values = rhs.Worker(worker);
    ASSERT_EQ(values, addresses[worker]);
    values[0] += 1.0;
    values[1] += 2.0;
    active[worker].store(false);
  });
  double target[] = {7.0, 11.0};
  rhs.Reduce(target);
  EXPECT_EQ(target[0], 135.0);
  EXPECT_EQ(target[1], 267.0);
  // Specify the new grouping/order boundary explicitly; do not promise the
  // locked path's interleaving or bitwise equality across worker schedules.
  addresses[0][0] = 1e16;
  addresses[1][0] = -1e16;
  addresses[2][0] = 1.0;
  target[0] = 0.0;
  rhs.Reduce(target);
  EXPECT_EQ(target[0], 1.0);
  PrivateSchurRhs next_call;
  ASSERT_TRUE(next_call.Init(2, 3));
  EXPECT_EQ(next_call.Worker(0)[0], 0.0);
}

TEST(PrivateSchurRhs, ReservedMatrixBudgetFallbackPreservesTarget) {
  class Sentinel final : public BlockRandomAccessMatrix {
   public:
    Sentinel() { cell.values = &value; }
    CellInfo* GetCell(int, int, int* row, int* col, int* rs, int* cs) final {
      ++calls;
      *row = *col = 0;
      *rs = *cs = 1;
      return &cell;
    }
    void SetZero() final { value = 0; }
    int num_rows() const final { return 1; }
    int num_cols() const final { return 1; }
    double value = 123.5;
    int calls = 0;
    CellInfo cell;
  } target;
  std::vector<Block> blocks(1);
  blocks[0].size = 1;
  PrivateSchurAccumulation matrix;
  EXPECT_FALSE(matrix.Init(&target, blocks, 0, 3, 256 * 1024 * 1024));
  EXPECT_EQ(target.calls, 0);
  EXPECT_EQ(target.value, 123.5);
  size_t reserved;
  ASSERT_TRUE(PrivateSchurRhs::RequiredBytes(1, 3, &reserved));
  PrivateSchurAccumulation eligible;
  EXPECT_TRUE(eligible.Init(&target, blocks, 0, 3, reserved));
  EXPECT_EQ(target.value, 123.5);
}

TEST(PrivateSchurRhs, SerialNoEAndRepeatedEliminationOracle) {
  ContextImpl context;
  context.EnsureMinimumThreads(3);
  auto* bs = new CompressedRowBlockStructure;
  for (int column = 0; column < 4; ++column) {
    bs->cols.emplace_back(1, column);
  }
  const int columns[5][2] = {{0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  for (int row = 0; row < 5; ++row) {
    CompressedRow value;
    value.block = Block(1, row);
    for (int cell = 0; cell < 2; ++cell) {
      Cell entry;
      entry.block_id = columns[row][cell];
      entry.position = row * 2 + cell;
      value.cells.push_back(entry);
    }
    bs->rows.push_back(value);
  }
  BlockSparseMatrix input(bs);
  const double values[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::copy_n(values, 10, input.mutable_values());
  const double b[] = {1, 2, 3, 4, 5};
  const double diagonal[] = {1, 1, 1, 1};
  std::vector<Block> retained{Block(1, 0), Block(1, 1)};
  for (int workers : {1, 3}) {
    LinearSolver::Options options;
    options.context = &context;
    options.num_threads = workers;
    options.elimination_groups.push_back(2);
    auto eliminator = SchurEliminatorBase::Create(options);
    eliminator->Init(2, true, bs);
    BlockRandomAccessDenseMatrix lhs(retained, &context, workers);
    for (bool regularized : {false, true}) {
      const double e0 = regularized ? 21.0 : 20.0;
      const double e1 = regularized ? 101.0 : 100.0;
      for (int repeat = 0; repeat < 2; ++repeat) {
        double rhs[] = {-123, -123};
        eliminator->Eliminate(BlockSparseMatrixData(input),
                              b,
                              regularized ? diagonal : nullptr,
                              &lhs,
                              rhs);
        EXPECT_NEAR(rhs[0],
                    3.0 * (1.0 - 20.0 / e0) + 7.0 * (3.0 - 300.0 / e1) + 50.0,
                    1e-12);
        EXPECT_NEAR(rhs[1],
                    5.0 * (2.0 - 40.0 / e0) + 9.0 * (4.0 - 400.0 / e1) + 55.0,
                    1e-12);
        // Independent upper-triangle matrix oracle includes serial NoE and
        // damping.
        const double damping = regularized ? 1.0 : 0.0;
        EXPECT_NEAR(
            lhs.values()[0], 158.0 + damping - 36.0 / e0 - 1764.0 / e1, 1e-12);
        EXPECT_NEAR(lhs.values()[1], 110.0 - 120.0 / e0 - 3024.0 / e1, 1e-12);
        EXPECT_NEAR(
            lhs.values()[3], 227.0 + damping - 400.0 / e0 - 5184.0 / e1, 1e-12);
        // Null RHS stays on the old path and still computes the matrix.
        eliminator->Eliminate(BlockSparseMatrixData(input),
                              nullptr,
                              regularized ? diagonal : nullptr,
                              &lhs,
                              nullptr);
      }
    }
  }
}

#endif

namespace {
class UpperTriangleSentinel final : public BlockRandomAccessMatrix {
 public:
  // mode0 stores all cells;1 stores upper only;2 stores diagonal only;
  // mode3 stores lower only;4 stores none. Each cell is an offset submatrix.
  explicit UpperTriangleSentinel(int mode) : mode_(mode) {
    for (int k = 0; k < 9; ++k) {
      values_[k].assign(42, 17.25);
      cells_[k].values = values_[k].data();
    }
  }
  CellInfo* GetCell(int i, int j, int* r, int* c, int* rs, int* cs) final {
    ++calls;
    if ((mode_ == 1 && i > j) || (mode_ == 2 && i != j) ||
        (mode_ == 3 && i <= j) || mode_ == 4)
      return nullptr;
    *r = 1;
    *c = 2;
    *rs = 6;
    *cs = 7;
    return &cells_[3 * i + j];
  }
  void SetZero() final { ADD_FAILURE() << "helper must not zero target"; }
  int num_rows() const final { return 6; }
  int num_cols() const final { return 6; }
  const std::vector<double>& values(int i, int j) const {
    return values_[3 * i + j];
  }
  int calls = 0;

 private:
  int mode_;
  std::vector<double> values_[9];
  CellInfo cells_[9];
};
}  // namespace

TEST(PrivateSchurUpperTriangle, SubmatrixAndWorkerZeroing) {
  const std::vector<Block> blocks = {{11, 0}, {1, 11}, {2, 12}, {3, 14}};
  UpperTriangleSentinel target(0);
  PrivateSchurAccumulation accumulation;
  ASSERT_TRUE(accumulation.Init(&target, blocks, 1, 3));
  EXPECT_EQ(target.calls, 9);  // Full original admission scan is retained.
  for (int t = 0; t < 3; ++t) {
    PrivateSchurAccumulation::WorkerMatrix worker(&accumulation, t, 6);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        int r, c, rs, cs;
        auto* cell = worker.GetCell(i, j, &r, &c, &rs, &cs);
        if (i > j) {
          EXPECT_EQ(cell, nullptr);
          continue;
        }
        ASSERT_NE(cell, nullptr);
        EXPECT_EQ(r, 0);
        EXPECT_EQ(c, 0);
        EXPECT_EQ(rs, i + 1);
        EXPECT_EQ(cs, j + 1);
        for (int row = 0; row < rs; ++row) {
          for (int col = 0; col < cs; ++col) {
            const int at = row * cs + col;
            EXPECT_EQ(cell->values[at], 0.0);
            cell->values[at] =
                (t + 1) * (1 + 3 * i + j) + row * 0.125 + col * 0.0625;
          }
        }
      }
    }
  }
  // Clear only worker1. Other worker contributions and target damping survive.
  PrivateSchurAccumulation::WorkerMatrix middle(&accumulation, 1, 6);
  middle.SetZero();
  accumulation.Reduce();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
          double expected = 17.25;
          if (i <= j && row >= 1 && row < i + 2 && col >= 2 && col < j + 3) {
            expected += 4 * (1 + 3 * i + j) + 2 * (row - 1) * 0.125 +
                        2 * (col - 2) * 0.0625;
          }
          EXPECT_EQ(target.values(i, j)[7 * row + col], expected)
              << i << ":" << j << ":" << row << ":" << col;
        }
      }
    }
  }
}

TEST(PrivateSchurUpperTriangle, MissingAndLowerOnlyCells) {
  const std::vector<Block> blocks = {{1, 0}, {2, 1}, {3, 3}};
  for (int mode : {1, 2, 3, 4}) {
    SCOPED_TRACE(mode);
    UpperTriangleSentinel target(mode);
    PrivateSchurAccumulation accumulation;
    EXPECT_EQ(accumulation.Init(&target, blocks, 0, 3), mode != 4);
    EXPECT_EQ(target.calls, 9);
    if (mode == 4) continue;
    PrivateSchurAccumulation::WorkerMatrix worker(&accumulation, 0, 6);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        int r, c, rs, cs;
        auto* cell = worker.GetCell(i, j, &r, &c, &rs, &cs);
        EXPECT_EQ(cell != nullptr,
                  i <= j && mode != 3 && (mode != 2 || i == j));
      }
    }
    worker.SetZero();
    accumulation.Reduce();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (double value : target.values(i, j)) EXPECT_EQ(value, 17.25);
  }
}

#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
TEST(PrivateSchurUpperTriangle, PreservesFullAdmissionBudget) {
  class ShapeOnly final : public BlockRandomAccessMatrix {
   public:
    CellInfo* GetCell(int, int, int* r, int* c, int* rs, int* cs) final {
      ++calls;
      *r = *c = 0;
      *rs = *cs = 1182;
      return &cell;
    }
    void SetZero() final { ADD_FAILURE(); }
    int num_rows() const final { return 1182; }
    int num_cols() const final { return 1182; }
    double sentinel = 123.5;
    CellInfo cell{&sentinel};
    int calls = 0;
  } target;
  const std::vector<Block> blocks = {{591, 0}, {591, 591}};
  PrivateSchurAccumulation accumulation;
  // Three upper cells alone fit, but the accepted full four-cell admission
  // with this reservation must still reject before worker allocation.
  EXPECT_FALSE(accumulation.Init(&target, blocks, 0, 24, 1024 * 1024));
  EXPECT_EQ(target.calls, 4);
  EXPECT_EQ(target.sentinel, 123.5);
}
#endif

TEST(PrivateSchurUpperTriangle, FirstSeenBufferOrderStillUsesUpperOutput) {
  auto* bs = new CompressedRowBlockStructure;
  bs->cols = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
  bs->rows.resize(3);
  const int order[] = {3, 1, 2};
  for (int r = 0; r < 3; ++r) {
    bs->rows[r].block = Block(1, r);
    bs->rows[r].cells = {{0, 2 * r}, {order[r], 2 * r + 1}};
  }
  BlockSparseMatrix A(bs);
  const double values[] = {1, 4, 2, 5, 3, 6};
  std::copy(values, values + 6, A.mutable_values());
  const double b[] = {7, 8, 9};
  const double D[] = {1, 0, 0, 0};
  const double ef[] = {10, 18, 4}, ff[] = {25, 36, 16}, fb[] = {40, 54, 28};
  ContextImpl context;
  context.EnsureMinimumThreads(3);
  for (int threads : {1, 3})
    for (bool damped : {false, true}) {
      LinearSolver::Options options;
      options.context = &context;
      options.num_threads = threads;
      auto eliminator = SchurEliminatorBase::Create(options);
      eliminator->Init(1, true, A.block_structure());
      BlockRandomAccessDenseMatrix lhs(
          {{1, 0}, {1, 1}, {1, 2}}, &context, threads);
      double rhs[] = {-123, -123, -123};
      eliminator->Eliminate(
          BlockSparseMatrixData(A), b, damped ? D : nullptr, &lhs, rhs);
      const double denominator = damped ? 15.0 : 14.0;
      for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(rhs[i], fb[i] - ef[i] * 50.0 / denominator, 1e-12);
        for (int j = i; j < 3; ++j)
          EXPECT_NEAR(lhs.values()[3 * i + j],
                      (i == j ? ff[i] : 0.0) - ef[i] * ef[j] / denominator,
                      1e-12);
      }
    }
}

namespace {

class AllocationFaultMatrix final : public BlockRandomAccessMatrix {
 public:
  CellInfo* GetCell(int i, int j, int* r, int* c, int* rs, int* cs) final {
    if (++calls == fail_on_call) {
      if (throw_other)
        throw std::runtime_error("non-allocation target failure");
      throw std::bad_alloc();
    }
    return target.GetCell(i, j, r, c, rs, cs);
  }
  void SetZero() final { ADD_FAILURE() << "Init must not zero target values"; }
  int num_rows() const final { return target.num_rows(); }
  int num_cols() const final { return target.num_cols(); }

  UpperTriangleSentinel target{0};
  int calls = 0;
  int fail_on_call = 0;
  bool throw_other = false;
};

bool InitAllocationFault(PrivateSchurAccumulation* accumulation,
                         AllocationFaultMatrix* target,
                         const std::vector<Block>& blocks
#if defined(CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE
                         ,
                         PrivateSchurMatrixScratch* scratch = nullptr
#endif
) {
  return accumulation->Init(target,
                            blocks,
                            0,
                            3
#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
                            ,
                            0
#endif
#if defined(CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE
                            ,
                            scratch
#endif
  );
}

void ExpectAllocationTargetUnchanged(const AllocationFaultMatrix& target) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (double value : target.target.values(i, j)) {
        EXPECT_EQ(value, 17.25);
      }
    }
  }
}

}  // namespace

TEST(PrivateSchurAllocationFallback, ClearsPartialStateAndPreservesTarget) {
  const std::vector<Block> blocks = {{1, 0}, {2, 1}, {3, 3}};
  for (const int fail_on_call : {1, 3, 9}) {
    SCOPED_TRACE(fail_on_call);
    AllocationFaultMatrix target;
    target.fail_on_call = fail_on_call;
    PrivateSchurAccumulation accumulation;
    bool admitted = true;
    EXPECT_NO_THROW(admitted =
                        InitAllocationFault(&accumulation, &target, blocks));
    EXPECT_FALSE(admitted);
    EXPECT_EQ(target.calls, fail_on_call);
    ExpectAllocationTargetUnchanged(target);

    // No lookup/cell/entry state may survive the failed initialization.
    PrivateSchurAccumulation::WorkerMatrix worker(&accumulation, 0, 6);
    int r, c, rs, cs;
    EXPECT_EQ(worker.GetCell(0, 0, &r, &c, &rs, &cs), nullptr);
    accumulation.Reduce();
    ExpectAllocationTargetUnchanged(target);

    // A new successful Init on the same helper must start from clean state.
    target.fail_on_call = 0;
    target.calls = 0;
    ASSERT_TRUE(InitAllocationFault(&accumulation, &target, blocks));
    EXPECT_EQ(target.calls, 9);
    for (int t = 0; t < 3; ++t) {
      PrivateSchurAccumulation::WorkerMatrix current(&accumulation, t, 6);
      auto* cell = current.GetCell(0, 0, &r, &c, &rs, &cs);
      ASSERT_NE(cell, nullptr);
      EXPECT_EQ(cell->values[0], 0.0);
    }
    accumulation.Reduce();
    ExpectAllocationTargetUnchanged(target);
  }
}

TEST(PrivateSchurAllocationFallback, DoesNotSwallowOtherExceptions) {
  AllocationFaultMatrix target;
  target.fail_on_call = 3;
  target.throw_other = true;
  PrivateSchurAccumulation accumulation;
  EXPECT_THROW(
      InitAllocationFault(&accumulation, &target, {{1, 0}, {2, 1}, {3, 3}}),
      std::runtime_error);
  ExpectAllocationTargetUnchanged(target);
}

#if defined(CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE
TEST(PrivateSchurAllocationFallback, ReleasesBorrowedNumericCapacity) {
  const std::vector<Block> blocks = {{1, 0}, {2, 1}, {3, 3}};
  PrivateSchurMatrixScratch scratch;
  AllocationFaultMatrix seed;
  {
    PrivateSchurAccumulation accumulation;
    ASSERT_TRUE(InitAllocationFault(&accumulation, &seed, blocks, &scratch));
  }
  ASSERT_GT(scratch.capacity_scalars(), 0);
  ASSERT_NE(scratch.data(), nullptr);
  ExpectAllocationTargetUnchanged(seed);

  AllocationFaultMatrix fault;
  fault.fail_on_call = 3;
  {
    PrivateSchurAccumulation accumulation;
    EXPECT_FALSE(InitAllocationFault(&accumulation, &fault, blocks, &scratch));
    EXPECT_EQ(scratch.capacity_scalars(), 0);
    EXPECT_EQ(scratch.data(), nullptr);
    PrivateSchurAccumulation::WorkerMatrix worker(&accumulation, 0, 6);
    int r, c, rs, cs;
    EXPECT_EQ(worker.GetCell(0, 0, &r, &c, &rs, &cs), nullptr);
    accumulation.Reduce();
  }
  EXPECT_EQ(scratch.capacity_scalars(), 0);
  EXPECT_EQ(scratch.data(), nullptr);
  ExpectAllocationTargetUnchanged(fault);
}
#endif

#if defined(CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE
namespace {
// Pass the existing RHS reservation only when its original feature is enabled.
bool InitReuse(PrivateSchurAccumulation* accumulation,
               PrivateSchurMatrixScratch* scratch,
               BlockRandomAccessMatrix* target,
               const std::vector<Block>& blocks,
               int workers,
               size_t reserved = 0) {
#if !defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) || \
    !CERES_PRIVATE_SCHUR_RHS_CANDIDATE
  (void)reserved;
#endif
  return accumulation->Init(target,
                            blocks,
                            0,
                            workers,
#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
                            reserved,
#endif
                            scratch);
}
}  // namespace

TEST(PrivateSchurMatrixReuse, RepeatedTargetsDirtyResetAndInactiveWorkers) {
  PrivateSchurMatrixScratch scratch;
  const std::vector<Block> blocks = {{1, 0}, {2, 1}, {3, 3}};
  double* same_shape_values = nullptr;
  for (int mode : {0, 0, 1, 2}) {
    UpperTriangleSentinel target(mode);
    {
      const size_t retained_capacity = scratch.capacity_scalars();
      const double* retained_values = scratch.data();
      if (same_shape_values != nullptr) {
        EXPECT_EQ(retained_values, same_shape_values);
        EXPECT_EQ(retained_capacity, 75);
      }
      PrivateSchurAccumulation accumulation;
      ASSERT_TRUE(InitReuse(&accumulation, &scratch, &target, blocks, 3));
      // Ownership/capacity transfers out of scratch; pointer equality alone
      // would also permit an allocator free/reallocate coincidence.
      EXPECT_EQ(scratch.capacity_scalars(), 0);
      EXPECT_EQ(scratch.data(), nullptr);
      EXPECT_EQ(target.calls, 9);
      for (int thread = 0; thread < 3; ++thread) {
        PrivateSchurAccumulation::WorkerMatrix worker(&accumulation, thread, 6);
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            int r, c, rs, cs;
            auto* cell = worker.GetCell(i, j, &r, &c, &rs, &cs);
            if (i > j || (mode == 2 && i != j)) {
              EXPECT_EQ(cell, nullptr);
              continue;
            }
            ASSERT_NE(cell, nullptr);
            for (int at = 0; at < rs * cs; ++at) {
              EXPECT_EQ(cell->values[at], 0.0);
              // Only worker0 contributes. The reduced inactive workers must
              // have been reset despite their dirty previous-call contents.
              cell->values[at] = thread == 0 ? 1.0 : 0.0;
            }
            if (thread == 0 && i == 0 && j == 0 && mode != 2) {
              if (same_shape_values != nullptr)
                EXPECT_EQ(cell->values, same_shape_values);
              same_shape_values = cell->values;
            }
          }
      }
      accumulation.Reduce();
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 7; ++c) {
              const bool live = i <= j && (mode != 2 || i == j) && r >= 1 &&
                                r < i + 2 && c >= 2 && c < j + 3;
              EXPECT_EQ(target.values(i, j)[7 * r + c],
                        17.25 + (live ? 1.0 : 0.0));
            }
      // Deliberately leave ALL buffers dirty, including inactive workers.
      for (int thread = 0; thread < 3; ++thread) {
        PrivateSchurAccumulation::WorkerMatrix worker(&accumulation, thread, 6);
        for (int i = 0; i < 3; ++i)
          for (int j = i; j < 3; ++j) {
            int r, c, rs, cs;
            auto* cell = worker.GetCell(i, j, &r, &c, &rs, &cs);
            if (cell) std::fill_n(cell->values, rs * cs, 99.0);
          }
      }
    }  // All target/cell/layout state dies before target and next invocation.
    EXPECT_GT(scratch.capacity_scalars(), 0);
  }
}

TEST(PrivateSchurMatrixReuse, RepeatedInitAndRefusalResetMetadata) {
  PrivateSchurMatrixScratch scratch;
  UpperTriangleSentinel first(0), second(2);
  const std::vector<Block> blocks = {{1, 0}, {2, 1}, {3, 3}};
  {
    PrivateSchurAccumulation accumulation;
    ASSERT_TRUE(InitReuse(&accumulation, &scratch, &first, blocks, 3));
    PrivateSchurAccumulation::WorkerMatrix old(&accumulation, 0, 6);
    int r, c, rs, cs;
    auto* value = old.GetCell(0, 1, &r, &c, &rs, &cs);
    ASSERT_NE(value, nullptr);
    std::fill_n(value->values, rs * cs, 999.0);
    ASSERT_TRUE(InitReuse(&accumulation, &scratch, &second, blocks, 3));
    EXPECT_EQ(first.calls, 9);
    EXPECT_EQ(second.calls, 9);
    PrivateSchurAccumulation::WorkerMatrix current(&accumulation, 0, 6);
    EXPECT_EQ(current.GetCell(0, 1, &r, &c, &rs, &cs), nullptr);
    accumulation.Reduce();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (double x : first.values(i, j)) EXPECT_EQ(x, 17.25);
    EXPECT_FALSE(InitReuse(&accumulation, &scratch, &second, blocks, 1));
    EXPECT_EQ(second.calls, 9);  // Original serial fallback performs no scan.
    EXPECT_EQ(current.GetCell(0, 0, &r, &c, &rs, &cs), nullptr);
  }
  EXPECT_EQ(scratch.capacity_scalars(), 0);
}

TEST(PrivateSchurMatrixReuse, ShapeThreadGrowthShrinkAndRelease) {
  PrivateSchurMatrixScratch scratch;
  // Sentinel backing is6x7 with origin(1,2): each3x3 block fits rows1..3,
  // columns2..4. No test writes beyond that declared submatrix.
  const std::vector<Block> large = {{3, 0}, {3, 3}, {3, 6}};
  const std::vector<Block> small = {{1, 0}, {1, 1}, {1, 2}};
  auto call = [&](const std::vector<Block>& blocks,
                  int workers,
                  size_t reserved = 0) {
    UpperTriangleSentinel target(0);
    PrivateSchurAccumulation accumulation;
    ASSERT_TRUE(
        InitReuse(&accumulation, &scratch, &target, blocks, workers, reserved));
    EXPECT_EQ(target.calls, 9);
    accumulation.Reduce();  // All configured workers must start at exact zero.
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (double x : target.values(i, j)) EXPECT_EQ(x, 17.25);
  };
  call(large, 3);
  EXPECT_EQ(scratch.capacity_scalars(), 162);
#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
  // Small-shape admission costs 2268 bytes on the tested 40/48-byte
  // Entry/CellInfo ABI; retaining 1296 numeric bytes cannot fit with metadata.
  // This exercises trim BEFORE metadata growth, followed by SAME
  // scan/admission.
  call(small, 3, 256 * 1024 * 1024 - 3000);
  EXPECT_EQ(scratch.capacity_scalars(), 18);
#endif
  call(small, 4);
  EXPECT_GE(scratch.capacity_scalars(), 24);
  call(small, 2);
  EXPECT_GE(scratch.capacity_scalars(), 12);
  call(large, 4);
  EXPECT_EQ(scratch.capacity_scalars(), 216);
  scratch
      .Release();  // Same operation used by structural SchurEliminator::Init.
  EXPECT_EQ(scratch.capacity_scalars(), 0);
}

#if defined(CERES_PRIVATE_SCHUR_RHS_CANDIDATE) && \
    CERES_PRIVATE_SCHUR_RHS_CANDIDATE
TEST(PrivateSchurMatrixReuse, OriginalBudgetFallbackDropsUnusedCache) {
  PrivateSchurMatrixScratch scratch;
  {
    UpperTriangleSentinel target(0);
    PrivateSchurAccumulation accumulation;
    ASSERT_TRUE(InitReuse(
        &accumulation, &scratch, &target, {{1, 0}, {1, 1}, {1, 2}}, 3));
  }
  ASSERT_GT(scratch.capacity_scalars(), 0);
  class ShapeOnly final : public BlockRandomAccessMatrix {
   public:
    CellInfo* GetCell(int, int, int* r, int* c, int* rs, int* cs) final {
      ++calls;
      *r = *c = 0;
      *rs = *cs = 1182;
      return &cell;
    }
    void SetZero() final { ADD_FAILURE(); }
    int num_rows() const final { return 1182; }
    int num_cols() const final { return 1182; }
    double sentinel = 123.5;
    CellInfo cell{&sentinel};
    int calls = 0;
  } target;
  {
    PrivateSchurAccumulation accumulation;
    EXPECT_FALSE(InitReuse(&accumulation,
                           &scratch,
                           &target,
                           {{591, 0}, {591, 591}},
                           24,
                           1024 * 1024));
    EXPECT_EQ(target.calls, 4);
    EXPECT_EQ(target.sentinel, 123.5);
    EXPECT_EQ(scratch.capacity_scalars(), 0);
  }
  EXPECT_EQ(scratch.capacity_scalars(), 0);
}
#endif
#endif

}  // namespace ceres::internal
