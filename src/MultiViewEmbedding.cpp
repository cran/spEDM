#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "CppStats.h"
#include "SimplexProjection.h"
#include <RcppThread.h>

// [[Rcpp::depends(RcppThread)]]

/**
 * Computes the multi-view embedding by evaluating multiple feature embeddings using simplex projection,
 * selecting top-performing embeddings, and aggregating their contributions.
 *
 * Parameters:
 *   - vectors: 2D vector where each row represents a sample and each column a feature.
 *   - target: Target spatial cross sectional series aligned with the samples in vectors.
 *   - lib_indices: Boolean flag indicating which samples to use for neighbor search.
 *   - pred_indices: Boolean flag indicating which samples to predict.
 *   - num_neighbors: Number of neighbors used for simplex projection.
 *   - top_num: Number of top-performing reconstructions to select.
 *   - threads: Number of threads used from the global pool.
 *
 * Returns:
 *   A vector<double> where each element is the predict value from selected embeddings columns.
 */
std::vector<double> MultiViewEmbedding(
    const std::vector<std::vector<double>>& vectors,
    const std::vector<double>& target,
    const std::vector<bool>& lib_indices,
    const std::vector<bool>& pred_indices,
    int num_neighbors,
    int top_num,
    int threads
) {
  // Configure threads
  size_t threads_sizet = static_cast<size_t>(std::abs(threads));
  threads_sizet = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), threads_sizet);

  // // Validate input dimensions
  // if (vectors.empty() || vectors[0].empty() ||
  //     vectors.size() != target.size() ||
  //     vectors.size() != lib_indices.size() ||
  //     vectors.size() != pred_indices.size()) {
  //   return {};
  // }

  const size_t num_samples = vectors.size();
  const size_t num_features = vectors[0].size();

  // // Verify uniform column count in vectors
  // for (const auto& row : vectors) {
  //   if (row.size() != num_features) {
  //     return {};
  //   }
  // }

  // Evaluate each feature column as a separate subset
  std::vector<std::vector<double>> pred_metrics(num_features, std::vector<double>(3));

  // // Iterate through each feature column
  // for (size_t col = 0; col < num_features; ++col) {
  //   // // Create subset matrix: each row contains a single feature value
  //   // std::vector<std::vector<double>> subset;
  //   // subset.reserve(num_samples);
  //
  //   // Create subset matrix with reserved size upon declaration
  //   std::vector<std::vector<double>> subset(num_samples, std::vector<double>(1));
  //
  //   for (size_t row = 0; row < num_samples; ++row) {
  //     subset.push_back({vectors[row][col]});
  //   }
  //
  //   // Get performance metrics for this subset
  //   auto metrics = SimplexBehavior(subset, target, lib_indices, pred_indices, num_neighbors);
  //   // if (metrics.size() != 3) continue;  // Skip invalid results
  //   pred_metrics.push_back(metrics);
  // }

  // Parallel loop over each feature column
  RcppThread::parallelFor(0, num_features, [&](size_t col) {
    // // Create subset matrix: each row contains a single feature value
    // std::vector<std::vector<double>> subset;
    // subset.reserve(num_samples);

    // Create subset matrix with reserved size upon declaration
    std::vector<std::vector<double>> subset(num_samples, std::vector<double>(1));

    for (size_t row = 0; row < num_samples; ++row) {
      subset.push_back({vectors[row][col]});
    }

    // Get performance metrics for this subset
    auto metrics = SimplexBehavior(subset, target, lib_indices, pred_indices, num_neighbors);
    // if (metrics.size() != 3) continue;  // Skip invalid results
    pred_metrics[col] = metrics;
  }, threads_sizet);

  // Create indices for sorting features by performance
  std::vector<size_t> indices(num_features);
  std::iota(indices.begin(), indices.end(), 0);

  // Sort indices based on performance metrics (ρ first, then RMSE, then MAE)
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    const auto& ma = pred_metrics[a];
    const auto& mb = pred_metrics[b];

    if (ma[0] != mb[0]) return ma[0] > mb[0];  // Higher ρ first
    if (ma[2] != mb[2]) return ma[2] < mb[2];  // Lower RMSE next
    return ma[1] < mb[1];                      // Lower MAE last
  });

  // Select top_num features (or fewer if not enough available)
  const size_t num_selected = std::min(top_num, static_cast<int>(indices.size()));
  std::vector<size_t> selected(indices.begin(), indices.begin() + num_selected);

  // // Aggregate selected features through column-wise summation
  // // std::vector<double> result;
  // // result.reserve(num_samples);
  // std::vector<double> result(num_samples);
  //
  // for (size_t row = 0; row < num_samples; ++row) {
  //   double sum = 0.0;
  //   size_t valid_count = 0; // Count of non-NaN elements
  //
  //   for (const auto& col : selected) {
  //     double value = vectors[row][col];
  //     if (!std::isnan(value)) { // Check if the value is not NaN
  //       sum += value;
  //       ++valid_count;
  //     }
  //   }
  //
  //   // If there are valid values, compute the average; otherwise, push NaN
  //   result.push_back(valid_count > 0 ? sum / valid_count : std::numeric_limits<double>::quiet_NaN());
  // }

  std::vector<std::vector<double>> selected_embeddings(num_samples,std::vector<double>(selected.size()));
  for (size_t row = 0; row < num_samples; ++row) {
    for (size_t col = 0; col < selected.size(); ++col){
      selected_embeddings[row][col] = vectors[row][selected[col]];
    }
  }
  std::vector<double> result = SimplexProjectionPrediction(selected_embeddings, target, lib_indices, pred_indices, num_neighbors);

  return result;
}

// /**
//  * Computes the multi-view embedding by evaluating multiple feature subsets using simplex projection,
//  * selecting top-performing subsets, and aggregating their contributions.
//  *
//  * Parameters:
//  *   - vectors: 2D vector where each row represents a sample and each column a feature.
//  *   - target: Target spatial cross sectional series aligned with the samples in vectors.
//  *   - lib_indices: Boolean flag indicating which samples to use for neighbor search.
//  *   - pred_indices: Boolean flag indicating which samples to predict.
//  *   - num_neighbors: Number of neighbors used for simplex projection.
//  *   - top_num: Number of top-performing reconstructions to select.
//  *   - threads: Number of threads used from the global pool.
//  *
//  * Returns:
//  *   A vector<double> where each element is the aggregated average of predict value from selected feature columns.
//  */
// std::vector<double> MultiViewEmbedding(
//     const std::vector<std::vector<double>>& vectors,
//     const std::vector<double>& target,
//     const std::vector<bool>& lib_indices,
//     const std::vector<bool>& pred_indices,
//     int num_neighbors,
//     int top_num,
//     int threads
// ) {
//   // Configure threads
//   size_t threads_sizet = static_cast<size_t>(threads);
//   threads_sizet = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), threads_sizet);
//
//   // // Validate input dimensions
//   // if (vectors.empty() || vectors[0].empty() ||
//   //     vectors.size() != target.size() ||
//   //     vectors.size() != lib_indices.size() ||
//   //     vectors.size() != pred_indices.size()) {
//   //   return {};
//   // }
//
//   const size_t num_samples = vectors.size();
//   const size_t num_features = vectors[0].size();
//
//   // // Verify uniform column count in vectors
//   // for (const auto& row : vectors) {
//   //   if (row.size() != num_features) {
//   //     return {};
//   //   }
//   // }
//
//   // Store the simplex projection result
//   std::vector<std::vector<double>> pred_res(num_samples, std::vector<double>(num_features,std::numeric_limits<double>::quiet_NaN()));
//
//   // Evaluate each feature column as a separate subset
//   std::vector<std::vector<double>> pred_metrics(num_features, std::vector<double>(3));
//
//   // Parallel loop over each feature column
//   RcppThread::parallelFor(0, num_features, [&](size_t col) {
//     // // Create subset matrix: each row contains a single feature value
//     // std::vector<std::vector<double>> subset;
//     // subset.reserve(num_samples);
//
//     // Create subset matrix with reserved size upon declaration
//     std::vector<std::vector<double>> subset(num_samples, std::vector<double>(1));
//
//     for (size_t row = 0; row < num_samples; ++row) {
//       subset.push_back({vectors[row][col]});
//     }
//
//     std::vector<double> local_pred = SimplexProjectionPrediction(subset, target, lib_indices, pred_indices, num_neighbors);
//     for (size_t row = 0; row < local_pred.size(); ++row) {
//       pred_res[row][col] = local_pred[row];
//     }
//
//     // Get performance metrics for this subset
//     // auto metrics = SimplexBehavior(subset, target, lib_indices, pred_indices, num_neighbors);
//     // // if (metrics.size() != 3) continue;  // Skip invalid results
//     // pred_metrics[col] = metrics;
//     double pearson = PearsonCor(local_pred, target, true);
//     double mae = CppMAE(local_pred, target, true);
//     double rmse = CppRMSE(local_pred, target, true);
//     pred_metrics[col] = {pearson, mae, rmse};
//
//   }, threads_sizet);
//
//   // Create indices for sorting features by performance
//   std::vector<size_t> indices(num_features);
//   std::iota(indices.begin(), indices.end(), 0);
//
//   // Sort indices based on performance metrics (ρ first, then RMSE, then MAE)
//   std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
//     const auto& ma = pred_metrics[a];
//     const auto& mb = pred_metrics[b];
//
//     if (ma[0] != mb[0]) return ma[0] > mb[0];  // Higher ρ first
//     if (ma[2] != mb[2]) return ma[2] < mb[2];  // Lower RMSE next
//     return ma[1] < mb[1];                      // Lower MAE last
//   });
//
//   // Select top_num features (or fewer if not enough available)
//   const size_t num_selected = std::min(top_num, static_cast<int>(indices.size()));
//   std::vector<size_t> selected(indices.begin(), indices.begin() + num_selected);
//
//   // Aggregate selected features through column-wise summation
//   // std::vector<double> result;
//   // result.reserve(num_samples);
//   std::vector<double> result(num_samples);
//
//   for (size_t row = 0; row < num_samples; ++row) {
//     double sum = 0.0;
//     size_t valid_count = 0; // Count of non-NaN elements
//
//     for (const auto& col : selected) {
//       double value = pred_res[row][col];
//       if (!std::isnan(value)) { // Check if the value is not NaN
//         sum += value;
//         ++valid_count;
//       }
//     }
//
//     // If there are valid values, compute the average; otherwise, push NaN
//     result.push_back(valid_count > 0 ? sum / valid_count : std::numeric_limits<double>::quiet_NaN());
//   }
//   return result;
// }
