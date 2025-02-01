#include <vector>
#include <cmath>
#include <algorithm> // Include for std::partial_sort
#include <numeric>
#include <utility>
#include <limits>
#include "CppStats.h"

// Function to compute the 'S-maps' prediction
// Returns a vectors: target_pred
std::vector<double> SMapPrediction(
    const std::vector<std::vector<double>>& vectors,  // Reconstructed state-space (each row is a separate vector/state)
    const std::vector<double>& target,                // Spatial cross-section series to be used as the target (should line up with vectors)
    const std::vector<bool>& lib_indices,             // Vector of T/F values (which states to include when searching for neighbors)
    const std::vector<bool>& pred_indices,            // Vector of T/F values (which states to predict from)
    int num_neighbors,                                // Number of neighbors to use for S-map
    double theta                                      // Weighting parameter for distances
) {
  // Convert num_neighbors to size_t
  size_t num_neighbors_sizet = static_cast<size_t>(num_neighbors);

  // Setup output
  std::vector<double> pred(target.size(), std::numeric_limits<double>::quiet_NaN());

  // Make predictions
  for (size_t p = 0; p < pred_indices.size(); ++p) {
    if (!pred_indices[p]) continue;

    // Create a local copy of lib_indices to modify
    std::vector<bool> local_lib_indices = lib_indices;
    bool temp_lib = local_lib_indices[p];
    local_lib_indices[p] = false;
    std::vector<size_t> libs;
    for (size_t i = 0; i < local_lib_indices.size(); ++i) {
      if (local_lib_indices[i]) libs.push_back(i);
    }

    // Compute distances
    std::vector<double> distances;
    for (size_t i : libs) {
      double sum_sq = 0.0;
      double sum_na = 0.0;
      for (size_t j = 0; j < vectors[p].size(); ++j) {
        if (!std::isnan(vectors[i][j]) && !std::isnan(vectors[p][j])) {
          sum_sq += std::pow(vectors[i][j] - vectors[p][j], 2);
          sum_na += 1.0;
        }
      }

      if (sum_na > 0) {
        distances.push_back(std::sqrt(sum_sq / sum_na));
      } else {
        distances.push_back(std::numeric_limits<double>::quiet_NaN());
      }
    }

    // Compute mean distance
    double mean_distance = 0.0;
    for (double dist : distances) {
      mean_distance += dist;
    }
    mean_distance /= distances.size();

    // Compute weights
    std::vector<double> weights(distances.size());
    for (size_t i = 0; i < distances.size(); ++i) {
      weights[i] = std::exp(-theta * distances[i] / mean_distance);
    }

    // Find nearest neighbors
    std::vector<size_t> neighbors(distances.size());
    std::iota(neighbors.begin(), neighbors.end(), 0);
    // Check if num_neighbors_sizet exceeds the size of neighbors
    // If so, adjust num_neighbors_sizet to the maximum allowed value (neighbors.size())
    if (num_neighbors_sizet > neighbors.size()) {
      num_neighbors_sizet = neighbors.size();
    }
    std::partial_sort(neighbors.begin(), neighbors.begin() + num_neighbors_sizet, neighbors.end(),
                      [&](size_t a, size_t b) { return distances[a] < distances[b]; });

    // Prepare data for SVD
    std::vector<std::vector<double>> A(num_neighbors_sizet, std::vector<double>(vectors[p].size() + 1, 0.0));
    std::vector<double> b(num_neighbors_sizet, 0.0);
    for (size_t i = 0; i < num_neighbors_sizet; ++i) {
      size_t idx = libs[neighbors[i]];
      for (size_t j = 0; j < vectors[p].size(); ++j) {
        A[i][j] = vectors[idx][j] * weights[neighbors[i]];
      }
      A[i][vectors[p].size()] = weights[neighbors[i]]; // Bias term
      b[i] = target[idx] * weights[neighbors[i]];
    }

    // Perform Singular Value Decomposition (SVD)
    std::vector<std::vector<std::vector<double>>> svd_result = CppSVD(A);
    std::vector<std::vector<double>> U = svd_result[0];
    std::vector<double> S = svd_result[1][0];
    std::vector<std::vector<double>> V = svd_result[2];

    // Remove singular values that are too small
    double max_s = *std::max_element(S.begin(), S.end());
    std::vector<double> S_inv(S.size(), 0.0);
    for (size_t i = 0; i < S.size(); ++i) {
      if (S[i] >= max_s * 1e-5) {
        S_inv[i] = 1.0 / S[i];
      }
    }

    // Compute the map coefficients
    std::vector<double> map_coeffs(vectors[p].size() + 1, 0.0);
    for (size_t i = 0; i < V.size(); ++i) {
      for (size_t j = 0; j < S_inv.size(); ++j) {
        map_coeffs[i] += V[i][j] * S_inv[j] * U[j][i];
      }
    }

    // Multiply by b to get the final coefficients
    for (size_t i = 0; i < map_coeffs.size(); ++i) {
      map_coeffs[i] *= b[i];
    }

    // Make prediction
    double prediction = 0.0;
    for (size_t i = 0; i < vectors[p].size(); ++i) {
      prediction += map_coeffs[i] * vectors[p][i];
    }
    prediction += map_coeffs[vectors[p].size()]; // Bias term
    pred[p] = prediction;

    // Restore the original lib_indices state
    local_lib_indices[p] = temp_lib;
  }

  return pred;
}

// Rho value by the 'S-Maps' prediction
double SMap(
    const std::vector<std::vector<double>>& vectors,  // Reconstructed state-space (each row is a separate vector/state)
    const std::vector<double>& target,                // Spatial cross-section series to be used as the target (should line up with vectors)
    const std::vector<bool>& lib_indices,             // Vector of T/F values (which states to include when searching for neighbors)
    const std::vector<bool>& pred_indices,            // Vector of T/F values (which states to predict from)
    int num_neighbors,                                // Number of neighbors to use for S-map
    double theta                                      // Weighting parameter for distances
) {
  std::vector<double> target_pred = SMapPrediction(vectors, target, lib_indices, pred_indices, num_neighbors, theta);
  return PearsonCor(target_pred, target, true);
}

// Description: Computes the S-Map prediction and returns a vector containing
//              Pearson correlation coefficient (PearsonCor), mean absolute error (MAE),
//              and root mean squared error (RMSE).
// Parameters:
//   - vectors: Reconstructed state-space (each row is a separate vector/state).
//   - target: Spatial cross-section series to be used as the target (should align with vectors).
//   - lib_indices: Vector of T/F values (which states to include when searching for neighbors).
//   - pred_indices: Vector of T/F values (which states to predict from).
//   - num_neighbors: Number of neighbors to use for S-Map.
//   - theta: Weighting parameter for distances.
// Returns: A vector<double> containing {PearsonCor, MAE, RMSE}.
std::vector<double> SMapBehavior(
    const std::vector<std::vector<double>>& vectors,
    const std::vector<double>& target,
    const std::vector<bool>& lib_indices,
    const std::vector<bool>& pred_indices,
    int num_neighbors,
    double theta
) {
  // Call SMapPrediction to get the prediction results
  std::vector<double> target_pred = SMapPrediction(vectors, target, lib_indices, pred_indices, num_neighbors, theta);

  // Compute PearsonCor, MAE, and RMSE
  double pearson = PearsonCor(target_pred, target, true);
  double mae = CppMAE(target_pred, target, true);
  double rmse = CppRMSE(target_pred, target, true);

  // Return the three metrics as a vector
  return {pearson, mae, rmse};
}
