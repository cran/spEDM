#ifndef SMap_H
#define SMap_H

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
);

// Rho value by the 'S-Maps' prediction
double SMap(
    const std::vector<std::vector<double>>& vectors,  // Reconstructed state-space (each row is a separate vector/state)
    const std::vector<double>& target,                // Spatial cross-section series to be used as the target (should line up with vectors)
    const std::vector<bool>& lib_indices,             // Vector of T/F values (which states to include when searching for neighbors)
    const std::vector<bool>& pred_indices,            // Vector of T/F values (which states to predict from)
    int num_neighbors,                                // Number of neighbors to use for S-map
    double theta                                      // Weighting parameter for distances
);

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
);

#endif // SMap_H
