#ifndef SimplexProjection_H
#define SimplexProjection_H

#include <vector>
#include <cmath>
#include <algorithm> // Include for std::partial_sort
#include <numeric>
#include <utility>
#include <limits>
#include "CppStats.h"

// Function to compute the simplex projection
double SimplexProjection(
    const std::vector<std::vector<double>>& vectors,  // Reconstructed state-space (each row is a separate vector/state)
    const std::vector<double>& target,                // Time series to be used as the target (should line up with vectors)
    const std::vector<bool>& lib_indices,             // Vector of T/F values (which states to include when searching for neighbors)
    const std::vector<bool>& pred_indices,            // Vector of T/F values (which states to predict from)
    int num_neighbors                                // Number of neighbors to use for simplex projection
);

#endif // SimplexProjection_H
