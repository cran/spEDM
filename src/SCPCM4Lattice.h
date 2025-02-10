#ifndef SCPCM4Lattice_H
#define SCPCM4Lattice_H

#include <vector>
#include <cmath>
#include <algorithm> // Include for std::partial_sort
#include <numeric>
#include <utility>
#include <limits>
#include <map>
#include "CppStats.h"
#include "CppLatticeUtils.h"
#include "SimplexProjection.h"
#include "SMap.h"
#include "spEDMDataStruct.h"
#include <RcppThread.h>

/*
 * Perform SCPCM on a single library and prediction set for lattice data.
 *
 * Parameters:
 *   - x_vectors: Reconstructed state-space (each row represents a separate vector/state).
 *   - y: Spatial cross-section series used as the target (should align with x_vectors).
 *   - controls: Cross-sectional data of control variables (stored by row).
 *   - nb_vec: Neighbor indices vector of the spatial units.
 *   - lib_indices: A boolean vector indicating which states to include when searching for neighbors.
 *   - lib_size: Size of the library used for cross mapping.
 *   - max_lib_size: Maximum size of the library.
 *   - possible_lib_indices: Indices of possible library states.
 *   - pred_indices: A boolean vector indicating which states to use for prediction.
 *   - conEs: Number of dimensions for attractor reconstruction with control variables.
 *   - taus: Spatial lag step for constructing lagged state-space vectors with control variables.
 *   - b: Number of neighbors to use for simplex projection.
 *   - simplex: If true, uses simplex projection for prediction; otherwise, uses s-mapping.
 *   - theta: Distance weighting parameter for local neighbors in the manifold (used in s-mapping).
 *   - cumulate: Whether to accumulate partial correlations.
 *
 * Returns:
 *   A vector of PartialCorRes objects, where each contains:
 *   - An integer representing the library size.
 *   - A double representing the Pearson correlation coefficient (rho).
 *   - A double representing the Partial correlation coefficient (pratial rho).
 */
std::vector<PartialCorRes> SCPCMSingle4Lattice(
    const std::vector<std::vector<double>>& x_vectors,  // Reconstructed state-space (each row is a separate vector/state)
    const std::vector<double>& y,                       // Spatial cross-section series to be used as the target (should line up with vectors)
    const std::vector<std::vector<double>>& controls,   // Cross-sectional data of control variables (**stored by row**)
    const std::vector<std::vector<int>>& nb_vec,        // Neighbor indices vector of the spatial units
    const std::vector<bool>& lib_indices,               // Vector of T/F values (which states to include when searching for neighbors)
    int lib_size,                                       // Size of the library
    int max_lib_size,                                   // Maximum size of the library
    const std::vector<int>& possible_lib_indices,       // Indices of possible library states
    const std::vector<bool>& pred_indices,              // Vector of T/F values (which states to predict from)
    const std::vector<int>& conEs,                      // Number of dimensions for the attractor reconstruction with control variables
    const std::vector<int>& taus,                       // Spatial lag step for constructing lagged state-space vectors with control variables
    int b,                                              // Number of neighbors to use for simplex projection
    bool simplex,                                       // Algorithm used for prediction; Use simplex projection if true, and s-mapping if false
    double theta,                                       // Distance weighting parameter for the local neighbours in the manifold
    bool cumulate                                       // Whether to cumulate the partial correlations
);

/**
 * Performs SCPCM on a spatial lattice dataset.
 *
 * Parameters:
 * - x: Spatial cross-section series used as the predictor variable (**cross mapping from**).
 * - y: Spatial cross-section series used as the target variable (**cross mapping to**).
 * - controls: Cross-sectional data of control variables (**stored by row**).
 * - nb_vec: A nested vector containing neighborhood information for lattice data.
 * - lib_sizes: A vector specifying different library sizes for SCPCM analysis.
 * - lib: A vector specifying the library indices (1-based in R, converted to 0-based in C++).
 * - pred: A vector specifying the prediction indices (1-based in R, converted to 0-based in C++).
 * - Es: A vector specifying the embedding dimensions for attractor reconstruction using x and control variables.
 * - taus: A vector specifying the spatial lag steps for constructing lagged state-space vectors using x and control variables.
 * - b: Number of nearest neighbors used for prediction.
 * - simplex: Boolean flag indicating whether to use simplex projection (true) or S-mapping (false) for prediction.
 * - theta: Distance weighting parameter used for weighting neighbors in the S-mapping prediction.
 * - threads: Number of threads to use for parallel computation.
 * - cumulate: Boolean flag indicating whether to cumulate partial correlations.
 * - progressbar: Boolean flag indicating whether to display a progress bar during computation.
 *
 * Returns:
 *    A 2D vector of results, where each row contains:
 *      - The library size.
 *      - The mean pearson cross-mapping correlation.
 *      - The statistical significance of the pearson correlation.
 *      - The lower bound of the pearson correlation confidence interval.
 *      - The upper bound of the pearson correlation confidence interval.
 *      - The mean partial cross-mapping partial correlation.
 *      - The statistical significance of the partial correlation.
 *      - The lower bound of the partial correlation confidence interval.
 *      - The upper bound of the partial correlation confidence interval.
 */
std::vector<std::vector<double>> SCPCM4Lattice(
    const std::vector<double>& x,                       // Spatial cross-section series to cross map from
    const std::vector<double>& y,                       // Spatial cross-section series to cross map to
    const std::vector<std::vector<double>>& controls,   // Cross-sectional data of control variables (**stored by row**)
    const std::vector<std::vector<int>>& nb_vec,        // Neighbor indices vector of the spatial units
    const std::vector<int>& lib_sizes,                  // Vector of library sizes to use
    const std::vector<int>& lib,                        // Vector specifying the library indices
    const std::vector<int>& pred,                       // Vector specifying the prediction indices
    const std::vector<int>& Es,                         // Number of dimensions for the attractor reconstruction with the x and control variables
    const std::vector<int>& taus,                       // Spatial lag step for constructing lagged state-space vectors with the x and control variables
    int b,                                              // Number of nearest neighbors to use for prediction
    bool simplex,                                       // Algorithm used for prediction; Use simplex projection if true, and s-mapping if false
    double theta,                                       // Distance weighting parameter for the local neighbours in the manifold
    int threads,                                        // Number of threads used from the global pool
    bool cumulate,                                      // Whether to cumulate the partial correlations
    bool progressbar                                    // Whether to print the progress bar
);

#endif // SCPCM4Lattice_H
