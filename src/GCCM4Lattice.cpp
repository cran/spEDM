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
#include <RcppThread.h>

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppThread)]]

/*
 * Perform GCCM on a single lib and pred for lattice data.
 *
 * Parameters:
 *   - x_vectors: Reconstructed state-space (each row represents a separate vector/state).
 *   - y: Spatial cross-section series used as the target (should align with x_vectors).
 *   - lib_size: Size of the library used for cross mapping.
 *   - max_lib_size: Maximum size of the library.
 *   - possible_lib_indices: Indices of possible library states.
 *   - pred_indices: A boolean vector indicating which states to use for prediction.
 *   - b: Number of neighbors to use for simplex projection.
 *   - simplex: If true, uses simplex projection for prediction; otherwise, uses s-mapping.
 *   - theta: Distance weighting parameter for local neighbors in the manifold (used in s-mapping).
 *   - threads: The number of threads to use for parallel processing.
 *   - parallel_level: Level of parallel computing: 0 for `lower`, 1 for `higher`.
 *
 * Returns:
 *   A vector of pairs, where each pair consists of:
 *   - An integer representing the library size.
 *   - A double representing the Pearson correlation coefficient (rho) between predicted and actual values.
 */
std::vector<std::pair<int, double>> GCCMSingle4Lattice(
    const std::vector<std::vector<double>>& x_vectors,
    const std::vector<double>& y,
    int lib_size,
    int max_lib_size,
    const std::vector<int>& possible_lib_indices,
    const std::vector<bool>& pred_indices,
    int b,
    bool simplex,
    double theta,
    size_t threads,
    int parallel_level
) {
  int n = x_vectors.size();

  // No possible library variation if using all vectors
  if (lib_size == max_lib_size) {
    std::vector<bool> lib_indices(n, false);
    for (int idx : possible_lib_indices) {
      lib_indices[idx] = true;
    }

    std::vector<std::pair<int, double>> x_xmap_y;

    // Run cross map and store results
    double rho = std::numeric_limits<double>::quiet_NaN();
    if (simplex) {
      rho = SimplexProjection(x_vectors, y, lib_indices, pred_indices, b);
    } else {
      rho = SMap(x_vectors, y, lib_indices, pred_indices, b, theta);
    }
    x_xmap_y.emplace_back(lib_size, rho);
    return x_xmap_y;
  } else if (parallel_level == 0){

    // Precompute valid indices for the library
    std::vector<std::vector<int>> valid_lib_indices;
    for (int start_lib = 0; start_lib < max_lib_size; ++start_lib) {
      std::vector<int> local_lib_indices;
      // Loop around to beginning of lib indices
      if (start_lib + lib_size > max_lib_size) {
        for (int i = start_lib; i < max_lib_size; ++i) {
          local_lib_indices.emplace_back(i);
        }
        int num_vectors_remaining = lib_size - (max_lib_size - start_lib);
        for (int i = 0; i < num_vectors_remaining; ++i) {
          local_lib_indices.emplace_back(i);
        }
      } else {
        for (int i = start_lib; i < start_lib + lib_size; ++i) {
          local_lib_indices.emplace_back(i);
        }
      }
      valid_lib_indices.emplace_back(local_lib_indices);
    }

    // Preallocate the result vector to avoid out-of-bounds access
    std::vector<std::pair<int, double>> x_xmap_y(valid_lib_indices.size());

    // Perform the operations using RcppThread
    RcppThread::parallelFor(0, valid_lib_indices.size(), [&](size_t i) {
      std::vector<bool> lib_indices(n, false);
      std::vector<int> local_lib_indices = valid_lib_indices[i];
      for(int& li : local_lib_indices){
        lib_indices[possible_lib_indices[li]] = true;
      }

      // Run cross map and store results
      double rho = std::numeric_limits<double>::quiet_NaN();
      if (simplex) {
        rho = SimplexProjection(x_vectors, y, lib_indices, pred_indices, b);
      } else {
        rho = SMap(x_vectors, y, lib_indices, pred_indices, b, theta);
      }

      std::pair<int, double> result(lib_size, rho); // Store the product of row and column library sizes
      x_xmap_y[i] = result;
    }, threads);

    return x_xmap_y;
  } else {

    std::vector<std::pair<int, double>> x_xmap_y;

    for (int start_lib = 0; start_lib < max_lib_size; ++start_lib) {
      std::vector<bool> lib_indices(n, false);
      // Setup changing library
      if (start_lib + lib_size > max_lib_size) { // Loop around to beginning of lib indices
        for (int i = start_lib; i < max_lib_size; ++i) {
          lib_indices[possible_lib_indices[i]] = true;
        }
        int num_vectors_remaining = lib_size - (max_lib_size - start_lib);
        for (int i = 0; i < num_vectors_remaining; ++i) {
          lib_indices[possible_lib_indices[i]] = true;
        }
      } else {
        for (int i = start_lib; i < start_lib + lib_size; ++i) {
          lib_indices[possible_lib_indices[i]] = true;
        }
      }

      // Run cross map and store results
      double rho = std::numeric_limits<double>::quiet_NaN();
      if (simplex) {
        rho = SimplexProjection(x_vectors, y, lib_indices, pred_indices, b);
      } else {
        rho = SMap(x_vectors, y, lib_indices, pred_indices, b, theta);
      }
      x_xmap_y.emplace_back(lib_size, rho);
    }

    return x_xmap_y;
  }
}

/**
 * Performs GCCM on a spatial lattice data.
 *
 * Parameters:
 * - x: Spatial cross-section series used as the predict variable (**cross mapping from**).
 * - y: Spatial cross-section series used as the target variable (**cross mapping to**).
 * - nb_vec: A nested vector containing neighborhood information for lattice data.
 * - lib_sizes: A vector specifying different library sizes for GCCM analysis.
 * - lib: A vector specifying the library indices (1-based in R, converted to 0-based in C++).
 * - pred: A vector specifying the prediction indices (1-based in R, converted to 0-based in C++).
 * - E: Embedding dimension for attractor reconstruction.
 * - tau: the step of spatial lags for prediction.
 * - b: Number of nearest neighbors used for prediction.
 * - simplex: Boolean flag indicating whether to use simplex projection (true) or S-mapping (false) for prediction.
 * - theta: Distance weighting parameter used for weighting neighbors in the S-mapping prediction.
 * - threads: Number of threads to use for parallel computation.
 * - parallel_level: Level of parallel computing: 0 for `lower`, 1 for `higher`.
 * - progressbar: Boolean flag to indicate whether to display a progress bar during computation.
 *
 * Returns:
 *    A 2D vector of results, where each row contains:
 *      - The library size.
 *      - The mean cross-mapping correlation.
 *      - The statistical significance of the correlation.
 *      - The upper bound of the confidence interval.
 *      - The lower bound of the confidence interval.
 */
std::vector<std::vector<double>> GCCM4Lattice(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<std::vector<int>>& nb_vec,
    const std::vector<int>& lib_sizes,
    const std::vector<int>& lib,
    const std::vector<int>& pred,
    int E,
    int tau,
    int b,
    bool simplex,
    double theta,
    int threads,
    int parallel_level,
    bool progressbar
) {
  // If b is not provided correctly, default it to E + 2
  if (b <= 0) {
    b = E + 2;
  }

  // Configure threads
  size_t threads_sizet = static_cast<size_t>(std::abs(threads));
  threads_sizet = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), threads_sizet);

  // Generate embeddings
  std::vector<std::vector<double>> x_vectors = GenLatticeEmbeddings(x, nb_vec, E, tau);

  int n = x_vectors.size();

  std::vector<int> possible_lib_indices;
  for (size_t i = 0; i < lib.size(); ++i) {
    possible_lib_indices.push_back(lib[i]-1);
  }
  int max_lib_size = static_cast<int>(possible_lib_indices.size()); // Maximum lib size

  std::vector<bool> pred_indices(n, false);
  for (size_t i = 0; i < pred.size(); ++i) {
    // // Do not strictly exclude spatial units with embedded state-space vectors containing NaN values from participating in cross mapping.
    // if (!checkOneDimVectorHasNaN(x_vectors[pred[i] - 1])){
    //   pred_indices[pred[i] - 1] = true;
    // }
    pred_indices[pred[i] - 1] = true; // Convert to 0-based index
  }

  std::vector<int> unique_lib_sizes(lib_sizes.begin(), lib_sizes.end());

  // Transform to ensure no size exceeds max_lib_size
  std::transform(unique_lib_sizes.begin(), unique_lib_sizes.end(), unique_lib_sizes.begin(),
                 [&](int size) { return std::min(size, max_lib_size); });

  // Ensure the minimum value in unique_lib_sizes is E + 2 (uncomment this section if required)
  // std::transform(unique_lib_sizes.begin(), unique_lib_sizes.end(), unique_lib_sizes.begin(),
  //                [&](int size) { return std::max(size, E + 2); });

  // Remove duplicates
  std::sort(unique_lib_sizes.begin(), unique_lib_sizes.end());
  unique_lib_sizes.erase(std::unique(unique_lib_sizes.begin(), unique_lib_sizes.end()), unique_lib_sizes.end());

  // Initialize the result container
  std::vector<std::pair<int, double>> x_xmap_y;

  if (parallel_level == 0){
    // Iterate over each library size
    if (progressbar) {
      RcppThread::ProgressBar bar(unique_lib_sizes.size(), 1);
      for (int lib_size : unique_lib_sizes) {
        auto results = GCCMSingle4Lattice(x_vectors,
                                          y,
                                          lib_size,
                                          max_lib_size,
                                          possible_lib_indices,
                                          pred_indices,
                                          b,
                                          simplex,
                                          theta,
                                          threads_sizet,
                                          parallel_level);
        x_xmap_y.insert(x_xmap_y.end(), results.begin(), results.end());
        bar++;
      }
    } else {
      for (int lib_size : unique_lib_sizes) {
        auto results = GCCMSingle4Lattice(x_vectors,
                                          y,
                                          lib_size,
                                          max_lib_size,
                                          possible_lib_indices,
                                          pred_indices,
                                          b,
                                          simplex,
                                          theta,
                                          threads_sizet,
                                          parallel_level);
        x_xmap_y.insert(x_xmap_y.end(), results.begin(), results.end());
      }
    }
  } else {
    // Perform the operations using RcppThread
    if (progressbar) {
      RcppThread::ProgressBar bar(unique_lib_sizes.size(), 1);
      RcppThread::parallelFor(0, unique_lib_sizes.size(), [&](size_t i) {
        int lib_size = unique_lib_sizes[i];
        auto results = GCCMSingle4Lattice(x_vectors,
                                          y,
                                          lib_size,
                                          max_lib_size,
                                          possible_lib_indices,
                                          pred_indices,
                                          b,
                                          simplex,
                                          theta,
                                          threads_sizet,
                                          parallel_level);
        x_xmap_y.insert(x_xmap_y.end(), results.begin(), results.end());
        bar++;
      }, threads_sizet);
    } else {
      RcppThread::parallelFor(0, unique_lib_sizes.size(), [&](size_t i) {
        int lib_size = unique_lib_sizes[i];
        auto results = GCCMSingle4Lattice(x_vectors,
                                          y,
                                          lib_size,
                                          max_lib_size,
                                          possible_lib_indices,
                                          pred_indices,
                                          b,
                                          simplex,
                                          theta,
                                          threads_sizet,
                                          parallel_level);
        x_xmap_y.insert(x_xmap_y.end(), results.begin(), results.end());
      }, threads_sizet);
    }
  }

  // Group by the first int(lib_size) and compute the mean (rho)
  std::map<int, std::vector<double>> grouped_results;
  for (const auto& result : x_xmap_y) {
    grouped_results[result.first].push_back(result.second);
  }

  std::vector<std::vector<double>> final_results;
  for (const auto& group : grouped_results) {
    double mean_value = CppMean(group.second, true);
    final_results.push_back({static_cast<double>(group.first), mean_value});
  }

  // Calculate significance and confidence interval for each result
  for (size_t i = 0; i < final_results.size(); ++i) {
    double rho = final_results[i][1];
    double significance = CppCorSignificance(rho, n);
    std::vector<double> confidence_interval = CppCorConfidence(rho, n);

    final_results[i].push_back(significance);
    final_results[i].push_back(confidence_interval[0]);
    final_results[i].push_back(confidence_interval[1]);
  }

  return final_results;
}
