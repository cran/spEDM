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

std::vector<std::pair<int, double>> GCCMSingle4Lattice(
    const std::vector<std::vector<double>>& x_vectors,  // Reconstructed state-space (each row is a separate vector/state)
    const std::vector<double>& y,                       // Spatial cross-section series to be used as the target (should line up with vectors)
    const std::vector<bool>& lib_indices,               // Vector of T/F values (which states to include when searching for neighbors)
    int lib_size,                                       // Size of the library
    int max_lib_size,                                   // Maximum size of the library
    const std::vector<int>& possible_lib_indices,       // Indices of possible library states
    const std::vector<bool>& pred_indices,              // Vector of T/F values (which states to predict from)
    int b,                                              // Number of neighbors to use for simplex projection
    bool simplex,                                       // Algorithm used for prediction; Use simplex projection if true, and s-mapping if false
    double theta                                        // Distance weighting parameter for the local neighbours in the manifold
) {
  int n = x_vectors.size();
  std::vector<std::pair<int, double>> x_xmap_y;
  double rho;

  if (lib_size == max_lib_size) { // No possible library variation if using all vectors
    std::vector<bool> lib_indices(n, false);
    for (int idx : possible_lib_indices) {
      lib_indices[idx] = true;
    }

    // Run cross map and store results
    if (simplex) {
      rho = SimplexProjection(x_vectors, y, lib_indices, pred_indices, b);
    } else {
      rho = SMap(x_vectors, y, lib_indices, pred_indices, b, theta);
    }
    x_xmap_y.emplace_back(lib_size, rho);
  } else {
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
      if (simplex) {
        rho = SimplexProjection(x_vectors, y, lib_indices, pred_indices, b);
      } else {
        rho = SMap(x_vectors, y, lib_indices, pred_indices, b, theta);
      }
      x_xmap_y.emplace_back(lib_size, rho);
    }
  }

  return x_xmap_y;
}

std::vector<std::vector<double>> GCCM4Lattice(
    const std::vector<std::vector<double>>& x_vectors,  // Reconstructed state-space (each row is a separate vector/state)
    const std::vector<double>& y,                       // Spatial cross-section series to cross map to
    const std::vector<int>& lib_sizes,                  // Vector of library sizes to use
    const std::vector<std::pair<int, int>>& lib,        // Matrix (n x 2) using n sequences of data to construct libraries
    const std::vector<std::pair<int, int>>& pred,       // Matrix (n x 2) using n sequences of data to predict from
    int E,                                              // Number of dimensions for the attractor reconstruction
    int tau,                                            // Spatial lag for the lagged-vector construction
    int b,                                              // Number of nearest neighbors to use for prediction
    bool simplex,                                       // Algorithm used for prediction; Use simplex projection if true, and s-mapping if false
    double theta,                                       // Distance weighting parameter for the local neighbours in the manifold
    int threads,                                        // Number of threads used from the global pool
    bool progressbar = true                             // Whether to print the progress bar
) {
  // If b is not provided correctly, default it to E + 2
  if (b <= 0) {
    b = E + 2;
  }

  size_t threads_sizet = static_cast<size_t>(threads);
  unsigned int max_threads = std::thread::hardware_concurrency();
  threads_sizet = std::min(static_cast<size_t>(max_threads), threads_sizet);

  int n = x_vectors.size();
  // Setup pred_indices
  std::vector<bool> pred_indices(n, false);
  for (const auto& p : pred) {
    int row_start = p.first + (E - 1) * tau;
    int row_end = p.second;
    if (row_end > row_start && row_start >= 0 && row_end < n) {
      std::fill(pred_indices.begin() + row_start, pred_indices.begin() + row_end + 1, true);
    }
  }

  // Setup lib_indices
  std::vector<bool> lib_indices(n, false);
  for (const auto& l : lib) {
    int row_start = l.first + (E - 1) * tau;
    int row_end = l.second;
    if (row_end > row_start && row_start >= 0 && row_end < n) {
      std::fill(lib_indices.begin() + row_start, lib_indices.begin() + row_end + 1, true);
    }
  }

  int max_lib_size = std::accumulate(lib_indices.begin(), lib_indices.end(), 0); // Maximum lib size
  std::vector<int> possible_lib_indices;
  for (int i = 0; i < n; ++i) {
    if (lib_indices[i]) {
      possible_lib_indices.push_back(i);
    }
  }

  std::vector<int> unique_lib_sizes(lib_sizes.begin(), lib_sizes.end());

  // Transform to ensure no size exceeds max_lib_size
  std::transform(unique_lib_sizes.begin(), unique_lib_sizes.end(), unique_lib_sizes.begin(),
                 [&](int size) { return std::min(size, max_lib_size); });

  // Ensure the minimum value in unique_lib_sizes is E + 2 (uncomment this section if required)
  // std::transform(unique_lib_sizes.begin(), unique_lib_sizes.end(), unique_lib_sizes.begin(),
  //                [&](int size) { return std::max(size, E + 2); });

  // Remove duplicates
  unique_lib_sizes.erase(std::unique(unique_lib_sizes.begin(), unique_lib_sizes.end()), unique_lib_sizes.end());

  // Initialize the result container
  std::vector<std::pair<int, double>> x_xmap_y;

  // // Sequential version of the for loop
  // for (int lib_size : unique_lib_sizes) {
  //   auto results = GCCMSingle4Lattice(x_vectors, y, lib_indices, lib_size, max_lib_size, possible_lib_indices, pred_indices, b, simplex, theta);
  //   x_xmap_y.insert(x_xmap_y.end(), results.begin(), results.end());
  // }

  // Perform the operations using RcppThread
  if (progressbar) {
    RcppThread::ProgressBar bar(unique_lib_sizes.size(), 1);
    RcppThread::parallelFor(0, unique_lib_sizes.size(), [&](size_t i) {
      int lib_size = unique_lib_sizes[i];
      auto results = GCCMSingle4Lattice(x_vectors, y, lib_indices, lib_size, max_lib_size, possible_lib_indices, pred_indices, b, simplex, theta);
      x_xmap_y.insert(x_xmap_y.end(), results.begin(), results.end());
      bar++;
    }, threads_sizet);
  } else {
    RcppThread::parallelFor(0, unique_lib_sizes.size(), [&](size_t i) {
      int lib_size = unique_lib_sizes[i];
      auto results = GCCMSingle4Lattice(x_vectors, y, lib_indices, lib_size, max_lib_size, possible_lib_indices, pred_indices, b, simplex, theta);
      x_xmap_y.insert(x_xmap_y.end(), results.begin(), results.end());
    }, threads_sizet);
  }

  // Group by the first int and compute the mean
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
