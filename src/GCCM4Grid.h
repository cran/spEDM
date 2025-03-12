#ifndef GCCM4Grid_H
#define GCCM4Grid_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <limits>
#include <map>
#include "CppStats.h"
#include "CppGridUtils.h"
#include "SimplexProjection.h"
#include "SMap.h"
#include <RcppThread.h>

/**
 * Perform Grid-based Geographical Convergent Cross Mapping (GCCM) for a single library size and pred indice.
 *
 * This function calculates the cross mapping between a predictor variable (xEmbedings) and a response variable (yPred)
 * over a 2D grid, using either Simplex Projection or S-Mapping.
 *
 * @param xEmbedings           A 2D matrix of the predictor variable's embeddings (spatial cross-section data).
 * @param yPred                A 1D vector of the response variable's values (spatial cross-section data).
 * @param lib_sizes            A vector of two integers, where the first element is the row-wise library size and the second element is the column-wise library size.
 * @param possible_lib_indices A boolean vector indicating which spatial units are valid for inclusion in the library.
 * @param pred_indices         A boolean vector indicating which spatial units to be predicted.
 * @param totalRow             The total number of rows in the 2D grid.
 * @param totalCol             The total number of columns in the 2D grid.
 * @param b                    The number of nearest neighbors to use for prediction.
 * @param simplex              If true, use Simplex Projection; if false, use S-Mapping.
 * @param theta                The distance weighting parameter for S-Mapping (ignored if simplex is true).
 * @param threads              The number of threads to use for parallel processing.
 * @param parallel_level       Level of parallel computing: 0 for `lower`, 1 for `higher`.
 * @param row_size_mark        If true, use the row-wise libsize to mark the libsize; if false, use col-wise libsize.
 *
 * @return  A vector of pairs, where each pair contains the library size and the corresponding cross mapping result.
 */
std::vector<std::pair<int, double>> GCCMSingle4Grid(
    const std::vector<std::vector<double>>& xEmbedings,
    const std::vector<double>& yPred,
    const std::vector<int>& lib_sizes,
    const std::vector<bool>& possible_lib_indices,
    const std::vector<bool>& pred_indices,
    int totalRow,
    int totalCol,
    int b,
    bool simplex,
    double theta,
    size_t threads,
    int parallel_level,
    bool row_size_mark
);

/**
 * Perform Grid-based Geographical Convergent Cross Mapping (GCCM) for a single library size.
 *
 * This function follows the same library construction logic as GCCMSingle4Lattice, where libraries
 * are created by selecting consecutive indices from possible_lib_indices with possible wraparound.
 *
 * @param xEmbedings           State-space embeddings for the predictor variable (each row is a spatial vector)
 * @param yPred                Target spatial cross-section series
 * @param lib_size             Number of consecutive spatial units to include in each library
 * @param max_lib_size         Maximum possible library size (total valid spatial units)
 * @param possible_lib_indices Integer vector indicating the indices of eligible spatial units for library construction
 * @param pred_indices         Boolean vector indicating spatial units to predict
 * @param totalRow             Total rows in spatial grid
 * @param totalCol             Total columns in spatial grid
 * @param b                    Number of nearest neighbors for prediction
 * @param simplex              Use simplex projection if true, S-mapping if false
 * @param theta                Distance weighting parameter for S-mapping
 * @param threads              The number of threads to use for parallel processing
 * @param parallel_level       Level of parallel computing: 0 for `lower`, 1 for `higher`
 *
 * @return A vector of pairs, where each pair contains the library size and the corresponding cross mapping result.
 */
std::vector<std::pair<int, double>> GCCMSingle4GridOneDim(
    const std::vector<std::vector<double>>& xEmbedings,
    const std::vector<double>& yPred,
    int lib_size,
    int max_lib_size,
    const std::vector<int>& possible_lib_indices,
    const std::vector<bool>& pred_indices,
    int totalRow,
    int totalCol,
    int b,
    bool simplex,
    double theta,
    size_t threads,
    int parallel_level
);

/**
 * Perform Geographical Convergent Cross Mapping (GCCM) for spatial grid data.
 *
 * This function calculates the cross mapping between predictor variables (xMatrix) and response variables (yMatrix)
 * over a 2D grid, using either Simplex Projection or S-Mapping. It supports parallel processing and progress tracking.
 *
 * @param xMatrix        A 2D matrix of the predictor variable's values (spatial cross-section data).
 * @param yMatrix        A 2D matrix of the response variable's values (spatial cross-section data).
 * @param lib_sizes      A 2D vector where the first sub-vector contains row-wise library sizes and the second sub-vector contains column-wise library sizes.
 * @param lib            A vector of pairs representing the indices (row, column) of spatial units to be the library.
 * @param pred           A vector of pairs representing the indices (row, column) of spatial units to be predicted.
 * @param E              The number of dimensions for attractor reconstruction.
 * @param tau            The step of spatial lags for prediction.
 * @param b              The number of nearest neighbors to use for prediction.
 * @param simplex        If true, use Simplex Projection; if false, use S-Mapping.
 * @param theta          The distance weighting parameter for S-Mapping (ignored if simplex is true).
 * @param threads        The number of threads to use for parallel processing.
 * @param parallel_level Level of parallel computing: 0 for `lower`, 1 for `higher`.
 * @param progressbar    If true, display a progress bar during computation.
 *
 * @return A 2D vector where each row contains the library size, mean cross mapping result,
 *         significance, and confidence interval bounds.
 */
std::vector<std::vector<double>> GCCM4Grid(
    const std::vector<std::vector<double>>& xMatrix,
    const std::vector<std::vector<double>>& yMatrix,
    const std::vector<std::vector<int>>& lib_sizes,
    const std::vector<std::pair<int, int>>& lib,
    const std::vector<std::pair<int, int>>& pred,
    int E,
    int tau,
    int b,
    bool simplex,
    double theta,
    int threads,
    int parallel_level,
    bool progressbar
);

/**
 * Perform Geographical Convergent Cross Mapping (GCCM) for spatial grid data.
 *
 * This function calculates the cross mapping between predictor variables (xMatrix) and response variables (yMatrix)
 * over a 2D grid, using either Simplex Projection or S-Mapping. It supports parallel processing and progress tracking.
 *
 * @param xMatrix        A 2D matrix of the predictor variable's values (spatial cross-section data).
 * @param yMatrix        A 2D matrix of the response variable's values (spatial cross-section data).
 * @param lib_sizes      Number of consecutive spatial units to include in each library.
 * @param lib            A vector of representing the indices of spatial units to be the library.
 * @param pred           A vector of representing the indices of spatial units to be predicted.
 * @param E              The number of dimensions for attractor reconstruction.
 * @param tau            The step of spatial lags for prediction.
 * @param b              The number of nearest neighbors to use for prediction.
 * @param simplex        If true, use Simplex Projection; if false, use S-Mapping.
 * @param theta          The distance weighting parameter for S-Mapping (ignored if simplex is true).
 * @param threads        The number of threads to use for parallel processing.
 * @param parallel_level Level of parallel computing: 0 for `lower`, 1 for `higher`.
 * @param progressbar    If true, display a progress bar during computation.
 *
 * @return A 2D vector where each row contains the library size, mean cross mapping result,
 *         significance, and confidence interval bounds.
 */
std::vector<std::vector<double>> GCCM4GridOneDim(
    const std::vector<std::vector<double>>& xMatrix,
    const std::vector<std::vector<double>>& yMatrix,
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
);

#endif // GCCM4Grid_H
