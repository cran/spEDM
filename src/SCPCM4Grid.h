#ifndef SCPCM4Grid_H
#define SCPCM4Grid_H

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
#include "spEDMDataStruct.h"
#include <RcppThread.h>

/**
 * Perform Grid-based Spatially Convergent Partial Cross Mapping (SCPCM) for a single library size.
 *
 * This function calculates the partial cross mapping between a predictor variable (xEmbedings) and a response
 * variable (yPred) over a 2D grid, using either Simplex Projection or S-Mapping.
 *
 * @param xEmbedings   A 2D matrix of the predictor variable's embeddings (spatial cross-section data).
 * @param yPred        A 1D vector of the response variable's values (spatial cross-section data).
 * @param controls     A 2D matrix that stores the control variables.
 * @param lib_size     The size of the library (number of spatial units) used for prediction.
 * @param pred         A vector of pairs representing the indices (row, column) of spatial units to be predicted.
 * @param conEs        Number of dimensions for the attractor reconstruction with control variables
 * @param taus:        Vector specifying the spatial lag step for constructing lagged state-space vectors with control variables.
 * @param totalRow     The total number of rows in the 2D grid.
 * @param totalCol     The total number of columns in the 2D grid.
 * @param b            The number of nearest neighbors to use for prediction.
 * @param simplex      If true, use Simplex Projection; if false, use S-Mapping.
 * @param theta        The distance weighting parameter for S-Mapping (ignored if simplex is true).
 * @param cumulate     Whether to cumulate the partial correlations.
 * @return             A vector contains the library size and the corresponding cross mapping and partial cross mapping result.
 */
std::vector<PartialCorRes> SCPCMSingle4Grid(
    const std::vector<std::vector<double>>& xEmbedings,
    const std::vector<double>& yPred,
    const std::vector<std::vector<double>>& controls,
    int lib_size,
    const std::vector<std::pair<int, int>>& pred,
    const std::vector<int>& conEs,
    const std::vector<int>& taus,
    int totalRow,
    int totalCol,
    int b,
    bool simplex,
    double theta,
    bool cumulate);

/**
 * Perform Grid-based Spatially Convergent Partial Cross Mapping (SCPCM) for multiple library sizes.
 *
 * This function estimates the cross mapping and partial cross mapping between predictor variables (`xMatrix`) and response
 * variables (`yMatrix`) over a 2D spatial grid, incorporating control variables (`zMatrixs`). It supports both Simplex Projection
 * and S-Mapping, with options for parallel computation and progress tracking.
 *
 * Parameters:
 * - xMatrix: A 2D matrix of predictor variable values (spatial cross-section data).
 * - yMatrix: A 2D matrix of response variable values (spatial cross-section data).
 * - zMatrixs: A 2D matrix storing the control variables.
 * - lib_sizes: A vector specifying different library sizes (number of spatial units) for SCPCM analysis.
 * - pred: A vector of pairs representing the indices (row, column) of spatial units to be predicted.
 * - Es: A vector specifying the embedding dimensions for attractor reconstruction using `xMatrix` and control variables.
 * - taus: A vector specifying the spatial lag steps for constructing lagged state-space vectors with control variables.
 * - b: Number of nearest neighbors used for prediction.
 * - simplex: Boolean flag indicating whether to use Simplex Projection (true) or S-Mapping (false) for prediction.
 * - theta: Distance weighting parameter used for weighting neighbors in the S-Mapping prediction.
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
std::vector<std::vector<double>> SCPCM4Grid(
    const std::vector<std::vector<double>>& xMatrix,     // Two dimension matrix of X variable
    const std::vector<std::vector<double>>& yMatrix,     // Two dimension matrix of Y variable
    const std::vector<std::vector<double>>& zMatrixs,    // 2D matrix that stores the control variables
    const std::vector<int>& lib_sizes,                   // Vector of library sizes to use
    const std::vector<std::pair<int, int>>& pred,        // Indices of spatial units to be predicted
    const std::vector<int>& Es,                          // Number of dimensions for the attractor reconstruction with the x and control variables
    const std::vector<int>& taus,                        // Vector specifying the spatial lag step for constructing lagged state-space vectors with control variables.
    int b,                                               // Number of nearest neighbors to use for prediction
    bool simplex,                                        // Algorithm used for prediction; Use simplex projection if true, and s-mapping if false
    double theta,                                        // Distance weighting parameter for the local neighbours in the manifold
    int threads,                                         // Number of threads used from the global pool
    bool cumulate,                                       // Whether to cumulate the partial correlations
    bool progressbar                                     // Whether to print the progress bar
);

#endif // SCPCM4Grid_H
