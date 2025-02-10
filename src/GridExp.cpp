#include <vector>
#include "CppGridUtils.h"
#include "Forecast4Grid.h"
#include "GCCM4Grid.h"
#include "SCPCM4Grid.h"
#include "IntersectionCardinality.h"
// 'Rcpp.h' should not be included and correct to include only 'RcppArmadillo.h'.
// #include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericMatrix RcppLaggedVar4Grid(Rcpp::NumericMatrix mat, int lagNum) {
  // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
  int numRows = mat.nrow();
  int numCols = mat.ncol();
  std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));

  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      cppMat[r][c] = mat(r, c);
    }
  }

  // Call the CppLaggedVar4Grid function
  std::vector<std::vector<double>> laggedMat = CppLaggedVar4Grid(cppMat, lagNum);

  // Convert the result back to Rcpp::NumericMatrix
  int laggedRows = laggedMat.size();
  int laggedCols = laggedMat[0].size();
  Rcpp::NumericMatrix result(laggedRows, laggedCols);

  for (int r = 0; r < laggedRows; ++r) {
    for (int c = 0; c < laggedCols; ++c) {
      result(r, c) = laggedMat[r][c];
    }
  }

  return result;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix RcppGenGridEmbeddings(Rcpp::NumericMatrix mat,
                                          int E, int tau) {
  // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
  int numRows = mat.nrow();
  int numCols = mat.ncol();
  std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));

  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      cppMat[r][c] = mat(r, c);
    }
  }

  // Call the GenGridEmbeddings function
  std::vector<std::vector<double>> embeddings = GenGridEmbeddings(cppMat, E, tau);

  // Convert std::vector<std::vector<double>> to Rcpp::NumericMatrix
  int rows = embeddings.size();
  int cols = embeddings[0].size();
  Rcpp::NumericMatrix result(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = embeddings[i][j];
    }
  }

  return result;
}

// [[Rcpp::export]]
int RcppLocateGridIndices(int curRow, int curCol,
                          int totalRow, int totalCol){
  int indices = LocateGridIndices(curRow,curCol,totalRow,totalCol);
  return indices + 1;
};

// [[Rcpp::export]]
Rcpp::NumericMatrix RcppSimplex4Grid(const Rcpp::NumericMatrix& mat,
                                     const Rcpp::IntegerMatrix& lib,
                                     const Rcpp::IntegerMatrix& pred,
                                     const Rcpp::IntegerVector& E,
                                     int tau,
                                     int b,
                                     int threads) {
  // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
  int numRows = mat.nrow();
  int numCols = mat.ncol();
  std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));

  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      cppMat[r][c] = mat(r, c);
    }
  }

  // Initialize lib_indices and pred_indices with all false
  std::vector<bool> pred_indices(numRows * numCols, false);
  std::vector<bool> lib_indices(numRows * numCols, false);

  // Convert lib and pred (1-based in R) to 0-based indices and set corresponding positions to true
  int currow;
  int curcol;

  for (int i = 0; i < lib.nrow(); ++i) {
    // Convert to 0-based index
    currow = lib(i,0);
    curcol = lib(i,1);
    if (!std::isnan(cppMat[currow-1][curcol-1])){
      lib_indices[LocateGridIndices(currow, curcol, numRows, numCols)] = true;
    }
  }

  for (int i = 0; i < pred.nrow(); ++i) {
    // Convert to 0-based index
    currow = pred(i,0);
    curcol = pred(i,1);
    if (!std::isnan(cppMat[currow-1][curcol-1])){
      pred_indices[LocateGridIndices(currow, curcol, numRows, numCols)] = true;
    }
  }

  // Convert Rcpp::IntegerVector to std::vector<int>
  std::vector<int> E_std = Rcpp::as<std::vector<int>>(E);

  std::vector<std::vector<double>> res_std = Simplex4Grid(
    cppMat,
    lib_indices,
    pred_indices,
    E_std,
    tau,
    b,
    threads);

  size_t n_rows = res_std.size();
  size_t n_cols = res_std[0].size();

  // Create an Rcpp::NumericMatrix with the same dimensions
  Rcpp::NumericMatrix result(n_rows, n_cols);

  // Fill the Rcpp::NumericMatrix with data from res_std
  for (size_t i = 0; i < n_rows; ++i) {
    for (size_t j = 0; j < n_cols; ++j) {
      result(i, j) = res_std[i][j];
    }
  }

  // Set column names for the result matrix
  Rcpp::colnames(result) = Rcpp::CharacterVector::create("E", "rho", "mae", "rmse");
  return result;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix RcppSMap4Grid(const Rcpp::NumericMatrix& mat,
                                  const Rcpp::IntegerMatrix& lib,
                                  const Rcpp::IntegerMatrix& pred,
                                  const Rcpp::NumericVector& theta,
                                  int E,
                                  int tau,
                                  int b,
                                  int threads) {
  // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
  int numRows = mat.nrow();
  int numCols = mat.ncol();
  std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));

  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      cppMat[r][c] = mat(r, c);
    }
  }

  // Initialize lib_indices and pred_indices with all false
  std::vector<bool> pred_indices(numRows * numCols, false);
  std::vector<bool> lib_indices(numRows * numCols, false);

  // Convert lib and pred (1-based in R) to 0-based indices and set corresponding positions to true
  int currow;
  int curcol;

  for (int i = 0; i < lib.nrow(); ++i) {
    // Convert to 0-based index
    currow = lib(i,0);
    curcol = lib(i,1);
    if (!std::isnan(cppMat[currow-1][curcol-1])){
      lib_indices[LocateGridIndices(currow, curcol, numRows, numCols)] = true;
    }
  }

  for (int i = 0; i < pred.nrow(); ++i) {
    // Convert to 0-based index
    currow = pred(i,0);
    curcol = pred(i,1);
    if (!std::isnan(cppMat[currow-1][curcol-1])){
      pred_indices[LocateGridIndices(currow, curcol, numRows, numCols)] = true;
    }
  }

  // Convert Rcpp::NumericMatrix to std::vector<double>
  std::vector<double> theta_std = Rcpp::as<std::vector<double>>(theta);

  std::vector<std::vector<double>> res_std = SMap4Grid(
    cppMat,
    lib_indices,
    pred_indices,
    theta_std,
    E,
    tau,
    b,
    threads);

  size_t n_rows = res_std.size();
  size_t n_cols = res_std[0].size();

  // Create an Rcpp::NumericMatrix with the same dimensions
  Rcpp::NumericMatrix result(n_rows, n_cols);

  // Fill the Rcpp::NumericMatrix with data from res_std
  for (size_t i = 0; i < n_rows; ++i) {
    for (size_t j = 0; j < n_cols; ++j) {
      result(i, j) = res_std[i][j];
    }
  }

  // Set column names for the result matrix
  Rcpp::colnames(result) = Rcpp::CharacterVector::create("theta", "rho", "mae", "rmse");
  return result;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix RcppGCCM4Grid(
    const Rcpp::NumericMatrix& xMatrix,
    const Rcpp::NumericMatrix& yMatrix,
    const Rcpp::IntegerVector& libsizes,
    const Rcpp::IntegerMatrix& pred,
    int E,
    int tau,
    int b,
    bool simplex,
    double theta,
    int threads,
    bool progressbar) {

  // Convert Rcpp NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> xMatrix_cpp(xMatrix.nrow(), std::vector<double>(xMatrix.ncol()));
  for (int i = 0; i < xMatrix.nrow(); ++i) {
    for (int j = 0; j < xMatrix.ncol(); ++j) {
      xMatrix_cpp[i][j] = xMatrix(i, j);
    }
  }

  // Convert Rcpp NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> yMatrix_cpp(yMatrix.nrow(), std::vector<double>(yMatrix.ncol()));
  for (int i = 0; i < yMatrix.nrow(); ++i) {
    for (int j = 0; j < yMatrix.ncol(); ++j) {
      yMatrix_cpp[i][j] = yMatrix(i, j);
    }
  }

  // // Convert Rcpp IntegerVector to std::vector<int>
  // std::vector<int> libsizes_cpp(libsizes.size());
  // for (int i = 0; i < libsizes.size(); ++i) {
  //   libsizes_cpp[i] = libsizes[i];
  // }

  // Convert Rcpp::IntegerVector to std::vector<int>
  std::vector<int> libsizes_cpp = Rcpp::as<std::vector<int>>(libsizes);

  // Convert Rcpp IntegerMatrix to std::vector<std::pair<int, int>>
  std::vector<std::pair<int, int>> pred_cpp(pred.nrow());
  for (int i = 0; i < pred.nrow(); ++i) {
    pred_cpp[i] = std::make_pair(pred(i, 0), pred(i, 1));
  }

  // Call the C++ function GCCM4Grid
  std::vector<std::vector<double>> result = GCCM4Grid(
    xMatrix_cpp,
    yMatrix_cpp,
    libsizes_cpp,
    pred_cpp,
    E,
    tau,
    b,
    simplex,
    theta,
    threads,
    progressbar
  );

  Rcpp::NumericMatrix resultMatrix(result.size(), 5);
  for (size_t i = 0; i < result.size(); ++i) {
    resultMatrix(i, 0) = result[i][0];
    resultMatrix(i, 1) = result[i][1];
    resultMatrix(i, 2) = result[i][2];
    resultMatrix(i, 3) = result[i][3];
    resultMatrix(i, 4) = result[i][4];
  }

  // Set column names for the result matrix
  Rcpp::colnames(resultMatrix) = Rcpp::CharacterVector::create("libsizes",
                 "x_xmap_y_mean","x_xmap_y_sig",
                 "x_xmap_y_upper","x_xmap_y_lower");
  return resultMatrix;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix RcppSCPCM4Grid(
    const Rcpp::NumericMatrix& xMatrix,
    const Rcpp::NumericMatrix& yMatrix,
    const Rcpp::NumericMatrix& zMatrix,
    const Rcpp::IntegerVector& libsizes,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerMatrix& pred,
    int b,
    bool simplex,
    double theta,
    int threads,
    bool cumulate,
    bool progressbar) {

  // Convert Rcpp NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> xMatrix_cpp(xMatrix.nrow(), std::vector<double>(xMatrix.ncol()));
  for (int i = 0; i < xMatrix.nrow(); ++i) {
    for (int j = 0; j < xMatrix.ncol(); ++j) {
      xMatrix_cpp[i][j] = xMatrix(i, j);
    }
  }

  // Convert Rcpp NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> yMatrix_cpp(yMatrix.nrow(), std::vector<double>(yMatrix.ncol()));
  for (int i = 0; i < yMatrix.nrow(); ++i) {
    for (int j = 0; j < yMatrix.ncol(); ++j) {
      yMatrix_cpp[i][j] = yMatrix(i, j);
    }
  }

  // Convert Rcpp NumericMatrix to std::vector of std::vectors
  std::vector<std::vector<double>> zMatrix_cpp(zMatrix.ncol());
  for (int i = 0; i < zMatrix.ncol(); ++i) {
    Rcpp::NumericVector covvar = zMatrix.column(i);
    zMatrix_cpp[i] = Rcpp::as<std::vector<double>>(covvar);
  }

  // // Convert Rcpp IntegerVector to std::vector<int>
  // std::vector<int> lib_sizes_cpp(lib_sizes.size());
  // for (int i = 0; i < lib_sizes.size(); ++i) {
  //   lib_sizes_cpp[i] = lib_sizes[i];
  // }
  //
  // // Convert Rcpp IntegerVector to std::vector<int>
  // std::vector<int> E_cpp(E.size());
  // for (int i = 0; i < E.size(); ++i) {
  //   E_cpp[i] = E[i];
  // }

  // Convert Rcpp::IntegerVector to std::vector<int>
  std::vector<int> libsizes_cpp = Rcpp::as<std::vector<int>>(libsizes);
  std::vector<int> E_cpp = Rcpp::as<std::vector<int>>(E);
  std::vector<int> tau_cpp = Rcpp::as<std::vector<int>>(tau);

  // Convert Rcpp IntegerMatrix to std::vector<std::pair<int, int>>
  std::vector<std::pair<int, int>> pred_cpp(pred.nrow());
  for (int i = 0; i < pred.nrow(); ++i) {
    pred_cpp[i] = std::make_pair(pred(i, 0), pred(i, 1));
  }

  // Call the C++ function SCPCM4Grid
  std::vector<std::vector<double>> result = SCPCM4Grid(
    xMatrix_cpp,
    yMatrix_cpp,
    zMatrix_cpp,
    libsizes_cpp,
    pred_cpp,
    E_cpp,
    tau_cpp,
    b,
    simplex,
    theta,
    threads,
    cumulate,
    progressbar
  );

  // Convert std::vector<std::vector<double>> to Rcpp::NumericMatrix
  Rcpp::NumericMatrix resultMatrix(result.size(), 9);
  for (size_t i = 0; i < result.size(); ++i) {
    resultMatrix(i, 0) = result[i][0];
    resultMatrix(i, 1) = result[i][1];
    resultMatrix(i, 2) = result[i][2];
    resultMatrix(i, 3) = result[i][3];
    resultMatrix(i, 4) = result[i][4];
    resultMatrix(i, 5) = result[i][5];
    resultMatrix(i, 6) = result[i][6];
    resultMatrix(i, 7) = result[i][7];
    resultMatrix(i, 8) = result[i][8];
  }

  // Set column names for the result matrix
  Rcpp::colnames(resultMatrix) = Rcpp::CharacterVector::create(
    "libsizes","T_mean","D_mean",
    "T_sig","T_upper","T_lower",
    "D_sig","D_upper","D_lower");
  return resultMatrix;
}

// Wrapper function to perform GCMC Grid and return a NumericVector
// [[Rcpp::export]]
Rcpp::NumericVector RcppGCMC4Grid(
    const Rcpp::NumericMatrix& xMatrix,
    const Rcpp::NumericMatrix& yMatrix,
    const Rcpp::IntegerMatrix& pred,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerVector& b,
    const Rcpp::IntegerVector& max_r,
    int threads,
    bool progressbar){
  // Convert Rcpp NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> xMatrix_cpp(xMatrix.nrow(), std::vector<double>(xMatrix.ncol()));
  for (int i = 0; i < xMatrix.nrow(); ++i) {
    for (int j = 0; j < xMatrix.ncol(); ++j) {
      xMatrix_cpp[i][j] = xMatrix(i, j);
    }
  }

  // Convert Rcpp NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> yMatrix_cpp(yMatrix.nrow(), std::vector<double>(yMatrix.ncol()));
  for (int i = 0; i < yMatrix.nrow(); ++i) {
    for (int j = 0; j < yMatrix.ncol(); ++j) {
      yMatrix_cpp[i][j] = yMatrix(i, j);
    }
  }

  // Convert Rcpp IntegerVector to std::vector<int>
  std::vector<int> E_std = Rcpp::as<std::vector<int>>(E);
  std::vector<int> tau_std = Rcpp::as<std::vector<int>>(tau);
  std::vector<int> b_std = Rcpp::as<std::vector<int>>(b);
  std::vector<int> maxr_std = Rcpp::as<std::vector<int>>(max_r);

  // Convert Rcpp IntegerMatrix to std::vector<int>
  int numRows = xMatrix.nrow();
  int numCols = xMatrix.ncol();
  std::vector<int> pred_std;
  for (int i = 0; i < pred.nrow(); ++i) {
    pred_std.push_back(LocateGridIndices(pred(i,0), pred(i,1), numRows, numCols));
  }

  // Generate embeddings
  std::vector<std::vector<double>> e1 = GenGridEmbeddings(xMatrix_cpp, E[0], tau_std[0]);
  std::vector<std::vector<double>> e2 = GenGridEmbeddings(yMatrix_cpp, E[1], tau_std[1]);

  // Perform GCMC For Grid
  double cs1 = IntersectionCardinality(e1,e2,pred_std,b_std[0],maxr_std[0],threads,progressbar);
  double cs2 = IntersectionCardinality(e2,e1,pred_std,b_std[1],maxr_std[1],threads,progressbar);

  Rcpp::NumericVector res_vec = Rcpp::NumericVector::create(
    Rcpp::Named("x_xmap_y",cs1),
    Rcpp::Named("y_xmap_x",cs2));

  return res_vec;
}
