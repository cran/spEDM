#include <vector>
#include "CppGridUtils.h"
#include "GCCM4Grid.h"
#include <Rcpp.h>

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
Rcpp::NumericMatrix RcppGenGridEmbeddings(Rcpp::NumericMatrix mat, int E) {
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
  std::vector<std::vector<double>> embeddings = GenGridEmbeddings(cppMat, E);

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
Rcpp::NumericMatrix RcppGCCM4Grid(
    const Rcpp::NumericMatrix& xMatrix,
    const Rcpp::NumericMatrix& yMatrix,
    const Rcpp::IntegerVector& lib_sizes,
    const Rcpp::IntegerMatrix& pred,
    int E,
    int tau,
    int b,
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

  // Convert Rcpp IntegerVector to std::vector<int>
  std::vector<int> lib_sizes_cpp(lib_sizes.size());
  for (int i = 0; i < lib_sizes.size(); ++i) {
    lib_sizes_cpp[i] = lib_sizes[i];
  }

  // Convert Rcpp IntegerMatrix to std::vector<std::pair<int, int>>
  std::vector<std::pair<int, int>> pred_cpp(pred.nrow());
  for (int i = 0; i < pred.nrow(); ++i) {
    pred_cpp[i] = std::make_pair(pred(i, 0), pred(i, 1));
  }

  // Call the C++ function GCCM4Grid
  std::vector<std::vector<double>> result = GCCM4Grid(
    xMatrix_cpp,
    yMatrix_cpp,
    lib_sizes_cpp,
    pred_cpp,
    E,
    tau,
    b,
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

  return resultMatrix;
}

// // [[Rcpp::export]]
// Rcpp::List RcppGenGridEmbeddings2(Rcpp::NumericMatrix mat, int E) {
//   // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
//   int numRows = mat.nrow();
//   int numCols = mat.ncol();
//   std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));
//
//   for (int r = 0; r < numRows; ++r) {
//     for (int c = 0; c < numCols; ++c) {
//       cppMat[r][c] = mat(r, c);
//     }
//   }
//
//   // Call the GenGridEmbeddings function
//   std::vector<std::vector<std::vector<double>>> embeddings = GenGridEmbeddings2(cppMat, E);
//
//   // Convert the result back to an Rcpp::List of Rcpp::NumericMatrix
//   Rcpp::List result(E + 1);
//
//   for (int i = 0; i <= E; ++i) {
//     int embeddingRows = embeddings[i].size();
//     int embeddingCols = embeddings[i][0].size();
//     Rcpp::NumericMatrix embeddingMat(embeddingRows, embeddingCols);
//
//     for (int r = 0; r < embeddingRows; ++r) {
//       for (int c = 0; c < embeddingCols; ++c) {
//         embeddingMat(r, c) = embeddings[i][r][c];
//       }
//     }
//
//     result[i] = embeddingMat;
//   }
//
//   return result;
// }
