#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>

/**
 * Calculate the one-dimensional index for a specified position in a 2D grid.
 *
 * This function converts a row and column position in a 2D grid into a corresponding
 * one-dimensional index, assuming the grid is stored in row-major order (i.e., rows are stored sequentially).
 *
 * @param curRow   The current row number (1-based indexing).
 * @param curCol   The current column number (1-based indexing).
 * @param totalRow The total number of rows in the grid.
 * @param totalCol The total number of columns in the grid.
 * @return         The calculated one-dimensional index (0-based indexing).
 */
int LocateGridIndices(int curRow, int curCol, int totalRow, int totalCol) {
  return (curRow - 1) * totalCol + curCol - 1;
}

// Function to save the grid data matrix format as a vector row by row
std::vector<double> GridMat2Vec(const std::vector<std::vector<double>>& Matrix){
  std::vector<double> vec;
  for (const auto& row : Matrix) {
    vec.insert(vec.end(), row.begin(), row.end());
  }
  return vec;
}

// Function to save the grid data vector format as a matrix by rows
std::vector<std::vector<double>> GridVec2Mat(const std::vector<double>& Vec,
                                             int NROW){
  // Calculate the number of columns based on the vector size and number of rows
  int NCOL = Vec.size() / NROW;

  // Create the resulting matrix with NROW rows and NCOL columns
  std::vector<std::vector<double>> matrix(NROW, std::vector<double>(NCOL));

  // Fill the matrix with values from the input vector
  for (int i = 0; i < NROW; ++i) {
    for (int j = 0; j < NCOL; ++j) {
      matrix[i][j] = Vec[i * NCOL + j];
    }
  }

  return matrix;
}

// Note that the return value is the value of the lagged order position, not the index.
std::vector<std::vector<double>> CppLaggedVar4Grid(
    const std::vector<std::vector<double>>& mat,
    int lagNum
) {
  int numRows = mat.size();
  int numCols = mat[0].size();
  int totalElements = numRows * numCols;
  std::vector<std::vector<double>> result(totalElements, std::vector<double>(8, std::numeric_limits<double>::quiet_NaN()));

  // Directions for Moore neighborhood (8 directions)
  const int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
  const int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      int elementIndex = r * numCols + c;
      int neighborIndex = 0;

      // Calculate positions that are exactly 'lagNum' units away
      for (int dir = 0; dir < 8; ++dir) {
        int dr = dx[dir] * lagNum;
        int dc = dy[dir] * lagNum;
        int rNeighbor = r + dr;
        int cNeighbor = c + dc;

        // Check if the neighbor position is within bounds
        if (rNeighbor >= 0 && rNeighbor < numRows && cNeighbor >= 0 && cNeighbor < numCols) {
          result[elementIndex][neighborIndex] = mat[rNeighbor][cNeighbor];
        }
        // Else, it remains NaN

        neighborIndex++;
        if (neighborIndex >= 8) {
          break;
        }
      }
    }
  }

  return result;
}

// Similar to GenLatticeEmbeddings
std::vector<std::vector<double>> GenGridEmbeddings(
    const std::vector<std::vector<double>>& mat,
    int E,
    bool includeself) {
  // Calculate the total number of elements in all subsets of mat
  int total_elements = 0;
  for (const auto& subset : mat) {
    total_elements += subset.size();
  }

  // Initialize the result matrix with total_elements rows and E columns
  std::vector<std::vector<double>> result(total_elements, std::vector<double>(E, 0.0));

  if (includeself) {
    // Fill the first column with the elements from mat
    int row = 0;
    for (const auto& subset : mat) {
      for (double value : subset) {
        result[row][0] = value;
        ++row;
      }
    }

    // Fill the remaining columns (2 to E) with the averaged lagged variables
    for (int lagNum = 1; lagNum < E; ++lagNum) {
      // Calculate the lagged variables for the current lagNum
      std::vector<std::vector<double>> lagged_vars = CppLaggedVar4Grid(mat, lagNum);

      // Fill the current column (lagNum + 1) with the averaged lagged variables
      row = 0;
      for (const auto& subset : lagged_vars) {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < 8; ++i) {
          double val = subset[i];
          if (!std::isnan(val)) {
            sum += val;
            ++count;
          }
        }
        if (count > 0) {
          result[row][lagNum] = sum / count;
        } else {
          result[row][lagNum] = std::numeric_limits<double>::quiet_NaN();
        }
        ++row;
      }
    }
  } else {
    int row = 0;
    for (int lagNum = 1; lagNum <= E; ++lagNum) {
      // Calculate the lagged variables for the current lagNum
      std::vector<std::vector<double>> lagged_vars = CppLaggedVar4Grid(mat, lagNum);

      // Fill the current column (lagNum) with the averaged lagged variables
      row = 0;
      for (const auto& subset : lagged_vars) {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < 8; ++i) {
          double val = subset[i];
          if (!std::isnan(val)) {
            sum += val;
            ++count;
          }
        }
        if (count > 0) {
          result[row][lagNum-1] = sum / count;
        } else {
          result[row][lagNum-1] = std::numeric_limits<double>::quiet_NaN();
        }
        ++row;
      }
    }
  }

  return result;
}

// std::vector<std::vector<std::vector<double>>> GenGridEmbeddings2(
//     const std::vector<std::vector<double>>& mat, int E) {
//   // Initialize a vector to store the embeddings
//   std::vector<std::vector<std::vector<double>>> xEmbeddings(E);
//
//   // The first embedding is the transpose of the input matrix
//   int numRows = mat.size();
//   int numCols = mat[0].size();
//   xEmbeddings[0].resize(numCols, std::vector<double>(numRows));
//
//   for (int r = 0; r < numRows; ++r) {
//     for (int c = 0; c < numCols; ++c) {
//       xEmbeddings[0][c][r] = mat[r][c]; // Transpose the matrix
//     }
//   }
//
//   // Generate the remaining embeddings using laggedVariableAs2Dim
//   for (int i = 1; i < E; ++i) {
//     xEmbeddings[i] = CppLaggedVar4Grid(mat, i);
//   }
//
//   return xEmbeddings;
// }
