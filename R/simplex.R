methods::setGeneric("simplex", function(data, ...) standardGeneric("simplex"))

.simplex_sf_method = \(data,target,lib,pred = lib,E = 1:10,tau = 1,
                       k = 4, nb = NULL, threads = detectThreads()){
  vec = .uni_lattice(data,target)
  lib = .check_indices(lib,length(vec))
  pred = .check_indices(pred,length(vec))
  if (is.null(nb)) nb = .internal_lattice_nb(data)
  res = RcppSimplex4Lattice(vec,nb,lib,pred,E,tau,k,threads)
  cat(paste0("The suggested embedding dimension E for variable ",target," is ",OptEmdedDim(res)), "\n")
  return(res)
}

.simplex_spatraster_method = \(data,target,lib,pred = lib,E = 1:10,tau = 1,
                               k = 4, threads = detectThreads()){
  mat = .uni_grid(data,target)
  res = RcppSimplex4Grid(mat,lib,pred,E,tau,k,threads)
  cat(paste0("The suggested embedding dimension E for variable ",target," is ",OptEmdedDim(res)), "\n")
  return(res)
}

#' simplex forecasting
#'
#' @inheritParams embedded
#' @param lib Row numbers(`vector`) of lattice data or row-column numbers(`matrix`) of grid data for creating the library from observations.
#' @param pred (optional) Row numbers(`vector`) of lattice data or row-column numbers(`matrix`) of grid data used for predictions.
#' @param k (optional) Number of nearest neighbors to use for prediction.
#' @param threads (optional) Number of threads.
#'
#' @return A matrix
#' @export
#'
#' @name simplex
#' @rdname simplex
#' @aliases simplex,sf-method
#'
#' @examples
#' columbus = sf::read_sf(system.file("shapes/columbus.gpkg", package="spData"))
#' \donttest{
#' simplex(columbus,target = "CRIME",lib = 1:49)
#' }
methods::setMethod("simplex", "sf", .simplex_sf_method)

#' @rdname simplex
methods::setMethod("simplex", "SpatRaster", .simplex_spatraster_method)
