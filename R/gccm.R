methods::setGeneric("gccm", function(data, ...) standardGeneric("gccm"))

.gccm_sf_method = \(data, cause, effect, libsizes, E = 3, tau = 1, k = E+2, theta = 1, algorithm = "simplex", lib = NULL, pred = NULL,
                    nb = NULL,threads = detectThreads(),parallel.level = "low",bidirectional = TRUE,trend.rm = TRUE,progressbar = TRUE){
  varname = .check_character(cause, effect)
  E = .check_inputelementnum(E,2)
  tau = .check_inputelementnum(tau,2)
  k = .check_inputelementnum(k,2)
  pl = .check_parallellevel(parallel.level)
  .varname = .internal_varname()
  if (is.null(nb)) nb = .internal_lattice_nb(data)
  coords = as.data.frame(sdsfun::sf_coordinates(data))
  data = sf::st_drop_geometry(data)
  data = data[,varname]
  names(data) = .varname

  if (trend.rm){
    data = .internal_trend_rm(data,.varname,coords)
  }
  cause = data[,"cause",drop = TRUE]
  effect = data[,"effect",drop = TRUE]

  if (is.null(lib)) lib = .internal_library(data)
  if (is.null(pred)) pred = lib

  simplex = ifelse(algorithm == "simplex", TRUE, FALSE)
  x_xmap_y = NULL
  if (bidirectional){
    x_xmap_y = RcppGCCM4Lattice(cause,effect,nb,libsizes,lib,pred,E[1],tau[1],k[1],simplex,theta,threads,pl,progressbar)
  }
  y_xmap_x = RcppGCCM4Lattice(effect,cause,nb,libsizes,lib,pred,E[2],tau[2],k[2],simplex,theta,threads,pl,progressbar)

  return(.bind_xmapdf(varname,x_xmap_y,y_xmap_x,bidirectional))
}

.gccm_spatraster_method = \(data, cause, effect, libsizes, E = 3, tau = 1, k = E+2, theta = 1, algorithm = "simplex", lib = NULL, pred = NULL,
                            threads = detectThreads(), parallel.level = "low", bidirectional = TRUE, trend.rm = TRUE, progressbar = TRUE){
  varname = .check_character(cause, effect)
  E = .check_inputelementnum(E,2)
  tau = .check_inputelementnum(tau,2)
  k = .check_inputelementnum(k,2)
  pl = .check_parallellevel(parallel.level)
  libsizes = as.matrix(libsizes)
  .varname = .internal_varname()
  data = data[[varname]]
  names(data) = .varname

  dtf = terra::as.data.frame(data,xy = TRUE,na.rm = FALSE)
  if (trend.rm){
    dtf = .internal_trend_rm(dtf,.varname)
  }
  causemat = matrix(dtf[,"cause"],nrow = terra::nrow(data),byrow = TRUE)
  effectmat = matrix(dtf[,"effect"],nrow = terra::nrow(data),byrow = TRUE)

  if (is.null(lib)) lib = .internal_library(dtf,TRUE)
  if (is.null(pred)) pred = lib

  simplex = ifelse(algorithm == "simplex", TRUE, FALSE)
  x_xmap_y = NULL
  if (bidirectional){
    x_xmap_y = RcppGCCM4Grid(causemat,effectmat,libsizes,lib,pred,E[1],tau[1],k[1],simplex,theta,threads,pl,progressbar)
  }
  y_xmap_x = RcppGCCM4Grid(effectmat,causemat,libsizes,lib,pred,E[2],tau[2],k[2],simplex,theta,threads,pl,progressbar)

  return(.bind_xmapdf(varname,x_xmap_y,y_xmap_x,bidirectional))
}

#' geographical convergent cross mapping
#'
#' @param data The observation data.
#' @param cause Name of causal variable.
#' @param effect Name of effect variable.
#' @param libsizes Number of spatial units used in prediction.
#' @param E (optional) Dimensions of the embedding.
#' @param tau (optional) Step of spatial lags.
#' @param k (optional) Number of nearest neighbors used in prediction.
#' @param theta (optional) Weighting parameter for distances, useful when `algorithm` is `smap`.
#' @param algorithm (optional) Algorithm used in prediction.
#' @param lib (optional) Libraries indices.
#' @param pred (optional) Predictions indices.
#' @param nb (optional) The neighbours list.
#' @param threads (optional) Number of threads.
#' @param parallel.level (optional) Level of parallelism, `low` or `high`.
#' @param bidirectional (optional) whether to examine bidirectional causality.
#' @param trend.rm (optional) Whether to remove the linear trend.
#' @param progressbar (optional) whether to show the progress bar.
#'
#' @return A list
#' \describe{
#' \item{\code{xmap}}{cross mapping results}
#' \item{\code{varname}}{names of causal and effect variable}
#' \item{\code{bidirectional}}{whether to examine bidirectional causality}
#' }
#' @export
#' @importFrom methods setGeneric
#' @importFrom methods setMethod
#' @name gccm
#' @rdname gccm
#' @aliases gccm,sf-method
#' @references
#' Gao, B., Yang, J., Chen, Z. et al. Causal inference from cross-sectional earth system data with geographical convergent cross mapping. Nat Commun 14, 5875 (2023).
#'
#' @examples
#' columbus = sf::read_sf(system.file("case/columbus.gpkg", package="spEDM"))
#' \donttest{
#' g = gccm(columbus,"hoval","crime",libsizes = seq(5,45,5),E = 6)
#' g
#' plot(g, ylimits = c(0,0.85))
#' }
methods::setMethod("gccm", "sf", .gccm_sf_method)

#' @rdname gccm
methods::setMethod("gccm", "SpatRaster", .gccm_spatraster_method)
