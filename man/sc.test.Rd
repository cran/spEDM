% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sctest.R
\name{sc.test}
\alias{sc.test}
\alias{sc.test,sf-method}
\alias{sc.test,SpatRaster-method}
\title{spatial causality test}
\usage{
\S4method{sc.test}{sf}(
  data,
  cause,
  effect,
  k,
  block = 3,
  boot = 399,
  seed = 42,
  base = 2,
  lib = NULL,
  pred = NULL,
  nb = NULL,
  threads = detectThreads(),
  trend.rm = TRUE,
  normalize = FALSE,
  progressbar = FALSE
)

\S4method{sc.test}{SpatRaster}(
  data,
  cause,
  effect,
  k,
  block = 3,
  boot = 399,
  seed = 42,
  base = 2,
  lib = NULL,
  pred = NULL,
  threads = detectThreads(),
  trend.rm = TRUE,
  normalize = FALSE,
  progressbar = FALSE
)
}
\arguments{
\item{data}{The observation data.}

\item{cause}{Name of causal variable.}

\item{effect}{Name of effect variable.}

\item{k}{(optional) Number of nearest neighbors used in symbolization.}

\item{block}{(optional) Number of blocks used in spatial block bootstrap.}

\item{boot}{(optional) Number of bootstraps to perform.}

\item{seed}{(optional) The random seed.}

\item{base}{(optional) Base of the logarithm.}

\item{lib}{(optional) Libraries indices.}

\item{pred}{(optional) Predictions indices.}

\item{nb}{(optional) The neighbours list.}

\item{threads}{(optional) Number of threads.}

\item{trend.rm}{(optional) Whether to remove the linear trend.}

\item{normalize}{(optional) Whether to normalize the result.}

\item{progressbar}{(optional) Whether to show the progress bar.}
}
\value{
A list
\describe{
\item{\code{sc}}{statistic for spatial causality}
\item{\code{varname}}{names of causal and effect variable}
}
}
\description{
spatial causality test
}
\examples{
columbus = sf::read_sf(system.file("case/columbus.gpkg", package="spEDM"))
\donttest{
sc.test(columbus,"hoval","crime", k = 15)
}
}
\references{
Herrera, M., Mur, J., & Ruiz, M. (2016). Detecting causal relationships between spatial processes. Papers in Regional Science, 95(3), 577–595.
}
