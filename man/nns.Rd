% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nns.R
\name{nns}
\alias{nns}
\title{Given a set of embeddings and a set of candidate neighbors, find the top N nearest
neighbors.}
\usage{
nns(
  x,
  N = 10,
  candidates = character(0),
  pre_trained,
  stem = FALSE,
  language = "porter",
  as_list = TRUE,
  show_language = TRUE
)
}
\arguments{
\item{x}{a \code{dem-class} or \code{fem-class} object.}

\item{N}{(numeric) number of nearest neighbors to return}

\item{candidates}{(character) vector of features to consider as candidates to be nearest neighbor
You may for example want to only consider features that meet a certain count threshold
or exclude stop words etc. To do so you can simply identify the set of features you
want to consider and supply these as a character vector in the \code{candidates} argument.}

\item{pre_trained}{(numeric) a F x D matrix corresponding to pretrained embeddings.
F = number of features and D = embedding dimensions.
rownames(pre_trained) = set of features for which there is a pre-trained embedding.}

\item{stem}{(logical) - whether to stem candidates when evaluating nns. Default is FALSE.
If TRUE, candidate stems are ranked by their average cosine similarity to the target.
We recommend you remove misspelled words from candidate set \code{candidates} as these can
significantly influence the average.}

\item{language}{the name of a recognized language, as returned by
     \code{\link[SnowballC]{getStemLanguages}}, or a two- or three-letter ISO-639
     code corresponding to one of these languages (see references for
     the list of codes).
  }

\item{as_list}{(logical) if FALSE all results are combined into a single data.frame
If TRUE, a list of data.frames is returned with one data.frame per group.}

\item{show_language}{(logical) if TRUE print out message with language used for stemming.}
}
\value{
a \code{data.frame} or list of data.frames (one for each target)
with the following columns:
\describe{
\item{\code{target}}{ (character) rownames of \code{x},
the labels of the ALC embeddings. \code{NA} if \code{is.null(rownames(x))}.}
\item{\code{feature}}{(character) features identified as nearest neighbors.}
\item{\code{rank}}{(character) rank of feature in terms of similarity with \code{x}.}
\item{\code{value}}{(numeric) cosine similarity between \code{x} and feature.}
}
}
\description{
Given a set of embeddings and a set of candidate neighbors, find the top N nearest
neighbors.
}
\examples{

library(quanteda)

# tokenize corpus
toks <- tokens(cr_sample_corpus)

# build a tokenized corpus of contexts sorrounding a target term
immig_toks <- tokens_context(x = toks, pattern = "immigr*", window = 6L)

# build document-feature matrix
immig_dfm <- dfm(immig_toks)

# construct document-embedding-matrix
immig_dem <- dem(immig_dfm, pre_trained = cr_glove_subset,
transform = TRUE, transform_matrix = cr_transform, verbose = FALSE)

# to get group-specific embeddings, average within party
immig_wv_party <- dem_group(immig_dem, groups = immig_dem@docvars$party)

# find nearest neighbors by party
# setting as_list = FALSE combines each group's
# results into a single tibble (useful for joint plotting)
immig_nns <- nns(immig_wv_party, pre_trained = cr_glove_subset,
N = 5, candidates = immig_wv_party@features, stem = FALSE, as_list = TRUE)
}
\keyword{nns}
