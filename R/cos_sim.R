#' Compute the cosine similarity between one or more ALC embeddings and a set of features.
#'
#' @param x a (quanteda) `dem-class` or `fem-class` object.
#' @param pre_trained (numeric) a F x D matrix corresponding to pretrained embeddings.
#' F = number of features and D = embedding dimensions.
#' rownames(pre_trained) = set of features for which there is a pre-trained embedding.
#' @param features (character) features of interest.
#' @param stem (logical) - If TRUE, both `features` and `rownames(pre_trained)`
#' are stemmed and average cosine similarities are reported.
#' We recommend you remove misspelled words from  `pre_trained` as these can
#' significantly influence the average.
#' @inheritParams SnowballC::wordStem
#' @param as_list (logical) if FALSE all results are combined into a single data.frame
#' If TRUE, a list of data.frames is returned with one data.frame per feature.
#' @param show_language (logical) if TRUE print out message with language used for stemming.
#'
#' @return a `data.frame` or list of data.frames (one for each target)
#' with the following columns:
#' \describe{
#'  \item{`target`}{ (character) rownames of `x`,
#'  the labels of the ALC embeddings.
#'  NA if is.null(rownames(x)).}
#'  \item{`feature`}{(character) feature terms defined in
#'  the `features` argument.}
#'  \item{`value`}{(numeric) cosine similarity between `x`
#'  and feature.}
#'  }
#'
#' @export
#' @rdname cos_sim
#' @keywords cos_sim
#' @examples
#'
#' library(quanteda)
#'
#' # tokenize corpus
#' toks <- tokens(cr_sample_corpus)
#'
#' # build a tokenized corpus of contexts sorrounding a target term
#' immig_toks <- tokens_context(x = toks, pattern = "immigr*", window = 6L)
#'
#' # build document-feature matrix
#' immig_dfm <- dfm(immig_toks)
#'
#' # construct document-embedding-matrix
#' immig_dem <- dem(immig_dfm, pre_trained = cr_glove_subset,
#' transform = TRUE, transform_matrix = cr_transform, verbose = FALSE)
#'
#' # to get group-specific embeddings, average within party
#' immig_wv_party <- dem_group(immig_dem, groups = immig_dem@docvars$party)
#'
#' # compute the cosine similarity between each party's embedding and a specific set of features
#' cos_sim(x = immig_wv_party, pre_trained = cr_glove_subset,
#' features = c('reform', 'enforcement'), as_list = FALSE)
cos_sim <- function(x, pre_trained, features = NULL, stem = FALSE, language = 'porter', as_list = TRUE, show_language = TRUE){

  # for single numeric vectors
  if(is.null(dim(x)) && length(x) == dim(pre_trained)[2]) x <- matrix(x, nrow = 1)

  # check features is non-empty
  if(is.null(features)) stop('the "features" argument cannot be NULL')

  # check features are in pre-trained embeddings
  if(stem){
    if (requireNamespace("SnowballC", quietly = TRUE)) {
      if(show_language) cat('Using', language, 'for stemming. To check available languages run "SnowballC::getStemLanguages()"', '\n')
      features <- SnowballC::wordStem(features)
      pre_trained_feats <- SnowballC::wordStem(rownames(pre_trained))
    } else stop('"SnowballC (>= 0.7.0)" package must be installed to use stemmming option.')
  } else pre_trained_feats <- rownames(pre_trained)

  # rename features in pre_trained (only changes if stem)
  rownames(pre_trained) <- pre_trained_feats

  # check which, if any features, are present
  feature_check <- features %in% pre_trained_feats
  if(!any(feature_check)) stop('none of features appear to have an embedding in the set of pre-trained embeddings provided, please select other features.', call. = FALSE)
  if(!all(feature_check)) warning('the following features do not appear to have an embedding in the set of pre-trained embeddings provided: ', paste(features[which(!feature_check)], collapse = ', '))
  features_present <- features[feature_check]

  # subset pre-trained embeddings to features of interest
  pre_trained_subset <- pre_trained[rownames(pre_trained) %in% features_present,]

  # compute cosine similarity
  cos_sim <- text2vec::sim2(as.matrix(x), as.matrix(pre_trained_subset), method = 'cosine', norm = 'l2')

  # convert to dataframe
  cos_sim <- reshape2::melt(as.matrix(cos_sim)) %>% setNames(c('target', 'feature', 'value'))
  if(is.null(rownames(x))) cos_sim$target <- NA

  # stemming
  if(stem) result <- cos_sim %>% dplyr::mutate(feature = SnowballC::wordStem(feature, language = language)) %>% dplyr::group_by(target, feature) %>% dplyr::summarise(dplyr::across(where(is.numeric), mean), .groups = "drop") %>% dplyr::ungroup()
  else result <- cos_sim

  # if !as_list return a list object with an item for each feature data.frame
  if(as_list) result <- lapply(unique(result$feature), function(i) result[result$feature == i,] %>% dplyr::mutate(feature = as.character(feature))) %>% setNames(unique(result$feature))

  return(result)
}
