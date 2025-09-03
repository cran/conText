## ----echo = FALSE-------------------------------------------------------------
knitr::opts_chunk$set(collapse = FALSE,
                      dpi = 72,
                      warning = FALSE,
                      message = FALSE,
                      comment = "##",
                      tidy = FALSE)

## ----eval=FALSE---------------------------------------------------------------
# install.packages("conText")

## ----eval=FALSE---------------------------------------------------------------
# devtools::install_github("prodriguezsosa/conText")

## ----message=FALSE------------------------------------------------------------
library(conText)
# other libraries used in this guide
library(quanteda)
library(dplyr)
library(text2vec)

## ----message = FALSE----------------------------------------------------------
# tokenize corpus removing unnecessary (i.e. semantically uninformative) elements
toks <- tokens(cr_sample_corpus, remove_punct=T, remove_symbols=T, remove_numbers=T, remove_separators=T)

# clean out stopwords and words with 2 or fewer characters
toks_nostop <- tokens_select(toks, pattern = stopwords("en"), selection = "remove", min_nchar=3)

# only use features that appear at least 5 times in the corpus
feats <- dfm(toks_nostop, tolower=T, verbose = FALSE) %>% dfm_trim(min_termfreq = 5) %>% featnames()

# leave the pads so that non-adjacent words will not become adjacent
toks_nostop_feats <- tokens_select(toks_nostop, feats, padding = TRUE)

## ----message = FALSE----------------------------------------------------------
# build a tokenized corpus of contexts surrounding the target term "immigration"
immig_toks <- tokens_context(x = toks_nostop_feats, pattern = "immigr*", window = 6L)
head(docvars(immig_toks), 3)

## ----message = FALSE----------------------------------------------------------
# build document-feature matrix
immig_dfm <- dfm(immig_toks)
immig_dfm[1:3,1:3]

## ----message = FALSE----------------------------------------------------------
# build a document-embedding-matrix
immig_dem <- dem(x = immig_dfm, pre_trained = cr_glove_subset, transform = TRUE, transform_matrix = cr_transform, verbose = TRUE)

# each document inherits its corresponding docvars
#head(immig_dem@docvars)

# you can check which documents were not embedded due to lack of overlapping features (in this example all documents are embedded)
# note: 'quanteda' functions like `docvars()` and `docnames()` don't work on `dem` objects, so you will have to call the attributes directly. 
#setdiff(docnames(immig_dfm), immig_dem@Dimnames$docs)

# vector of features used to create the embeddings
#head(immig_dem@features)

## ----message = FALSE----------------------------------------------------------
# to get a single "corpus-wide" embedding, take the column average
immig_wv <- matrix(colMeans(immig_dem), ncol = ncol(immig_dem)) %>%  `rownames<-`("immigration")
dim(immig_wv)

## ----message = FALSE----------------------------------------------------------
# to get group-specific embeddings, average within party
immig_wv_party <- dem_group(immig_dem, groups = immig_dem@docvars$party)
dim(immig_wv_party)

## -----------------------------------------------------------------------------
# find nearest neighbors by party
# setting as_list = FALSE combines each group's results into a single tibble (useful for joint plotting)
immig_nns <- nns(immig_wv_party, pre_trained = cr_glove_subset, N = 5, candidates = immig_wv_party@features, as_list = TRUE)

# check out results for Republican party
immig_nns[["R"]]

## -----------------------------------------------------------------------------
# compute the cosine similarity between each party's embedding and a specific set of features
cos_sim(immig_wv_party, pre_trained = cr_glove_subset, features = c('reform', 'enforcement'), as_list = FALSE)

## -----------------------------------------------------------------------------
# compute the cosine similarity between each party's embedding and a specific set of features
nns_ratio(x = immig_wv_party, N = 5, numerator = "R", candidates = immig_wv_party@features, pre_trained = cr_glove_subset, verbose = FALSE)

## -----------------------------------------------------------------------------

# compute the cosine similarity between each party's embedding and a set of tokenized contexts
immig_ncs <- ncs(x = immig_wv_party, contexts_dem = immig_dem, contexts = immig_toks, N = 5, as_list = TRUE)

# nearest contexts to Republican embedding of target term
# note, these may included contexts originating from Democrat speakers
immig_ncs[["R"]]

# you can limit candidate contexts to those of a specific party
immig_ncs <- ncs(x = immig_wv_party["R",], contexts_dem = immig_dem[immig_dem@docvars$party == "R",], contexts = immig_toks, N = 5, as_list = FALSE)


## ----warning=F, message=F-----------------------------------------------------
# extract candidate features from the dem object
immig_feats <- immig_wv_party@features

# check spelling. toupper avoids names being considered misspelled
if (requireNamespace("hunspell", quietly = TRUE)) { 
  library(hunspell) # spell check library
  spellcheck <-  hunspell_check(toupper(immig_feats), dict = hunspell::dictionary("en_US"))
  immig_feats <- immig_feats[spellcheck]
  }

# find nearest neighbors by party using stemming
immig_nns_stem <- nns(immig_wv_party, pre_trained = cr_glove_subset, N = 5, candidates = immig_feats, stem = TRUE, as_list = TRUE)

# check out results for Republican party
immig_nns_stem[["R"]]

## ----message = FALSE----------------------------------------------------------
# build a corpus of contexts surrounding the target term "immigration"
mkws_toks <- tokens_context(x = toks_nostop_feats, pattern = c("immigration", "welfare", "immigration reform", "economy"), window = 6L, verbose = FALSE)

# create document-feature matrix
mkws_dfm <- dfm(mkws_toks)

# create document-embedding matrix using a la carte
mkws_dem <- dem(x = mkws_dfm, pre_trained = cr_glove_subset, transform = TRUE, transform_matrix = cr_transform, verbose = FALSE)

# get embeddings for each pattern
mkws_wvs <- dem_group(mkws_dem, groups = mkws_dem@docvars$pattern)

# find nearest neighbors for each keyword
mkws_nns <- nns(mkws_wvs, pre_trained = cr_glove_subset, N = 3, candidates = mkws_wvs@features, as_list = TRUE)

# to check results for a given pattern
mkws_nns[["immigration reform"]]

## ----message = FALSE----------------------------------------------------------
# build a corpus of contexts surrounding the immigration related words
topical_toks <- tokens_context(x = toks_nostop_feats, pattern = c("immigration", "immigrant", "immigration reform"), window = 6L, verbose = FALSE)

# create document-feature matrix
topical_dfm <- dfm(topical_toks)

# create document-embedding matrix using a la carte
topical_dem <- dem(x = topical_dfm, pre_trained = cr_glove_subset, transform = TRUE, transform_matrix = cr_transform, verbose = FALSE)

# get "topical" embeddings for each party
topical_wvs <- dem_group(topical_dem, groups = topical_dem@docvars$party)

# find nearest neighbors for each keyword
nns(topical_wvs, pre_trained = cr_glove_subset, N = 5, candidates = topical_wvs@features, stem = TRUE, as_list = FALSE)

## ----message = FALSE----------------------------------------------------------

# we limit candidates to features in our corpus
feats <- featnames(dfm(immig_toks))

# compare nearest neighbors between groups
set.seed(2021L)
immig_party_nns <- get_nns(x = immig_toks, N = 10,
        groups = docvars(immig_toks, 'party'),
        candidates = feats,
        pre_trained = cr_glove_subset,
        transform = TRUE,
        transform_matrix = cr_transform,
        bootstrap = TRUE,
        num_bootstraps = 100, 
        confidence_level = 0.95,
        as_list = TRUE)

# nearest neighbors of "immigration" for Republican party
immig_party_nns[["R"]]

## ----message = FALSE----------------------------------------------------------

# compute the cosine similarity between each group's embedding and a specific set of features
set.seed(2021L)
get_cos_sim(x = immig_toks,
            groups = docvars(immig_toks, 'party'),
            features = c("reform", "enforce"),
            pre_trained = cr_glove_subset,
            transform = TRUE,
            transform_matrix = cr_transform,
            bootstrap = TRUE,
            num_bootstraps = 100,
            as_list = FALSE)

## ----message = FALSE----------------------------------------------------------

# we limit candidates to features in our corpus
feats <- featnames(dfm(immig_toks))

# compute ratio
set.seed(2021L)
immig_nns_ratio <- get_nns_ratio(x = immig_toks, 
              N = 10,
              groups = docvars(immig_toks, 'party'),
              numerator = "R",
              candidates = feats,
              pre_trained = cr_glove_subset,
              transform = TRUE,
              transform_matrix = cr_transform,
              bootstrap = TRUE,
              num_bootstraps = 100,
              permute = TRUE,
              num_permutations = 100,
              verbose = FALSE)

head(immig_nns_ratio)

## ----eval=FALSE---------------------------------------------------------------
# plot_nns_ratio(x = immig_nns_ratio, alpha = 0.01, horizontal = TRUE)

## ----message = FALSE----------------------------------------------------------
# compare nearest neighbors between groups
set.seed(2021L)
immig_party_ncs <- get_ncs(x = immig_toks,
                            N = 10,
                            groups = docvars(immig_toks, 'party'),
                            pre_trained = cr_glove_subset,
                            transform = TRUE,
                            transform_matrix = cr_transform,
                            bootstrap = TRUE,
                            num_bootstraps = 100,
                            as_list = TRUE)

# nearest neighbors of "immigration" for Republican party
immig_party_ncs[["R"]]

## ----message = FALSE----------------------------------------------------------
# two factor covariates
set.seed(2021L)
model1 <- conText(formula = immigration ~ party + gender,
                  data = toks_nostop_feats,
                  pre_trained = cr_glove_subset,
                  transform = TRUE, transform_matrix = cr_transform,
                  jackknife = FALSE, confidence_level = 0.95,
                  cluster_variable = NULL,
                  permute = TRUE, num_permutations = 100,
                  window = 6, case_insensitive = TRUE,
                  verbose = TRUE)

# notice, non-binary covariates are automatically "dummified"
rownames(model1)

## ----message=FALSE, eval=FALSE------------------------------------------------
# require(doParallel)
# registerDoParallel(4)
# set.seed(2021L)
# model1 <- conText(formula = immigration ~ party + gender,
#                   data = toks_nostop_feats,
#                   pre_trained = cr_glove_subset,
#                   transform = TRUE, transform_matrix = cr_transform,
#                   jackknife = TRUE, confidence_level = 0.95,
#                   parallel = TRUE,
#                   cluster_variable = NULL,
#                   permute = TRUE, num_permutations = 100,
#                   window = 6, case_insensitive = TRUE,
#                   verbose = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# require(devtools)
# install_version("conText", version = "1.4.2", repos = "http://cran.us.r-project.org")

## ----message = FALSE----------------------------------------------------------

# D-dimensional beta coefficients
# the intercept in this case is the ALC embedding for female Democrats
# beta coefficients can be combined to get each group's ALC embedding
DF_wv <- model1['(Intercept)',] # (D)emocrat - (F)emale 
DM_wv <- model1['(Intercept)',] + model1['gender_M',] # (D)emocrat - (M)ale 
RF_wv <- model1['(Intercept)',] + model1['party_R',]  # (R)epublican - (F)emale 
RM_wv <- model1['(Intercept)',] + model1['party_R',] + model1['gender_M',] # (R)epublican - (M)ale 

# nearest neighbors
nns(rbind(DF_wv,DM_wv), N = 10, pre_trained = cr_glove_subset, candidates = model1@features)

## ----message = FALSE----------------------------------------------------------
model1@normed_coefficients

## ----message = FALSE----------------------------------------------------------

# continuous covariate
set.seed(2021L)
model2 <- conText(formula = immigration ~ nominate_dim1,
                  data = toks_nostop_feats,
                  pre_trained = cr_glove_subset,
                  transform = TRUE, transform_matrix = cr_transform,
                  jackknife = FALSE, confidence_level = 0.95,
                  permute = TRUE, num_permutations = 100,
                  window = 6, case_insensitive = TRUE,
                  verbose = FALSE)

# look at percentiles of nominate
percentiles <- quantile(docvars(cr_sample_corpus)$nominate_dim1, probs = seq(0.05,0.95,0.05))
percentile_wvs <- lapply(percentiles, function(i) model2["(Intercept)",] + i*model2["nominate_dim1",]) %>% do.call(rbind,.)
percentile_sim <- cos_sim(x = percentile_wvs, pre_trained = cr_glove_subset, features = c("reform", "enforce"), as_list = TRUE)

# check output
rbind(head(percentile_sim[["reform"]], 2),tail(percentile_sim[["reform"]], 2))
rbind(head(percentile_sim[["enforce"]], 2),tail(percentile_sim[["enforce"]], 2))

## ----message=FALSE------------------------------------------------------------
library(text2vec)

#---------------------------------
# estimate glove model
#---------------------------------

# construct the feature co-occurrence matrix for our toks_nostop_feats object (see above)
toks_fcm <- fcm(toks_nostop_feats, context = "window", window = 6, count = "frequency", tri = FALSE) # important to set tri = FALSE

# estimate glove model using text2vec
glove <- GlobalVectors$new(rank = 300, 
                           x_max = 10,
                           learning_rate = 0.05)
wv_main <- glove$fit_transform(toks_fcm, n_iter = 10,
                               convergence_tol = 1e-3, 
                               n_threads = 2) # set to 'parallel::detectCores()' to use all available cores


wv_context <- glove$components
local_glove <- wv_main + t(wv_context) # word vectors

# qualitative check
find_nns(local_glove['immigration',], pre_trained = local_glove, N = 5, candidates = feats)

## ----message = FALSE----------------------------------------------------------
# compute transform
# weighting = 'log' works well for smaller corpora
# for large corpora use a numeric value e.g. weighting = 500
# see: https://arxiv.org/pdf/1805.05388.pdf
local_transform <- compute_transform(x = toks_fcm, pre_trained = local_glove, weighting = 'log')

## ----message = FALSE----------------------------------------------------------
#---------------------------------
# check
#---------------------------------

# create document-embedding matrix using our locally trained GloVe embeddings and transformation matrix
immig_dem_local <- dem(x = immig_dfm, pre_trained = local_glove, transform = TRUE, transform_matrix = local_transform, verbose = TRUE)

# take the column average to get a single "corpus-wide" embedding
immig_wv_local <- colMeans(immig_dem_local)

# find nearest neighbors for overall immigration embedding
find_nns(immig_wv_local, pre_trained = local_glove, N = 10, candidates = immig_dem_local@features)

# we can also compare to corresponding pre-trained embedding
sim2(x = matrix(immig_wv_local, nrow = 1), y = matrix(local_glove['immigration',], nrow = 1), method = 'cosine', norm = 'l2')

## ----message = FALSE----------------------------------------------------------
# create feature co-occurrence matrix for each party (set tri = FALSE to work with fem)
fcm_D <- fcm(toks_nostop_feats[docvars(toks_nostop_feats, 'party') == "D",], context = "window", window = 6, count = "frequency", tri = FALSE)
fcm_R <- fcm(toks_nostop_feats[docvars(toks_nostop_feats, 'party') == "R",], context = "window", window = 6, count = "frequency", tri = FALSE)

## ----message = FALSE----------------------------------------------------------

# compute feature-embedding matrix
fem_D <- fem(fcm_D, pre_trained = cr_glove_subset, transform = TRUE, transform_matrix = cr_transform, verbose = FALSE)
fem_R <- fem(fcm_R, pre_trained = cr_glove_subset, transform = TRUE, transform_matrix = cr_transform, verbose = FALSE)

# cr_fem will contain an embedding for each feature
fem_D[1:5,1:3]

## ----message = FALSE----------------------------------------------------------

# compute "horizontal" cosine similarity
feat_comp <- feature_sim(x = fem_R, y = fem_D)

# least similar features
head(feat_comp)

# most similar features
tail(feat_comp)

## ----message = FALSE----------------------------------------------------------

# identify documents with fewer than 100 words
short_toks <- toks_nostop_feats[sapply(toks_nostop_feats, length) <= 100,]

# run regression on full documents
model3 <- conText(formula = . ~ party,
                  data = short_toks,
                  pre_trained = cr_glove_subset,
                  transform = TRUE, transform_matrix = cr_transform,
                  jackknife = FALSE, confidence_level = 0.95,
                  permute = TRUE, num_permutations = 100,
                  window = 6, case_insensitive = TRUE,
                  verbose = FALSE)


