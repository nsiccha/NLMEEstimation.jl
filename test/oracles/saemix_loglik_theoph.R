# Test-only likelihood oracle: saemix's importance-sampling and Gauss-Hermite
# log-likelihoods at its SAEM estimate on theoph.csv, for the model of
# test/oracles/saemix_theoph.R (same data, model, settings and seed).
# Regenerate from the repository root with
#   Rscript --vanilla test/oracles/saemix_loglik_theoph.R
# Requires saemix 3.5. It writes test/data/saemix_loglik_theoph.csv; the suite
# compares NLMEEstimation's marginal likelihood at these estimates and needs no R.
suppressPackageStartupMessages(library(saemix))

data <- read.csv(file.path("test", "data", "theoph.csv"))
data$Dose <- data$Dose * data$Wt
data$logW <- log(data$Wt / 70)
saemix.data <- saemixData(name.data = data, header = TRUE,
  name.group = "Subject", name.predictors = c("Dose", "Time"),
  name.response = "conc", name.covariates = "logW", name.X = "Time",
  verbose = FALSE)

one_compartment <- function(psi, id, xidep) {
  dose <- xidep[, 1]
  time <- xidep[, 2]
  ka <- psi[id, 1]
  CL <- psi[id, 2]
  V <- psi[id, 3]
  ke <- CL / V
  dose * ka / (V * (ka - ke)) * (exp(-ke * time) - exp(-ka * time))
}

saemix.model <- saemixModel(model = one_compartment,
  psi0 = matrix(c(1.57, 2.75, 31.5, 0, 0, 0), ncol = 3, byrow = TRUE,
    dimnames = list(NULL, c("ka", "CL", "V"))),
  transform.par = c(1, 1, 1),
  covariate.model = matrix(c(0, 1, 0), nrow = 1),
  covariance.model = diag(3),
  omega.init = diag(c(0.36, 0.0676, 0.0196)),
  error.model = "constant", error.init = c(0.7, 0), verbose = FALSE)
control <- saemixControl(nbiter.saemix = c(150, 80), nbiter.burn = 20,
  nbiter.mcmc = c(2, 2, 0, 0), alpha.sa = 0.8, fix.seed = TRUE, seed = 23456,
  map = TRUE, fim = FALSE, ll.is = TRUE, ll.gq = TRUE, nmc.is = 50000,
  nnodes.gq = 12, nsd.gq = 4,
  print = FALSE, save = FALSE, save.graphs = FALSE)
fit <- saemix(saemix.model, saemix.data, control)
stopifnot(fit@results@status == "fitted")
r <- fit@results
result <- data.frame(quantity = c(r@name.fixed, "residual_sd",
    "omega_ka", "omega_CL", "omega_V", "ll_is", "ll_gq", "aic_is", "bic_is",
    "aic_gq", "bic_gq", "nobs", "nsubjects"),
  value = c(r@fixed.effects, r@respar[1], diag(r@omega), r@ll.is, r@ll.gq,
    r@aic.is, r@bic.is, r@aic.gq, r@bic.gq, fit@data@ntot.obs, fit@data@N))
write.csv(result, file.path("test", "data", "saemix_loglik_theoph.csv"),
  row.names = FALSE, quote = FALSE)
print(result, digits = 12)
cat(sprintf("saemix %s, R %s\n", packageVersion("saemix"), getRversion()))
