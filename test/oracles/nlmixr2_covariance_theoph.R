# Test-only oracle: nlmixr2 FOCEi estimates and covariance-step matrices on the
# public theophylline data, for the one-compartment oral model with additive error
# and diagonal Ω of test/oracles/nlmixr2_posthoc_theoph.R.
#
# Regenerate from the repository root with
#   Rscript --vanilla test/oracles/nlmixr2_covariance_theoph.R
# It writes test/data/nlmixr2_focei_theoph_estimates.csv (estimates on nlmixr2's
# scale, Ω as variances, and the objective) and one covariance file
# per covariance method, test/data/nlmixr2_focei_theoph_cov_<method>.csv, over the
# full parameter vector (covFull = TRUE). test/covariance.jl compares
# NLMEEstimation's covariance step at these estimates against them; the suite
# needs no R.
suppressPackageStartupMessages(library(nlmixr2))

one_compartment <- function() {
  ini({
    tka <- log(1.57)
    tcl <- log(2.75)
    tv <- log(31.5)
    eta.ka ~ 0.36
    eta.cl ~ 0.0676
    eta.v ~ 0.0196
    add.sd <- 0.7
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    linCmt() ~ add(add.sd)
  })
}

methods <- c(rs = "r,s", r = "r", s = "s")
estimates <- NULL
for (name in names(methods)) {
  fit <- nlmixr2(one_compartment, nlmixr2data::theo_sd, est = "focei",
    control = foceiControl(print = 0, covMethod = methods[[name]]))
  cat(sprintf("covMethod %s: fit$covMethod = %s\n", methods[[name]], fit$covMethod))
  current <- data.frame(parameter = c(names(fit$theta), rownames(fit$omega)),
    estimate = c(unname(fit$theta), unname(diag(fit$omega))))
  if (is.null(estimates)) {
    estimates <- current
  } else {
    stopifnot(isTRUE(all.equal(estimates$estimate, current$estimate, tolerance = 1e-12)))
  }
  cov <- as.matrix(fit$cov)
  write.csv(data.frame(parameter = rownames(cov), cov, check.names = FALSE),
    file.path("test", "data", sprintf("nlmixr2_focei_theoph_cov_%s.csv", name)),
    row.names = FALSE, quote = FALSE)
  print(cov, digits = 10)
}
# nlmixr2's objective, OBJF = -2 log-likelihood - N_obs log(2 pi), as the last row.
estimates <- rbind(estimates, data.frame(parameter = "objective", estimate = fit$objective))
write.csv(estimates, file.path("test", "data", "nlmixr2_focei_theoph_estimates.csv"),
  row.names = FALSE, quote = FALSE)
print(estimates, digits = 12)
cat(sprintf("objective %.10f with nlmixr2 %s, nlmixr2est %s, rxode2 %s, R %s\n",
  fit$objective, packageVersion("nlmixr2"), packageVersion("nlmixr2est"),
  packageVersion("rxode2"), getRversion()))
