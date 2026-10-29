# Test-only oracle: nlmixr2 SAEM estimates and its default covariance, "sa" (the
# stochastic-approximation Fisher information by Louis's identity, from a covariance
# phase that resimulates the individual parameters at the estimate), on the public
# theophylline data, for the one-compartment oral model with additive error and
# diagonal Ω of test/oracles/nlmixr2_covariance_theoph.R.
#
# Regenerate from the repository root with
#   Rscript --vanilla test/oracles/nlmixr2_saem_sa_theoph.R
# It writes test/data/nlmixr2_saem_theoph_estimates.csv (estimates on nlmixr2's
# scale, Ω as variances) and test/data/nlmixr2_saem_theoph_cov_sa{500,5000}.csv (the full
# covariance, covFull = TRUE), once with the default 500 covariance-phase iterations
# and once with 5,000. test/louis.jl compares LouisInformation at these estimates
# against them; the suite needs no R.
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

estimates <- NULL
for (n in c(500L, 5000L)) {
  set.seed(20261029)
  fit <- nlmixr2(one_compartment, nlmixr2data::theo_sd, est = "saem",
    control = saemControl(print = 0, seed = 20261029, covMethod = "sa", nSaCov = n))
  cat(sprintf("nSaCov %d: fit$covMethod = %s\n", n, fit$covMethod))
  current <- data.frame(parameter = c(names(fit$theta), rownames(fit$omega)),
    estimate = c(unname(fit$theta), unname(diag(fit$omega))))
  if (is.null(estimates)) {
    estimates <- current
  } else {
    stopifnot(isTRUE(all.equal(estimates$estimate, current$estimate, tolerance = 1e-12)))
  }
  cov <- as.matrix(fit$cov)
  write.csv(data.frame(parameter = rownames(cov), cov, check.names = FALSE),
    file.path("test", "data", sprintf("nlmixr2_saem_theoph_cov_sa%d.csv", n)),
    row.names = FALSE, quote = FALSE)
  print(cov, digits = 10)
}
write.csv(estimates, file.path("test", "data", "nlmixr2_saem_theoph_estimates.csv"),
  row.names = FALSE, quote = FALSE)
print(estimates, digits = 12)
cat(sprintf("nlmixr2 %s, nlmixr2est %s, rxode2 %s, R %s\n",
  packageVersion("nlmixr2"), packageVersion("nlmixr2est"), packageVersion("rxode2"),
  getRversion()))
