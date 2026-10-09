# Test-only oracle: nlmixr2 posthoc (empirical Bayes) estimates on the public
# theophylline data, at the fixed parameters `θ_THEOPH`, `σ_THEOPH`, `Ω_THEOPH` of
# test/common.jl (one-compartment oral model, additive error, no covariates).
#
# Regenerate from the repository root with
#   Rscript --vanilla test/oracles/nlmixr2_posthoc_theoph.R
# It writes test/data/nlmixr2_posthoc_theoph.csv (EBEs) and
# test/data/nlmixr2_posthoc_theoph_predictions.csv (IPRED at the EBEs, PRED at η = 0);
# test/empirical_bayes.jl compares against both, so the test suite itself needs no R.
#
# nlmixr2's posthoc modes are converged only to its internal inner tolerance, which
# neither `sigdig` nor `innerOpt` changes (all give bit-identical modes): on the
# shared objective they sit up to ~3e-6 in log-density, ~6e-4 in η, short of the
# optimum (measured 2026-10-09, nlmixr2 7.0.1).
suppressPackageStartupMessages(library(nlmixr2))

one_compartment <- function() {
  ini({
    tka <- log(1.57)
    tcl <- log(2.75)
    tv <- log(31.5)
    eta.ka ~ 0.36     # 0.60^2
    eta.cl ~ 0.0676   # 0.26^2
    eta.v ~ 0.0196    # 0.14^2
    add.sd <- 0.7
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    linCmt() ~ add(add.sd)
  })
}

fit <- nlmixr2(one_compartment, nlmixr2data::theo_sd, est = "posthoc",
  control = foceiControl(print = 0))
eta <- fit$eta[order(fit$eta$ID), c("ID", "eta.ka", "eta.cl", "eta.v")]
out <- file.path("test", "data", "nlmixr2_posthoc_theoph.csv")
write.csv(eta, out, row.names = FALSE, quote = FALSE)
predictions <- as.data.frame(fit)[, c("ID", "TIME", "IPRED", "PRED")]
write.csv(predictions, file.path("test", "data", "nlmixr2_posthoc_theoph_predictions.csv"),
  row.names = FALSE, quote = FALSE)
cat(sprintf("wrote %s with nlmixr2 %s, nlmixr2est %s, rxode2 %s, R %s\n", out,
  packageVersion("nlmixr2"), packageVersion("nlmixr2est"), packageVersion("rxode2"),
  getRversion()))
print(eta, digits = 10)
