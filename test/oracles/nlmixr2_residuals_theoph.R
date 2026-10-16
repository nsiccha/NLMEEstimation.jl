# Test-only oracle: nlmixr2 FOCEi residual diagnostics on the public theophylline
# data, for the model of test/oracles/nlmixr2_covariance_theoph.R. Regenerate from
# the repository root with
#   Rscript --vanilla test/oracles/nlmixr2_residuals_theoph.R
# It writes test/data/nlmixr2_residuals_theoph.csv: per observation PRED, IPRED,
# IWRES, CWRES and nlmixr2's simulation-based NPDE (addNpde, 1000 replicates),
# at the fit's estimates (test/data/nlmixr2_focei_theoph_estimates.csv). The
# suite compares its own diagnostics at those estimates and needs no R.
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

fit <- nlmixr2(one_compartment, nlmixr2data::theo_sd, est = "focei",
  control = foceiControl(print = 0), table = tableControl(cwres = TRUE, npde = TRUE))
fit <- addNpde(fit)
d <- as.data.frame(fit)
keep <- intersect(c("ID", "TIME", "DV", "PRED", "IPRED", "RES", "IRES", "IWRES",
  "WRES", "CPRED", "CRES", "CWRES", "EPRED", "NPDE", "NPD", "PDE", "PD"), names(d))
write.csv(d[, keep], file.path("test", "data", "nlmixr2_residuals_theoph.csv"),
  row.names = FALSE, quote = FALSE)
print(head(d[, keep]), digits = 8)
cat(sprintf("columns: %s\nnlmixr2 %s, nlmixr2est %s, rxode2 %s, R %s\n",
  paste(keep, collapse = ","), packageVersion("nlmixr2"), packageVersion("nlmixr2est"),
  packageVersion("rxode2"), getRversion()))
