diagnostics = residual_diagnostics(model, fit; replicates=0)
println((subject_modes_converged=all(fit.ebes.converged),
    largest_scaled_gradient=maximum(fit.ebes.gradient_norm),
    optimizer_status=fit.optimizer_info.status))
println((residual_sd_mg_per_L=round(exp(only(fit.sigma)); digits=3),
    covariance_fixed=fit.omega_matrix ≈ Ω))
