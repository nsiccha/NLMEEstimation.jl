begin
    brm_correlated_group_effects(g, K, eta) = begin
            tau[1:K] .~ Exponential.(1)
            L ~ LKJCholesky(K, eta)
            z[1:length(levels(g)), 1:K] .~ Normal.(0, 1)
            return z * transpose(tau .* L)
        end
    brm_population_effects(X, ncoef, loc, scale) = begin
            beta_pop[1:ncoef] .~ Normal.(loc, scale)
            return X * beta_pop
        end
    begin
        b_pk_subject ~ brm_correlated_group_effects(subject, 2, 1.0)
        subject_level = brm_level_indices(subject, subject)
        ranef_log_CL_pk_subject = b_pk_subject[subject_level, 1]
        ranef_log_V_pk_subject = b_pk_subject[subject_level, 2]
        X_log_CL = hcat(ones(length(subject)), weight)
        pop_log_CL ~ brm_population_effects(X_log_CL, 2, [0.2, 0.0], [1.0, 0.5])
        log_CL = pop_log_CL .+ ranef_log_CL_pk_subject
        log_V_Intercept ~ Normal(1.6, 1.0)
        log_V = log_V_Intercept .+ ranef_log_V_pk_subject
        sigma ~ Exponential(1.0)
        #= line 0 =# @plate for i = eachindex(dose)
                CL = exp(log_CL[i])
                V = exp(log_V[i])
                pred_cell = (dose[i] / V) .* exp.(-(CL / V) .* t[i])
                pred_cells[i] = pred_cell
                pred_argument_dv_1[i] = pred_cell
                pred_argument_dv_2[i] = sigma .+ 0.03 .* pred_cell
            end
        pred = brm_flatten_cells(pred_cells)
        #= line 0 =# @plate for i = eachindex(dv)
                dv[i] .~ Normal.(pred_argument_dv_1[i], pred_argument_dv_2[i])
            end
    end
end