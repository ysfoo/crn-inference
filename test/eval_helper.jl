###################################
# Helper functions for evaluation # 
###################################

## Plotting utilities
include(joinpath(@__DIR__, "../src/plot_helper.jl"));


## Extract subvector of reaction rate estimates from ODEInferenceSol
function mask_kvec(isol, inferred_rxs)
	est_kvec = zeros(length(isol.kvec))
	est_kvec[inferred_rxs] .= isol.kvec[inferred_rxs];
	return est_kvec
end


## Functions for trajectory-based evaluation metrics

# Errors in trajectory reconstruction as a matrix of dimensions n_species x n_grid
# est_kvec  : Vector of estimated rate constants
# true_kvec : Vector of true rate constants
# oprob     : ODEProblem
# k         : Symbolics object for reaction rate constants
# t_grid    : Grid of time points to compute trajectory errors
function get_traj_err(est_kvec, true_kvec, oprob, k, t_grid; u0=oprob.u0)
	true_oprob = remake(oprob, p=[k => true_kvec], u0=u0)
    true_sol_grid = solve(true_oprob)(t_grid).u
    est_oprob = remake(oprob, p=[k => est_kvec], u0=u0)
    est_sol_grid = solve(est_oprob)(t_grid).u
	return reduce(hcat, est_sol_grid .- true_sol_grid)
end

# Maximum trajectory error (sum over species first)
# traj_err : Matrix of trajectory errors, dimensions are n_species x n_grid
function traj_max_err(traj_err)
	return maximum(sum(abs.(traj_err), dims=1))
end

# Mean trajectory error (sum over species first)
# traj_err : Matrix of trajectory errors, dimensions are n_species x n_grid
function traj_mean_err(traj_err)
	return mean(sum(abs.(traj_err), dims=1))
end
