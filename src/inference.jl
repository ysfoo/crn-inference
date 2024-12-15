#####################################
### Main module for CRN inference ###
#####################################

using OrdinaryDiffEq

using DelimitedFiles

using LineSearches
using Optim

# oprob        : ODEProblem to be passed to `remake` whenever loss function is evaluated
# t_obs        : Sequence of observation times
# data         : Concentration observations in a matrix of dimensions n_species * n_obs
# tf           : Trnasformation function from rate space to optimisation space, e.g. log
# itf          : Inverse transformation function from optimisation space back to rate space
# penalty_func : Penalty function on rate constants, e.g. L1 penalty
# optim_func   : Function of rate constants to be optimised for parameter inference
# NB: `penalty_func` is defined on the original space, ` optim_func` is defined on the transformed space
struct ODEInferenceProb
	oprob::ODEProblem
	t_obs::AbstractVector{<:Real}
	data::AbstractMatrix{<:Real}
	tf::Function
	itf::Function
	penalty_func::Function
	optim_func::Function
end

### Functions to be directly called when using this module


# Create an ODEInferenceProb struct
# See section under "Helper functions" for specifics about penalty functions, e.g. defaults for hyperparameters
# See comments around the definition of ODEInferenceProb for definitions of `oprob`, `t_obs`, `data`
# penalty_str : Choice of penalty function, one of < L1 | logL1 | approxL0 | hslike >
# k           : Symbolics object for reaction rate constants, e.g. as defined in `../test/define_networks.jl`
# lower_bound : Lower bound, e.g. 1e-10 to clip parameters away from 0 to prevent issues such as log(0)
# pen_hyp     : Hyperparameter for penalty function
# log_opt     : Whether to perform optimisation in log space, default to true
function make_iprob(oprob, t_obs, data, penalty_str, lower_bound, k, pen_hyp; log_opt=true)
	penalty_func = make_penalty_func(penalty_str, pen_hyp)
	if log_opt
		tf = u -> log(max(lower_bound, u))
		itf = exp
	else
		tf = u -> max(lower_bound, u)
		itf = identity
	end
	optim_func = u -> loss_func(itf.(u), oprob, t_obs, data, penalty_func, k);
	return ODEInferenceProb(oprob, t_obs, data, tf, itf, penalty_func, optim_func)
end


# Performs multi-start optimisation from different initial points for parameter inference
# iprob         : ODEInferenceProb struct
# lbs           : Vector of lower bounds for parameters
# ubs           : Vector of upper bounds for parameters
# init_vec      : Vector of initial points for optimisation runs
# opt_alg       : Optimisation algorithm from `Optim.jl`
# callback_func : Function for displaying progress of each run, takes in run index (integer) and optimisation result as input
function optim_iprob(
	iprob, lbs, ubs, init_vec; 
	opt_alg=Fminbox(BFGS(linesearch = LineSearches.BackTracking())),
	callback_func=(i, res) -> nothing
)
	res_vec = Vector{Optim.MultivariateOptimizationResults}([]);
	for (i, init_params) in enumerate(init_vec)
		res = optimize(
			iprob.optim_func, iprob.tf.(lbs), iprob.tf.(ubs), iprob.tf.(init_params),
			opt_alg; autodiff = :forward
		)
		callback_func(i, res)
		push!(res_vec, res)
	end
	return res_vec
end

# Store estimated rate constants (parameters)
# res_vec : Vector of optimisation results
# iprob   : ODEInferenceProb struct
# dirname : Directory to store estimated rate constants in
# fname   : Filename for estimated rate constants
function export_estimates(res_vec, iprob, dirname, fname="inferred_rates.txt")
	kmat = reduce(hcat, [iprob.itf.(r.minimizer) for r in res_vec]); # dimensions are n_rx * n_runs
	writedlm(joinpath(dirname, fname), kmat);
	return kmat
end

### Helper functions that do not need to be directly called when using this module

# Loss function to minimise for parameter estimation, note that `penalty_func` is generic
function loss_func(params, oprob, t_obs, data, penalty_func, k)
	oprob = remake(oprob, p=[k => params]);
	sol = try 
		solve(oprob);
	catch e
		return Inf
	end
	sum(abs2.(data .- reduce(hcat, sol(t_obs).u))) / (2σ^2) + sum(penalty_func.(params))
end

# Penalty functions on parameters
# Some of these can be interpreted as the negative log density of some prior distribution
# Increasing the hyperparameter `hyp` penalises model complexity more heavily
function L1_pen(x, hyp) # Exponential(scale = 1/hyp)
	hyp*x
end
function logL1_pen(x, hyp) # Exponential(scale = 1/hyp) for log rates
	hyp*log(x)
end
function approxL0_pen(x, hyp) # hyp * no of params, which is approximated by x^0.1
	hyp*x^0.1
end
function hslike_pen(x, hyp) # horseshoe-like prior, scale = 1/hyp
	-log(log(1 + abs2(1/(hyp*x))))
end

# Create specific penalty function
function make_penalty_func(penalty_str, hyp)
	penalty_func = eval(Meta.parse(penalty_str * "_pen"));
	return x -> penalty_func(x, hyp)
end



