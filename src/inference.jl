#####################################
### Main module for CRN inference ###
#####################################

using DelimitedFiles

using OrdinaryDiffEq
using SymbolicIndexingInterface
using SciMLStructures: Tunable, canonicalize, replace
using PreallocationTools

using BSplineKit, SparseArrays
import RegularizationTools

using Statistics
using Distributions

using LineSearches
using Optim


### Functions for data pre-processing

# u      : Univariate time series
# t      : Timepoints corresponding to the time series
# d      : Order of derivative to penalise
# t_itp  : Interpolation points to penalise d-th order derivative at
# alg    : Algorithm for determining smoothing hyperparameter, see RegularizationSmooth from DataInterpolations.jl
function smooth_data(u, t; d=2, t_itp=range(extrema(t)..., 50), alg=:L_curve)
	basis = BSplineBasis(BSplineKit.BSplineOrder(4), t_itp);
	B = collocation_matrix(basis, t, BSplineKit.Derivative(0), SparseMatrixCSC{Float64});
	D = collocation_matrix(basis, t_itp, BSplineKit.Derivative(d), SparseMatrixCSC{Float64});
	Ψ = RegularizationTools.setupRegularizationProblem(B, collect(D))
	β = RegularizationTools.solve(Ψ, u; alg=alg).x
	return (basis, β)
end

function eval_spline(basis, β, t, d=0)
	B = collocation_matrix(basis, t, BSplineKit.Derivative(d), SparseMatrixCSC{Float64});
	return B*β
end

function estim_σ(diffs)
	std(diffs; corrected=false, mean=0.)
end

function get_scale_fcts(smooth_resvec, t, species_vec, rx_vec, k)
	est_derivs = Dict(
		x => eval_spline(basis, β, t, 1)
		for (x, (basis, β)) in zip(species_vec, smooth_resvec)
	);
	est_trajs = [eval_spline(basis, β, t) for (basis, β) in smooth_resvec];

	rates_vec = substitute.(oderatelaw.(rx_vec), Ref([k => ones(length(k))]));
	n_itp = length(est_trajs[1]);
	return [
		begin
			itp_rates = substitute.(
				Ref(rates), [
					Dict(x => traj[i] for (x, traj) in zip(species_vec, est_trajs))
					for i in 1:n_itp]);
			rate_min, rate_max = extrema(itp_rates);
			deriv_ranges = [
				begin 
					deriv_min, deriv_max = extrema(est_derivs[x] ./ stoich); 
					deriv_max - deriv_min 
				end for (x, stoich) in rx.netstoich]
			maximum(deriv_ranges) / (rate_max - rate_min)
		end for (rx, rates) in zip(rx_vec, rates_vec)
	]
end


### Functions to be called for parameter inference

# Wrapper for function to be passed to `Optim` to reduce compilation time
struct FunctionWrapper
    f::Function
end
# ((wrapper::FunctionWrapper)(arg::AbstractVector{T})::T) where {T<:Real} = wrapper.f(arg);
(wrapper::FunctionWrapper)(arg) = wrapper.f(arg);

# oprob        : ODEProblem to be passed to `remake` whenever loss function is evaluated
# k            : Symbolics object for reaction rate constants
# t_obs        : Sequence of observation times
# data         : Concentration observations in a matrix of dimensions n_species * n_obs
# scale_fcts   : Scaling factors for rate constants
# tf           : Transformation function from rate space to optimisation space
# itf          : Inverse transformation function from optimisation space back to rate space
# loss_func    : Loss function called with rate constants 
# pen_func     : Penalty function called with scaled rate constants, e.g. L1 penalty
# optim_func   : Objective function called with log-transformed parameters
struct ODEInferenceProb
	oprob
	k
	t_obs
	data
	scale_fcts
	tf
	itf
	loss_func
	pen_func
	optim_func
end

# Create an ODEInferenceProb struct
# See section under "Helper functions" for specifics about penalty functions, e.g. defaults for hyperparameters
# See comments around the definition of ODEInferenceProb for definitions of `oprob`, `k`, `t_obs`, `data`, `scale_fcts`
# pen_str     : Choice of penalty function, one of < L1 | logL1 | approxL0 | hslike >
# pen_hyp     : Hyperparameter for penalty function
# lower_bound : Lower bound of scaled rate constants
function make_iprob(oprob, k, t_obs, data, pen_str, pen_hyp; scale_fcts=ones(length(k)), lower_bound=1e-10)
	n_species = size(data, 1)
	loss_func = make_loss_func(oprob, k, t_obs, data)
	pen_func = make_pen_func(pen_str, pen_hyp, lower_bound)
	tf = k -> log.(max.(lower_bound, k ./ scale_fcts))
	itf = u -> exp.(u) .* scale_fcts
	optim_func = FunctionWrapper(u -> begin 
		σs = u[1:n_species]
		k_unscaled = exp.(u[n_species+1:end]); 
		loss_func(k_unscaled .* scale_fcts, σs) + sum(pen_func, k_unscaled) 
	end);
	return ODEInferenceProb(oprob, k, t_obs, data, scale_fcts, tf, itf, loss_func, pen_func, optim_func)
end

# Performs multi-start optimisation from different initial points for parameter inference
# iprob         : ODEInferenceProb struct
# lbs           : Vector of lower bounds for parameters
# ubs           : Vector of upper bounds for parameters
# init_vec      : Vector of initial points for normalised rate constants for optimisation runs
# init_σs       : Vector of initial points for ODE parameters for optimisation runs
# optim_alg     : Optimisation algorithm from `Optim.jl`
# optim_opts    : Optimisation options of type `Optim.Options`
# callback_func : Function for displaying progress of each run, takes in run index (integer) and optimisation result as input
function optim_iprob(
	iprob, lbs, ubs, init_vec, init_σs; 
	optim_alg=Fminbox(BFGS(linesearch = LineSearches.BackTracking())),
	optim_opts=Optim.Options(x_abstol=1e-10, f_abstol=1e-10, outer_x_abstol=1e-10, outer_f_abstol=1e-10),
	callback_func=(i, res) -> nothing
)
	res_vec = Vector{Optim.MultivariateOptimizationResults}([]);
	for (i, init_params) in enumerate(init_vec)
		res = optimize(
			iprob.optim_func, 
			[0.01.*init_σs; log.(lbs)], 
			[100.0.*init_σs; log.(ubs)],
			[init_σs; log.(clamp.(init_params, lbs, ubs))],
			optim_alg, optim_opts; autodiff = :forward
		)
		callback_func(i, res)
		push!(res_vec, res)
	end
	return res_vec
end

# Store estimated parameters
# res_vec : Vector of optimisation results
# dirname : Directory to store estimates in
function export_estimates(res_vec, dirname)
	est_mat = reduce(hcat, [r.minimizer for r in res_vec]);
	writedlm(joinpath(dirname, "estimates.txt"), est_mat);
	return est_mat
end

function get_kmat(iprob, est_mat)
	n_species = size(iprob.data, 1)
	return reduce(hcat, [
		iprob.itf(est[n_species+1:end]) for est in eachcol(est_mat)
	])
end

struct ODEInferenceSol
	iprob
	est
	kvec
	σs
end

function make_isol(iprob, est_mat)
	n_species = size(iprob.data, 1)
	optim_vals = iprob.optim_func.(eachcol(est_mat))
	est = est_mat[:,argmin(optim_vals)] # extract best run
	σs = est[1:n_species]
	kvec = iprob.itf(est[n_species+1:end])
	return ODEInferenceSol(iprob, est, kvec, σs)
end


### Functions to be called for structural inference

function infer_reactions_prev(isol; thres=cquantile(Chisq(1), 1e-6)/2, print_diff=false)
    loss_val = isol.iprob.loss_func(isol.kvec, isol.σs)
    tmp_kvec = zeros(length(isol.kvec))
    inferred = Vector{Int}()
    for idx in reverse(sortperm(isol.kvec ./ isol.iprob.scale_fcts))
		push!(inferred, idx)
        tmp_kvec[idx] = isol.kvec[idx]
		loss_diff = isol.iprob.loss_func(tmp_kvec, isol.σs) - loss_val
		if print_diff
			println(idx, " ", loss_diff)
		end
        if loss_diff < thres
            break
        end		
    end
	return (inferred, isol.iprob.loss_func(tmp_kvec, isol.σs))
end

# Decide which reactions are present in the system given estimated rate constants
function infer_reactions(isol, species_vec, rx_vec; n_itp=50, thres=cquantile(Chisq(1), 1e-4)/2, print_diff=false)
	setter = setp(isol.iprob.oprob, isol.iprob.k)
	n_species, n_obs = size(isol.iprob.data)
	
	ratelaws = substitute.(oderatelaw.(rx_vec), Ref(Dict(k => isol.kvec)));
	t_itp = range(extrema(isol.iprob.t_obs)..., n_itp)
	setter(isol.iprob.oprob, isol.kvec)
	sol = solve(isol.iprob.oprob, AutoTsit5(Rosenbrock23()); saveat=t_itp)
	rates_itp = [
		substitute.(ratelaws, Ref(Dict(species_vec .=> pt)))
		for pt in sol.u];
	sorted_idxs = sortperm([sum(getindex.(rates_itp, idx)) for idx in 1:length(rx_vec)])
	
	loss_val = isol.iprob.loss_func(isol.kvec, isol.σs)
	fit_val = 0.0
    tmp_kvec = zeros(length(isol.kvec))
    inferred = Vector{Int}()
    for idx in reverse(sorted_idxs)
		push!(inferred, idx)
        tmp_kvec[idx] = isol.kvec[idx]
		# remade = remake(isol.iprob.oprob; p=[isol.iprob.k => tmp_kvec])		
		# sol = solve(remade, AutoTsit5(Rosenbrock23()); saveat=isol.iprob.t_obs)
		setter(isol.iprob.oprob, tmp_kvec)
		sol = solve(isol.iprob.oprob, AutoTsit5(Rosenbrock23()); saveat=isol.iprob.t_obs)
		σs_est = std.(eachrow(sol[1:n_species,:] .- isol.iprob.data); corrected=false, mean=0.)
		fit_val = 0.5*n_species*n_obs + n_obs*sum(log, σs_est)
		loss_diff = fit_val - loss_val
		if print_diff
			println(idx, " ", loss_diff)
		end
        if loss_diff < thres
            break
        end		
    end
	return (inferred, fit_val)
end

# Hyperparameter tuning via BIC
function tune_hyp_bic(inferred_vec, fit_vec)
	best = 1
	best_inferred = inferred_vec[1]
	best_fit = fit_vec[1]
	for idx in 2:length(fit_vec)
		inferred = inferred_vec[idx]
		fit = fit_vec[idx]
		n_diff = length(best_inferred) - length(inferred)
		if n_diff > 0
			# if difference in loss is not larger than critical value, favour simpler model
			if fit - best_fit < log(303)/2*n_diff
				best = idx
				best_inferred, best_fit = inferred, fit
			end
		elseif n_diff == 0 && fit < best_fit
			best = idx
			best_inferred, best_fit = inferred, fit
		end
	end
	return best
end

# Hyperparameter tuning via identifying plateau
# thres  : Maximum difference in model fit heuristic to be considered as part of the same plateau
function tune_hyp_plateau(inferred_vec, fit_vec, thres=cquantile(Chisq(1), 1e-4)/2)
	segments = []
	run = 0
	curr_val = fit_vec[1]
	for (i, loss_val) in enumerate(fit_vec)
		if abs(loss_val - curr_val) > thres
			push!(segments, (run, i-run))
			run = 0
		end
		run += 1
		curr_val = fit_vec[i]
	end
	push!(segments, (run, length(fit_vec)+1-run))
	plat_len, plat_start = maximum(segments)
	ns = length.(inferred_vec)
	idx = argmin(ns[plat_start:plat_start+plat_len-1])
	return plat_start + idx - 1
end



### Helper functions that do not need to be directly called when using this module

# Loss function to minimise for parameter estimation, note that `pen_func` is generic
function make_loss_func(oprob, k, t_obs, data)
	mtk_p = parameter_values(oprob) # MTK parameters object
	param_vals, repack, _ = canonicalize(Tunable(), mtk_p)
    cache = DiffCache(copy(param_vals));
    setter = setp(oprob, k);
    alg = AutoTsit5(Rosenbrock23());
	idxs = Tuple.(CartesianIndices(data))
	n_obs = length(t_obs)
    return function loss_func(x::Vector{T}, σs::Vector{T})::T where T
        buffer = get_tmp(cache, x) # get pre-allocated buffer
        copyto!(buffer, param_vals) # copy current values to buffer
        repacked_p = repack(buffer) # replace tunable portion of mtk_p with buffer
        setter(repacked_p, x) # set the updated values
        remade_oprob = remake(oprob; p = repacked_p)
        sol = solve(remade_oprob, alg; saveat=t_obs);
        if !SciMLBase.successful_retcode(sol)
            return Inf
        end
		loss = sum(log, σs) * n_obs
		for (i, j) in idxs
			@inbounds loss += 0.5*abs2((data[i,j] - sol.u[j][i]) / σs[i])
		end
		return loss
    end
end

# Penalty functions on parameters
# Some of these can be interpreted as the negative log density of some prior distribution
# Increasing the hyperparameter `hyp` penalises model complexity more heavily

function L1_pen(x, hyp) # Exponential(scale = 1/hyp)
	hyp*x
end

function logL1_pen(x, hyp) # Exponential(scale = 1/hyp) for log rates
	hyp*(log(x))
end

function approxL0_pen(x, hyp) # hyp * no of params, which is approximated by x^2/(x^2+eps) for eps << 1
	hyp*x^0.1
end

function hslike_pen(x, hyp) # horseshoe-like prior, scale = 1/hyp
	-log(log(1 + abs2(1/(hyp*x))))
end

# Create specific penalty function
function make_pen_func(pen_str, hyp, lower_bound)
	pen_func = eval(Meta.parse(pen_str * "_pen"));
	return x -> pen_func(x, hyp) - pen_func(lower_bound, hyp)
end



