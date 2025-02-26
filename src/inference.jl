#####################################
### Main module for CRN inference ###
#####################################

using DelimitedFiles

using DataInterpolations
import RegularizationTools

using OrdinaryDiffEq
using SymbolicIndexingInterface
using SciMLStructures: Tunable, canonicalize, replace
using PreallocationTools

using Statistics
using Distributions

using LineSearches
using Optim


### Functions for data pre-processing

function smooth_data(series, t; n_itp=50, d=3, alg=:gcv_svd)
	t_itp = range(extrema(t)..., n_itp)
	return RegularizationSmooth(collect(series), t, t_itp, d; alg=alg)
end

function estim_σ(series, smoother)
	std(series .- smoother.(smoother.t))
end

function get_scale_fcts(smoothers, species_vec, rx_vec, k)
	est_derivs = Dict(
		x => DataInterpolations.derivative.(Ref(smoother), smoother.t̂, 1) 
		for (x, smoother) in zip(species_vec, smoothers)
	);
	est_trajs = [smoother.û for smoother in smoothers];

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
	t_obs
	data
	scale_fcts
	σs_init
	tf
	itf
	loss_func
	pen_func
	optim_func
end

# Create an ODEInferenceProb struct
# See section under "Helper functions" for specifics about penalty functions, e.g. defaults for hyperparameters
# See comments around the definition of ODEInferenceProb for definitions of `oprob`, `t_obs`, `data`
# k           : Symbolics object for reaction rate constants, e.g. as defined in `../test/define_networks.jl`
# σs          : Vector of standard deviations
# pen_str     : Choice of penalty function, one of < L1 | logL1 | approxL0 | hslike >
# pen_hyp     : Hyperparameter for penalty function
# lower_bound : Lower bound of scaled rate constants
function make_iprob(oprob, k, t_obs, data, σs_init, pen_str, pen_hyp; scale_fcts=ones(length(k)), lower_bound=1e-10)
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
	return ODEInferenceProb(oprob, t_obs, data, scale_fcts, σs_init, tf, itf, loss_func, pen_func, optim_func)
end

# Performs multi-start optimisation from different initial points for parameter inference
# iprob         : ODEInferenceProb struct
# lbs           : Vector of lower bounds for parameters
# ubs           : Vector of upper bounds for parameters
# init_vec      : Vector of initial points for optimisation runs
# optim_alg     : Optimisation algorithm from `Optim.jl`
# optim_opts    : Optimisation options of type `Optim.Options`
# callback_func : Function for displaying progress of each run, takes in run index (integer) and optimisation result as input
function optim_iprob(
	iprob, lbs, ubs, init_vec; 
	optim_alg=Fminbox(BFGS(linesearch = LineSearches.BackTracking())),
	optim_opts=Optim.Options(x_abstol=1e-10, f_abstol=1e-10, outer_x_abstol=1e-10, outer_f_abstol=1e-10),
	callback_func=(i, res) -> nothing
)
	res_vec = Vector{Optim.MultivariateOptimizationResults}([]);
	for (i, init_params) in enumerate(init_vec)
		res = optimize(
			iprob.optim_func, 
			[0.1.*iprob.σs_init; log.(lbs)], 
			[10.0.*iprob.σs_init; log.(ubs)],
			[iprob.σs_init; log.(clamp.(init_params, lbs, ubs))],
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

# Decide which reactions are present in the system given estimated rate constants
function infer_reactions(isol; thres=cquantile(Chisq(1), 1e-6)/2, print_diff=false)
    loss_val = isol.iprob.loss_func(isol.kvec, isol.σs)
    tmp_kvec = zeros(length(isol.kvec))
    reactions = Vector{Int}()
    for idx in reverse(sortperm(isol.kvec ./ isol.iprob.scale_fcts))
		push!(reactions, idx)
        tmp_kvec[idx] = isol.kvec[idx]
		loss_diff = isol.iprob.loss_func(tmp_kvec, isol.σs) - loss_val
		if print_diff
			println(idx, " ", loss_diff)
		end
        if loss_diff < thres
            break
        end		
    end
	return (reactions, isol.iprob.loss_func(tmp_kvec, isol.σs))
end

# Hyperparameter tuning via likelihood ratio test
function tune_hyp_lrt(isol_vec, alpha=1e-6)
	infer_vec = infer_reactions.(isol_vec)
	best = 1
	best_infer, best_fit_val = infer_vec[1]
	for idx in 2:length(infer_vec)
		infer, fit_val = infer_vec[idx]
		n_diff = length(best_infer) - length(infer)
		if n_diff > 0
			# if difference in loss is not larger than critical value, favour simpler model
			if fit_val - best_fit_val < cquantile(Chisq(n_diff), alpha)/2
				best = idx
				best_infer, best_fit_val = infer_vec[idx]
			end
		elseif n_diff == 0 && fit_val < best_fit_val
			best = idx
			best_infer, best_fit_val = infer_vec[idx]
		end
	end
	return best
end

# Hyperparameter tuning via identifying plateau
# thres  : Maximum difference in model fit heuristic to be considered as part of the same plateau
function tune_hyp_plateau(isols, thres=log(3))
	loss_vec = [isol.iprob.loss_func(isol.kvec, isol.σs) for isol in isols]
	segments = []
	run = 0
	curr_val = loss_vec[1]
	for (i, loss_val) in enumerate(loss_vec)
		if abs(loss_val - curr_val) > thres
			push!(segments, (run, i-1))
			run = 0
		end
		run += 1
		curr_val = loss_vec[i]
	end
	push!(segments, (run, length(loss_vec)))
	return maximum(segments)[2]
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
    return function loss_func(x, σs)
        buffer = get_tmp(cache, x) # get pre-allocated buffer
        copyto!(buffer, param_vals) # copy current values to buffer
        repacked_p = repack(buffer) # replace tunable portion of mtk_p with buffer
        setter(repacked_p, x) # set the updated values
        remade_oprob = remake(oprob; p = repacked_p)
        sol = solve(remade_oprob, alg; saveat=t_obs);
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



