#####################################
### Main module for CRN inference ###
#####################################

using OrdinaryDiffEq

using DelimitedFiles

using LineSearches
using Optim

include(joinpath(@__DIR__, "plot_helper.jl"));

# oprob        : ODEProblem to be passed to `remake` whenever loss function is evaluated
# t_obs        : Sequence of observation times
# data         : Concentration observations in a matrix of dimensions n_species x n_obs
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
# k           : Symbolics object for reaction rate constants, e.g. as defined in `full_network.jl`
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
	kmat = reduce(hcat, [iprob.itf.(r.minimizer) for r in res_vec]); # dimensions are n_rx x n_runs
	writedlm(joinpath(dirname, fname), kmat);
	return kmat
end


# Generate plots for results
# iprob     : ODEInferenceProb struct
# kmat      : Matrix of estimated rate constants, dimensions are n_rx x n_runs
# true_kvec : Vector of true rate constants
# k         : Symbolics object for reaction rate constants, e.g. as defined in `full_network.jl`
# dirname   : Directory to store plots in
function make_plots(iprob, kmat, true_kvec, k, dirname)
	n_rx, n_runs = size(kmat)
	n_species = length(iprob.oprob.u0)
	# Optimised value of loss function for each run
	optim_loss = iprob.optim_func.(eachcol(iprob.tf.(kmat)))
	# Run indices ranked by optimised value of loss function
	ranked_order = sortperm(optim_loss);
	# Loss function evaluated for the ground truth parameters
	true_loss = iprob.optim_func(iprob.tf.(true_kvec))

	# 1. Heatmap of estimated rate constants (all runs on same plot)
	f = Figure();
	ax = Axis(
		f[1,1], 
		title="Estimated rate constants for each run", 
		xlabel="Reaction index", 
		ylabel="Loss offset (relative to loss under true rate constants)",
		yticks=(1:n_runs, pyfmt.(FMT_2DP, optim_loss[ranked_order].-true_loss))
	);
	if maximum(kmat) > 1.5*maximum(true_kvec)
		hm = heatmap!(
			kmat[:,ranked_order], colormap=:Blues, colorscale=sqrt, 
			highclip=:black, colorrange=(0, 1.5*maximum(true_kvec)),
		)		
	else
		hm = heatmap!(kmat[:,ranked_order], colormap=:Blues, colorscale=sqrt)		
	end
	Colorbar(f[:, end+1], hm, ticks=get_pos_sqrt_ticks(min(1.5*maximum(true_kvec), maximum(kmat))))
	
	# A bizzare hack to draw boxes to appear on top of the axis spines for aesthetic reasons
	true_rxs = Observable(findall(>(0.), true_kvec))
	rects_screenspace = lift(true_rxs, ax.finallimits, ax.scene.viewport) do xs, lims, pxa
		map(xs) do x
			Rect(
				pxa.origin[1] + pxa.widths[1]*(x-0.5-lims.origin[1])/lims.widths[1],
				pxa.origin[2],
				pxa.widths[1] / n_rx,
				pxa.widths[2]	
			)
		end
	end
	boxes = poly!(ax.blockscene, rects_screenspace, color=(:white, 0.0), strokewidth=3)
	translate!(boxes, 0, 0, 100)
	f
	save(joinpath(dirname, "inferred_rates_heatmap.png"), f);

	# 2. Histogram of all estimated rate constants aggregated over all runs
	bin_edges = 10 .^ range(-10, 2, 25);
	f = hist(
		vec(kmat), bins=bin_edges, label="All runs"; 
		axis=(;
			:title=>"Histogram of estimated rate constants (aggregated over all runs)", 
			:xlabel=>"Estimated rate constants", 
			:xscale=>log10,
			:xticks=>LogTicks(LinearTicks(7)),
		)
	)
	hist!(kmat[:,ranked_order[1]], bins=bin_edges, label="Best run")
	axislegend(position=:rt)
	ylims!(0., 32.);
	save(joinpath(dirname, "inferred_rates_histogram.png"), f);

	# 3. Dotplot of estimated rate constants compared to ground truth (one plot per run)
	for run_idx in 1:n_runs
		kvec = kmat[:,run_idx]
		loss_offset = pyfmt(FMT_2DP, optim_loss[run_idx] - true_loss)	
		f = Figure()
		title = "True and estimated reaction rate constants\n(Run $run_idx, loss offset = $loss_offset)"
		ax = Axis(
			f[1,1], title=title, xlabel="Reaction index", ylabel="Reaction rate constant", 
			yscale=symsqrt, yticks=get_pos_sqrt_ticks(maximum([true_kvec; kvec]))
		)
		scatter!(1:n_rx, true_kvec, alpha=0.7, label="Ground truth")
		scatter!(1:n_rx, kvec, alpha=0.7, label="Estimated")
		f[2, 1] = Legend(f, ax, orientation=:horizontal);		
		save(joinpath(dirname, "inferred_rates_run$(run_idx).png"), f)
	end

	# 4. Trajectories reconstructed from estimates compared to ground truth (one plot per run)
	t_grid = range(t_span..., 1001);
	true_oprob = remake(iprob.oprob, p=[k => true_kvec])
	true_sol_grid = solve(true_oprob)(t_grid);
	for run_idx in 1:n_runs
		kvec = kmat[:,run_idx]
		loss_offset = pyfmt(FMT_2DP, optim_loss[run_idx] - true_loss)
		est_oprob = remake(iprob.oprob, p=[k => kvec])
		est_sol_grid = solve(est_oprob)(t_grid);
		f = Figure()
		title = "ODE trajectories (Run $run_idx, loss offset = $loss_offset)"
		ax = Axis(f[1,1], title=title, xlabel=L"t", ylabel="Concentration")
		for i in 1:n_species
			lines!(t_grid, [pt[i] for pt in true_sol_grid], color=palette[i], alpha = 0.8, label=L"True $X_%$i$")
		end
		for i in 1:n_species
			lines!(t_grid, [pt[i] for pt in est_sol_grid], color=palette[i], linestyle=:dash, label=L"Est. $X_%$i$")
		end
		axislegend(position=:rc);
		save(joinpath(dirname, "inferred_trajs_run$(run_idx).png"), f)
	end
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



