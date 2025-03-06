###########################################
# Evaluation metrics and plotting results # 
###########################################

include(joinpath(@__DIR__, "../src/plot_helper.jl"));


function mask_kvec(isol, inferred_rxs)
	est_kvec = zeros(length(isol.kvec))
	est_kvec[inferred_rxs] .= isol.kvec[inferred_rxs];
	return est_kvec
end


## Functions for evaluation metrics

# Errors in trajectory reconstruction as a matrix of dimensions n_species x n_grid
# est_kvec  : Vector of estimated rate constants
# true_kvec : Vector of true rate constants
# oprob     : ODEProblem
# k         : Symbolics object for reaction rate constants, e.g. as defined in `define_networks.jl`
# t_grid    : Grid of time points to compute trajectory errors
function get_traj_err(est_kvec, true_kvec, oprob, k, t_grid; u0=oprob.u0)
	true_oprob = remake(oprob, p=[k => true_kvec], u0=u0)
    true_sol_grid = solve(true_oprob)(t_grid).u
    est_oprob = remake(oprob, p=[k => est_kvec], u0=u0)
    est_sol_grid = solve(est_oprob)(t_grid).u
	return reduce(hcat, est_sol_grid .- true_sol_grid)
end

# L_∞ norm of trajectory errors (sum over species first)
# traj_err : Matrix of trajectory errors, dimensions are n_species x n_grid
function get_traj_Linf(traj_err, ws=nothing)
	return maximum(sum(abs.(traj_err), dims=1))
end

# L_1 norm of trajectory errors (sum over species first)
# traj_err : Matrix of trajectory errors, dimensions are n_species x n_grid
function get_traj_L1(traj_err, ws=ones(size(traj_err, 2)))
	L1_rows = [
		0.5 * sum(ws .* (err_row[begin:end-1] .+ err_row[begin+1:end]))
		for err_row in eachrow(abs.(traj_err))
	]
	return sum(L1_rows)
end


## Generate plots for results pertaining to each optimisation run
# iprob      : ODEInferenceProb struct
# est_mat    : Matrix of estimates, dimensions are (n_species+n_rx) * n_runs
# true_kvec  : Ground-truth rate constants
# true_σs    : Ground-truth noise standard deviations
# k          : Symbolics object for reaction rate constants, e.g. as defined in `define_networks.jl`
# dirname    : Directory to store plots in
# indiv_runs : Whether to also plot the graphs for individual runs
function make_plots_runs(iprob, est_mat, true_kvec, true_σs, k, dirname, indiv_runs=false)
	kmat = get_kmat(iprob, est_mat)
	n_rx, n_runs = size(kmat)
	n_species = length(iprob.oprob.u0)

	true_θ = [true_σs; iprob.tf(true_kvec)];
	
	# Optimised value of penalised loss for each run
	optim_vals = iprob.optim_func.(eachcol(est_mat))
	# Run indices ranked by optimised value of penalised loss
	ranked_order = sortperm(optim_vals);
	# Penalised loss evaluated for the ground truth parameters
	true_val = iprob.optim_func(true_θ)

	# 1. Heatmap of estimated rate constants (all runs on same plot)
	f = Figure();
	ax = Axis(
		f[1,1], 
		title="Estimated rate constants for each run", 
		xlabel="Reaction index", 
		ylabel="Penalised loss diff. (relative to ground-truth parameters)",
		yticks=(1:n_runs, pyfmt.(FMT_2DP, optim_vals[ranked_order].-true_val))
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
	normalised_kvecs = [kvec ./ iprob.scale_fcts for kvec in eachcol(kmat)]
	f = hist(
		reduce(vcat, normalised_kvecs), bins=bin_edges, label="All runs"; 
		axis=(;
			:title=>"Histogram of estimated rate constants (normalised)", 
			:xlabel=>"Estimated rate constants", 
			:xscale=>log10,
			:xticks=>LogTicks(LinearTicks(7)),
		)
	)
	hist!(normalised_kvecs[ranked_order[1]], bins=bin_edges, label="Best run")
	axislegend(position=:rt)
	ylims!(0., 32.);
	save(joinpath(dirname, "inferred_rates_histogram.png"), f);

	# Terminate if graphs for individual runs are not needed
	if !indiv_runs
		return
	end

	# 3. Dotplot of estimated rate constants compared to ground truth (one plot per run)
	for run_idx in 1:n_runs
		kvec = kmat[:,run_idx]
		penloss_diff = pyfmt(FMT_2DP, optim_vals[run_idx] - true_val)	
		f = Figure()
		title = "True and estimated reaction rate constants\n(Run $run_idx, diff. = $penloss_diff)"
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
	true_sol_grid = solve(true_oprob)(t_grid).u;
	for run_idx in 1:n_runs
		kvec = kmat[:,run_idx]
		penloss_diff = pyfmt(FMT_2DP, optim_vals[run_idx] - true_val)
		est_oprob = remake(iprob.oprob, p=[k => kvec])
		est_sol_grid = solve(est_oprob)(t_grid).u;
		f = Figure()
		title = "ODE trajectories (Run $run_idx, diff. = $penloss_diff)"
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