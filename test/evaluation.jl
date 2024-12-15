###########################################
# Evaluation metrics and plotting results # 
###########################################

include(joinpath(@__DIR__, "../src/plot_helper.jl"));


# Text to appear on plots
pen_names = ["\$L_1\$", "log \$L_1\$", "approx. \$L_0\$", "horseshoe"];
hyp_names = ["halved", "default", "doubled"];


## Functions for evaluation metrics

# Decide which reactions are present in the system given estimated rate constants
# Sort the reaction rate constants in log scale, and find the largest gap
# Reactions with rate constants on the higher side of the gap are inferred to be present
function infer_reactions(kvec)
    log_kvec = sort(log.(kvec))
    max_diff = maximum(diff(log_kvec))
    idx = findlast(==(max_diff), diff(log_kvec))
    thres = log_kvec[idx]
    return findall(k -> (log(k) > thres), kvec)
end

# Errors in trajectory reconstruction as a matrix of dimensions n_species x n_grid
# est_kvec  : Vector of estimated rate constants
# true_kvec : Vector of true rate constants
# oprob     : ODEProblem
# k         : Symbolics object for reaction rate constants, e.g. as defined in `define_networks.jl`
# t_grid    : Grid of time points to compute trajectory errors
function get_traj_err(est_kvec, true_kvec, oprob, k, t_grid)
	true_oprob = remake(oprob, p=[k => true_kvec])
    true_sol_grid = solve(true_oprob)(t_grid).u
    est_oprob = remake(oprob, p=[k => est_kvec])
    est_sol_grid = solve(est_oprob)(t_grid).u
	return reduce(hcat, est_sol_grid .- true_sol_grid)
end

# L_∞ norm of trajectory errors (sum over species first)
# traj_err : Matrix of trajectory errors, dimensions are n_species x n_grid
function get_traj_Linf(traj_err)
	return maximum(sum(abs.(traj_err), dims=1))
end


## Generate plots for results pertaining to each optimisation run
# iprob     : ODEInferenceProb struct
# kmat      : Matrix of estimated rate constants, dimensions are n_rx * n_runs
# true_kvec : Vector of true rate constants
# k         : Symbolics object for reaction rate constants, e.g. as defined in `define_networks.jl`
# dirname   : Directory to store plots in
function make_plots_runs(iprob, kmat, true_kvec, k, dirname)
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
	true_sol_grid = solve(true_oprob)(t_grid).u;
	for run_idx in 1:n_runs
		kvec = kmat[:,run_idx]
		loss_offset = pyfmt(FMT_2DP, optim_loss[run_idx] - true_loss)
		est_oprob = remake(iprob.oprob, p=[k => kvec])
		est_sol_grid = solve(est_oprob)(t_grid).u;
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