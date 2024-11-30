##################################################
# Setup for command-line arguments
# Skip this when interacting with this script
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
	"--pen"
        help = "penalty function, one of [L1|logL1|approxL0|hslike]"
		required = true
		range_tester = x -> x in ["L1", "logL1", "approxL0", "hslike"]
	"--uselog"
        help = "whether to perform optimisation in log space, either [y/n]"
		required = true
		range_tester = x -> lowercase(x) in ["y", "n"]
	"--skipopt"
		help = "if this flag is invoked, skip optimisation and only export results (requires optimisation to have previously run)"
		action = :store_true
end
parsed_args = parse_args(s)
##################################################

### If interacting, start here (define optimisation options)
if @isdefined parsed_args # command-line mode
	PEN_STR = parsed_args["pen"]
	USE_LOG = lowercase(parsed_args["uselog"]) == "y"
	SKIP_OPT = parsed_args["skipopt"]
else # interactive mode
	### NB: these optimisation options need to be manually changed
	PEN_STR = "approxL0"; # L1, logL1, approxL0, hslike
	USE_LOG = true; # run optimisation on the original scale of parameters or on the log scale?
	SKIP_OPT = false; # if true, skip optimisation and only export results
end

# Directory name for these optimisation options
OPT_DIR = PEN_STR * (USE_LOG ? "_uselog" : "_nolog");


using Catalyst
using OrdinaryDiffEq

include((@__DIR__) * "/plot_helper.jl");
t = default_t(); # time variable


### Define full CRN
# Species and complexes
x_species = @species X1(t) X2(t) X3(t);
complexes_vec = [[X1], [X2], [X3], [X1, X2], [X2, X3], [X1, X3]];

# Reactions
rct_prd_pairs = [
	(reactants, products) for reactants in complexes_vec for products in complexes_vec 
	if reactants !== products
]; # reactants-products pair for each reaction
n_rx = length(rct_prd_pairs); # number of reactions
@parameters k[1:n_rx] # reaction rate constants
reactions = [
	Reaction(kval, reactants, products) for ((reactants, products), kval) in zip(rct_prd_pairs, k)
];

# Full CRN
@named full_network = ReactionSystem(reactions, t)
full_network = complete(full_network)

# Export all reactions
out_file = open((@__DIR__) * "/output/reactions.txt", "w")
redirect_stdout(out_file) do
    println.(reactions);
end
close(out_file)


### Read in data
using DelimitedFiles
fullmat = readdlm((@__DIR__) * "/output/data.txt");
n_obs = size(fullmat, 1);
t_obs = fullmat[:,1];
data = fullmat[:,2:end]';
n_obs = length(t_obs);
t_span = extrema(t_obs);


### Optional: Verify full CRN can reproduce 'ground truth' CRN (sim_data.jl)
true_kvec = zeros(n_rx);
true_kvec[18] = true_kvec[13] = true_kvec[1] = 1.;
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.];

oprob = ODEProblem(full_network, x0map, t_span, [k => true_kvec]);
sol = solve(oprob);

# These two sets of values should be similar (sanity check)
sol.u[end] # final point for trajectory simulated from full model
data[:,end] # final data point (noisy)


### Parameter estimation, assuming known initial conditions

σ = 0.01; # assume noise SD is known

# `penalty_func` is generic, interpreted as the negative log density of some prior on parameters
function loss_func(params, oprob, y, t_obs, penalty_func)
	oprob = remake(oprob, p=[k => params]);
	sol = try 
		solve(oprob);
	catch e
		return Inf
	end

	sum(abs2.(y .- reduce(hcat, sol(t_obs).u))) / (2σ^2) + sum(penalty_func.(params))
end

LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);

# Penalty functions
L1_pen(x) = x/0.01; # Exponential(scale = 0.01)
logL1_pen(x) = log(max(LB, x))*(1/1.0); # Exponential(scale = 1.0) for log rates
approxL0_pen(x) = log(n_obs)/2*x^0.1; # x^0.1 approximates #params, coefficient is inspired by BIC
hslike_pen(x) = -log(log(1 + abs2(0.01/x))); # horseshoe-like prior, scale = 0.01

chosen_pen = eval(Meta.parse(PEN_STR * "_pen"));

# `max(LB, u)` clamps values to be >= LB since some penalty functions are undefined at 0
if USE_LOG
	tf(u) = log(max(LB, u)); # log transform
	itf(u) = exp(u) # inverse of log transform, i.e. exp
	optim_func = u -> loss_func(itf.(u), oprob, data, t_obs, chosen_pen);
else
	tf(u) = max(LB, u); itf(u) = u;
	optim_func = u -> loss_func(u, oprob, data, t_obs, chosen_pen);	
end

# Sanity check: Second value should be much smaller than the first value
optim_func(tf.(ones(n_rx)))
true_loss = optim_func(tf.(true_kvec))

# Perform 10 runs of optimisation, show runtime and minimum value found
begin
	if SKIP_OPT @goto after_opt end	

	using LineSearches
	using Optim
	using Random

	# First run of optimisation takes longer
	# @time opt_res = optimize(
	# 	optim_func, tf.(lbs), tf.(ubs), tf.(rand(n_rx)),
	# 	Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
	# 	autodiff = :forward
	# )

	# 10 runs of optimisation, show runtime and loss offset
	res_vec = Vector{Optim.MultivariateOptimizationResults}([]);
	n_runs = 10;
	for s in 1:n_runs
		Random.seed!(s)
		init_params = rand(n_rx) # randomise the optimisation starting point
		@time local opt_res = optimize(
			optim_func, tf.(lbs), tf.(ubs), tf.(init_params),
			Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
			autodiff = :forward
		)
		# loss offset is optimised loss - loss for true parameter values
		println("loss offset = $(pyfmt(FMT_2DP, opt_res.minimum - true_loss))")
		push!(res_vec, opt_res)
	end

	# Store estimated rate constants (each column of matrix is an optimisation run)
	kmat = reduce(hcat, [itf.(r.minimizer) for r in res_vec]);
	writedlm((@__DIR__) * "/output/$(OPT_DIR)/inferred_rates.txt", kmat);
	@label after_opt
end

### Export visual results
# Read in parameter estimates
kmat = readdlm((@__DIR__) * "/output/$(OPT_DIR)/inferred_rates.txt");
n_runs = size(kmat, 2);

# Optimised value of loss function for each run
optim_loss = optim_func.(eachcol(tf.(kmat)))
ranked_order = sortperm(optim_loss);

# Show heatmap of reaction rate constants for all runs
f = Figure();
ax = Axis(
	f[1,1], 
	title="Estimated rate constants for each run", 
	xlabel="Reaction index", 
	ylabel="Loss offset (relative to loss under true parameter values)",
	yticks=(1:n_runs,pyfmt.(FMT_2DP, sort(optim_loss).-true_loss))
);
hm = heatmap!(kmat[:,ranked_order], colormap=:Blues);
Colorbar(f[:, end+1], hm);
# A bizzare hack to draw boxes that can appear on top of the axis spines
true_rxs = Observable([18, 13, 1])
rects_screenspace = lift(true_rxs, ax.finallimits, ax.scene.viewport) do xs, lims, pxa
    rects = map(xs) do x
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
save((@__DIR__) * "/output/$(OPT_DIR)/inferred_rates_heatmap.png", f);

# for col in eachcol(kmat)
# 	display(extrema([k for k in col if k < 0.1]))
# end

bin_edges = 10 .^ range(-10, 2, 25);
f = hist(
	vec(kmat), bins=bin_edges, label="All runs"; 
	axis=(;
		:title=>"Histogram of estimated rate constants (aggregated over all reactions)", 
		:xlabel=>"Estimated rate constants", 
		:xscale=>log10,
		:xticks=>LogTicks(LinearTicks(7)),
	)
)
hist!(kmat[:,ranked_order[1]], bins=bin_edges, label="Best run")
axislegend(position=:rt)
ylims!(0., 32.);
save((@__DIR__) * "/output/$(OPT_DIR)/inferred_rates_histogram.png", f);

# Visualise estimated reaction rates and trajectories simulated from these rates
t_grid = range(t_span..., 1001);
for run_idx in 1:n_runs
	kvec = kmat[:,run_idx]
	loss_offset = pyfmt(FMT_2DP, optim_loss[run_idx] - true_loss)

	local f = Figure()
	title = "True and estimated reaction rate constants\n(Run $run_idx, loss offset = $loss_offset)"
	local ax = Axis(f[1,1], title=title, xlabel="Reaction index", ylabel="Reaction rate constant")
	scatter!(1:n_rx, true_kvec, alpha=0.7, label="Ground truth")
	scatter!(1:n_rx, kvec, alpha=0.7, label="Estimated")
	f[2, 1] = Legend(f, ax, orientation=:horizontal);
	save((@__DIR__) * "/output/$(OPT_DIR)/inferred_rates_run$(run_idx).png", f)

	est_oprob = remake(oprob, p=[k => kvec])
	est_sol = solve(est_oprob);
	sol_grid = sol(t_grid);
	est_sol_grid = est_sol(t_grid);
	local f = Figure()
	title = "ODE trajectories (Run $run_idx, loss offset = $loss_offset)"
	local ax = Axis(f[1,1], title=title, xlabel=L"t", ylabel="Concentration")
	for i in 1:3
		lines!(t_grid, [pt[i] for pt in sol_grid], color=palette[i], alpha = 0.8, label=L"True $X_%$i$")
	end
	for i in 1:3
		lines!(t_grid, [pt[i] for pt in est_sol_grid], color=palette[i], linestyle=:dash, label=L"Est. $X_%$i$")
	end
	axislegend(position=:rc);
	save((@__DIR__) * "/output/$(OPT_DIR)/inferred_trajs_run$(run_idx).png", f)
end