### Define full CRN

using Catalyst
using OrdinaryDiffEq

t = default_t(); # time variable

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

### Verify full CRN can reproduce 'ground truth' CRN (sim_data.jl)

include((@__DIR__) * "/plot_helper.jl");

true_kvec = zeros(n_rx);
true_kvec[18] = true_kvec[13] = true_kvec[1] = 1.;
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.];
t_span = (0., 10.);

oprob = ODEProblem(full_network, x0map, t_span, [k => true_kvec]);
sol = solve(oprob);

f = Figure()
ax = Axis(f[1,1], xlabel=L"t")
for i in 1:3
	lines!(sol.t, [pt[i] for pt in sol.u], label=L"X_%$i")
end
axislegend(position=:lt);
f


### Parameter estimation
using LineSearches
using Optim
using DelimitedFiles

data = readdlm((@__DIR__) * "/data.txt");
σ = 0.01; # assume noise SD is known

# penalty_func is generic, can interpret as the negative log density of some prior on parameters
function loss_func(params, oprob, y, t_obs, penalty_func)
	oprob = remake(oprob, p=[k => params]);
	sol = try 
		solve(oprob);
	catch e
		return Inf
	end

	sum(abs2.(y .- reduce(hcat, sol(t_obs).u))) / (2σ^2) + sum(penalty_func.(params))
end

L1_loss(x) = 10*x; # negative log density of Exp(rate = 10)

loss_func(ones(n_rx), oprob, data, t_obs, L1_loss)
loss_func(true_kvec, oprob, data, t_obs, L1_loss) # this should be much smaller than the line above

@time solve(oprob);

# 10 runs of optimisation, show runtime and minimum value found
res_vec = [begin
	lbs = 1e-14 .* ones(n_rx);
	ubs = 100.0 .* ones(n_rx);
	Random.seed!(s)
	init_params = rand(n_rx) # randomise the optimisation starting point

	@time opt_res = optimize(
		params -> loss_func(params, oprob, data, t_obs, L1_loss), 
		lbs, ubs, init_params,
		Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
		autodiff = :forward
	)
	display(opt_res.minimum)
	opt_res
end for s in 1:10
];

[r.minimum for r in res_vec]

kmat = reduce(hcat, [r.minimizer for r in res_vec]);
fig, ax, hm = heatmap(log10.(kmat), colormap=:Blues);
Colorbar(fig[:, end+1], hm)
fig

## Visualise estimated reaction rates and trajectories simulated from these rates
t_grid = range(t_span..., 1001);
for (run_idx, opt_res) in enumerate(res_vec)
	f = Figure()
	ax = Axis(f[1,1], xlabel="Reaction index", ylabel="Reaction rate constant")
	scatter!(1:n_rx, true_kvec, alpha=0.7, label="Ground truth")
	scatter!(1:n_rx, opt_res.minimizer, label="Estimated")
	axislegend(position=:rt);
	save((@__DIR__) * "/output/inferred_rates_run$(run_idx).png", f)

	est_oprob = remake(oprob, p=[k => opt_res.minimizer])
	est_sol = solve(est_oprob);
	f = Figure()
	ax = Axis(f[1,1], xlabel=L"t")
	for i in 1:3
		lines!(sol.t, [pt[i] for pt in sol.u], color=palette[i], alpha = 0.8, label=L"True $X_%$i$")
	end
	for i in 1:3
		lines!(est_sol.t, [pt[i] for pt in est_sol.u], color=palette[i], linestyle=:dash, label=L"Est. $X_%$i$")
	end
	axislegend(position=:lt);
	save((@__DIR__) * "/output/inferred_trajs_run$(run_idx).png", f)
end