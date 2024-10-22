using Catalyst
using OrdinaryDiffEq
using Random
using SciMLBase, Symbolics

t = default_t();
# x_species = @species X1(t) X2(t) X3(t);
x_species = @species A(t) B(t) C(t);

# complexes_vec = [[X1], [X2], [X3], [X1, X2], [X2, X3], [X1, X3]]
complexes_vec = [[A], [B], [C], [A, B], [B, C], [A, C]]
reaction_tuples = [
	(reactants, products) for reactants in complexes_vec for products in complexes_vec 
	if reactants !== products
]

# @parameters k[1:30]

k_syms = [Symbol("k$i") for i in 1:30]
k_nums = eval(Symbolics._parse_vars(
	:parameters, Real, 
	k_syms,
	ModelingToolkit.toparam)
)

reactions = [
	Reaction(k_nums[i], reactants, products) for (i, (reactants, products)) in enumerate(reaction_tuples)
]

@named full_network = ReactionSystem(reactions, t)
full_network = complete(full_network)

full_pmap  = [k_num => rand()*0.1 for k_num in k_nums];
full_x0map = [:A => 3, :B => 2, :C => 1.];

# time interval to solve on
# tspan = (0., 20.);

# create the ODEProblem we want to solve
full_oprob = ODEProblem(full_network, full_x0map, tspan, full_pmap);
full_sol = solve(full_oprob, Tsit5());
series(full_sol.t, reduce(hcat, full_sol.u))


# look at equations
ode_sys = convert(ODESystem, full_network; combinatoric_ratelaws=false);
equations(ode_sys)





### Parameter estimation

using LineSearches
using Optim

# assume known initial condition
known_x0map = [:A => 3., :B => 2., :C => 1.];

### Try making the ODE problem
true_ks = zeros(30);
true_ks[[18, 13, 1]] .= last.(true_pmap);
true_kmap = [k_num => k_val for (k_num, k_val) in zip(k_nums, true_ks)];
oprob = remake(full_oprob; p=true_kmap, u0=known_x0map);
sol = solve(oprob, Tsit5());
series(sol.t, reduce(hcat, sol.u))

function full_loss_func(params, t_obs, y, λ)
	pmap = [k_num => k_val for (k_num, k_val) in zip(k_nums, params)];
	# x0map = [:A => params[4], :B => params[5], :C => params[6]];
	oprob = remake(full_oprob, p=pmap, u0=known_x0map);
	# oprob = ODEProblem(full_network, known_x0map, tspan, (:k => params,));
	sol = try 
		solve(oprob, Tsit5());
	catch e
		return Inf
	end
	# return oprob

	return(sum(abs2.(log.(y) .- log.(reduce(hcat, sol(t_obs).u)))) + λ*sum(params))
end

full_loss_func(ones(30), t_obs, data, 1e-6)

full_loss_func(true_ks, t_obs, data, 1e-5)

res_vec = [begin
	lbs = 0.0 .* ones(30);
	ubs = 10 .* ones(30);
	Random.seed!(s)
	init_params = rand(30)
	# init_params = 0.01 .* ones(30);

	λ = 1e-3
	opt_res = optimize(
		params -> full_loss_func(params, t_obs, data, λ), 
		lbs, ubs, init_params,
		Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
		autodiff = :forward
	)


	display(opt_res.minimum)

	# est_kmap = [k_num => k_val for (k_num, k_val) in zip(k_nums, opt_res.minimizer)];
	# est_oprob = ODEProblem(full_network, full_x0map, tspan, est_kmap);
	# est_sol = solve(est_oprob, Tsit5());

	# f = Figure();
	# ax = Axis(f[1,1]);
	# series!(est_sol.t, reduce(hcat, est_sol.u), labels=[L"X_1", L"X_2", L"X_3"]);
	# f[2, 1] = Legend(f, ax, orientation=:horizontal);
	# f

	# f = Figure();
	# ax = Axis(f[1,1]);
	# series!(true_sol.t, reduce(hcat, true_sol.u), labels=[L"X_1", L"X_2", L"X_3"]);
	# f[2, 1] = Legend(f, ax, orientation=:horizontal);
	# f

	### Visualise estimated reaction rates
	# f = scatter(1:30, true_ks)
	# scatter!(1:30, opt_res.minimizer)
	# display(f)
	opt_res
end for s in 1:10
];

# save((@__DIR__) * "/learned.png", f)

est_kmap = [k_num => k_val for (k_num, k_val) in zip(k_nums, res_vec[end].minimizer)];
est_oprob = ODEProblem(full_network, full_x0map, (0,10), est_kmap);
est_sol = solve(est_oprob, Tsit5());

f = Figure();
ax = Axis(f[1,1]);
series!(true_sol.t, reduce(hcat, true_sol.u), labels=[L"X_1", L"X_2", L"X_3"]);
series!(est_sol.t, reduce(hcat, est_sol.u), linestyle=:dash, labels=[L"Est X_1", L"Est X_2", L"Est X_3"]);
f[2, 1] = Legend(f, ax, orientation=:horizontal);
f