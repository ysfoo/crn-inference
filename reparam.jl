using CairoMakie
using Catalyst
using LineSearches
using Optim
using OrdinaryDiffEq
using Random

true_rn = @reaction_network begin
	k1, A --> B
	k2, B --> ∅
end

true_rn = complete(true_rn);

true_pmap  = (:k1 => 1e1, :k2 => 1e-2);
true_x0map = [:A => 3, :B => 2];

# time interval to solve on
tspan = (0., 0.1);

# create the ODEProblem we want to solve
true_oprob = ODEProblem(true_rn, true_x0map, tspan, true_pmap);
true_sol = solve(true_oprob, QNDF(), abstol=1e-12);

series(true_sol.t, reduce(hcat, true_sol.u))

ab_traj = reduce(hcat, true_sol.u);
abc_traj = [ab_traj; sum(ab_traj, dims=1)]
series(true_sol.t, abc_traj)

t_obs = range(0.0, 0.1, 21);
y_noiseless = reduce(hcat, true_sol(t_obs).u)
y_noiseless = max.(y_noiseless, 1e-18)

for s in 11:20
	Random.seed!(s);
	data = y_noiseless .* exp.(0.05 .* randn(size(y_noiseless)))

	function loss_func(params, t_obs, y)
		pmap = (:k1 => params[1], :k2 => params[2]);
		x0map = [:A => params[3], :B => params[4]];
		oprob = remake(true_oprob, p=pmap, u0=x0map);
		sol = try 
			solve(oprob, QNDF(), abstol=1e-12);
		catch e
			return Inf
		end

		sol_capped = max.(reduce(hcat, sol(t_obs).u), 1e-18)
		# return(sum(abs2.(log.(y) .- log.(sol_capped))))
		return(sum(abs2.(y .- sol_capped)))
	end

	true_params = [collect(last.(true_pmap)); collect(last.(true_x0map))]
	loss_func(true_params, t_obs, data)

	wrong_pmap  = (:k1 => 11, :k2 => 1.01);
	wrong_x0map = [:A => 3, :B => 2];

	wrong_params = [collect(last.(wrong_pmap)); collect(last.(wrong_x0map))]
	loss_func(wrong_params, t_obs, data)

	lbs = 0.0 .* ones(4);
	ubs = 100 .* ones(4);
	init_params = ones(4);
	init_params[3] = 3;
	init_params[4] = 2;

	opt_res = optimize(
		params -> loss_func(params, t_obs, data), 
		lbs, ubs, init_params,
		Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
		autodiff = :forward
	)

	# println("Seed $s: no reparam")
	# display(opt_res.minimizer)
	no_repar_slow_est = opt_res.minimizer[2]





	### Reparameterised ODE

	function reparam_ode(dx, x, p, t)
		# a' = -k1*a
		# c' = -k2*(c-a)
		dx[1] = -p[1]*x[1]
		dx[2] = -p[2]*(x[2]-x[1])
	end

	reparam_oprob = ODEProblem(reparam_ode, [3, 5], tspan, [1e1, 1e-2]);
	solve(reparam_oprob, QNDF(), abstol=1e-12).u

	function repar_loss_func(params, t_obs, y)
		oprob = remake(reparam_oprob, p=params[1:2], u0=params[3:4]);
		sol = try 
			solve(oprob, QNDF(), abstol=1e-12);
		catch e
			return Inf
		end
		# return reduce(hcat, sol(t_obs).u)

		sol_capped = max.(reduce(hcat, sol(t_obs).u), 1e-18)
		# return(sum(abs2.(log.(y) .- log.(sol_capped))))
		return(sum(abs2.(y .- sol_capped)))
	end

	repar_true_params = [collect(last.(true_pmap)); collect(last.(true_x0map))]
	repar_true_params[4] = 3 + 2;

	repar_data = [data[1,:]'; sum(data, dims=1)]

	repar_loss_func(repar_true_params, t_obs, repar_data)

	wrong_pmap  = (:k1 => 11, :k2 => 1.01);
	wrong_x0map = [:A => 3, :B => 2];

	wrong_params = [collect(last.(wrong_pmap)); collect(last.(wrong_x0map))]
	loss_func(wrong_params, t_obs, repar_data)

	lbs = 0.0 .* ones(4);
	ubs = 100 .* ones(4);
	init_params = ones(4);
	init_params[3] = 3;
	init_params[4] = 5;

	opt_res = optimize(
		params -> repar_loss_func(params, t_obs, repar_data), 
		lbs, ubs, init_params,
		Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
		autodiff = :forward
	)

	opt_res.minimizer

	# println("Seed $s: with reparam")
	# display(opt_res.minimizer)

	with_repar_slow_est = opt_res.minimizer[2]

	println("Seed $s: $no_repar_slow_est vs $with_repar_slow_est")
end