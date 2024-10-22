# create model and data from `data.jl` first

using LineSearches
using Optim
using OrdinaryDiffEq

true_oprob = ODEProblem(true_rn, true_x0map, tspan, true_pmap);
true_sol = solve(true_oprob, Tsit5());



wrong_pmap = (:kAB => 5.2, :kC => 1.2, :kA => 2.2);
wrong_x0map = [:A => 3, :B => 2, :C => 1];

wrong_oprob = remake(true_oprob, p=wrong_pmap, u0=wrong_x0map);
wrong_sol = solve(wrong_oprob, Tsit5());

fig = series(true_sol.t, reduce(hcat, true_sol.u));
series!(wrong_sol.t, reduce(hcat, wrong_sol.u), linestyle=:dash);
fig

function loss_func(params, t_obs, y)
	pmap = (:kAB => params[1], :kC => params[2], :kA => params[3]);
	x0map = [:A => params[4], :B => params[5], :C => params[6]];
	oprob = remake(true_oprob, p=pmap, u0=x0map);
	sol = try 
		solve(oprob, Tsit5());
	catch e
		return Inf
	end

	return(sum(abs2.(log.(y) .- log.(reduce(hcat, sol(t_obs).u)))))
end

true_params = [collect(last.(true_pmap)); collect(last.(true_x0map))]
loss_func(true_params, t_obs, data)

wrong_params = [collect(last.(wrong_pmap)); collect(last.(wrong_x0map))]
loss_func(wrong_params, t_obs, data)

lbs = 0.1 .* ones(6);
ubs = 10 .* ones(6);
init_params = ones(6);

opt_res = optimize(
	params -> loss_func(params, t_obs, data), 
	lbs, ubs, init_params,
	Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
	autodiff = :forward
)

opt_res.minimizer
true_params




### Informal: extra reaction

wrong_rn = @reaction_network begin
	kAB, A + B --> C
	kC, C --> A + B
	kA, A --> B
	kB, B --> A
end

wrong_rn = complete(wrong_rn);

wrong_pmap  = (:kAB => 5., :kC => 1., :kA => 2., :kB => 4.);
wrong_x0map = [:A => 3, :B => 2, :C => 1.];

# time interval to solve on
tspan = (0., 20.);

# create the ODEProblem we want to solve
wrong_oprob = ODEProblem(wrong_rn, wrong_x0map, tspan, wrong_pmap);

function L1_loss_func(params, t_obs, y, λ)
	pmap = (:kAB => params[1], :kC => params[2], :kA => params[3], :kB => params[4]);
	x0map = [:A => params[5], :B => params[6], :C => params[7]];
	oprob = remake(wrong_oprob, p=pmap, u0=x0map);
	sol = try 
		solve(oprob, Tsit5());
	catch e
		return Inf
	end

	return(sum(abs2.(log.(y) .- log.(reduce(hcat, sol(t_obs).u)))) + λ*sum(params[1:4]))
end

lbs = 0.0 .* ones(7);
ubs = 10 .* ones(7);
init_params = ones(7);

opt_res = optimize(
	params -> L1_loss_func(params, t_obs, data, 1e-6), 
	lbs, ubs, init_params,
	Fminbox(BFGS(linesearch = LineSearches.BackTracking()));
	autodiff = :forward
)

opt_res.minimizer