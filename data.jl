using CairoMakie
using Catalyst
using OrdinaryDiffEq
using Random

true_rn = @reaction_network begin
	kAB, A + B --> C # reaction number 18
	kC, C --> A + B # reaction number 13
	kA, A --> B # reaction number 1
end

true_rn = complete(true_rn);

# true_pmap  = (:kAB => 5., :kC => 3., :kA => 1.);
true_pmap  = (:kAB => 5., :kC => 1., :kA => 2.);
true_x0map = [:A => 3, :B => 2, :C => 1.];

ode_sys = convert(ODESystem, true_rn; combinatoric_ratelaws=false);
equations(ode_sys)

# time interval to solve on
tspan = (0., 10.);

# create the ODEProblem we want to solve
true_oprob = ODEProblem(true_rn, true_x0map, tspan, true_pmap);
true_sol = solve(true_oprob, Tsit5());

f = Figure()
ax = Axis(f[1,1], xlabel=L"t")
series!(true_sol.t, reduce(hcat, true_sol.u), labels=[L"A", L"B", L"C"]);
axislegend(position=:lt);
f
save((@__DIR__) * "/true_traj.png", f)

t_obs = range(tspan..., 501);
y_noiseless = reduce(hcat, true_sol(t_obs).u)

Random.seed!(5);
data = y_noiseless .* exp.(0.01 .* randn(size(y_noiseless)))

fig = series(true_sol.t, reduce(hcat, true_sol.u));
for data_row in eachrow(data)
	scatter!(t_obs, data_row)
end
fig