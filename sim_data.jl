using Catalyst
using OrdinaryDiffEq
using Random

include((@__DIR__) * "/plot_helper.jl");

# Indices of reaction rate constants follow the reaction indices in full_network.jl
true_rn = @reaction_network begin
	k18, X1 + X2 --> X3
	k13, X3 --> X1 + X2
	k1, X1 --> X2
end
true_rn = complete(true_rn);

true_pmap  = (:k18 => 1., :k13 => 1., :k1 => 1.); # reaction rate constants
true_x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # initial conditions
t_span = (0., 10.); # time interval to solve on

true_oprob = ODEProblem(true_rn, true_x0map, t_span, true_pmap);
true_sol = solve(true_oprob, saveat=range(t_span..., 1001)); # ODE solution

# Plot ground truth trajectories
f = Figure()
ax = Axis(f[1,1], xlabel=L"t", ylabel="Concentration", title="Ground truth trajectories")
for i in 1:3
	lines!(true_sol.t, [pt[i] for pt in true_sol.u], label=L"X_%$i")
end
axislegend(position=:rc);
f
save((@__DIR__) * "/output/true_traj.png", f)

# Observed data
n_obs = 101; # number of time points
t_obs = range(t_span..., n_obs);
y_noiseless = reduce(hcat, true_sol(t_obs).u);

Random.seed!(2024);
σ = 0.01; # noise SD
data = y_noiseless .+ σ .* randn(size(y_noiseless)); # additive noise
data = max.(0.0, data); # clamp negative values to 0

# Plot noisy data with ground truth trajectories
f = Figure()
ax = Axis(f[1,1], xlabel=L"t", ylabel="Concentration", title="Ground truth trajectories and observed data")
for i in 1:3
	scatter!(t_obs, data[i,:], color=lighten(palette[i], 0.7))
	lines!(true_sol.t, [pt[i] for pt in true_sol.u], label=L"X_%$i")	
end
axislegend(position=:rc);
f
save((@__DIR__) * "/output/data.png", f)

# Export data
using DelimitedFiles
writedlm((@__DIR__) * "/output/data.txt", [t_obs data']);