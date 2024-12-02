#####################################################################
### Define ground truth network and functions for data simulation ###
#####################################################################

using Catalyst
using OrdinaryDiffEq

using Random

using DelimitedFiles

include(joinpath(@__DIR__, "plot_helper.jl"));

# Indices of reaction rate constants follow the reaction indices in full_network.jl
default_rn = @reaction_network begin
	k1, X1 --> X2
	(k18, k13), X1 + X2 <--> X3	
end
default_rn = complete(default_rn);

default_kmap  = (:k1 => 1., :k18 => 1., :k13 => 1.); # reaction rate constants
default_x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # initial conditions
default_t_span = (0., 10.); # time interval to solve on

# Simulates and exports data
# n_obs   : Number of observation points
# σ       : Noise standard deviation
# rn      : Reaction network, which is a Catalyst.ReactionSystem
# kmap    : Dictionary mapping reaction rate symbols to their values
# x0map   : Dictionary mapping species symbols to their initial concentrations
# t_span  : Timespan for ODE problem
# dirname : Directory name to store simulated data
function sim_data(;
	n_obs=101, σ = 0.01,
	rn=default_rn, kmap=default_kmap, x0map=default_x0map, t_span=default_t_span, 
	dirname=joinpath(@__DIR__, "output"), seed=nothing
)
	n_species = length(x0map)
	oprob = ODEProblem(rn, x0map, t_span, kmap);
	sol = solve(oprob, saveat=range(t_span..., 1001)); # ODE solution

	# Plot ground truth trajectories
	f = Figure()
	ax = Axis(f[1,1], xlabel=L"t", ylabel="Concentration", title="Ground truth trajectories")
	for i in 1:n_species
		lines!(sol.t, [pt[i] for pt in sol.u], label=L"X_%$i")
	end
	axislegend(position=:rc);
	f
	save(joinpath(dirname, "traj.png"), f)

	# Observed data
	t_obs = range(t_span..., n_obs);
	y_noiseless = reduce(hcat, sol(t_obs).u);

	Random.seed!(seed);
	data = y_noiseless .+ σ .* randn(size(y_noiseless)); # additive noise
	data = max.(0.0, data); # clamp negative values to 0

	# Plot noisy data with ground truth trajectories
	f = Figure()
	ax = Axis(f[1,1], xlabel=L"t", ylabel="Concentration", title="Ground truth trajectories and observed data")
	for i in 1:n_species
		scatter!(t_obs, data[i,:], color=lighten(palette[i], 0.7))
		lines!(sol.t, [pt[i] for pt in sol.u], label=L"X_%$i")	
	end
	axislegend(position=:rc);
	f
	save(joinpath(dirname, "data.png"), f)

	# Export data
	# NB: First column is time, each column after the first corresponds to a speicies
	writedlm(joinpath(dirname, "data.txt"), [t_obs data']);
end