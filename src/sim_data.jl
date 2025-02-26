#####################################
### Functions for data simulation ###
#####################################

using Catalyst
using OrdinaryDiffEq

using Random

using DelimitedFiles

include(joinpath(@__DIR__, "plot_helper.jl"));

# Simulates and exports data
# rn      : Reaction network, which is a Catalyst.ReactionSystem
# kmap    : Dictionary mapping reaction rate symbols to their values
# x0map   : Dictionary mapping species symbols to their initial concentrations
# t_span  : Timespan for ODE problem
# n_obs   : Number of observation points
# σ       : Noise standard deviation
# dirname : Directory name to store simulated data
# rng     : Random number generator
function sim_data(rn, kmap, x0map, t_span, n_obs, σ, dirname, rng=Random.default_rng())
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

	data = y_noiseless .+ σ .* randn(rng, size(y_noiseless)); # additive noise
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