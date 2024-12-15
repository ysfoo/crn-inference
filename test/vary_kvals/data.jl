#########################################################
### Simulate data where all rate constants are varied ###
#########################################################

include(joinpath(@__DIR__, "../../src/sim_data.jl")); # imports `sim_data` for simulating data
include(joinpath(@__DIR__, "setup.jl")); # imports helper function `get_data_dir` and defines `true_rn`

# Simulate data for different ground truth rate constants
# NB: k13 is fixed at 1.0
for k1 in [0.1, 0.3, 1., 3., 10.]
	for k18 in [0.1, 0.3, 1., 3., 10.]
		k1_str = pyfmt(".0e", k1)
		k18_str = pyfmt(".0e", k18)
		kmap  = (:k1 => k1, :k18 => k18, :k13 => 1.); # reaction rate constants
		
		# Make directory
		data_dir = get_data_dir(k1, k18)
		mkpath(data_dir)

		# Simulate and export simulated data
		sim_data(
			true_rn, kmap, x0map, t_span, n_obs, σ,
			data_dir, 2024
		);
	end
end