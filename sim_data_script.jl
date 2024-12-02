include(joinpath(@__DIR__, "sim_data.jl"));; # imports `sim_data` for simulating data

# Simulate data under default setup, where all rate constants are 1.0
sim_data(dirname=joinpath(@__DIR__, "output"), seed=2024);

# Simulate data for different ground truth rate constants
# NB: k13 is fixed at 1.0
for k1 in [0.1, 0.3, 1., 3., 10.]
	for k18 in [0.1, 0.3, 1., 3., 10.]
		k1_str = pyfmt(".0e", k1)
		k18_str = pyfmt(".0e", k18)
		kmap  = (:k1 => k1, :k18 => k18, :k13 => 1.); # reaction rate constants
		
		# Make directory
		dirname = joinpath(@__DIR__, "output/vary_kvals/k1_$(k1_str)_k18_$(k18_str)")
		mkpath(dirname)
		
		# Export true reaction rates
		open(joinpath(dirname, "true_rates.txt"), "w") do io
			show(io, kmap)
		end

		# Simulate and export simulated data
		sim_data(kmap=kmap, dirname=dirname, seed=2024);
	end
end