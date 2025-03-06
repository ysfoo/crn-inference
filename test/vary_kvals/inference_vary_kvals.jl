############################################################################################
### Performs inference for different datasets and optimisation settings (multi-threaded) ###
############################################################################################

using StableRNGs

using LinearAlgebra
BLAS.set_num_threads(1);

### Try to make multi-threading more efficient
using ThreadPinning
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);
###

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports key functions for inference

# Estimating scale constants uses `Symbolics.substitute`, which is not thread-safe
true_σs_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
init_σs_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
	true_σs_dict[(k1, k18)] = vec(readdlm(joinpath(data_dir, "stds_true.txt")))

    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
	init_σs_dict[(k1, k18)] = [
		estim_σ(datarow .- eval_spline(res..., t_obs)) 
		for (datarow, res) in zip(eachrow(data), smooth_resvec)
	]
	scale_dict[(k1, k18)] = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)
end

# Perform multi-start optimisation and export reaction rates
@Threads.threads for (k1, k18, pen_str, hyp_val) in settings_vec
	# Random initial points for optimisation runs (same points for same dataset)
	rng = StableRNG(hash((k1, k18)));
	init_vec = [rand(n_rx) for _ in 1:N_RUNS];

    # Define ground truth
    true_kvec = make_true_kvec(k1, k18);

    # Import synthetic data    
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	init_σs = init_σs_dict[(k1, k18)]
	scale_fcts = scale_dict[(k1, k18)]
	
    # Make directory
	opt_dir = get_opt_dir(k1, k18, pen_str, hyp_val) # directory for storing results
	mkpath(opt_dir);

    # Optimisation
	iprob = make_iprob(oprob, k, t_obs, data, pen_str, hyp_val; scale_fcts);
	true_σs = true_σs_dict[(k1, k18)]
	true_θ = [true_σs; iprob.tf(true_kvec)];
	true_val = iprob.optim_func(true_θ) # objective evaluated for ground-truth parameters
	
	open(joinpath(opt_dir, "optim_progress.txt"), "w"; lock=false) do io
		function optim_callback(i, res)
			time_str = pyfmt(FMT_2DP, res.time_run)
			penloss_diff_str = pyfmt(FMT_2DP, res.minimum - true_val)    
			write(io, "Run $i: $(time_str) seconds, diff = $(penloss_diff_str)\n")
			flush(io)
		end
		res_vec = optim_iprob(iprob, lbs, ubs, init_vec, init_σs; callback_func=optim_callback)
		export_estimates(res_vec, opt_dir);
	end
end
