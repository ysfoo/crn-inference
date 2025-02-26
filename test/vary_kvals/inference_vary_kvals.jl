############################################################################################
### Performs inference for different datasets and optimisation settings (multi-threaded) ###
############################################################################################

using StableRNGs

using LinearAlgebra
BLAS.set_num_threads(1);

### only used for SLURM HPC
using ThreadPinning
pinthreads(:affinitymask)
###

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports key functions for inference

# Estimating scale constants uses `Symbolics.substitute`, which is not thread-safe
σs_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smoothers = smooth_data.(eachrow(data), Ref(t_obs));
	σs_dict[(k1, k18)] = estim_σ.(eachrow(data), smoothers);
	scale_dict[(k1, k18)] = get_scale_fcts(smoothers, species_vec, rx_vec, k);
end

# Perform multi-start optimisation and export reaction rates
# Comment out this block if optimisation is already performed (e.g. when redoing plots)
@Threads.threads for (k1, k18, pen_str, hyp_val) in settings_vec
	# Random initial points for optimisation runs (same points for same dataset)
	rng = StableRNG(hash((k1, k18)));
	init_vec = [rand(n_rx) for _ in 1:N_RUNS];

    # Define ground truth
    true_kvec = make_true_kvec(k1, k18);

    # Import synthetic data    
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	σs_init = σs_dict[(k1, k18)]
	scale_fcts = scale_dict[(k1, k18)]
	
    # Make directory
	opt_dir = get_opt_dir(k1, k18, pen_str, hyp_val) # directory for storing results
	mkpath(opt_dir);

    # Optimisation
	iprob = make_iprob(oprob, k, t_obs, data, σs_init, pen_str, hyp_val; scale_fcts);
	true_θ = [σs_true; iprob.tf(true_kvec)];
	true_val = iprob.optim_func(true_θ) # objective evaluated for ground-truth parameters
	
	open(joinpath(opt_dir, "optim_progress.txt"), "w"; lock=false) do io
		function optim_callback(i, res)
			time_str = pyfmt(FMT_2DP, res.time_run)
			penloss_diff_str = pyfmt(FMT_2DP, res.minimum - true_val)    
			write(io, "Run $i: $(time_str) seconds, diff = $(penloss_diff_str)\n")
			flush(io)
		end
		res_vec = optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback)
		export_estimates(res_vec, opt_dir);
	end
end
