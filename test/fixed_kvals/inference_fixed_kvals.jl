##########################################
### Performs inference for one dataset ###
##########################################

using StableRNGs

using LinearAlgebra
BLAS.set_num_threads(1);

### only used for SLURM HPC
using ThreadPinning
pinthreads(:affinitymask)
###

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports key functions for inference

# Import synthetic data
t_obs, data = read_data(joinpath(@__DIR__, "data.txt"));
n_species, n_obs = size(data);

smoothers = smooth_data.(eachrow(data), Ref(t_obs));
σs_init = estim_σ.(eachrow(data), smoothers);
scale_fcts = get_scale_fcts(smoothers, species_vec, rx_vec, k);

# Random initial points for optimisation runs
rng = StableRNG(1);
init_vec = [rand(rng, n_rx) for _ in 1:N_RUNS];

# Export reaction rates
@Threads.threads for (pen_str, hyp_val) in opt_options
	opt_dir = get_opt_dir(pen_str, hyp_val) # directory for storing results
	mkpath(opt_dir); # create directory
	
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
