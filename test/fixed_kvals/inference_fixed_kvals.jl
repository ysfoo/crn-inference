##########################################
### Performs inference for one dataset ###
##########################################

using StableRNGs

using LinearAlgebra
BLAS.set_num_threads(1);

### Try to make multi-threading more efficient
using ThreadPinning
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);
###

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports key functions for inference

# Import synthetic data
t_obs, data = read_data(joinpath(@__DIR__, "data.txt"));
n_species, n_obs = size(data);

smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
init_σs = [estim_σ(datarow .- eval_spline(res..., t_obs)) for (datarow, res) in zip(eachrow(data), smooth_resvec)]
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)

# Random initial points for optimisation runs
rng = StableRNG(1);
init_vec = [rand(rng, n_rx) for _ in 1:N_RUNS];

# Export reaction rates
@Threads.threads for (pen_str, hyp_val) in opt_options
	opt_dir = get_opt_dir(pen_str, hyp_val) # directory for storing results
	mkpath(opt_dir); # create directory
	
	iprob = make_iprob(oprob, k, t_obs, data, pen_str, hyp_val; scale_fcts);
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
