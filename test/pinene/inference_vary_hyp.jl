using QuasiMonteCarlo
using StableRNGs
using LinearAlgebra
BLAS.set_num_threads(1);

# Try to make multi-threading more efficient
using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

using ProgressMeter
using StableRNGs
using Format
FMT_3DP = ".3f";

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference

# Smooth data
smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
init_σs = [estim_σ(datarow .- eval_spline(res..., t_obs)) for (datarow, res) in zip(eachrow(data), smooth_resvec)]
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)

# rng = StableRNG(1);
# init_vecs = [rand(rng, n_rx) for _ in 1:N_RUNS];

# Export reaction rates
@Threads.threads for (pen_str, hyp_val) in opt_options
# @showprogress @Threads.threads for (pen_str, hyp_val) in opt_options
	opt_dir = get_opt_dir(pen_str, hyp_val) # directory for storing results
	mkpath(opt_dir); # create directory

	rng = StableRNG(hyp_val);
	init_vecs = eachcol(0.0001 .^ QuasiMonteCarlo.sample(
		N_RUNS, n_rx, SobolSample(R=OwenScramble(base=2, pad=32, rng=rng))
	));
	
	# pen_func = make_pen_func(pen_str, hyp_val)
	iprob = make_iprob(
		oprob, k, t_obs, data, pen_str, hyp_val; scale_fcts, abstol=1e-10
		# scale_fcts, custom_pen_func=(x)->pen_func(sum(x[13:16])*sum(x[21:24]))
	);
	
	open(joinpath(opt_dir, "optim_progress.txt"), "w"; lock=false) do io
		function optim_callback(i, res)
			time_str = pyfmt(FMT_3DP, res.time_run)
			σs = exp.(res.minimizer[end-n_species+1:end])
            σs_str = join(pyfmt.(Ref(FMT_3DP), σs), " ")
            write(io, "Run $i: $(time_str) seconds, σs = [$σs_str]\n")
            flush(io)
		end
		res_vec = optim_iprob(
			iprob, [lbs; σ_lbs], [ubs; σ_ubs], 
			init_vecs, init_σs; callback_func=optim_callback,
			# optim_opts=Optim.Options(outer_x_abstol=1e-10, outer_f_abstol=1e-10)
		);
		export_estimates(res_vec, opt_dir);
	end
end
