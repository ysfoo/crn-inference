using QuasiMonteCarlo
using StableRNGs
using LinearAlgebra
BLAS.set_num_threads(1);

# Try to make multi-threading more efficient
using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

using ProgressMeter
using Format
FMT_3DP = ".3f";

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference

# init_vecs = eachcol(QuasiMonteCarlo.sample(N_RUNS, n_rx, SobolSample()))

# Random initial points for optimisation runs
# rng = StableRNG(1);
# init_vecs = [rand(rng, n_rx) for _ in 1:N_RUNS];

# Export reaction rates
@showprogress @Threads.threads for (pen_str, hyp_val) in opt_options
	opt_dir = get_opt_dir(pen_str, hyp_val) # directory for storing results
	mkpath(opt_dir); # create directory

	rng = StableRNG(hyp_val);
	init_vecs = eachcol(0.0001 .^ QuasiMonteCarlo.sample(
		N_RUNS, n_rx, SobolSample(R=OwenScramble(base=2, pad=32, rng=rng))
	));
	
	# pen_func = make_pen_func(pen_str, hyp_val)
	iprob = make_iprob(
		oprob, k, t_obs, data, pen_str, hyp_val;
		scale_fcts, alg=AutoVern7(KenCarp4()), abstol=1e-10, verbose=false
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
			init_vecs, init_σs; 
			optim_opts=Optim.Options(outer_x_abstol=1e-10, outer_f_abstol=1e-10),
			callback_func=optim_callback
		);
		export_estimates(res_vec, opt_dir);
	end
end
