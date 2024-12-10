##############################################################################
### Performs inference for different optimisation options (multi-threaded) ###
##############################################################################

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports key functions for inference


# Random initial points for optimisation runs
Random.seed!(1);
init_vec = [rand(n_rx) for _ in 1:N_RUNS]

# Perform multi-start optimisation and export reaction rates
Threads.@threads for (pen_str, log_opt) in collect(opt_options)
	opt_dirname = get_opt_dir(pen_str, log_opt) # directory for storing results
	mkpath(opt_dirname); # create directory

	pen_hyp = HYP_DICT[pen_str]
	iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, pen_hyp; log_opt=log_opt)
	true_loss = iprob.optim_func(iprob.tf.(true_kvec)) # loss evaluated for ground truth parameters

	open(joinpath(opt_dirname, "optim_progress.txt"), "w") do io
		function optim_callback(i, res)
			time_str = pyfmt(FMT_2DP, res.time_run)
			# loss offset = diff b/w the local minimum found and the loss evaluated at the ground truth rate constants
			loss_offset_str = pyfmt(FMT_2DP, res.minimum - true_loss)    
			write(io, "Run $i: $(time_str) seconds, loss offset = $(loss_offset_str)\n")
			flush(io)
		end
		res_vec = optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback)
		export_estimates(res_vec, iprob, opt_dirname, "inferred_rates.txt");
	end
end
