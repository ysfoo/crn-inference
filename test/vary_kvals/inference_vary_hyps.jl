############################################################################################
### Performs inference for different datasets and optimisation settings (multi-threaded) ###
############################################################################################

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports key functions for inference

# Random initial points for optimisation runs
Random.seed!(1);
init_vec = [rand(n_rx) for _ in 1:N_RUNS];

# Perform multi-start optimisation and export reaction rates
# Comment out this block if optimisation is already performed (e.g. when redoing plots)
Threads.@threads for (k1, k18, pen_str, hyp_str) in settings_vec
    # Define ground truth
    true_kvec = make_true_kvec(k1, k18);

    # Import synthetic data    
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(data_dir);
    n_obs = length(t_obs);
	
    # Make directory
	opt_dir = get_opt_dir(k1, k18, pen_str, hyp_str) # directory for storing results
	mkpath(opt_dir);

    # Optimisation
    pen_hyp = get_hyp(pen_str, hyp_str)
	iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, pen_hyp)
	true_loss = iprob.optim_func(iprob.tf.(true_kvec))

	open(joinpath(opt_dir, "optim_progress.txt"), "w") do io
		function optim_callback(i, res)
			time_str = pyfmt(FMT_2DP, res.time_run)
			# loss offset = diff b/w the local minimum found and the loss evaluated at the ground truth rate constants
			loss_offset_str = pyfmt(FMT_2DP, res.minimum - true_loss)    
			write(io, "Run $i: $(time_str) seconds, loss offset = $(loss_offset_str)\n")
			flush(io)
		end
		res_vec = optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback)
		export_estimates(res_vec, iprob, opt_dir, "inferred_rates.txt");
	end
end
