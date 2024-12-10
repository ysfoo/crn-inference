########################################################
### Plots results for different optimisation options ###
########################################################

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl")); # import `make_plots`

# Visualise results
for (pen_str, log_opt) in collect(opt_options)
	opt_dirname = get_opt_dir(pen_str, log_opt) # directory for storing results

	pen_hyp = HYP_DICT[pen_str]
	iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, pen_hyp; log_opt=log_opt)

	kmat = readdlm(joinpath(opt_dirname, "inferred_rates.txt"));
	make_plots(iprob, kmat, true_kvec, k, opt_dirname);
end
