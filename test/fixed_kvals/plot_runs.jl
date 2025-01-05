####################################################################
### Plots results (all runs) for different optimisation settings ###
####################################################################

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../evaluation.jl"));

# Import synthetic data
t_obs, data = read_data(joinpath(@__DIR__, "data.txt"));

# Visualise results
for (pen_str, log_opt) in opt_options
	opt_dir = get_opt_dir(pen_str, log_opt) # directory for storing results

	pen_hyp = HYP_DICT[pen_str]
	iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, pen_hyp; log_opt=log_opt)

	kmat = readdlm(joinpath(opt_dir, "inferred_rates.txt"));
	make_plots_runs(iprob, kmat, true_kvec, k, opt_dir);
end
