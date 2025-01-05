#################################################################################
### Plots results (all runs) for different datasets and optimisation settings ###
#################################################################################

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../evaluation.jl"));

# Visualise results (no multi-threading as plotting is not thread-safe, boo!)
for (k1, k18, pen_str, hyp_str) in settings_vec
    true_kvec = make_true_kvec(k1, k18);

    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
    
	opt_dir = get_opt_dir(k1, k18, pen_str, hyp_str)
    pen_hyp = get_hyp(pen_str, hyp_str)
	iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, pen_hyp)
    
	kmat = readdlm(joinpath(opt_dir, "inferred_rates.txt"));	
	make_plots_runs(iprob, kmat, true_kvec, k, opt_dir);
    GC.gc(); # garbage collection, just in case
end