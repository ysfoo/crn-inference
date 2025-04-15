#################################################################################
### Plots results (all runs) for different datasets and optimisation settings ###
#################################################################################

using ProgressMeter

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
	scale_dict[(k1, k18)] = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)
end

# Visualise results (no multi-threading as plotting is not thread-safe, boo!)
@showprogress dt=5. for (k1, k18, pen_str, hyp_val) in settings_vec
    true_kvec = make_true_kvec(k1, k18);

    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));

    scale_fcts = scale_dict[(k1, k18)]

	opt_dir = get_opt_dir(k1, k18, pen_str, hyp_val)

    iprob = make_iprob(oprob, k, t_obs, data, pen_str, hyp_val; scale_fcts);
	est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
    true_σs = vec(readdlm(joinpath(data_dir, "stds_true.txt")))
	make_plots_runs(iprob, est_mat, true_kvec, true_σs, k, opt_dir);
    GC.gc(); # garbage collection, just in case
end
