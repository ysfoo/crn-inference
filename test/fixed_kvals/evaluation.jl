#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

fig_dir = joinpath(@__DIR__, "output", "vary_hyp_figs");
mkpath(fig_dir);

# Import synthetic data
t_obs, data = read_data(joinpath(@__DIR__, "data.txt"));
n_species, n_obs = size(data);

smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)

pen_names = Dict(
    "L1" => "\\mathbf{L_1}", 
    "logL1" => "\\textbf{log} \\mathbf{L_1}", 
    "approxL0" => "\\textbf{Approximate }\\mathbf{L_0}", 
    "hslike" => "\\textbf{Horseshoe}"
);
for pen_str in PEN_STRS
    opt_dirs = get_opt_dir.(Ref(pen_str), HYP_VALS)
    iprobs = [
        iprob = make_iprob(oprob, k, t_obs, data, pen_str, hyp_val; scale_fcts)
        for hyp_val in HYP_VALS]
    isol_vec = [
        begin
            est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
            make_isol(iprob, est_mat)
        end for (opt_dir, iprob) in zip(opt_dirs, iprobs)]
    kvecs = getproperty.(isol_vec, :kvec)
    
    infer_vec = infer_reactions.(isol_vec, Ref(species_vec), Ref(rx_vec))
    inferred_rx_vec = first.(infer_vec)
    fit_vec = last.(infer_vec)
    tune_idx = tune_hyp_bic(inferred_rx_vec, fit_vec)

    ns = length.(inferred_rx_vec)
    nmin, nmax = extrema(ns)
    f = Figure()
    ax1 = Axis(
        f[1, 1], yticklabelcolor=:dodgerblue4, xscale=log10, 
        xlabel=L"\lambda", ylabel="Model fit heuristic",
        title=L"$%$(pen_names[pen_str])$ \textbf{penalty}"
    )
    ax2 = Axis(
        f[1, 1], yticklabelcolor=:darkorange3, yticks=nmin:nmax,
        xscale=log10, ylabel="Number of reactions inferred", yaxisposition = :right
    )
    hidespines!(ax2)
    hidexdecorations!(ax2)
    scatterlines!(
        ax1, HYP_VALS, fit_vec
    )
    scatterlines!(
        ax2, HYP_VALS, ns, 
        color=palette[2]
    )
    scatter!(ax1, HYP_VALS[tune_idx], fit_vec[tune_idx]; color=:dodgerblue4, marker=:star5, markersize=20)
    scatter!(ax2, HYP_VALS[tune_idx], ns[tune_idx]; color=:darkorange4, marker=:star5, markersize=20)
    save(joinpath(fig_dir, "$pen_str.png"), f)
    display(current_figure())

    # # Penalty func vs reaction
    # heatmap(reduce(hcat, kvecs), highclip=:black, colorrange=(0, 1.5))
    # display(current_figure())

    display(inferred_rx_vec[tune_idx])
end
