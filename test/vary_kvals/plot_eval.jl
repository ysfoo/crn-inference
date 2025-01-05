#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../evaluation.jl"));

# Directory for figures
fig_dir = joinpath(@__DIR__, "output/eval_figs");
mkpath(fig_dir);

# Text to appear on plots
pen_names = ["\$L_1\$", "log \$L_1\$", "approx. \$L_0\$", "horseshoe"];
hyp_names = ["halved", "default", "doubled"];

# Read in estimation results (takes ~10s)
kmat_dict = Dict{Tuple, AbstractMatrix}();
loffset_dict = Dict{Tuple, AbstractVector}();
best_kvec_dict = Dict{Tuple, AbstractVector}();
best_loffset_dict = Dict{Tuple, Float64}();
traj_err_dict = Dict{Tuple, AbstractMatrix}();

t_grid = range(t_span..., 1001);

for settings in settings_vec
    k1, k18, pen_str, hyp_str = settings
    t_obs, data = read_data(joinpath(get_data_dir(k1, k18), "data.txt"));
    iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, get_hyp(pen_str, hyp_str))
    kmat = readdlm(joinpath(get_opt_dir(k1, k18, pen_str, hyp_str), "inferred_rates.txt"));
    true_kvec = make_true_kvec(k1, k18)
    
    # Optimised value of loss function for each run
    loss_vec = iprob.optim_func.(eachcol(iprob.tf.(kmat)))
	# Loss function evaluated for the ground truth parameters
	true_loss = iprob.optim_func(iprob.tf.(true_kvec))
    # Difference between optimised loss values and loss value evaluated at the ground truth
    loffset_vec = loss_vec .- true_loss

    kmat_dict[settings] = kmat
    loffset_dict[settings] = loffset_vec
    min_idx = argmin(loffset_vec)
    best_kvec_dict[settings] = kmat[:,min_idx]
    best_loffset_dict[settings] = loffset_vec[min_idx]

    traj_err_dict[settings] = get_traj_err(kmat[:,min_idx], true_kvec, oprob, k, t_grid)
end


## Trajectory reconstruction

traj_Linf_dict = Dict(settings => get_traj_Linf(traj_err) for (settings, traj_err) in traj_err_dict);
traj_Linf_max = maximum(values(traj_Linf_dict))

fig = Figure(size=(1200, 900))
for ax_i in eachindex(hyp_choice), ax_j in eachindex(pen_choice)
    ax = Axis(
        fig[ax_i, ax_j], aspect=DataAspect(),
        xminorgridcolor = :black,
        xminorgridvisible = true,
        xminorticks = eachindex(k1_choice) .- 0.5,
        yminorgridcolor = :black,
        yminorgridvisible = true,
        yminorticks = eachindex(k1_choice) .- 0.5,
        xticks = (eachindex(k1_choice), string.(k1_choice)),
        yticks = (eachindex(k18_choice), string.(k18_choice)),
        xlabel = L"Ground truth $k_1$",
        ylabel = L"Ground truth $k_{18}$",
        title = L"%$(pen_names[ax_j]), %$(hyp_names[ax_i]) $\lambda$"
    )
    mat = [traj_Linf_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
    hm = heatmap!(mat, colormap=(:Reds, 0.8), colorrange=(0, traj_Linf_max))
    if ax_i == length(hyp_choice) && ax_j == length(pen_choice)
        Colorbar(fig[:, end+1], hm, height = Relative(0.9))
    end
end
Label(
    fig[0, 1:length(pen_choice)], "Absolute trajectory reconstruction error", 
    font = :bold, fontsize=24
)
current_figure()
save(joinpath(fig_dir, "traj_err.png"), fig)


## Network reconstruction

inferred_rxs_dict = Dict(settings => infer_reactions(kvec) for (settings, kvec) in best_kvec_dict);

true_rxs = [1, 18, 13];
tpos_dict = Dict(settings => length(rxs ∩ true_rxs) for (settings, rxs) in inferred_rxs_dict); # true positives
fpos_dict = Dict(settings => length(setdiff(rxs, true_rxs)) for (settings, rxs) in inferred_rxs_dict); # false positives
fneg_dict = Dict(settings => length(setdiff(true_rxs, rxs)) for (settings, rxs) in inferred_rxs_dict); # false negatives

# True positives
fig = Figure(size=(1200, 900))
for ax_i in eachindex(hyp_choice), ax_j in eachindex(pen_choice)
    ax = Axis(
        fig[ax_i, ax_j], aspect=DataAspect(),
        xminorgridcolor = :black,
        xminorgridvisible = true,
        xminorticks = eachindex(k1_choice) .- 0.5,
        yminorgridcolor = :black,
        yminorgridvisible = true,
        yminorticks = eachindex(k1_choice) .- 0.5,
        xticks = (eachindex(k1_choice), string.(k1_choice)),
        yticks = (eachindex(k18_choice), string.(k18_choice)),
        xlabel = L"Ground truth $k_1$",
        ylabel = L"Ground truth $k_{18}$",
        title = L"%$(pen_names[ax_j]), %$(hyp_names[ax_i]) $\lambda$"
    )
    mat = [tpos_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
    hm = heatmap!(mat, colormap=(:greens, 0.8), colorrange=(0, 3))
    for i in eachindex(k1_choice), j in eachindex(k18_choice)
        text!(ax, "$(mat[i, j])", position = (i, j),
            color = :black, align = (:center, :center))
    end
end
Label(
    fig[0, 1:length(pen_choice)], "Number of reactions correctly identified from best optimisation run", 
    font = :bold, fontsize=24
)
current_figure()
save(joinpath(fig_dir, "true_positives.png"), fig)

# False positives
fpos_max = maximum(maximum.(values(fpos_dict)))
fig = Figure(size=(1200, 900))
for ax_i in eachindex(hyp_choice), ax_j in eachindex(pen_choice)
    ax = Axis(
        fig[ax_i, ax_j], aspect=DataAspect(),
        xminorgridcolor = :black,
        xminorgridvisible = true,
        xminorticks = eachindex(k1_choice) .- 0.5,
        yminorgridcolor = :black,
        yminorgridvisible = true,
        yminorticks = eachindex(k1_choice) .- 0.5,
        xticks = (eachindex(k1_choice), string.(k1_choice)),
        yticks = (eachindex(k18_choice), string.(k18_choice)),
        xlabel = L"Ground truth $k_1$",
        ylabel = L"Ground truth $k_{18}$",
        title = L"%$(pen_names[ax_j]), %$(hyp_names[ax_i]) $\lambda$"
    )
    mat = [fpos_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
    hm = heatmap!(mat, colormap=(:reds, 0.8), colorrange=(0, fpos_max), colorscale=sqrt)
    for i in eachindex(k1_choice), j in eachindex(k18_choice)
        text!(ax, "$(mat[i, j])", position = (i, j),
            color = :black, align = (:center, :center))
    end
end
Label(
    fig[0, 1:length(pen_choice)], "Number of reactions extraneously identified from best optimisation run", 
    font = :bold, fontsize=24
)
current_figure()
save(joinpath(fig_dir, "false_positives.png"), fig)


## Rate constant estimation

function make_kerr_plot(k_idx, true_k_dict)
    k_err_dict = Dict(
        settings => begin
            true_k = true_k_dict[settings];
            (kvec[k_idx] - true_k) / true_k
        end for (settings, kvec) in best_kvec_dict
    );
    fig = Figure(size=(1200, 900))
    for ax_i in eachindex(hyp_choice), ax_j in eachindex(pen_choice)
        ax = Axis(
            fig[ax_i, ax_j], aspect=DataAspect(),
            xminorgridcolor = :black,
            xminorgridvisible = true,
            xminorticks = eachindex(k1_choice) .- 0.5,
            yminorgridcolor = :black,
            yminorgridvisible = true,
            yminorticks = eachindex(k1_choice) .- 0.5,
            xticks = (eachindex(k1_choice), string.(k1_choice)),
            yticks = (eachindex(k18_choice), string.(k18_choice)),
            xlabel = L"Ground truth $k_1$",
            ylabel = L"Ground truth $k_{18}$",
            title = L"%$(pen_names[ax_j]), %$(hyp_names[ax_i]) $\lambda$"
        )
        mat = [k_err_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
        hm = heatmap!(mat, colormap=:curl, colorrange=(-1, 1))
    end
    Colorbar(fig[:, end+1], limits=(-1, 1), colormap=:curl, height = Relative(0.9))
    Label(
        fig[0, 1:length(pen_choice)], L"$\textbf{Relative error of } \mathbf{k_{%$(k_idx)}}$", 
        fontsize=24,
    )
    save(joinpath(fig_dir, "k$(k_idx)_err.png"), fig)
    return fig;
end

true_k1_dict = Dict(settings => settings[1] for settings in settings_vec);
true_k13_dict = Dict(settings => 1.0 for settings in settings_vec);
true_k18_dict = Dict(settings => settings[2] for settings in settings_vec);

make_kerr_plot(1, true_k1_dict);
make_kerr_plot(13, true_k13_dict);
make_kerr_plot(18, true_k18_dict);