using Format
FMT_2DP = ".2f"; # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "full_network.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)
include(joinpath(@__DIR__, "inference.jl"));
include(joinpath(@__DIR__, "plot_helper.jl"));

# Setup
σ = 0.01; # assume noise SD is known
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # assume initial conditions are known
t_span = (0., 10.);
rx_vec = Catalyst.reactions(full_network); # list of reactions
n_rx = length(rx_vec);
oprob = ODEProblem(full_network, x0map, t_span, zeros(n_rx)); # all rates here are zero, no dynamics

LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)

HYP_DICT = Dict(
	"L1" => 20.0, 
	"logL1" => 1.0, 
	"approxL0" => log(303), # total number of data points  
	"hslike" => 20.0
);

MULT_DICT = Dict("orig_hyps" => 1.0, "half_hyps" => 0.5, "double_hyps" => 2.0);

# All possible setups
k1_choice = [0.1, 0.3, 1., 3., 10.];
k18_choice = [0.1, 0.3, 1., 3., 10.];
pen_choice = ["L1", "logL1", "approxL0", "hslike"]; # penalty_function
hyp_choice = ["half_hyps", "orig_hyps", "double_hyps"] # hyperparameter values (relative to default)
setups = collect(Iterators.product(
    k1_choice, k18_choice, pen_choice, hyp_choice
));

pen_names = ["\$L_1\$", "log \$L_1\$", "approx. \$L_0\$", "horseshoe"];
hyp_names = ["halved", "default", "doubled"];

# Helper functions for reading in estimation results
function make_data_dir(setup)
    k1, k18, _, _ = setup
    k1_str = pyfmt(".0e", k1)
    k18_str = pyfmt(".0e", k18)
    return joinpath(@__DIR__, "output/vary_kvals/k1_$(k1_str)_k18_$(k18_str)")
end

function make_opt_dir(setup)
    _, _, pen_str, hyp_str = setup
    data_dir = make_data_dir(setup)
    joinpath(data_dir, hyp_str, pen_str * "_uselog")
end

function make_true_kvec(setup)
    k1, k18, _, _ = setup
    true_kvec = zeros(n_rx);
    true_kvec[1] = k1; true_kvec[18] = k18; true_kvec[13] = 1.;
    return true_kvec
end

function get_hyp(setup)
    _, _, pen_str, hyp_str = setup
    return HYP_DICT[pen_str] * MULT_DICT[hyp_str]
end

# Read in estimation results (takes ~10s)
kmat_dict = Dict{Tuple, AbstractMatrix}();
loffset_dict = Dict{Tuple, AbstractVector}();
best_kvec_dict = Dict{Tuple, AbstractVector}();
best_loffset_dict = Dict{Tuple, Float64}();

t_grid = range(t_span..., 1001);
traj_err_dict = Dict{Tuple, Float64}();

for setup in setups
    k1, k18, pen_str, hyp_str = setup
    t_obs, data = read_data(make_data_dir(setup));
    iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, get_hyp(setup))
    kmat = readdlm(joinpath(make_opt_dir(setup), "inferred_rates.txt"));
    true_kvec = make_true_kvec(setup)
    
    # Optimised value of loss function for each run
    loss_vec = iprob.optim_func.(eachcol(iprob.tf.(kmat)))
	# Loss function evaluated for the ground truth parameters
	true_loss = iprob.optim_func(iprob.tf.(true_kvec))
    # Difference between optimised loss values and loss value evaluated at the ground truth
    loffset_vec = loss_vec .- true_loss

    kmat_dict[setup] = kmat
    loffset_dict[setup] = loffset_vec
    min_idx = argmin(loffset_vec)
    best_kvec_dict[setup] = kmat[:,min_idx]
    best_loffset_dict[setup] = loffset_vec[min_idx]

    true_oprob = remake(iprob.oprob, p=[k => true_kvec])
    true_sol_grid = solve(true_oprob)(t_grid)
    est_oprob = remake(iprob.oprob, p=[k => kmat[:,min_idx]])
    est_sol_grid = solve(est_oprob)(t_grid)
    traj_err_dict[setup] = sum(abs.(true_sol_grid .- est_sol_grid)) / length(true_sol_grid)
end


traj_err_max = maximum(values(traj_err_dict))

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
    mat = [traj_err_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
    hm = heatmap!(mat, colormap=(:Reds, 0.8), colorrange=(0, traj_err_max))
    if ax_i == length(hyp_choice) && ax_j == length(pen_choice)
        Colorbar(fig[:, end+1], hm, height = Relative(0.9))
    end
end
Label(
    fig[0, 1:length(pen_choice)], "Absolute trajectory reconstruction error", 
    font = :bold, fontsize=24
)
current_figure()
save(joinpath(@__DIR__, "eval_figs/traj_err.png"), fig)


# Decide which reactions are present in the system given estimated rate constants
# Sort the reaction rate constants in log scale, and find the largest gap
# Reactions with rate constants on the higher side of the gap are inferred to be present
function infer_reactions(kvec)
    log_kvec = sort(log.(kvec))
    max_diff = maximum(diff(log_kvec))
    idx = findlast(==(max_diff), diff(log_kvec))
    thres = log_kvec[idx]
    return findall(k -> (log(k) > thres), kvec)
end

inferred_rxs_dict = Dict(setup => infer_reactions(kvec) for (setup, kvec) in best_kvec_dict);

true_rxs = [1, 18, 13];
tpos_dict = Dict(setup => length(rxs ∩ true_rxs) for (setup, rxs) in inferred_rxs_dict); # true positives
fpos_dict = Dict(setup => length(setdiff(rxs, true_rxs)) for (setup, rxs) in inferred_rxs_dict); # false positives
fneg_dict = Dict(setup => length(setdiff(true_rxs, rxs)) for (setup, rxs) in inferred_rxs_dict); # false negatives

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
save(joinpath(@__DIR__, "eval_figs/true_positives.png"), fig)

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
save(joinpath(@__DIR__, "eval_figs/false_positives.png"), fig)

ticks = [-1, -0.5, -0.2, 0.1, 0, 0.1, 0.2, 0.5, 1];

k13_error_dict = Dict(setup => kvec[13] - 1. for (setup, kvec) in best_kvec_dict);
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
    mat = [k13_error_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
    hm = heatmap!(mat, colormap=:curl, colorrange=(-1, 1))
end
Colorbar(fig[:, end+1], limits=(-1, 1), colormap=:curl, height = Relative(0.9))
Label(
    fig[0, 1:length(pen_choice)], L"$\textbf{Relative error of } \mathbf{k_{13}}$", 
    fontsize=24,
)
current_figure()
save(joinpath(@__DIR__, "eval_figs/k13_err.png"), fig)

k1_error_dict = Dict(setup => (kvec[1] - setup[1])/setup[1] for (setup, kvec) in best_kvec_dict);
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
    mat = [k1_error_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
    hm = heatmap!(mat, colormap=:curl, colorrange=(-1, 1))
end
Colorbar(fig[:, end+1], limits=(-1, 1), colormap=:curl, height = Relative(0.9))
Label(
    fig[0, 1:length(pen_choice)], L"$\textbf{Relative error of } \mathbf{k_{1}}$", 
    fontsize=24,
)
current_figure()
save(joinpath(@__DIR__, "eval_figs/k1_err.png"), fig)

k18_error_dict = Dict(setup => (kvec[18] - setup[2])/setup[2] for (setup, kvec) in best_kvec_dict);
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
    mat = [k18_error_dict[(k1, k18, pen_choice[ax_j], hyp_choice[ax_i])] for k1 in k1_choice, k18 in k18_choice]
    hm = heatmap!(mat, colormap=:curl, colorrange=(-1, 1))
end
Colorbar(fig[:, end+1], limits=(-1, 1), colormap=:curl, height = Relative(0.9))
Label(
    fig[0, 1:length(pen_choice)], L"$\textbf{Relative error of } \mathbf{k_{18}}$", 
    fontsize=24,
)
current_figure()
save(joinpath(@__DIR__, "eval_figs/k18_err.png"), fig)
