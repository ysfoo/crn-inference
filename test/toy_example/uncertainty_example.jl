#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter
using LogExpFunctions, SpecialFunctions, StatsBase

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

# NO_CROSS = false
# EST_FNAME = NO_CROSS ? "nocross_estimates.txt" : "refined_estimates.txt"
# PLT_FNAME = NO_CROSS ? "top_crns_nocross.png" : "top_crns_crossover.png"

true_rxs = [1, 13, 18];
equiv_crns = [[1,11,14,18],[1,12,15,18],[1,13,16,19],[1,13,17,20]];
# [[true_rxs]; equiv_crns]
t_grid = range(t_span..., 1000);

scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
	scale_dict[(k1, k18)] = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)
end

## Text to appear on plots
pen_names = Dict(
    "L1" => "\\mathbf{L_1}", 
    "logL1" => "\\textbf{log } \\mathbf{L_1}", 
    "approxL0" => "\\textbf{Approximate }\\mathbf{L_0}", 
    "hslike" => "\\textbf{Horseshoe}"
);

pen_names_reg = Dict(
    "L1" => "L_1",
    "logL1" => "\\text{log }L_1",
    "approxL0" => "\\text{Approx. }L_0",
    "hslike" => "\\text{Horseshoe}"
);

logprior(n) = -logabsbinomial(n_rx, n)[1] - log(n_rx+1)

bic_dict_lookup = Dict{Tuple, Dict{Vector{Int64},Float64}}();
post_dict_lookup = Dict{Tuple, Dict{Vector{Int64},Float64}}();
isol_dict_lookup = Dict{Tuple, Dict{Vector{Int64},ODEInferenceSol}}();
top_95_dict = Dict{Tuple, Vector{Vector{Int64}}}();
crn_mode_dict = Dict{Tuple, Vector{Int64}}();
base_crns_dict = Dict{Tuple, Set{Vector{Int64}}}();

@showprogress for (k1, k18, pen_str) in Iterators.product(K1_VALS, K18_VALS, PEN_STRS)
    settings = (k1, k18, pen_str);
    data_dir = get_data_dir(k1, k18);
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
    scale_fcts = scale_dict[(k1, k18)];
    iprob = make_iprob(
        oprob, k, t_obs, data, pen_str, HYP_VALS[1]; 
        scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
    )
    bic_by_rxs = Dict{Vector{Int64},Float64}();
    isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
    est_mat = readdlm(joinpath(data_dir, pen_str, "refined_estimates.txt"));
    # est_mat[findall(est_mat .≈ log(1e-6))] .= -Inf
    for est in eachcol(est_mat)
        rxs = findall(exp.(est[1:n_rx]) .> 1e-6 + eps(Float64))
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)      
        bic = 2*iprob.loss_func(kvec, σs) + length(rxs)*log(length(data))
        if (get(bic_by_rxs, rxs, Inf) < bic) continue end
        bic_by_rxs[rxs] = bic
        isol_by_rxs[rxs] = isol
    end
    bic_dict_lookup[settings] = bic_by_rxs
    isol_dict_lookup[settings] = isol_by_rxs

    logps = Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_dict_lookup[settings])
    logZ = logsumexp(collect(values(logps)))
    post_dict_lookup[settings] = posts = Dict(rxs => exp(logp-logZ) for (rxs, logp) in logps)

    cutoff_idx = findfirst((>)(0.95), cumsum(sort(collect(values(posts)), rev=true)))
    cutoff_logp = sort(collect(values(logps)), rev=true)[cutoff_idx]
    top_95_dict[settings] = [rxs for (rxs, logp) in logps if logp >= cutoff_logp]
    crn_mode_dict[settings] = argmax(x->posts[x], keys(posts))

    base_crns = Set{Vector{Int64}}();
    opt_dir = joinpath(@__DIR__, "output", pen_str)
    est_mat = readdlm(joinpath(data_dir, pen_str, "nocross_estimates.txt"))
    # est_mat[findall(est_mat .≈ log(1e-6))] .= -Inf
    for est in eachcol(est_mat)
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        rxs = findall(isfinite.(est[1:n_rx]))
        push!(base_crns, sort(rxs))
    end
    base_crns_dict[settings] = base_crns
end

x0_altmap = [:X1 => 1., :X2 => 0., :X3 => 0.];

using GraphMakie, Graphs
using Format
FMT_2DP = ".2f";
FMT_3DP = ".3f";
FMT_4DP = ".4f";

MIN_SIZE = 3;

struct Node
    p_sum::Float64
    post_dict::Dict{Vector{Int64},Float64}
    is_leaf::Bool
    branch_rx::Int64
    rx_subset::Vector{Int64}
end

function walk_tree(post_dict, n_rx; p_thres=0.5)
    g = SimpleDiGraph()
    nodes = Node[]
    walk_tree!(g, nodes, post_dict, Set(1:n_rx); p_thres)
    return (g, nodes)
end    

function walk_tree!(g, nodes, post_dict, branch_set, 
                    branched_by=0, rx_subset=Int64[]; p_thres=0.5)
    add_vertex!(g)
    p_sum = sum(values(post_dict)); 
    curr = vertices(g)[end]

    # find reaction to branch on
    marg_dict = Dict(r => sum(p for (crn, p) in post_dict if r ∈ crn; init=0.0) for r in branch_set)
    # println((length(marg_dict), length(post_dict)))
    r_max, p_max = argmax(x->(x.second,-x.first), collect(marg_dict))
    has_r = Dict(crn => p for (crn, p) in post_dict if r_max ∈ crn)
    no_r = Dict(crn => p for (crn, p) in post_dict if r_max ∉ crn)

    # check if leaf    
    min_crn = argmin(length, keys(post_dict))
    # is_leaf = all(Ref(min_crn) .⊆ keys(post_dict)) || p_max/p_sum < p_thres
    is_leaf = (
        length(min_crn) == length(rx_subset)
        || length(has_r) == 1
        # || length(post_dict) <= MIN_SIZE 
        # || p_max/p_sum < p_thres
    )
    push!(nodes, Node(p_sum, post_dict, is_leaf, branched_by, rx_subset))
    if is_leaf return curr end
    
    # remove reaction from branching candidates
    delete!(branch_set, r_max)
    # left child
    child1 = walk_tree!(g, nodes, has_r, branch_set, r_max, [rx_subset;r_max]; p_thres)    
    add_edge!(g, curr, child1)
    # right child
    if !isempty(no_r)
        child2 = walk_tree!(g, nodes, no_r, branch_set, r_max, rx_subset; p_thres)      
        add_edge!(g, curr, child2)  
    end
    # restore reaction to branching candidates
    push!(branch_set, r_max)

    return curr
end

function get_nlabel(node, incl_crns=true, excl_line=true)
    n_crns = length(node.post_dict)    
    suppress_crns = minimum(length, collect(keys(node.post_dict))) == length(node.rx_subset)
    firstline = "$(pyfmt(FMT_3DP, node.p_sum))($n_crns)"
    crn_strs = Ref("{") .* join.(
        sort(setdiff.(collect(keys(node.post_dict)), Ref(excl_line ? node.rx_subset : Int64[]))),
        Ref(",")) .* Ref("}"
    );
    if node.is_leaf && (incl_crns || !suppress_crns)
        nwline = "\n";
        return "$firstline\n$(join(crn_strs, nwline))"
    end
    return firstline
end

function get_einfo_rx(g, nodes)
    elabels = AbstractString[]
    eshifts = Float64[]
    esides = Symbol[]
    ealigns = Tuple{Symbol, Symbol}[]
    edists = Float64[]
    for (parent, child_vec) in enumerate(g.fadjlist)
        if length(child_vec) == 0 continue end
        child = nodes[child_vec[1]]
        push!(elabels, L"%$(string(child.branch_rx))\in R")      
        push!(esides, :left)
        push!(ealigns, (:right, :center))
        if length(child_vec) == 1
            push!(eshifts, 0.5)       
            push!(edists, -10.)     
        else
            push!(elabels, L"%$(string(child.branch_rx))\,\notin\, R")
            append!(eshifts, fill(0.4, 2))
            push!(esides, :right)
            push!(ealigns, (:left, :center))  
            append!(edists, fill(-10., 2))
        end
    end
    return elabels, eshifts, esides, ealigns, edists
end

function get_subdict(d, subkeys)
    return Dict(k => d[k] for k in subkeys)
end

# pen_str = "hslike";
for pen_str in PEN_STRS
settings = (0.1,10,pen_str);
top_95_dict[settings]
post_dict = post_dict_lookup[settings];
marg_post = [sum(p for (crn, p) in post_dict if r ∈ crn) for r in 1:n_rx];
rx_subset = findall(marg_post .> 0.01)
n_sub = length(rx_subset);
tmp_d = get_subdict(post_dict, top_95_dict[settings]);

wpairs = collect(post_dict);
ws = ProbabilityWeights(last.(wpairs));
hasrx_vec = [Int.(rx .∈ first.(wpairs)) for rx in 1:n_rx];
cmat = cor(reduce(hcat, hasrx_vec), ws);

data_dir = get_data_dir(settings[1:2]...);
t_obs, data = read_data(joinpath(data_dir, "data.txt"));
scale_fcts = scale_dict[settings[1:2]];
iprob = make_iprob(
    oprob, k, t_obs, data, settings[3], HYP_VALS[1]; 
    scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
);

fig = begin
fig = Figure(size = (1080, 1080), figure_padding=1)
fa = fig[1, 1] = GridLayout()
fb = fig[1, 2] = GridLayout()
fc = fig[2, 1:2] = GridLayout()

a1 = Axis(fa[1,1], xlabel=L"t", ylabel=L"x_1(t)")
a2 = Axis(fa[2,1], xlabel=L"t", ylabel=L"x_1(t)")
scatter!(a1, t_obs, data[1,:], markersize=8, color=(:grey40, 0.8), label="Data")
for crn in keys(tmp_d)
    isol = isol_dict_lookup[settings][crn]
    est_kvec =  mask_kvec(isol, crn)
    est_oprob = remake(iprob.oprob, p=[k => est_kvec], u0=x0_map)
    est_sol_grid = solve(est_oprob)(t_grid).u
    lines!(a1, t_grid, first.(est_sol_grid), color=(palette[1], 0.6))
    pred_oprob = remake(iprob.oprob, p=[k => est_kvec], u0=x0_altmap)
    pred_sol_grid = solve(pred_oprob)(t_grid).u
    lines!(a2, t_grid, first.(pred_sol_grid), color=(palette[1], 0.6))
end
axislegend(a1, position=:rb)
rowgap!(fa, 0)
# linkyaxes!(a1, a2)

a3 = Axis(
    fb[1,1], xscale=log10, xautolimitmargin=(0.,0.05), xreversed=true, yaxisposition=:right,
    yautolimitmargin=(0.,0.), ylabel="Reaction index", #xlabel="Posterior prob.",
    yticks=(1:n_sub, string.(rx_subset)),
    limits=(nothing, (0.5, n_sub+0.5)), 
    title="Posterior probabilities", titlesize=18,
)
a4 = Axis(
    fb[1,2], aspect=DataAspect(), xlabel="Reaction index",
    yticks=1:n_sub, yticklabelsvisible=false,
    xticks=(1:n_sub, string.(rx_subset)), 
    title="Posterior correlations", titlesize=18,
)
barplot!(
    a3, marg_post[rx_subset], direction=:x,
    color=[palette[r∈true_rxs ? 2 : 1] for r in rx_subset]
)
Legend(
    fb[1,1],
    [PolyElement(color=palette[2])],
    ["Ground truth"],
    gridshalign=:left, labelsize=16,
    tellheight = false, halign=:center,
    tellwidth = false, valign=:bottom,
    margin = (10, 10, -60, 10), patchsize=(15.,15.),
    # padding = (6, 6, 6, 6),
)
hm = heatmap!(a4, cmat[rx_subset,rx_subset], colormap=:RdBu, colorrange=(-1,1))
linkyaxes!(a3, a4)
Colorbar(
    fb[1,3], hm, ticklabelsize=16,
    #label="Posterior correlation", labelsize=16, labelrotation=π/2
    )
colsize!(fb, 2, Fixed(350))
colgap!(fb, 1, -5)
colgap!(fb, 2, 0)

a = Axis(fc[1,1])
g, nodes = walk_tree(tmp_d, n_rx);
nlabels = get_nlabel.(nodes, false)
get_nsize(node) = 4 + 1.0*length(node.post_dict)^0.8;
nsizes = get_nsize.(nodes)
nlabels_align = [(:right, :center) for v in vertices(g)];
elabels, eshifts, esides, ealigns, edists = get_einfo_rx(g, nodes)
p = graphplot!(
    a, g;
    layout=GraphMakie.Buchheim(),
    # arrow_show=false,
    # edge_plottype=:linesegments,
    nlabels=nlabels,
    nlabels_distance=10,
    nlabels_align,
    node_size=nsizes,
    elabels=elabels,
    elabels_shift=eshifts,
    elabels_side=esides,
    elabels_align=ealigns,
    elabels_distance=edists,
    elabels_fontsize=16,
    elabels_rotation=0,
    tangents=((0,-1),(0,-1)),
    arrow_size=16,
);
hidedecorations!(a); hidespines!(a);
for v in vertices(g)
    if isempty(inneighbors(g, v)) # root
        nlabels_align[v] = (:center,:bottom)
    elseif isempty(outneighbors(g, v)) #leaf
        nlabels_align[v] = (:center,:top)
    else
        self = p[:node_pos][][v]
        curr = p[:node_pos][][inneighbors(g, v)[1]]
        if self[1] < curr[1] # left branch
            nlabels_align[v] = (:right,:center)
        end
    end
end
p.nlabels_align = nlabels_align;
# autolimits!(a)
autolimits!(a)
tmp_lims = a.finallimits[]
pad = Dict(
    "L1" => (0., 0., 1., 0.3),
    "logL1" => (0., 0., 1.6, 0.3),
    "approxL0" => (0.1, 0., 1.2, .3),
    "hslike" => (0., 0., 2.6, 0.3),
)[pen_str];
xlims!(a, a.finallimits.val.origin[1]-pad[1], a.finallimits.val.origin[1]+a.finallimits.val.widths[1]+pad[2])
ylims!(a, a.finallimits.val.origin[2]-pad[3], a.finallimits.val.origin[2]+a.finallimits.val.widths[2]+pad[4])
labels = [
    begin 
        s = string(rx_vec[rx])
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "\\rightarrow")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        # "\$$s\$"
        L"%$rx: %$s"
    end for rx in [1;11:15;16:20;30]
];
n_labels = length(labels)

Label(fa[1, 1, Top()], L"\textbf{Reconstructed}\;\mathbf{X_1}\;\textbf{(95% HPD set)}", valign = :bottom,
    font = :bold, fontsize=18,
    padding = (0, 0, 5, 0))
Label(fa[2, 1, Top()], L"\textbf{Predicted}\;\mathbf{X_1}\;\textbf{(95% HPD set)}", valign = :bottom,
    font = :bold, fontsize=18,
    padding = (0, 0, 5, 0))
# Label(fb[1, :, Top()], "Reaction-level summary of posterior", valign = :bottom,
#     font = :bold, fontsize=18,
#     padding = (0, 0, 5, 0))
Label(fc[1, 1, Top()], "Composition of 95% HPD set", valign = :bottom,
    font = :bold, fontsize=20,
    padding = (0, 0, 0, 0))
Legend(
    fc[1,1],
    fill(MarkerElement(marker=:circle, markersize=0), n_labels),
    labels, #"Ground-truth CRN",
    markerpoints=fill(Point2f(0., 0.), n_labels),
    gridshalign=:left, labelsize=16,
    tellheight = false, halign=:left,
    tellwidth = false, valign=:bottom,
    margin = (10, 10, 10, 10), patchsize=(0.,20.),
    padding = (6, 12, 6, 6),
    nbanks=6, orientation=:horizontal
)
for (label, layout) in zip(["A", "B", "C"], [fa, fb, fc])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 24,
        font = :bold,
        padding = (0, "label" == "C" ? 0 : 15, 10, 0),
        halign = :right)
end
colsize!(fig.layout, 1, Auto(0.5))
rowsize!(fig.layout, 2, Auto(2.0))
rowgap!(fig.layout, 1, 15)
resize_to_layout!(fig)
fig
end

## Directory for evaluation plots
eval_dir = joinpath(@__DIR__, "output/eval_figs");
mkpath(eval_dir);
save(joinpath(eval_dir, "post_summary_$(pen_str).svg"), fig);
end