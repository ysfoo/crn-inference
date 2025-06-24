#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter
using LogExpFunctions
using SpecialFunctions
using StatsBase

using StableRNGs

using Format
FMT_3DP = ".3f";

READ_GOLD = true;

include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "../../src/plot_helper.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "gold_std.jl"));

# NO_CROSS = false
# EST_FNAME = NO_CROSS ? "nocross_estimates.txt" : "refined_estimates.txt"
# PLT_FNAME = NO_CROSS ? "top_crns_nocross.png" : "top_crns_crossover.png"

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

# bb = elicit_bb_prior(n_rx, 12, 4)
# logprior(n) = -logabsbinomial(n_rx, n)[1] + logpdf(bb, n)

calc_jac(x, y) = length(x ∩ y) / length(x ∪ y);

bic_dict_lookup = Dict{String, Dict{Vector{Int64},Float64}}();
post_dict_lookup = Dict{String, Dict{Vector{Int64},Float64}}();
isol_dict_lookup = Dict{String, Dict{Vector{Int64},ODEInferenceSol}}();
top_95_dict = Dict{String, Vector{Vector{Int64}}}();
refine_func_dict = Dict{Vector{Int64}, FunctionWrapper}();

iprob = make_iprob(
    oprob, k, t_obs, data, PEN_STRS[1], HYP_VALS[1]; 
    scale_fcts, alg=AutoVern7(KenCarp4()), abstol=1e-10, verbose=false
);
loss_func = iprob.loss_func;

for pen_str in PEN_STRS
    bic_by_rxs = Dict{Vector{Int64},Float64}();
    isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
    opt_dir = joinpath(@__DIR__, "output", pen_str)
    est_mat = readdlm(joinpath(opt_dir, "refined_estimates.txt"))
    est_mat[findall(est_mat .≈ log(1e-6))] .= -Inf
    for est in eachcol(est_mat)
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        rxs = findall(isfinite.(est[1:n_rx]))
        bic = 2*loss_func(kvec, σs) + length(rxs)*log(length(data))
        if (get(bic_by_rxs, rxs, Inf) < bic) continue end
        bic_by_rxs[rxs] = bic
        isol_by_rxs[rxs] = isol
    end

    bic_dict_lookup[pen_str] = bic_by_rxs
    isol_dict_lookup[pen_str] = isol_by_rxs
    
    logps = Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_dict_lookup[pen_str])
    logZ = logsumexp(collect(values(logps)))
    post_dict_lookup[pen_str] = posts = Dict(rxs => exp(logp-logZ) for (rxs, logp) in logps)

    cutoff_idx = findfirst((>)(0.95), cumsum(sort(collect(values(posts)), rev=true)))
    cutoff_logp = sort(collect(values(logps)), rev=true)[cutoff_idx]
    top_95_dict[pen_str] = [rxs for (rxs, logp) in logps if logp >= cutoff_logp]
end

using GraphMakie, Graphs
using Format
FMT_2DP = ".2f";
FMT_3DP = ".3f";
FMT_4DP = ".4f";

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
    sorted_crns = sort(setdiff.(collect(keys(node.post_dict)), Ref(excl_line ? node.rx_subset : Int64[])))
    crn_strs = Ref("\$\\{") .* join.(
        [[rx ∈ gold_idxs ? "\\mathbf{$rx}" : string(rx) for rx in crn] for crn in sorted_crns],
        Ref(",")) .* Ref("\\}\$"
    );
    if node.is_leaf && (incl_crns || !suppress_crns)
        nwline = "\\n";
        return L"%$firstline%$nwline%$(join(crn_strs, nwline))"
    end
    return firstline
end

function format_rx(rx)
    rx ∈ gold_idxs ? "\\mathbf{$rx}" : string(rx)
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
        push!(elabels, L"%$(format_rx(child.branch_rx))\in R")
        push!(esides, :left)
        push!(ealigns, (:right, :center))
        if length(child_vec) == 1
            push!(eshifts, 0.5)       
            push!(edists, -6.)     
        else
            push!(elabels, L"%$(format_rx(child.branch_rx))\,\notin\, R")
            append!(eshifts, fill(0.4, 2))
            push!(esides, :right)
            push!(ealigns, (:left, :center))  
            append!(edists, fill(-8., 2))
        end
    end
    return elabels, eshifts, esides, ealigns, edists
end

function get_deriv_midpt(pth)
    p0 = pth.commands[1].p
    p1 = pth.commands[2].c1
    p2 = pth.commands[2].c2
    p3 = pth.commands[2].p
    return @. 0.75(p1-p0+2p2-2p1+p3-p2)
end

function get_subdict(d, subkeys)
    return Dict(k => d[k] for k in subkeys)
end

pen_str = "logL1"
post_dict = post_dict_lookup[pen_str];
marg_post = [sum(p for (crn, p) in post_dict if r ∈ crn) for r in 1:n_rx];
rx_subset = findall(marg_post .> 0.01)
n_sub = length(rx_subset);
tmp_d = get_subdict(post_dict, top_95_dict[pen_str]);

wpairs = collect(post_dict);
ws = ProbabilityWeights(last.(wpairs));
hasrx_vec = [Int.(rx .∈ first.(wpairs)) for rx in 1:n_rx];
cmat = cor(reduce(hcat, hasrx_vec), ws);

iprob = make_iprob(
    oprob, k, t_obs, data, pen_str, HYP_VALS[1]; 
    scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
);

fig, ax, p, g, nodes = begin
fig = Figure(size = (1200, 1500), figure_padding=1)
a = Axis(fig[1,1])
g, nodes = walk_tree(tmp_d, n_rx);
nlabels = get_nlabel.(nodes, false)
get_nsize(node) = 4 + 1.0*length(node.post_dict)^0.5;
nsizes = get_nsize.(nodes)
nlabels_align = [(:right, :center) for v in vertices(g)];
elabels, eshifts, esides, ealigns, edists = get_einfo_rx(g, nodes)

# adjust positions manually
layout = GraphMakie.Buchheim()(g.fadjlist);
idx = findfirst(pt->pt[1]!=0.0, layout)-1
shift = layout[idx][2] * 0.4
for i in 2:idx
    pt = layout[i]
    layout[i] = Point(pt[1], pt[2] * 0.6)
end
for i in (idx+1):length(layout)
    pt = layout[i]
    layout[i] = Point(pt[1], pt[2]-shift)
end
for i in 32:33
    pt = layout[i]
    layout[i] = Point(pt[1]+1, pt[2])
end
for i in 10:34
    pt = layout[i]
    layout[i] = Point(pt[1], pt[2]-1.)
end
pt = layout[34]
layout[34] = Point(pt[1]+1.25, pt[2])

p = graphplot!(
    a, g;
    layout=layout,
    arrow_show=false,
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
p.elabels_distance = [
    begin
        dx, dy = get_deriv_midpt(pth);
        dy > 0.5 ? -16. : orig_dst
    end for (pth, orig_dst) in zip(p.edge_paths[], p.elabels_distance[])
]
p.nlabels_align = nlabels_align;
# autolimits!(a)
autolimits!(a)
tmp_lims = a.finallimits[]
xlims!(a, a.finallimits.val.origin[1]-1.1, a.finallimits.val.origin[1]+a.finallimits.val.widths[1]-0.15)
ylims!(a, a.finallimits.val.origin[2]-1.9, a.finallimits.val.origin[2]+a.finallimits.val.widths[2]-0.4)
labels = [
    begin 
        s = string(rx_vec[rx])
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "\\rightarrow")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        # "\$$s\$"
        rx ∈ gold_idxs ? L"\mathbf{%$rx: %$s}" : L"%$rx: %$s"
    end for rx in 1:n_rx
];
n_labels = length(labels);

# Label(fa[1, 1, Top()], L"\textbf{Reconstructed }\mathbf{X_1}", valign = :bottom,
#     font = :bold, fontsize=18,
#     padding = (0, 0, 5, 0))
# Label(fb[1, 1, Top()], L"\textbf{Predicted }\mathbf{X_1}", valign = :bottom,
#     font = :bold, fontsize=18,
#     padding = (0, 0, 5, 0))
# Label(fc[1, 1, Top()], "Reaction-level summary of posterior", valign = :bottom,
#     font = :bold, fontsize=18,
#     padding = (0, 0, 5, 0))
Label(fig[1, 1, Top()], L"\textbf{Composition of 95% HPD set (}%$(pen_names[pen_str])\;\textbf{penalty)}", valign = :bottom,
    font = :bold, fontsize=20,
    padding = (0, 0, 0, 0))
Legend(
    fig[1,1],
    fill(MarkerElement(marker=:circle, markersize=0), n_labels),
    labels, "Candidate reactions",
    markerpoints=fill(Point2f(0., 0.), n_labels),
    nbanks=17, orientation=:horizontal,
    padding=(6.,12.,6.,6.),
    gridshalign=:left, labelsize=17, titlesize=18,
    tellheight = false, halign=:left,
    tellwidth = false, valign=:top,
    margin = (25, 10, 10, 10), patchsize=(0.,20.),
)
# for (label, layout) in zip(["A", "B", "C", "D"], [fa, fb, fc, fd])
#     Label(layout[1, 1, TopLeft()], label,
#         fontsize = 24,
#         font = :bold,
#         padding = (0, "label" == "D" ? 0 : 15, 10, 0),
#         halign = :right)
# end
# colsize!(fig.layout, 3, Auto(2.9))
# rowsize!(fig.layout, 2, Auto(2.5))
# rowgap!(fig.layout, 5)
# colgap!(fig.layout, 2, 0)
# Box(fig[1, 1], color = (:red, 0.2), strokewidth = 0)
resize_to_layout!(fig)
display(fig)
fig, a, p, g, nodes
end;

## Directory for evaluation plots
eval_dir = joinpath(@__DIR__, "output/eval_figs");
mkpath(eval_dir);
save(joinpath(eval_dir, "hpd_tree_$(pen_str).svg"), fig);