using MAT
using StableRNGs
using OrdinaryDiffEq

SRC_DIR = joinpath(@__DIR__, "../../src");
include(joinpath(SRC_DIR, "plot_helper.jl"));

datasets = matopen(joinpath(@__DIR__, "Analysis_Timeseries.mat")) do file
    mat_var = read(file, "Analysis_Timeseries")
    mat_var[:,4]
end;

n_data = size(datasets[1], 2)
t_data = 0.0:(n_data-1)

data_idx = 89
data = datasets[data_idx][:,1:end-1] # exclude last point for prediction
final_pt = datasets[data_idx][:,end]
n_species, n_obs = size(data)
t_obs = 0.0:1.0:(n_obs-1);
t_grid = range(0, n_obs, 1000);

# Basic plotting

using DataInterpolations
using RegularizationTools
using StatsBase
d = 2;
smooth_λ = 1.0;

σ_vec = [];
t_itp = range(extrema(t_obs)..., 1000);
f = Figure();
for i in 1:n_species
    ax = Axis(f[div(i+1,2),mod(i-1,2)+1])
    
    u = data[i,:]
    A = RegularizationSmooth(u, t_obs, d; λ = smooth_λ, alg = :fixed);
    û = A.û;
    push!(σ_vec, std(u .- û))

    u_itp = A.(t_itp)
    scatter!(t_obs, u)
    lines!(t_itp, u_itp)
end
display(current_figure())
σ_vec

make_pairs(n_species) = [(i, j) for j in 1:n_species, i in 1:n_species if i !== j];

function make_LV_func(n_species)
    pairs = make_pairs(n_species)
    function LV_func!(dx, x, p, t)
        idx = 1
        # growth
        for i in 1:n_species
            @inbounds dx[i] = p[idx]
            idx += 1
        end
        # self-inhibition
        for i in 1:n_species
            @inbounds dx[i] -= p[idx]*x[i]
            idx += 1
        end
        # positive/negative interactions
        for (i, j) in pairs
            @inbounds dx[i] += (p[idx] - p[idx+1])*x[j]
            idx += 2
        end
        # multiply factor of x[i]
        for i in 1:n_species
            @inbounds dx[i] *= x[i]
        end
    end
    return LV_func!
end

using OrdinaryDiffEq

LV_func! = make_LV_func(n_species);
u0 = zeros(n_species)
tspan = extrema(t_obs)
n_p = 2*n_species^2+n_species
p = zeros(n_p);
prob = ODEProblem(LV_func!, u0, tspan, p);

LB = 1e-4;
UB = 50;
lbs = ones(n_p) .* LB;
ubs = ones(n_p) .* UB;

tf(u) = log(max(u, LB));
itf(u) = exp(u);

function make_loss_func(prob, data, σ_vec)
    n_species = size(data, 1)
    idxs = Tuple.(CartesianIndices(data))
    function loss_func(p)
        remade_prob = remake(prob, p=p, u0=p[end-n_species+1:end]);
        sol = OrdinaryDiffEq.solve(remade_prob; verbose=false, saveat=t_obs); # simulate the ODE
        if !SciMLBase.successful_retcode(sol)
            return Inf
        end
        loss = 0.0
        for (i, j) in idxs
            @inbounds loss += abs2((data[i,j] - sol.u[j][i])/σ_vec[i])
        end
        return loss / 2
    end
    return loss_func
end

loss_func = make_loss_func(prob, data, σ_vec)

λ = 1.0
pen_range = (n_species+1):(2*n_species^2)
penalised_func(θ) = begin
    k = itf.(θ);
    loss_func(k) + λ*sum(log, k)
end

using Optim
using LineSearches
opt_args = (
    Fminbox(BFGS(linesearch = LineSearches.BackTracking())), # optimisation algorithm
    Optim.Options(x_abstol=1e-10, f_abstol=1e-10, outer_x_abstol=1e-10, outer_f_abstol=1e-10) # optimisation options
);

struct FunctionWrapper
    f::Function
end
(wrapper::FunctionWrapper)(arg) = wrapper.f(arg);

exp_pred(t, t0, p) = p[1]*exp(p[2]*(t-t0));
function fit_exp(xs, ts)    
    t0 = ts[1]
    loss_func(p) = sum(abs2.(xs .- exp_pred.(ts, t0, Ref(p))))
    optimize(FunctionWrapper(loss_func), [xs[1], 0.0])
end


start_lbs = [LB.*ones(2*n_species^2); max.(0.01, 0.5.*data[:,1])]
start_ubs = [3.0.*ones(2*n_species); 1.5.*ones(2*n_species^2-2*n_species); max.(0.1, 1.5.*data[:,1])]

rng = StableRNG(1);
start_pts = [];
results = [];
n_starts = 100;
for _ in 1:n_starts
    start_pt = start_lbs .+ (start_ubs .- start_lbs) .* rand(rng, n_p)
    push!(start_pts, start_pt)
end

loss_func(itf.(start_pts[1]))

for start_pt in start_pts
    @time res = optimize(FunctionWrapper(penalised_func), log.(lbs), log.(ubs), log.(start_pt), opt_args...; autodiff = :forward)
    push!(results, res)
end

est_ps = [itf.(res.minimizer) for res in results];
loss_vec = loss_func.(est_ps);

finite_loss_vec = filter(isfinite, loss_vec);
hist(finite_loss_vec)
scatter(sort(finite_loss_vec))

sort(loss_vec)[1:20]

function get_growth_params(p, n)
    return [p[1:n] -p[n+1:2*n]]'
end

function get_interaction_params(p, n)
    pairs = make_pairs(n)
    mat = zeros(n, n)
    for (idx, (i, j)) in enumerate(pairs)
        mat[i,j] = p[2*(n+idx-1)+1]-p[2*(n+idx-1)+2]
    end
    return mat
end

get_growth_params(est_ps[1], n_species)
get_interaction_params(est_ps[1], n_species)

# est_prob = remake(prob, p=est, u0=est[end-n_species+1:end]);
# est_sol = OrdinaryDiffEq.solve(est_prob);

t_ext = range(n_obs / 2, n_obs, 100);
for idx in sortperm(loss_vec)[1:10]
    est_p = est_ps[idx]
    est_prob = remake(prob, p=est_p, u0=est_p[end-n_species+1:end]);
    est_sol = OrdinaryDiffEq.solve(est_prob);
    f = Figure(size=(800, 1000))
    for i in 1:n_species
        ic = mod(i-1,2)+1
        ax = Axis(f[div(i+1,2),(2*ic-1):(2*ic)])
        exp_coefs = fit_exp(data[i,10:end], t_obs[10:end]).minimizer
        lines!(t_ext, exp_pred.(t_ext, t_obs[10], Ref(exp_coefs)), linestyle=:dash)
        lines!(t_grid, [v[i] for v in est_sol(t_grid).u])        
        scatter!(t_obs, data[i,:])    
        scatter!(n_obs, datasets[data_idx][i,end], color=:black)        
    end
    subgl = GridLayout(f[3, :])
    ax = Axis(subgl[1,1], aspect=DataAspect(), yreversed=true)
    heatmap!(
        get_growth_params(est_p, n_species), colormap=:PRGn_5, colorrange=(-2, 2),
        highclip=colorant"rgb(64,0,75)", lowclip=colorant"rgb(0,68,27)"
    )
    ax = Axis(subgl[1,2:3], aspect=DataAspect(), yreversed=true)
    heatmap!(
        get_interaction_params(est_p, n_species), colormap=:PRGn_5, colorrange=(-2, 2),
        highclip=colorant"rgb(64,0,75)", lowclip=colorant"rgb(0,68,27)"
    )
    Colorbar(
        subgl[1,4], limits=(-2, 2), colormap=:PRGn_5, height = Relative(0.9),
        highclip=colorant"rgb(64,0,75)", lowclip=colorant"rgb(0,68,27)"
    )
    resize_to_layout!(f)
    display(current_figure())
end

get_interaction_params(est_ps[sortperm(loss_vec)[3]], n_species)

for idx in sortperm(loss_vec)[1:10]
    ip = get_interaction_params(est_ps[idx], n_species)
    display((maximum(est_ps[idx][1:2*n_species]), maximum(est_ps[idx][pen_range]), sum(abs.(ip).>2e-4)))
end