using StableRNGs
rng = StableRNG(1)

SRC_DIR = joinpath(@__DIR__, "../../src");
include(joinpath(SRC_DIR, "plot_helper.jl"));

n_species = 2
n_obs = 18
t_obs = 0.0:1.0:(n_obs-1)

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

### Toy example

using OrdinaryDiffEq

LV_func! = make_LV_func(2);

u0 = [1.0; 1.0];
tspan = (0.0, n_obs);
p = [2.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, u0...];

# du = similar(u0);
# using BenchmarkTools
# @btime LV_func!(du, u0, p, 0.0);

prob = ODEProblem(LV_func!, u0, tspan, p);
sol = solve(prob);

# t_grid = range(0, n_obs, 1000);
# f = Figure();
# ax = Axis(f[1,1]);
# for i in 1:2
#     lines!(t_grid, [v[i] for v in sol(t_grid).u])
# end
# display(current_figure())

σ = 0.2
tmp_data = reduce(hcat, sol(t_obs).u) .+ 0.2 .* randn(rng, (2, n_obs))
# for row in eachrow(tmp_data)
#     scatter(t_obs, row)
#     display(current_figure())
# end

LB = 1e-10;
UB = 50;
lbs = ones(10) .* LB;
ubs = ones(10) .* UB;

tf(u) = log(max(u, LB));
itf(u) = exp(u);

function make_loss_func(prob, data, σ)
    n_species = size(data, 1)
    function loss_func(p)
        q = copy(p)
        q[2] = -q[2]
        remade_prob = remake(prob, p=q, u0=p[end-n_species+1:end]);
        sol = solve(remade_prob, saveat=t_obs); # simulate the ODE
        if !SciMLBase.successful_retcode(sol)
            return Inf
        end
        # loss = 0.0
        # for (i, j) in idxs
        #     @inbounds loss += abs2(data[i,j] - sol.u[j][i])
        # end
        # return loss / (2*σ^2)
        return sum(abs2, data .- reduce(hcat, sol.u)) / (2*σ^2)
    end
    return loss_func
end

loss_func = make_loss_func(prob, tmp_data, σ)
q = copy(p);q[2] = -q[2];loss_func(q)

using BenchmarkTools
@btime loss_func($q)

using ForwardDiff
du = similar(q)
@btime ForwardDiff.gradient!($du, loss_func, $q)

# λ = 1.0
# penalised_func(θ) = loss_func(itf.(θ)) + λ*sum(log.(itf.(θ[3:8])));

# using Optim
# using LineSearches
# opt_args = (
#     Fminbox(BFGS(linesearch = LineSearches.BackTracking())), # optimisation algorithm
#     Optim.Options(x_abstol=1e-10, f_abstol=1e-10, outer_x_abstol=1e-10, outer_f_abstol=1e-10) # optimisation options
# );

# struct FunctionWrapper
#     f::Function
# end
# (wrapper::FunctionWrapper)(arg) = wrapper.f(arg);

# start_pt = [ones(8).*0.1; tmp_data[1,1]; tmp_data[2,1]]
# res = optimize(FunctionWrapper(penalised_func), log.(lbs), log.(ubs), log.(start_pt), opt_args...; autodiff = :forward)

# est = itf.(res.minimizer);
# loss_func(itf.(res.minimizer))
# est[2] = -est[2];
# res.minimum

# est_prob = remake(prob, p=est, u0=est[end-1:end]);
# est_sol = solve(est_prob);

# t_grid = range(0, n_obs, 1000);
# f = Figure();
# ax = Axis(f[1,1]);
# for i in 1:2
#     lines!(t_grid, [v[i] for v in est_sol(t_grid).u])
#     scatter!(t_obs, tmp_data[i,:])
# end
# display(current_figure())