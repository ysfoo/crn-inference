using QuasiMonteCarlo
using LinearAlgebra
# BLAS.set_num_threads(1);

# Try to make multi-threading more efficient
# using ThreadPinning
# isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
# isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

using ProgressMeter
using StableRNGs
using Format
FMT_3DP = ".3f";

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem

@parameters gold_k[1:n_gold_rx]

gold_rx_vec = [
	begin
        rct, rct_stoich = remove_pentane(rct_tuple);
        prd, prd_stoich = remove_pentane(prd_tuple);
        Reaction(kval, rct, prd, rct_stoich, prd_stoich) 
    end for ((rct_tuple, prd_tuple), kval) in zip(rxs_no_k[gold_idxs], gold_k)
];
@named gold_network = ReactionSystem(gold_rx_vec, t, species_vec, gold_k; combinatoric_ratelaws=false)
gold_network = complete(gold_network)

gold_oprob = ODEProblem(gold_network, x0_map, t_span, zeros(n_gold_rx));
loss_func = make_loss_func(
    gold_oprob, gold_k, t_obs, data;
    alg=AutoVern7(KenCarp4()), 
    abstol=1e-10, verbose=false
)
optim_func = let gold_k=gold_k, loss_func=loss_func, n_gold_rx=n_gold_rx, n_species=n_species, scale_fcts=scale_fcts[gold_idxs]
    function tmp_func(u::Vector{T})::T where T
        kvec = exp.(u[1:n_gold_rx]) .* scale_fcts
        σs = exp.(u[end-n_species+1:end])
        # σs = clamp.(exp.(u[end-n_species+1:end]), σ_lbs, σ_ubs)
        loss_func(kvec, σs)
    end    
end;
refine_func = let gold_k=gold_k, loss_func=loss_func, n_gold_rx=n_gold_rx, n_species=n_species, scale_fcts=scale_fcts[gold_idxs], σ_lbs=σ_lbs, σ_ubs=σ_ubs
    function tmp_func(u::Vector{T})::T where T
        kvec = exp.(u[1:n_gold_rx]) .* scale_fcts
        # σs = exp.(u[end-n_species+1:end])
        σs = clamp.(exp.(u[end-n_species+1:end]), σ_lbs, σ_ubs)
        loss_func(kvec, σs)
    end    
end;

gold_est = if !(@isdefined READ_GOLD)
    using StableRNGs
    res_vec = Vector{Optim.MultivariateOptimizationResults}([]);
    @showprogress for seed in 1:64
        rng = StableRNG(seed);
        init_pt = log.([rand(rng, n_gold_rx); strict_clamp.(init_σs, σ_lbs, σ_ubs)]);
        gold_lbs, gold_ubs =  fill(log(1e-5), length(init_pt)), fill(log(1e2), length(init_pt));
        gold_res = optimize(
            FunctionWrapper(optim_func), 
            gold_lbs, gold_ubs, init_pt,
            Fminbox(BFGS(linesearch=LineSearches.BackTracking())),
            Optim.Options(x_abstol=1e-10, f_abstol=1e-10, outer_x_abstol=1e-10, outer_f_abstol=1e-10);
            autodiff = :forward
        )
        refine_res = optimize(
            FunctionWrapper(refine_func), 
            gold_res.minimizer,
            BFGS(),
            Optim.Options(iterations=10^4, time_limit=10.);
            autodiff = :forward
        )
        push!(res_vec, refine_res)
    end
    res = argmin((x)->x.minimum, res_vec)
    fname = joinpath(@__DIR__, "output", "gold_estimates.txt")
    est = res.minimizer
    est[end-n_species+1:end] .= clamp.(est[end-n_species+1:end],  log.(σ_lbs), log.(σ_ubs))
    writedlm(fname, est);
    res.minimizer
else
    fname = joinpath(@__DIR__, "output", "gold_estimates.txt")
    vec(readdlm(fname))
end;

gold_logp = -refine_func(gold_est)
gold_kvec = exp.(gold_est[1:n_gold_rx]) .* scale_fcts[gold_idxs]
gold_σs = exp.(gold_est[end-n_species+1:end])
gold_remade = remake(gold_oprob, p=[gold_k=>gold_kvec]);
gold_osol = solve(gold_remade, saveat=t_grid);