#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter
using LogExpFunctions, SpecialFunctions
using StatsBase

using StableRNGs
using ForwardDiff

using Format
FMT_3DP = ".3f";

include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "../../src/plot_helper.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));
include(joinpath(@__DIR__, "setup.jl"));

smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k);

pen_names = Dict(
    "L1" => "\\mathbf{L_1}", 
    "logL1" => "\\textbf{log } \\mathbf{L_1}", 
    "approxL0" => "\\textbf{Approximate }\\mathbf{L_0}", 
    "hslike" => "\\textbf{Horseshoe}"
);

alg = AutoVern7(KenCarp4());
abstol = eps(Float64);
glob_loss_func = make_loss_func(oprob, k, t_obs, data; alg, abstol);
glob_optim_func = let n_species=n_species, scale_fcts=scale_fcts, loss_func=glob_loss_func
    FunctionWrapper(u -> begin 		
		k_unscaled = exp.(u[1:end-n_species]); 
		σs = exp.(u[end-n_species+1:end])
		loss_func(k_unscaled .* scale_fcts, σs)
	end);
end;
glob_cfg = ForwardDiff.HessianConfig(glob_optim_func, zeros(n_rx+n_species));
glob_setter = setp(oprob, k);

function calc_hess_eigvals(t_obs, data, oprob, setter, isol, idxs,
                           loss_func, optim_func, cfg)
    n_rx = length(isol.kvec)
    n_species = size(data, 1)
    refine_func = make_refine_func(loss_func, isol, idxs);
    res = optimize(
        refine_func, 
        log.([fill(1e-10, length(idxs)); fill(1e-5, n_species)]),
        log.([fill(1e2, length(idxs)); fill(1e-1, n_species)]),
        [isol.est[idxs];isol.est[end-n_species+1:end]],
        Fminbox(BFGS()); autodiff = :forward
    )

    masked = fill(-Inf, n_rx+n_species)
    masked[idxs] .= res.minimizer[1:length(idxs)]
    # masked[end-n_species+1:end] .= log.(est_σs_given_kvec(
    #     t_obs, data, oprob, setter, mask_kvec(isol, idxs); alg, kwargs...
    # ))
    masked[end-n_species+1:end] .= res.minimizer[end-n_species+1:end]
    hess = ForwardDiff.hessian(optim_func, masked, cfg);
    res, hess
end

idxs = sort_by_w[1].first
@time res, hess = calc_hess_eigvals(
    t_obs, data, oprob, glob_setter, isol_dict_lookup[pen_str][idxs], idxs,
    glob_loss_func, glob_optim_func, glob_cfg
);

[isol_dict_lookup[pen_str][idxs].θ[idxs] res.minimizer[1:length(idxs)]]
[isol_dict_lookup[pen_str][idxs].θ[end-n_species+1] res.minimizer[end-n_species+1]]

res
eigvals(hess)

eigvals(hess)[n_rx-length(idxs)+1:end]

tmp_isol = isol_dict_lookup[pen_str][idxs];
est_σs = est_σs_given_kvec(
    t_obs, data, oprob, glob_setter, mask_kvec(tmp_isol, idxs); alg, abstol
)
g = ForwardDiff.gradient((s)->glob_loss_func(convert(typeof(s), mask_kvec(tmp_isol, idxs)),s), est_σs)

glob_setter(oprob, mask_kvec(tmp_isol, idxs));
sol = solve(oprob, alg; saveat=t_obs, abstol);

std.(eachrow(sol[1:n_species,:] .- data); corrected=false, mean=0.) == est_σs

sqrt.(mean.(eachrow(abs2.(sol[1:n_species,:] .- data)))) == est_σs

glob_loss_func(mask_kvec(tmp_isol, idxs), est_σs) - glob_loss_func(mask_kvec(tmp_isol, idxs), est_σs .- 1e-14*g)
new_loss_func(mask_kvec(tmp_isol, idxs), est_σs) - new_loss_func(mask_kvec(tmp_isol, idxs), est_σs .- 1e-14*g)


@time glob_loss_func(mask_kvec(tmp_isol, idxs), est_σs)
new_loss_func = make_loss_func(oprob, k, t_obs, data; alg, abstol);
@time new_loss_func(mask_kvec(tmp_isol, idxs), est_σs)