include(joinpath(@__DIR__, "inference.jl")); # imports key functions
include(joinpath(@__DIR__, "plot_helper.jl"));

xs = 10 .^ range(-10, 2, 1001);
L1_vals = L1_pen.(xs, 20.) .- L1_pen(1e-10, 20.);
logL1_vals = logL1_pen.(xs, 1.) .- logL1_pen(1e-10, 1.);
approxL0_vals = approxL0_pen.(xs, log(303)) .- approxL0_pen(1e-10, log(303));
hslike_vals = hslike_pen.(xs, 20.) .- hslike_pen(1e-10, 20.);

f = Figure();
ax = Axis(
    f[1,1], xlabel=L"Parameter value $\theta$", ylabel=L"\text{pen}(\theta) - \text{pen}(10^{-10})",
    xscale=log10, xticks=LogTicks(LinearTicks(6))
);
lines!(xs, L1_vals, label=L"L_1");
lines!(xs, logL1_vals, label=L"log scale $L_1$");
lines!(xs, approxL0_vals, label=L"approximate $L_0$");
lines!(xs, hslike_vals, label="horseshoe-like");
xlims!(1e-10, 1e2);
ylims!(-1., 26);
axislegend(position=:lt);
f
save(joinpath(@__DIR__, "output/penalty_funcs.png"), f)
