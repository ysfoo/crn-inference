using Catalyst

### Define full CRN using Catalyst.jl

t = default_t(); # time variable
@species X1(t) X2(t) X3(t);
complexes_vec = [[X1], [X2], [X3], [X1, X2], [X2, X3], [X1, X3]];

# Reactions
rct_prd_pairs = [
	(reactants, products) for reactants in complexes_vec for products in complexes_vec 
	if reactants !== products
]; # reactants-products pair for each reaction
n_rx = length(rct_prd_pairs); # number of reactions
@parameters k[1:n_rx] # reaction rate constants
rx_vec = [
	Reaction(kval, reactants, products) for ((reactants, products), kval) in zip(rct_prd_pairs, k)
];

# Full CRN
@named full_network = ReactionSystem(rx_vec, t)
full_network = complete(full_network)

### Export reaction list (uncomment to run)
# out_file = open(joinpath(@__DIR__, "output/reactions.txt"), "w");
# redirect_stdout(out_file) do
#     println.(rx_vec);
# end;
# close(out_file)