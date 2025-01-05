################################################################################################
### Define ground truth reaction network and the reaction library network used for inference ###
################################################################################################

using Catalyst

### Define full CRN (i.e. library used for inference)

t = default_t(); # time variable
@species X1(t) X2(t) X3(t);
complexes_vec = [[X1], [X2], [X3], [X1, X2], [X2, X3], [X1, X3]]; # all possible complexes

# All possible reactions between ordered pairs of complexes
rct_prd_pairs = [
	(reactants, products) for reactants in complexes_vec for products in complexes_vec 
	if reactants !== products
]; # reactants-products pair for each reaction
n_rx = length(rct_prd_pairs); # number of reactions
@parameters k[1:n_rx] # reaction rate constants
rx_vec = [
	Reaction(kval, reactants, products) for ((reactants, products), kval) in zip(rct_prd_pairs, k)
];

# CRN
@named full_network = ReactionSystem(rx_vec, t)
full_network = complete(full_network)

# Export reaction list (uncomment to run)
# function export_reaction_vec(rx_vec, filename=joinpath(@__DIR__, "reactions.txt"))
# 	out_file = open(filename, "w");
# 	redirect_stdout(out_file) do
# 		println.(rx_vec);
# 	end;
# 	close(out_file)
# end


### Define true CRN (used for data simulation and inference evaluation)

# Indices of reaction rate constants follow the reaction indices in the full network above
true_rn = @reaction_network begin
	k1, X1 --> X2
	(k18, k13), X1 + X2 <--> X3	
end
true_rn = complete(true_rn);