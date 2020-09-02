module PrettyEinsum
using TensorOperations

export @peinp!_str, @peinew_str, @peinp!#, @peinew

include("einsum.jl")

end
