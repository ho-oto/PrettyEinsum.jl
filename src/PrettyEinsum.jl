module PrettyEinsum
using TensorOperations

export @peinp!_str, @peinew_str, @peinp!#, @peinew

include("parse.jl")
include("macrostr.jl")
include("macro.jl")

end
