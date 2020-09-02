macro peinp!_str(s::AbstractString)
    tensor_lhs_sym, tensors_rhs_sym, ex_rhs_lhs, ex_rhs_rhs = parser_strmac(s)
    ex_rhs =
        macroexpand(TensorOperations, :(TensorOperations.@tensor $ex_rhs_lhs = $ex_rhs_rhs))
    ex_lhs = Expr(:tuple, tensor_lhs_sym, tensors_rhs_sym...)
    return :($ex_lhs -> $ex_rhs)
end

macro peinew_str(s::AbstractString)
    tensor_lhs_sym, tensors_rhs_sym, ex_rhs_lhs, ex_rhs_rhs = parser_strmac(s)
    ex_rhs = macroexpand(
        TensorOperations,
        :(TensorOperations.@tensor $ex_rhs_lhs := $ex_rhs_rhs),
    )
    ex_lhs = Expr(:tuple, tensors_rhs_sym...)
    return :($ex_lhs -> $ex_rhs)
end

function parser_strmac(s::AbstractString)
    tensors_rhs = parse_tensors_rhs(s)
    indices_lhs = parse_indices_lhs(s)
    pairs_index = parse_pairs_index(s)
    tree_contract = parse_tree_contract(s)
    for ((kind, klab), v) in pairs_index
        replace!(tensors_rhs[klab].indices, (kind, klab) => v)
    end
    indices_all = union([x.indices for x in tensors_rhs]...)
    indices_sym = Dict(x => gensym() for x in indices_all)
    tensors_rhs_sym = [gensym() for _ = 1:length(tensors_rhs)]
    tensors_rhs_ex = Vector{Expr}()
    for (i, a) in enumerate(tensors_rhs)
        ex = Expr(:ref, tensors_rhs_sym[i], [indices_sym[i] for i in a.indices]...)
        ex = a.isconj ? Expr(:call, :conj, ex) : ex
        push!(tensors_rhs_ex, ex)
    end
    tensor_lhs_sym = gensym()
    ex_rhs_lhs = Expr(:ref, tensor_lhs_sym, [indices_sym[i] for i in indices_lhs]...)
    ex_rhs_rhs = if isnothing(tree_contract)
        Expr(:call, :(*), tensors_rhs_ex...)
    else
        evaltree(i, j) = Expr(
            :call,
            :(*),
            i isa Int ? tensors_rhs_ex[i] : evaltree(i...),
            j isa Int ? tensors_rhs_ex[j] : evaltree(j...),
        )
        evaltree(i, j, k...) = evaltree((i, j), k...)
        evaltree(tree_contract...)
    end
    return tensor_lhs_sym, tensors_rhs_sym, ex_rhs_lhs, ex_rhs_rhs
end