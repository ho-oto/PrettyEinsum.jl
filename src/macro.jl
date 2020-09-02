macro einΣ!(ex::Expr)
    ex.head == :string && isodd(length(ex.args)) || error("")
    s = join(ex.args[1:2:end])
    tensors_rhs = parse_tensors_rhs(s)
    indices_lhs = parse_indices_lhs(s)
    pairs_index = parse_pairs_index(s)
    tree_contract = parse_tree_contract(s)
    for ((kind, klab), v) in pairs_index
        replace!(tensors_rhs[klab].indices, (kind, klab) => v)
    end
    indices_all = union([x.indices for x in tensors_rhs]...)
    indices_sym = Dict(x => gensym() for x in indices_all)

    tensors_rhs_sym = Vector{Symbol}(undef, length(tensors_rhs))
    tensor_lhs_sym = Ref{Symbol}()
    for (i, (str, sym)) in enumerate(zip(ex.args[3:2:end], ex.args[2:2:end]))
        label = match(r"\s*@??\s*(?<label>[0-9>]+)", str)[:label]
        if label ≠ ">"
            tensors_rhs_sym[parse(Int, label)] = sym
        else
            tensor_lhs_sym[] = sym
        end
    end

    tensors_rhs_ex = Vector{Expr}()
    for (i, a) in enumerate(tensors_rhs)
        ex = Expr(:ref, esc(tensors_rhs_sym[i]), [indices_sym[i] for i in a.indices]...)
        ex = a.isconj ? Expr(:call, :conj, ex) : ex
        push!(tensors_rhs_ex, ex)
    end

    ex_rhs_lhs = Expr(:ref, esc(tensor_lhs_sym[]), [indices_sym[i] for i in indices_lhs]...)
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

    ex_rhs =
        macroexpand(TensorOperations, :(TensorOperations.@tensor $ex_rhs_lhs = $ex_rhs_rhs))
    return :($ex_rhs)
end

macro einΣ(ex::Expr)
    ex.head == :string && isodd(length(ex.args)) || error("")
    s = join(ex.args[1:2:end])
    tensors_rhs = parse_tensors_rhs(s)
    indices_lhs = parse_indices_lhs(s)
    pairs_index = parse_pairs_index(s)
    tree_contract = parse_tree_contract(s)
    for ((kind, klab), v) in pairs_index
        replace!(tensors_rhs[klab].indices, (kind, klab) => v)
    end
    indices_all = union([x.indices for x in tensors_rhs]...)
    indices_sym = Dict(x => gensym() for x in indices_all)

    tensors_rhs_sym = Vector{Symbol}(undef, length(tensors_rhs))
    tensor_lhs_sym = gensym()
    for (i, (str, sym)) in enumerate(zip(ex.args[3:2:end], ex.args[2:2:end]))
        label = match(r"\s*@??\s*(?<label>[0-9]+)", str)[:label]
        tensors_rhs_sym[parse(Int, label)] = sym
    end

    tensors_rhs_ex = Vector{Expr}()
    for (i, a) in enumerate(tensors_rhs)
        ex = Expr(:ref, esc(tensors_rhs_sym[i]), [indices_sym[i] for i in a.indices]...)
        ex = a.isconj ? Expr(:call, :conj, ex) : ex
        push!(tensors_rhs_ex, ex)
    end

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

    ex_rhs =
        macroexpand(TensorOperations, :(TensorOperations.@tensor $ex_rhs_lhs := $ex_rhs_rhs))
    return :($ex_rhs)
end
