struct TensorInformation
    indices::Vector{Tuple{Symbol,Int}}
    isconj::Bool
end

macro peinp!_str(s::AbstractString)
    tensor_lhs_sym, tensors_rhs_sym, ex_rhs_lhs, ex_rhs_rhs = parser(s)
    ex_rhs = macroexpand(
        TensorOperations,
        :(TensorOperations.@tensor $ex_rhs_lhs = $ex_rhs_rhs),
    )
    ex_lhs = Expr(:tuple, tensor_lhs_sym, tensors_rhs_sym...)
    return :($ex_lhs -> $ex_rhs)
end

macro peinew_str(s::AbstractString)
    tensor_lhs_sym, tensors_rhs_sym, ex_rhs_lhs, ex_rhs_rhs = parser(s)
    ex_rhs = macroexpand(
        TensorOperations,
        :(TensorOperations.@tensor $ex_rhs_lhs := $ex_rhs_rhs),
    )
    ex_lhs = Expr(:tuple, tensors_rhs_sym...)
    return :($ex_lhs -> $ex_rhs)
end

function parser(s::AbstractString)
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

function parse_tensors_rhs(s::AbstractString)
    rhs_matched = collect(eachmatch(
        r"\[\s*(?<label>[0-9]+)\s*(?<isconj>[\*⋆]??)\s*\|" *
        r"(?<index>[,_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}\s]+?)\]",
        s;
        overlap = false,
    ))
    isempty(rhs_matched) && error("no argument tensors")
    tensors_rhs = Vector{TensorInformation}(undef, length(rhs_matched))
    for x in rhs_matched
        label = parse(Int, x[:label])
        isconj = !isempty(x[:isconj])
        indices = Vector{Tuple{Symbol,Int}}()
        for n in split(x[:index], ",")
            sym = Meta.parse(n)
            sym isa Symbol || error("invalid index syntax @$(label)")
            push!(indices, (sym, label))
        end
        tensors_rhs[label] = TensorInformation(indices, isconj)
    end
    isassigned(tensors_rhs) || error("labels of tensors should be in 1:(#tensor)")
    return tensors_rhs
end

function parse_indices_lhs(s::AbstractString)
    lhs_matched = collect(eachmatch(
        r"\[\s*>\s*\|(?<index>[0-9@,_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}\s]+?)\]",
        s;
        overlap = false,
    ))
    isempty(lhs_matched) && error("no definition of return tensor")
    length(lhs_matched) > 1 && error("duplicate definition of return tensor")
    lhs_matched = first(lhs_matched)
    indices_lhs = Vector{Tuple{Symbol,Int}}()
    for n in split(lhs_matched[:index], ",")
        index, label = split(n, "@")
        sym = Meta.parse(index)
        sym isa Symbol || error("invalud index syntax")
        label = parse(Int, label)
        push!(indices_lhs, (sym, label))
    end
    return indices_lhs
end

function parse_pairs_index(s::AbstractString)
    pairs_matched = collect(eachmatch(
        r"\(\s*(?<kind>[_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}]+)\s*@\s*(?<klab>[0-9]+)\s*" *
        r"\s*=\s*(?<vind>[_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}\s]+)\s*@\s*(?<vlab>[0-9]+)\s*\)",
        s;
        overlap = false,
    ))
    pairs_index = Dict{Tuple{Symbol,Int},Tuple{Symbol,Int}}()
    for bond in pairs_matched
        kind = Meta.parse(bond[:kind])
        klab = parse(Int, bond[:klab])
        kind isa Symbol || error("invalid index syntax")
        vind = Meta.parse(bond[:vind])
        vlab = parse(Int, bond[:vlab])
        vind isa Symbol || error("invalid index syntax")
        pairs_index[(kind, klab)] = (vind, vlab)
    end
    length(union(keys(pairs_index), values(pairs_index))) == 2 * length(pairs_matched) ||
        error("duplication in index pairs")
    return pairs_index
end

function parse_tree_contract(s::AbstractString)
    t = collect(eachmatch(r"\[(?<tree>[0-9\s,\(\)]+?)\]", s; overlap = false))
    isempty(t) && return nothing
    length(t) > 1 && error("duplicate definition of contraction tree")
    return Meta.eval(Meta.parse(t[1][:tree]))
end
