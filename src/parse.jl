struct TensorInformation
    indices::Vector{Tuple{Symbol,Int}}
    isconj::Bool
end

function parse_tensors_rhs(s::AbstractString)
    rhs_matched = collect(eachmatch(
        r"\[\s*@??\s*(?<label>[0-9]+)\s*(?<isconj>[\*⋆]??)\s*\|" *
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
        r"\[\s*@??\s*>\s*\|(?<index>[0-9@,_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}\s]+?)\]",
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
