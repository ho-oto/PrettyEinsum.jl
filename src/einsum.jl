struct TensorInformation
    indices::Vector{Tuple{String,Int}}
    rank::Int
    isconj::Bool
end

s = filter(!isspace, s)
nodes = collect(eachmatch(
    r"\[(?<isconj>(|'))(?<label>[0-9]+)\|"*
        r"(?<index>[,_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}]+?)\]",
    s;
    overlap=false
))
edges = collect(eachmatch(
    r"\((?<index1>[_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}]+)@(?<label1>[0-9]+)"*
    r"=(?<index2>[_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}]+)@(?<label2>[0-9]+)\)",
    s;
    overlap=false
))
result = collect(eachmatch(
    r"\[>\|(?<index>[0-9@,_\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}]+?)\]",
    s;
    overlap=false
))
tree = Meta.parse(match(r"\{(?<tree>[,0-9\(\)]+?)\}",s)[:tree])

tensorin = Vector{TensorInformation}(undef,length(nodes))
tensorout = Ref{TensorInformation}()
indexpair = NTuple{2,Tuple{String,Int}}[]

for node in nodes
    i = parse(Int, node[:label])
    inds = split(node[:index], ",")
    inds = [(ind, i) for ind in inds]
    tensorin[i] = TensorInformation(inds, length(inds), node[:isconj] == "'")
end
isassigned(tensorin) || error("duplication of tensor number")

for edge in edges
    push!(indexpair,
        (
            (edge[:index1], parse(Int, edge[:label1])),
            (edge[:index2], parse(Int, edge[:label2]))
        )
    )
end
