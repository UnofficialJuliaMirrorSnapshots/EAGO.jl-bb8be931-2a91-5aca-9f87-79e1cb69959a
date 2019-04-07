#

mutable struct Pattern
    Center::Int
    Vertices::Vector{Symbol}
    Meta::Vector{Symbol}
    Edges::Tuple{Int,Int}
end

function LexiographicOrder(x::Pattern)
end

function FillMetaData()
end

function AdjMatch(subs,pattern,sto)
end

function OpsMatch(subs,pattern,sto)
end

function MetaMatch(subs,pattern,sto)
end

function Templates_Match(subs,pattern,sto)
    AdjMatch(subs,pattern,sto) && OpsMatch(subs,pattern,sto) && MetaMatch(subs,pattern,sto)
end
