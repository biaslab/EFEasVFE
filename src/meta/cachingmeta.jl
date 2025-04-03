using Tullio
using TensorOperations
using RxInfer

struct CachingMeta
    cache::Dict{Symbol,AbstractArray}

    CachingMeta() = new(Dict{Symbol,AbstractArray}())
end

# Implement basic dictionary interface
Base.getindex(meta::CachingMeta, key::Symbol) = getindex(meta.cache, key)
Base.haskey(meta::CachingMeta, key::Symbol) = haskey(meta.cache, key)
Base.setindex!(meta::CachingMeta, value::AbstractArray, key::Symbol) = setindex!(meta.cache, value, key)

# Reset function to clear cache
function reset_cache!(meta::CachingMeta)
    empty!(meta.cache)
end

reset_cache!(node) = reset_cache!(node, node.properties.fform)
reset_cache!(node, fform) = nothing
reset_cache!(node, fform::Type{DiscreteTransition}) = reset_cache!(node, fform, RxInfer.GraphPPL.getextra(node, :meta, nothing))

reset_cache!(node, fform, any) = nothing

function reset_cache!(node, fform::Type{DiscreteTransition}, meta::CachingMeta)
    q_out = nothing
    if haskey(meta, :out)
        q_out = meta[:out]
    end
    reset_cache!(meta)
    if !isnothing(q_out)
        meta[:out] = q_out
    end
end

function after_iteration_callback(model, iteration)
    gppl = model.model
    for node in RxInfer.GraphPPL.factor_nodes(gppl)
        reset_cache!(gppl[node])
    end
end
