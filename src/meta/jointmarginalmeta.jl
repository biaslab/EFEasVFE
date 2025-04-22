mutable struct JointMarginalStorage{C}
    joint_marginal::C
end

get_marginal(jms::JointMarginalStorage) = jms.joint_marginal

function set_marginal!(jms::JointMarginalStorage{C}, marginal::C) where {C}
    jms.joint_marginal = marginal
    return jms
end

struct JointMarginalMetaComponent{C,OutDim,InDim,SumDims}
    jms::JointMarginalStorage{C}
    out_dim::Val{OutDim}
    in_dim::Val{InDim}
    sumdims::SumDims
end

function JointMarginalMetaComponent(jms::JointMarginalStorage{C}, out_dim::Int, in_dim::Int) where {C}
    joint_marginal = components(get_marginal(jms))
    dims = ntuple(identity, ndims(joint_marginal))
    sumdims = ntuple(i -> i ∉ (out_dim, in_dim) ? i : nothing, ndims(joint_marginal))
    sumdims = Tuple(filter(x -> x !== nothing, collect(sumdims)))

    return JointMarginalMetaComponent(jms, Val(out_dim), Val(in_dim), sumdims)
end

struct JointMarginalMeta
    components::Vector{<:JointMarginalMetaComponent}
end


getcomponents(meta::JointMarginalMeta) = meta.components