using LogExpFunctions
using RxInfer

function conditional_entropy(marginal::AbstractArray{T,2}, out_dim::Int, in_dim::Int) where {T}
    q_in = sum(marginal, dims=out_dim)
    q_given_in = marginal ./ q_in
    joint_entropies = entropy.(eachslice(q_given_in, dims=in_dim))
    return joint_entropies
end


function conditional_entropy(marginal::AbstractArray{T,N}, out_dim::Int, in_dim::Int) where {T,N}
    sum_dims = setdiff(1:N, in_dim)
    q_in = sum(marginal, dims=sum_dims)
    q_given_in = marginal ./ q_in
    joint_entropies = entropy.(eachslice(q_given_in, dims=in_dim))
    q_marginalized_y_given_in = sum(q_given_in, dims=out_dim)
    marginal_entropies = entropy.(eachslice(q_marginalized_y_given_in, dims=in_dim))
    joint_entropies .-= marginal_entropies
    return joint_entropies
end

function conditional_entropy(jmmc::JointMarginalMetaComponent{C,OutDim,InDim,SumDims}) where {C,OutDim,InDim,SumDims}
    joint_marginal = components(get_marginal(jmmc.jms))
    result = conditional_entropy(joint_marginal, OutDim, InDim)
    return result
end

@rule Exploration(:out, Marginalisation) (q_in::Any, meta::JointMarginalMeta,) = begin
    entropies = mapreduce(conditional_entropy, +, meta.components)
    softmax!(entropies, entropies)
    return Categorical(entropies)
end
