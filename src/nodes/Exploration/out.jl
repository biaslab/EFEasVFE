using LogExpFunctions
using RxInfer

function conditional_entropy(jmmc::JointMarginalMetaComponent{C,OutDim,InDim,SumDims}) where {C,OutDim,InDim,SumDims}
    joint_marginal = components(get_marginal(jmmc.jms))

    # Use the precomputed sumdims for summation
    q_u_out = dropdims(sum(joint_marginal, dims=jmmc.sumdims), dims=jmmc.sumdims) # q(out, u) is now a matrix
    q_u = sum(q_u_out, dims=1) # q(u) is now a vector
    q_u_out ./= q_u # q(out | u) is now a matrix

    entropies = entropy.(eachslice(q_u_out, dims=2))
    return entropies
end

@rule Exploration(:out, Marginalisation) (q_in::Any, meta::JointMarginalMeta,) = begin
    entropies = mapreduce(conditional_entropy, +, meta.components)
    return Categorical(softmax!(entropies, entropies); check_args=false)
end
