using LogExpFunctions
using RxInfer

@rule Ambiguity(:out, Marginalisation) (q_in::Any, meta::JointMarginalMeta,) = begin
    entropies = mapreduce(c -> -conditional_entropy(c), +, meta.components)
    softmax!(entropies, entropies)
    # @show "Ambiguity message: $entropies"
    return Categorical(entropies)
end