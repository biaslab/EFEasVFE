using LogExpFunctions
using RxInfer

@rule Ambiguity(:out, Marginalisation) (q_in::Any, meta::JointMarginalMeta,) = begin
    entropies = mapreduce(c -> -conditional_entropy(c), +, meta.components)
    return Categorical(softmax!(entropies, entropies); check_args=false)
end