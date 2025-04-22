using RxInfer

export Exploration

struct Exploration end

@node Exploration Stochastic [out, in]

@average_energy Exploration (q_out::Any, q_in::Any, meta::Any) = begin
    return 0.0
end