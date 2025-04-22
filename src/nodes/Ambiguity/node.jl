using RxInfer

export Ambiguity

struct Ambiguity end

@node Ambiguity Stochastic [out, in];

@average_energy Ambiguity (q_out::Any, q_in::Any, meta::Any) = begin
    return 0.0
end