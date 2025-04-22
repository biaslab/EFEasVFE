using RxInfer

export OneWay

struct OneWay end

@node OneWay Deterministic [out, in]

@rule OneWay(:out, Marginalisation) (m_in::Any,) = m_in

@rule OneWay(:in, Marginalisation) (m_out::Any,) = Uninformative()

@marginalrule OneWay(:out_in) (m_out::Any, m_in::Any) = begin
    return (out=m_in, in=m_in)
end

@average_energy OneWay (q_out_in::Any,) = begin
    return 0.0
end