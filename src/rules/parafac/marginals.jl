struct CPTensorMarginal{T,N}
    cp::CPTensor{T,N}
    incoming_msg::NTuple{N,Vector{T}}
end

using RxInfer

# Rules for observation model (q_out is pointmass)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    return CPTensorMarginal(q_a.point, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4)))
end

# Rules for observation model 
@marginalrule DiscreteTransition(:in_T1_T2_T3_T4) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    return CPTensorMarginal(q_a.point, (probvec(q_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4)))
end

# Rules for transition model (7 interfaces)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4_T5) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    return CPTensorMarginal(q_a.point, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(m_T5)))
end

# Rules for transition model 
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, q_T5::PointMass, meta::Any) = begin
    return CPTensorMarginal(q_a.point, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(q_T5)))
end

RxInfer.ReactiveMP.sdtype(any::RxInfer.ReactiveMP.StandaloneDistributionNode) = ReactiveMP.Stochastic()
RxInfer.BayesBase.entropy(cp::CPTensorMarginal) = 0.0

@average_energy ReactiveMP.DiscreteTransition (q_out::PointMass{Vector{T}}, q_in_T1_T2_T3_T4::CPTensorMarginal{T,6}, q_a::PointMass{CPTensor{T,6}},) where {T} = begin
    return -complete_mode_product(q_a.point, q_in_T1_T2_T3_T4.incoming_msg)
end


@average_energy ReactiveMP.DiscreteTransition (q_out_in_T1_T2_T3_T4::CPTensorMarginal{T,6}, q_a::PointMass{CPTensor{T,6}},) where {T} = begin
    return -complete_mode_product(q_a.point, q_out_in_T1_T2_T3_T4.incoming_msg)
end

@average_energy ReactiveMP.DiscreteTransition (q_out_in_T1_T2_T3_T4_T5::CPTensorMarginal{T,7}, q_a::PointMass{CPTensor{T,7}},) where {T} = begin
    return -complete_mode_product(q_a.point, q_out_in_T1_T2_T3_T4_T5.incoming_msg)
end

@average_energy ReactiveMP.DiscreteTransition (q_out_in_T1_T2_T3_T4::CPTensorMarginal{T,7}, q_a::PointMass{CPTensor{T,7}}, q_T5::PointMass{Vector{T}},) where {T} = begin
    return -complete_mode_product(q_a.point, q_out_in_T1_T2_T3_T4.incoming_msg)
end