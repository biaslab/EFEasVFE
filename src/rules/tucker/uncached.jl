using TensorOperations

# Rules for observation model (q_out is pointmass)
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(q_out)      # out dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension

    @tensor reduced[in] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[2] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    vout = tucker_tensor.factors[1]' * probvec(q_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension

    @tensor reduced[t1] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[3] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    vout = tucker_tensor.factors[1]' * probvec(q_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension

    @tensor reduced[t2] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[4] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    vout = tucker_tensor.factors[1]' * probvec(q_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension

    @tensor reduced[t3] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[5] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    vout = tucker_tensor.factors[1]' * probvec(q_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension

    @tensor reduced[t4] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[6] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    vout = tucker_tensor.factors[1]' * probvec(q_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension

    @tensor reduced[t5] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4]

    # Project back to original space
    result = tucker_tensor.factors[7] * reduced
    return Categorical(normalize!(result, 1))
end



# Rules for transition model (8 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    vt6 = tucker_tensor.factors[8]' * probvec(m_T6)      # t6 dimension
    @tensor reduced[out] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5] * vt6[t6]

    # Project back to original space
    result = tucker_tensor.factors[1] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    vt6 = tucker_tensor.factors[8]' * probvec(m_T6)      # t6 dimension
    @tensor reduced[in] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vout[out] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5] * vt6[t6]

    # Project back to original space
    result = tucker_tensor.factors[2] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    vt6 = tucker_tensor.factors[8]' * probvec(m_T6)      # t6 dimension
    @tensor reduced[t1] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vout[out] * vin[in] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5] * vt6[t6]

    # Project back to original space
    result = tucker_tensor.factors[3] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    vt6 = tucker_tensor.factors[8]' * probvec(m_T6)      # t6 dimension
    @tensor reduced[t2] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vout[out] * vin[in] * vt1[t1] * vt3[t3] * vt4[t4] * vt5[t5] * vt6[t6]

    # Project back to original space
    result = tucker_tensor.factors[4] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    vt6 = tucker_tensor.factors[8]' * probvec(m_T6)      # t6 dimension
    @tensor reduced[t3] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt4[t4] * vt5[t5] * vt6[t6]

    # Project back to original space
    result = tucker_tensor.factors[5] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    vt6 = tucker_tensor.factors[8]' * probvec(m_T6)      # t6 dimension
    @tensor reduced[t4] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt5[t5] * vt6[t6]

    # Project back to original space
    result = tucker_tensor.factors[6] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T6::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt6 = tucker_tensor.factors[8]' * probvec(m_T6)      # t6 dimension
    @tensor reduced[t5] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4] * vt6[t6]

    # Project back to original space
    result = tucker_tensor.factors[7] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T6, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    @tensor reduced[t6] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5, t6] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[8] * reduced
    return Categorical(normalize!(result, 1))
end


# Rules for transition model (8 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, q_T6::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:out, Marginalisation) (m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a, m_T6=Categorical(probvec(q_T6)), meta=meta)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, q_T6::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:in, Marginalisation) (m_out=m_out, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a, m_T6=Categorical(probvec(q_T6)), meta=meta)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, q_T6::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T1, Marginalisation) (m_out=m_out, m_in=m_in, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a, m_T6=Categorical(probvec(q_T6)), meta=meta)
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, q_T6::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T2, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a, m_T6=Categorical(probvec(q_T6)), meta=meta)
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, q_T6::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T3, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T4=m_T4, m_T5=m_T5, q_a=q_a, m_T6=Categorical(probvec(q_T6)), meta=meta)
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, q_T6::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T4, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T5=m_T5, q_a=q_a, m_T6=Categorical(probvec(q_T6)), meta=meta)
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, q_T6::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T5, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, q_a=q_a, m_T6=Categorical(probvec(q_T6)), meta=meta)
end

# Rules for transition model (6 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # out dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension

    @tensor reduced[out] := tucker_tensor.core[out, in, t1, t2, t3, t4] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4]

    # Project back to original space
    result = tucker_tensor.factors[1] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension

    @tensor reduced[in] := tucker_tensor.core[out, in, t1, t2, t3, t4] * vout[out] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4]

    # Project back to original space
    result = tucker_tensor.factors[2] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension

    @tensor reduced[t1] := tucker_tensor.core[out, in, t1, t2, t3, t4] * vout[out] * vin[in] * vt2[t2] * vt3[t3] * vt4[t4]

    # Project back to original space
    result = tucker_tensor.factors[3] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension

    @tensor reduced[t2] := tucker_tensor.core[out, in, t1, t2, t3, t4] * vout[out] * vin[in] * vt1[t1] * vt3[t3] * vt4[t4]

    # Project back to original space
    result = tucker_tensor.factors[4] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension

    @tensor reduced[t3] := tucker_tensor.core[out, in, t1, t2, t3, t4] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt4[t4]

    # Project back to original space
    result = tucker_tensor.factors[5] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension

    @tensor reduced[t4] := tucker_tensor.core[out, in, t1, t2, t3, t4] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3]

    # Project back to original space
    result = tucker_tensor.factors[6] * reduced
    return Categorical(normalize!(result, 1))
end

# Rules for transition model (6 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:TuckerTensor}, q_T4::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:out, Marginalisation) (m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, q_a=q_a, m_T4=Categorical(probvec(q_T4)), meta=meta)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:TuckerTensor}, q_T4::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:in, Marginalisation) (m_out=m_out, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, q_a=q_a, m_T4=Categorical(probvec(q_T4)), meta=meta)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:TuckerTensor}, q_T4::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T1, Marginalisation) (m_out=m_out, m_in=m_in, m_T2=m_T2, m_T3=m_T3, q_a=q_a, m_T4=Categorical(probvec(q_T4)), meta=meta)
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, q_a::PointMass{<:TuckerTensor}, q_T4::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T2, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T3=m_T3, q_a=q_a, m_T4=Categorical(probvec(q_T4)), meta=meta)
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, q_a::PointMass{<:TuckerTensor}, q_T4::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T3, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, q_a=q_a, m_T4=Categorical(probvec(q_T4)), meta=meta)
end

# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # out dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension

    @tensor reduced[out] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[1] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    @tensor reduced[in] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[2] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    @tensor reduced[t1] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt2[t2] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[3] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    @tensor reduced[t2] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt3[t3] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[4] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    @tensor reduced[t3] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt4[t4] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[5] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt5 = tucker_tensor.factors[7]' * probvec(m_T5)      # t5 dimension
    @tensor reduced[t4] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt5[t5]

    # Project back to original space
    result = tucker_tensor.factors[6] * reduced
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, meta::Any) = begin
    tucker_tensor = q_a.point
    # Multiply factor matrices with input vectors
    vout = tucker_tensor.factors[1]' * probvec(m_out)      # out dimension
    vin = tucker_tensor.factors[2]' * probvec(m_in)      # in dimension
    vt1 = tucker_tensor.factors[3]' * probvec(m_T1)      # t1 dimension
    vt2 = tucker_tensor.factors[4]' * probvec(m_T2)      # t2 dimension
    vt3 = tucker_tensor.factors[5]' * probvec(m_T3)      # t3 dimension
    vt4 = tucker_tensor.factors[6]' * probvec(m_T4)      # t4 dimension
    @tensor reduced[t5] := tucker_tensor.core[out, in, t1, t2, t3, t4, t5] * vout[out] * vin[in] * vt1[t1] * vt2[t2] * vt3[t3] * vt4[t4]

    # Project back to original space
    result = tucker_tensor.factors[7] * reduced
    return Categorical(normalize!(result, 1))
end

# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, q_T5::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:out, Marginalisation) (m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, q_a=q_a, m_T5=Categorical(probvec(q_T5)), meta=meta)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, q_T5::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:in, Marginalisation) (m_out=m_out, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, q_a=q_a, m_T5=Categorical(probvec(q_T5)), meta=meta)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, q_T5::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T1, Marginalisation) (m_out=m_out, m_in=m_in, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, q_a=q_a, m_T5=Categorical(probvec(q_T5)), meta=meta)
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, q_T5::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T2, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T3=m_T3, m_T4=m_T4, q_a=q_a, m_T5=Categorical(probvec(q_T5)), meta=meta)
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:TuckerTensor}, q_T5::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T3, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T4=m_T4, q_a=q_a, m_T5=Categorical(probvec(q_T5)), meta=meta)
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:TuckerTensor}, q_T5::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:T4, Marginalisation) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, q_a=q_a, m_T5=Categorical(probvec(q_T5)), meta=meta)
end


# Marginal rule for the transition model (8 interfaces)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4_T5_T6) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tullio eloga[a, b, c, d, e, f, g, h] = eloga[a, b, c, d, e, f, g, h] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g] * probvec(m_T6)[h]
    return Contingency(eloga)
end

# Marginal rule for the transition model (6 interfaces)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tullio eloga[a, b, c, d, e, f] = eloga[a, b, c, d, e, f] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Contingency(eloga)
end

# Marginal rule for the transition model (6 interfaces)
@marginalrule DiscreteTransition(:in_T1_T2_T3_T4) (q_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tullio eloga[a, b, c, d, e, f] = eloga[a, b, c, d, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Contingency(eloga)
end
