using TensorOperations

relu(x) = max(x, tiny)
BayesBase.mean(q_a::PointMass) = q_a.point

# Rules for observation model (q_out is pointmass)
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(q_out), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4)), (1, 3, 4, 5, 6))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(q_out), probvec(m_in), probvec(m_T2), probvec(m_T3), probvec(m_T4)), (1, 2, 4, 5, 6))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(q_out), probvec(m_in), probvec(m_T1), probvec(m_T3), probvec(m_T4)), (1, 2, 3, 5, 6))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(q_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T4)), (1, 2, 3, 4, 6))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(q_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3)), (1, 2, 3, 4, 5))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end



# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(m_T5)), (2, 3, 4, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(m_T5)), (1, 3, 4, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(m_T5)), (1, 2, 4, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    if length(probvec(m_T4)) == 3
        return missing
    end
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T3), probvec(m_T4), probvec(m_T5)), (1, 2, 3, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    return missing
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    if length(probvec(m_T2)) == 4
        return missing
    end
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T5)), (1, 2, 3, 4, 5, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4)), (1, 2, 3, 4, 5, 6))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end


# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, q_T5::PointMass, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(q_T5)), (2, 3, 4, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, q_T5::PointMass, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(q_T5)), (1, 3, 4, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, q_T5::PointMass, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T2), probvec(m_T3), probvec(m_T4), probvec(q_T5)), (1, 2, 4, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, q_T5::PointMass, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T3), probvec(m_T4), probvec(q_T5)), (1, 2, 3, 5, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, q_T5::PointMass, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T4), probvec(q_T5)), (1, 2, 3, 4, 6, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:CPTensor}, q_T5::PointMass, meta::Any) = begin
    cptensor = q_a.point
    result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(q_T5)), (1, 2, 3, 4, 5, 7))
    result .= relu.(result)
    return Categorical(normalize!(result, 1))
end

# # Rules for transition model (6 interfaces)
# @rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4)), (2, 3, 4, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(m_T4)), (1, 3, 4, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T2), probvec(m_T3), probvec(m_T4)), (1, 2, 4, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T3), probvec(m_T4)), (1, 2, 3, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T4)), (1, 2, 3, 4, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:CPTensor}, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3)), (1, 2, 3, 4, 5))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# # Rules for transition model (6 interfaces)
# @rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:CPTensor}, q_T4::PointMass, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(q_T4)), (2, 3, 4, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:CPTensor}, q_T4::PointMass, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_T1), probvec(m_T2), probvec(m_T3), probvec(q_T4)), (1, 3, 4, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:CPTensor}, q_T4::PointMass, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T2), probvec(m_T3), probvec(q_T4)), (1, 2, 4, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, q_a::PointMass{<:CPTensor}, q_T4::PointMass, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T3), probvec(q_T4)), (1, 2, 3, 5, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end

# @rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, q_a::PointMass{<:CPTensor}, q_T4::PointMass, meta::Any) = begin
#     cptensor = q_a.point
#     result = mode_product_inner(cptensor, (probvec(m_out), probvec(m_in), probvec(m_T1), probvec(m_T2), probvec(q_T4)), (1, 2, 3, 4, 6))
#     result .= relu.(result)
#     return Categorical(normalize!(result, 1))
# end
