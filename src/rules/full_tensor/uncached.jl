using Tullio, LoopVectorization
# relu(x) = max(x, tiny)
guard(x) = all(iszero(x)) ? x .+ 1.0 : x
# Rules for observation model (q_out is pointmass)
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, q_T1::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))
    T1_idx = findfirst(isone, probvec(q_T1))
    veloga = view(eloga, out_idx, :, T1_idx, :, :, :)

    @tullio out[i] := veloga[i, d, e, f] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, q_T1::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))
    T1_idx = findfirst(isone, probvec(q_T1))
    veloga = view(eloga, out_idx, :, T1_idx, :, :, :)

    @tullio out[i] := veloga[b, i, d, e] * probvec(m_in)[b] * probvec(m_T3)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T3, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, q_T1::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))
    T1_idx = findfirst(isone, probvec(q_T1))
    veloga = view(eloga, out_idx, :, T1_idx, :, :, :)

    @tullio out[i] := veloga[b, d, i, e] * probvec(m_in)[b] * probvec(m_T2)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T4, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, q_T1::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))
    T1_idx = findfirst(isone, probvec(q_T1))
    veloga = view(eloga, out_idx, :, T1_idx, :, :, :)

    @tullio out[i] := veloga[b, d, e, i] * probvec(m_in)[b] * probvec(m_T2)[d] * probvec(m_T3)[e]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[i, a, c, d, e, f, g] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, i, c, d, e, f, g] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, i, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, c, i, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, c, d, i, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, c, d, e, i, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T5)[g]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, c, d, e, f, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

# Rules for transition model (7 interfaces, q_T5 PointMass)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))
    veloga = view(eloga, :, :, :, :, :, :, out_idx)

    @tullio out[i] := veloga[i, a, c, d, e, f] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))
    veloga = view(eloga, :, :, :, :, :, :, out_idx)

    @tullio out[i] := veloga[a, i, c, d, e, f] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))
    veloga = view(eloga, :, :, :, :, :, :, out_idx)

    @tullio out[i] := veloga[a, b, i, d, e, f] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))
    veloga = view(eloga, :, :, :, :, :, :, out_idx)

    @tullio out[i] := veloga[a, b, c, i, e, f] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))
    veloga = view(eloga, :, :, :, :, :, :, out_idx)

    @tullio out[i] := veloga[a, b, c, d, i, f] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[f]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))
    veloga = view(eloga, :, :, :, :, :, :, out_idx)

    @tullio out[i] := veloga[a, b, c, d, e, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e]
    return Categorical(normalize!(guard(out), 1); check_args=false)
end

# Rules for observation model (q_out is pointmass)
@marginalrule DiscreteTransition(:in_T2_T3_T4) (q_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, q_T1::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))
    T1_idx = findfirst(isone, probvec(q_T1))
    veloga = view(eloga, out_idx, :, T1_idx, :, :, :)
    @tullio result[b, d, e, f] := veloga[b, d, e, f] * probvec(m_in)[b] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end

# Rules for observation model (q_out is pointmass)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio result[a, b, c, d, e, f] := eloga[a, b, c, d, e, f] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end


# Rules for observation model (q_out is pointmass)
@marginalrule DiscreteTransition(:in_T1_T2_T3_T4) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio result[b, c, d, e, f] := eloga[a, b, c, d, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end

# Rules for transition model (7 interfaces)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4_T5) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio result[a, b, c, d, e, f, g] := eloga[a, b, c, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end

# Rules for transition model 
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio result[a, b, c, d, e, f] := eloga[a, b, c, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(q_T5)[g]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end

# Rules for observation model for future observations (m_out is categorical)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    size_result = size(eloga)[1]
    return Categorical(fill(T(1 / size_result), size_result); check_args=false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    size_result = size(eloga)[2]
    return Categorical(fill(T(1 / size_result), size_result); check_args=false)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    size_result = size(eloga)[3]
    return Categorical(fill(T(1 / size_result), size_result); check_args=false)
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    size_result = size(eloga)[4]
    return Categorical(fill(T(1 / size_result), size_result); check_args=false)
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    size_result = size(eloga)[5]
    return Categorical(fill(T(1 / size_result), size_result); check_args=false)
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    size_result = size(eloga)[6]
    return Categorical(fill(T(1 / size_result), size_result); check_args=false)
end

# Marginals for JointMarginalMeta

# Rules for observation model (q_out is pointmass)
@marginalrule DiscreteTransition(:in_T2_T3_T4) (q_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, q_T1::PointMass{<:AbstractVector}, meta::JointMarginalStorage) where {T} = begin
    marginal = @call_marginalrule DiscreteTransition(:in_T2_T3_T4) (q_out=q_out, m_in=m_in, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, q_a=q_a, q_T1=q_T1, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end

@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,6}}, meta::JointMarginalStorage) where {T} = begin
    marginal = @call_marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, q_a=q_a, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end

# Rules for transition model (7 interfaces)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4_T5) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, meta::JointMarginalStorage) where {T} = begin
    marginal = @call_marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4_T5) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end

# Rules for transition model 
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass{<:AbstractArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::JointMarginalStorage) where {T} = begin
    marginal = @call_marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, q_a=q_a, q_T5=q_T5, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end

# Rules for transition model 
@marginalrule DiscreteTransition(:out_in_T1) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, q_a::PointMass{<:AbstractArray{T,3}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    @tullio result[a, b, c] := eloga[a, b, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end


# Rules for transition model 
@marginalrule DiscreteTransition(:out_in_T1) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, q_a::PointMass{<:AbstractArray{T,3}}, meta::JointMarginalStorage) where {T} = begin
    marginal = @call_marginalrule DiscreteTransition(:out_in_T1) (m_out=m_out, m_in=m_in, m_T1=m_T1, q_a=q_a, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end


# Rules for transition model 
@marginalrule DiscreteTransition(:out_in) (m_out::Categorical, m_in::Categorical, q_a::PointMass{<:AbstractArray{T,2}}, meta::JointMarginalStorage) where {T} = begin
    marginal = @call_marginalrule DiscreteTransition(:out_in) (m_out=m_out, m_in=m_in, q_a=q_a, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end
