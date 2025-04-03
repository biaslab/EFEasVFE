
# Rules for observation model (q_out is pointmass)
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, i, c, d, e, f, g] * probvec(q_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, i, c, d, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, i, d, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, i, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, i, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (8 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[i, a, c, d, e, f, g, h] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g] * probvec(m_T6)[h]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, i, c, d, e, f, g, h] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g] * probvec(m_T6)[h]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, i, c, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, i, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, i, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, i, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T6::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, f, i, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T6, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, f, g, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (6 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[i, a, c, d, e, f] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, i, c, d, e, f] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, i, c, d, e] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, c, i, d, e] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, c, d, i, e] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, c, d, e, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[i, a, c, d, e, f, g] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, i, c, d, e, f, g] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, i, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, i, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, i, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, i, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, f, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(out, 1))
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