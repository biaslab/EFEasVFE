using Tullio
using TensorOperations
using RxInfer

struct CachingMeta
    cache::Dict{Symbol,AbstractArray}

    CachingMeta() = new(Dict{Symbol,AbstractArray}())
end

# Implement basic dictionary interface
Base.getindex(meta::CachingMeta, key::Symbol) = getindex(meta.cache, key)
Base.haskey(meta::CachingMeta, key::Symbol) = haskey(meta.cache, key)
Base.setindex!(meta::CachingMeta, value::AbstractArray, key::Symbol) = setindex!(meta.cache, value, key)

# Reset function to clear cache
function reset!(meta::CachingMeta)
    empty!(meta.cache)
end

reset_cache!(node) = reset_cache!(node, node.properties.fform)
reset_cache!(node, fform) = nothing
reset_cache!(node, fform::Type{DiscreteTransition}) = reset_cache!(node, fform, RxInfer.GraphPPL.getextra(node, :meta, nothing))

reset_cache!(node, fform, any) = nothing

function reset_cache!(node, fform::Type{DiscreteTransition}, meta::CachingMeta)
    q_out = nothing
    if haskey(meta, :out)
        q_out = meta[:out]
    end
    reset!(meta)
    if !isnothing(q_out)
        meta[:out] = q_out
    end
end

function after_iteration_callback(model, iteration)
    gppl = model.model
    for node in RxInfer.GraphPPL.factor_nodes(gppl)
        reset_cache!(gppl[node])
    end
end




# Rules for observation model (q_out is pointmass)
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_T3_T4_T5
    if !haskey(meta, :out_T3_T4_T5)
        # Get or compute q_out_reduced first
        if !haskey(meta, :out)
            @tensor q_out_reduced[in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_out)[out]
            meta[:out] = q_out_reduced
        end
        q_out_reduced = meta[:out]

        @tensor out_T3_T4_T5[in, t1, t2] := q_out_reduced[in, t1, t2, t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:out_T3_T4_T5] = out_T3_T4_T5
    end
    q_out_T3_T4_T5 = meta[:out_T3_T4_T5]

    # Final computation using cached result
    @tensor out[in] := q_out_T3_T4_T5[in, t1, t2] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_T3_T4_T5
    if !haskey(meta, :out_T3_T4_T5)
        # Get or compute q_out_reduced first
        if !haskey(meta, :out)
            @tensor q_out_reduced[in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_out)[out]
            meta[:out] = q_out_reduced
        end
        q_out_reduced = meta[:out]

        @tensor out_T3_T4_T5[in, t1, t2] := q_out_reduced[in, t1, t2, t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:out_T3_T4_T5] = out_T3_T4_T5
    end
    q_out_T3_T4_T5 = meta[:out_T3_T4_T5]

    # Final computation using cached result
    @tensor out[t1] := q_out_T3_T4_T5[in, t1, t2] * probvec(m_in)[in] * probvec(m_T2)[t2]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_T3_T4_T5
    if !haskey(meta, :out_T3_T4_T5)
        # Get or compute q_out_reduced first
        if !haskey(meta, :out)
            @tensor q_out_reduced[in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_out)[out]
            meta[:out] = q_out_reduced
        end
        q_out_reduced = meta[:out]

        @tensor out_T3_T4_T5[in, t1, t2] := q_out_reduced[in, t1, t2, t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:out_T3_T4_T5] = out_T3_T4_T5
    end
    q_out_T3_T4_T5 = meta[:out_T3_T4_T5]

    # Final computation using cached result
    @tensor out[t2] := q_out_T3_T4_T5[in, t1, t2] * probvec(m_in)[in] * probvec(m_T1)[t1]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_out_reduced first
        if !haskey(meta, :out)
            @tensor q_out_reduced[in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_out)[out]
            meta[:out] = q_out_reduced
        end
        q_out_reduced = meta[:out]

        @tensor out_in_T1_T2[t3, t4, t5] := q_out_reduced[in, t1, t2, t3, t4, t5] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t3] := q_out_in_T1_T2[t3, t4, t5] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_out_reduced first
        if !haskey(meta, :out)
            @tensor q_out_reduced[in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_out)[out]
            meta[:out] = q_out_reduced
        end
        q_out_reduced = meta[:out]

        @tensor out_in_T1_T2[t3, t4, t5] := q_out_reduced[in, t1, t2, t3, t4, t5] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t4] := q_out_in_T1_T2[t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_out_reduced first
        if !haskey(meta, :out)
            @tensor q_out_reduced[in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_out)[out]
            meta[:out] = q_out_reduced
        end
        q_out_reduced = meta[:out]

        @tensor out_in_T1_T2[t3, t4, t5] := q_out_reduced[in, t1, t2, t3, t4, t5] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t5] := q_out_in_T1_T2[t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
    return Categorical(normalize!(out, 1))
end




# Rules for transition model (8 interfaces, q_T6 is pointmass)

@rule DiscreteTransition(:out, Marginalisation) (m_in::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, q_T6::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_T3_T4_T5
    if !haskey(meta, :out_T3_T4_T5)
        # Get or compute q_T6_reduced first
        if !haskey(meta, :T6)
            @tensor q_T6_reduced[out, in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5, t6] * probvec(q_T6)[t6]
            meta[:T6] = q_T6_reduced
        end
        q_T6_reduced = meta[:T6]

        @tensor out_T3_T4_T5[out, in, t1, t2] := q_T6_reduced[out, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:out_T3_T4_T5] = out_T3_T4_T5
    end
    q_out_T3_T4_T5 = meta[:out_T3_T4_T5]

    # Final computation using cached result
    @tensor result[out] := q_out_T3_T4_T5[out, in, t1, t2] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
    return Categorical(normalize!(result, 1))
end


@rule DiscreteTransition(:in, Marginalisation) (m_out::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, q_T6::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_T3_T4_T5
    if !haskey(meta, :out_T3_T4_T5)
        # Get or compute q_T6_reduced first
        if !haskey(meta, :T6)
            @tensor q_T6_reduced[out, in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5, t6] * probvec(q_T6)[t6]
            meta[:T6] = q_T6_reduced
        end
        q_T6_reduced = meta[:T6]

        @tensor out_T3_T4_T5[out, in, t1, t2] := q_T6_reduced[out, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:out_T3_T4_T5] = out_T3_T4_T5
    end
    q_out_T3_T4_T5 = meta[:out_T3_T4_T5]

    # Final computation using cached result
    @tensor result[in] := q_out_T3_T4_T5[out, in, t1, t2] * probvec(m_out)[out] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, q_T6::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_T3_T4_T5
    if !haskey(meta, :out_T3_T4_T5)
        # Get or compute q_T6_reduced first
        if !haskey(meta, :T6)
            @tensor q_T6_reduced[out, in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5, t6] * probvec(q_T6)[t6]
            meta[:T6] = q_T6_reduced
        end
        q_T6_reduced = meta[:T6]

        @tensor out_T3_T4_T5[out, in, t1, t2] := q_T6_reduced[out, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:out_T3_T4_T5] = out_T3_T4_T5
    end
    q_out_T3_T4_T5 = meta[:out_T3_T4_T5]

    # Final computation using cached result
    @tensor result[t1] := q_out_T3_T4_T5[out, in, t1, t2] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T2)[t2]
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, q_T6::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_T3_T4_T5
    if !haskey(meta, :out_T3_T4_T5)
        # Get or compute q_T6_reduced first
        if !haskey(meta, :T6)
            @tensor q_T6_reduced[out, in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5, t6] * probvec(q_T6)[t6]
            meta[:T6] = q_T6_reduced
        end
        q_T6_reduced = meta[:T6]

        @tensor out_T3_T4_T5[out, in, t1, t2] := q_T6_reduced[out, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:out_T3_T4_T5] = out_T3_T4_T5
    end
    q_out_T3_T4_T5 = meta[:out_T3_T4_T5]

    # Final computation using cached result
    @tensor out[t2] := q_out_T3_T4_T5[out, in, t1, t2] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T1)[t1]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, q_T6::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_T6_reduced first
        if !haskey(meta, :T6)
            @tensor q_T6_reduced[out, in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5, t6] * probvec(q_T6)[t6]
            meta[:T6] = q_T6_reduced
        end
        q_T6_reduced = meta[:T6]

        @tensor out_in_T1_T2[t3, t4, t5] := q_T6_reduced[out, in, t1, t2, t3, t4, t5] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t3] := q_out_in_T1_T2[t3, t4, t5] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass, q_T6::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_T6_reduced first
        if !haskey(meta, :T6)
            @tensor q_T6_reduced[out, in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5, t6] * probvec(q_T6)[t6]
            meta[:T6] = q_T6_reduced
        end
        q_T6_reduced = meta[:T6]

        @tensor out_in_T1_T2[t3, t4, t5] := q_T6_reduced[out, in, t1, t2, t3, t4, t5] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t4] := q_out_in_T1_T2[t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, q_T6::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_T6_reduced first
        if !haskey(meta, :T6)
            @tensor q_T6_reduced[out, in, t1, t2, t3, t4, t5] := eloga[out, in, t1, t2, t3, t4, t5, t6] * probvec(q_T6)[t6]
            meta[:T6] = q_T6_reduced
        end
        q_T6_reduced = meta[:T6]

        @tensor out_in_T1_T2[t3, t4, t5] := q_T6_reduced[out, in, t1, t2, t3, t4, t5] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t5] := q_out_in_T1_T2[t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (7 interfaces, q_T5 is pointmass)

@rule DiscreteTransition(:out, Marginalisation) (m_in::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, q_T5::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4
    if !haskey(meta, :T2_T3_T4)
        # Get or compute q_T5_reduced first
        if !haskey(meta, :T5)
            @tensor q_T5_reduced[out, in, t1, t2, t3, t4] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_T5)[t5]
            meta[:T5] = q_T5_reduced
        end
        q_T5_reduced = meta[:T5]

        @tensor T2_T3_T4[out, in, t1] := q_T5_reduced[out, in, t1, t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
        meta[:T2_T3_T4] = T2_T3_T4
    end
    q_T2_T3_T4 = meta[:T2_T3_T4]

    # Final computation using cached result
    @tensor result[out] := q_T2_T3_T4[out, in, t1] * probvec(m_in)[in] * probvec(m_T1)[t1]
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, q_T5::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4
    if !haskey(meta, :T2_T3_T4)
        # Get or compute q_T5_reduced first
        if !haskey(meta, :T5)
            @tensor q_T5_reduced[out, in, t1, t2, t3, t4] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_T5)[t5]
            meta[:T5] = q_T5_reduced
        end
        q_T5_reduced = meta[:T5]

        @tensor T2_T3_T4[out, in, t1] := q_T5_reduced[out, in, t1, t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
        meta[:T2_T3_T4] = T2_T3_T4
    end
    q_T2_T3_T4 = meta[:T2_T3_T4]

    # Final computation using cached result
    @tensor result[in] := q_T2_T3_T4[out, in, t1] * probvec(m_out)[out] * probvec(m_T1)[t1]
    return Categorical(normalize!(result, 1))
end


@rule DiscreteTransition(:T1, Marginalisation) (m_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, q_T5::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4
    if !haskey(meta, :T2_T3_T4)
        # Get or compute q_T5_reduced first
        if !haskey(meta, :T5)
            @tensor q_T5_reduced[out, in, t1, t2, t3, t4] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_T5)[t5]
            meta[:T5] = q_T5_reduced
        end
        q_T5_reduced = meta[:T5]

        @tensor T2_T3_T4[out, in, t1] := q_T5_reduced[out, in, t1, t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
        meta[:T2_T3_T4] = T2_T3_T4
    end
    q_T2_T3_T4 = meta[:T2_T3_T4]

    # Final computation using cached result
    @tensor result[t1] := q_T2_T3_T4[out, in, t1] * probvec(m_out)[out] * probvec(m_T1)[t1]
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, q_T5::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_T5_reduced first
        if !haskey(meta, :T5)
            @tensor q_T5_reduced[out, in, t1, t2, t3, t4] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_T5)[t5]
            meta[:T5] = q_T5_reduced
        end
        q_T5_reduced = meta[:T5]

        @tensor out_in_T1[t2, t3, t4] := q_T5_reduced[out, in, t1, t2, t3, t4] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor result[t2] := q_out_in_T1[t2, t3, t4] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass, q_T5::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_T5_reduced first
        if !haskey(meta, :T5)
            @tensor q_T5_reduced[out, in, t1, t2, t3, t4] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_T5)[t5]
            meta[:T5] = q_T5_reduced
        end
        q_T5_reduced = meta[:T5]

        @tensor out_in_T1[t2, t3, t4] := q_T5_reduced[out, in, t1, t2, t3, t4] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor result[t3] := q_out_in_T1[t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T4)[t4]
    return Categorical(normalize!(result, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass, q_T5::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_T5_reduced first
        if !haskey(meta, :T5)
            @tensor q_T5_reduced[out, in, t1, t2, t3, t4] := eloga[out, in, t1, t2, t3, t4, t5] * probvec(q_T5)[t5]
            meta[:T5] = q_T5_reduced
        end
        q_T5_reduced = meta[:T5]

        @tensor out_in_T1[t2, t3, t4] := q_T5_reduced[out, in, t1, t2, t3, t4] * probvec(m_out)[out] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor result[t4] := q_out_in_T1[t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3]
    return Categorical(normalize!(result, 1))
end


# Rules for transition model (8 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T3_T4_T5_T6
    if !haskey(meta, :T3_T4_T5_T6)
        # Get or compute q_out_reduced first
        @tensor T3_T4_T5_T6[o, in, t1, t2] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5] * probvec(m_T6)[t6]
        meta[:T3_T4_T5_T6] = T3_T4_T5_T6
    end
    q_T3_T4_T5_T6 = meta[:T3_T4_T5_T6]

    # Final computation using cached result
    @tensor out[o] := q_T3_T4_T5_T6[o, in, t1, t2] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T3_T4_T5_T6
    if !haskey(meta, :T3_T4_T5_T6)
        # Get or compute q_out_reduced first
        @tensor T3_T4_T5_T6[o, in, t1, t2] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5] * probvec(m_T6)[t6]
        meta[:T3_T4_T5_T6] = T3_T4_T5_T6
    end
    q_T3_T4_T5_T6 = meta[:T3_T4_T5_T6]

    # Final computation using cached result
    @tensor out[in] := q_T3_T4_T5_T6[o, in, t1, t2] * probvec(m_out)[o] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T3_T4_T5_T6
    if !haskey(meta, :T3_T4_T5_T6)
        # Get or compute q_out_reduced first
        @tensor T3_T4_T5_T6[o, in, t1, t2] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5] * probvec(m_T6)[t6]
        meta[:T3_T4_T5_T6] = T3_T4_T5_T6
    end
    q_T3_T4_T5_T6 = meta[:T3_T4_T5_T6]

    # Final computation using cached result
    @tensor out[t1] := q_T3_T4_T5_T6[o, in, t1, t2] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T2)[t2]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T3_T4_T5_T6
    if !haskey(meta, :T3_T4_T5_T6)
        # Get or compute q_out_reduced first
        @tensor T3_T4_T5_T6[o, in, t1, t2] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5] * probvec(m_T6)[t6]
        meta[:T3_T4_T5_T6] = T3_T4_T5_T6
    end
    q_T3_T4_T5_T6 = meta[:T3_T4_T5_T6]

    # Final computation using cached result
    @tensor out[t2] := q_T3_T4_T5_T6[o, in, t1, t2] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_out_reduced first
        @tensor out_in_T1_T2[t3, t4, t5, t6] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t3] := q_out_in_T1_T2[t3, t4, t5, t6] * probvec(m_T4)[t4] * probvec(m_T5)[t5] * probvec(m_T6)[t6]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_out_reduced first
        @tensor out_in_T1_T2[t3, t4, t5, t6] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t4] := q_out_in_T1_T2[t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T5)[t5] * probvec(m_T6)[t6]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T6::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_out_reduced first
        @tensor out_in_T1_T2[t3, t4, t5, t6] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t5] := q_out_in_T1_T2[t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T6)[t6]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T6, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1_T2
    if !haskey(meta, :out_in_T1_T2)
        # Get or compute q_out_reduced first
        @tensor out_in_T1_T2[t3, t4, t5, t6] := eloga[o, in, t1, t2, t3, t4, t5, t6] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1] * probvec(m_T2)[t2]
        meta[:out_in_T1_T2] = out_in_T1_T2
    end
    q_out_in_T1_T2 = meta[:out_in_T1_T2]

    # Final computation using cached result
    @tensor out[t6] := q_out_in_T1_T2[t3, t4, t5, t6] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end


# Rules for transition model (6 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4
    if !haskey(meta, :T2_T3_T4)
        # Get or compute q_out_reduced first
        @tensor T2_T3_T4[o, in, t1] := eloga[o, in, t1, t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
        meta[:T2_T3_T4] = T2_T3_T4
    end
    q_T2_T3_T4 = meta[:T2_T3_T4]

    # Final computation using cached result
    @tensor out[o] := q_T2_T3_T4[o, in, t1] * probvec(m_in)[in] * probvec(m_T1)[t1]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4
    if !haskey(meta, :T2_T3_T4)
        # Get or compute q_out_reduced first
        @tensor T2_T3_T4[o, in, t1] := eloga[o, in, t1, t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
        meta[:T2_T3_T4] = T2_T3_T4
    end
    q_T2_T3_T4 = meta[:T2_T3_T4]

    # Final computation using cached result
    @tensor out[in] := q_T2_T3_T4[o, in, t1] * probvec(m_out)[o] * probvec(m_T1)[t1]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4
    if !haskey(meta, :T2_T3_T4)
        # Get or compute q_out_reduced first
        @tensor T2_T3_T4[o, in, t1] := eloga[o, in, t1, t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
        meta[:T2_T3_T4] = T2_T3_T4
    end
    q_T2_T3_T4 = meta[:T2_T3_T4]

    # Final computation using cached result
    @tensor out[t1] := q_T2_T3_T4[o, in, t1] * probvec(m_out)[o] * probvec(m_in)[in]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_out_reduced first
        @tensor out_in_T1[t2, t3, t4] := eloga[o, in, t1, t2, t3, t4] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor out[t2] := q_out_in_T1[t2, t3, t4] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_out_reduced first
        @tensor out_in_T1[t2, t3, t4] := eloga[o, in, t1, t2, t3, t4] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor out[t3] := q_out_in_T1[t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T4)[t4]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_out_reduced first
        @tensor out_in_T1[t2, t3, t4] := eloga[o, in, t1, t2, t3, t4] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor out[t4] := q_out_in_T1[t2, t3, t4] * probvec(m_T2)[t2] * probvec(m_T3)[t3]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4_T5
    if !haskey(meta, :T2_T3_T4_T5)
        # Get or compute q_out_reduced first
        @tensor T2_T3_T4_T5[o, in, t1] := eloga[o, in, t1, t2, t3, t4, t5] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:T2_T3_T4_T5] = T2_T3_T4_T5
    end
    q_T2_T3_T4_T5 = meta[:T2_T3_T4_T5]

    # Final computation using cached result
    @tensor out[o] := q_T2_T3_T4_T5[o, in, t1] * probvec(m_in)[in] * probvec(m_T1)[t1]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4_T5
    if !haskey(meta, :T2_T3_T4_T5)
        # Get or compute q_out_reduced first
        @tensor T2_T3_T4_T5[o, in, t1] := eloga[o, in, t1, t2, t3, t4, t5] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:T2_T3_T4_T5] = T2_T3_T4_T5
    end
    q_T2_T3_T4_T5 = meta[:T2_T3_T4_T5]

    # Final computation using cached result
    @tensor out[in] := q_T2_T3_T4_T5[o, in, t1] * probvec(m_out)[o] * probvec(m_T1)[t1]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_T2_T3_T4_T5
    if !haskey(meta, :T2_T3_T4_T5)
        # Get or compute q_out_reduced first
        @tensor T2_T3_T4_T5[o, in, t1] := eloga[o, in, t1, t2, t3, t4, t5] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
        meta[:T2_T3_T4_T5] = T2_T3_T4_T5
    end
    q_T2_T3_T4_T5 = meta[:T2_T3_T4_T5]

    # Final computation using cached result
    @tensor out[t1] := q_T2_T3_T4_T5[o, in, t1] * probvec(m_out)[o] * probvec(m_in)[in]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_out_reduced first
        @tensor out_in_T1[t2, t3, t4, t5] := eloga[o, in, t1, t2, t3, t4, t5] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor out[t2] := q_out_in_T1[t2, t3, t4, t5] * probvec(m_T3)[t3] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_out_reduced first
        @tensor out_in_T1[t2, t3, t4, t5] := eloga[o, in, t1, t2, t3, t4, t5] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor out[t3] := q_out_in_T1[t2, t3, t4, t5] * probvec(m_T2)[t2] * probvec(m_T4)[t4] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_out_reduced first
        @tensor out_in_T1[t2, t3, t4, t5] := eloga[o, in, t1, t2, t3, t4, t5] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor out[t4] := q_out_in_T1[t2, t3, t4, t5] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T5)[t5]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass, meta::CachingMeta) = begin
    eloga = mean(q_a)

    # Get or compute q_out_in_T1
    if !haskey(meta, :out_in_T1)
        # Get or compute q_out_reduced first
        @tensor out_in_T1[t2, t3, t4, t5] := eloga[o, in, t1, t2, t3, t4, t5] * probvec(m_out)[o] * probvec(m_in)[in] * probvec(m_T1)[t1]
        meta[:out_in_T1] = out_in_T1
    end
    q_out_in_T1 = meta[:out_in_T1]

    # Final computation using cached result
    @tensor out[t5] := q_out_in_T1[t2, t3, t4, t5] * probvec(m_T2)[t2] * probvec(m_T3)[t3] * probvec(m_T4)[t4]
    return Categorical(normalize!(out, 1))
end