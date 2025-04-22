"""
Optimized rules for SparseArray tensors with minimal allocations.
"""
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))

    # Fixed dimensions tuple
    fixed_dims = ((1, out_idx),)

    # Call specialized function with explicit arguments for each dimension
    return sparse_tensor_marginalize_6d(
        eloga,
        2,                  # out_dim 
        fixed_dims,
        nothing,            # p1 (fixed by out_idx)
        nothing,            # p2 (output dimension)
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
    )
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))

    # Fixed dimensions tuple
    fixed_dims = ((1, out_idx),)

    # Call specialized function with explicit arguments for each dimension
    return sparse_tensor_marginalize_6d(
        eloga,
        3,                  # out_dim 
        fixed_dims,
        nothing,            # p1 (fixed by out_idx)
        probvec(m_in),      # p2 
        nothing,            # p3 (output dimension)
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
    )
end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))

    # Fixed dimensions tuple
    fixed_dims = ((1, out_idx),)

    # Call specialized function with explicit arguments for each dimension
    return sparse_tensor_marginalize_6d(
        eloga,
        4,                  # out_dim 
        fixed_dims,
        nothing,            # p1 (fixed by out_idx)
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        nothing,            # p4 (output dimension)
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
    )
end

@rule DiscreteTransition(:T3, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))

    # Fixed dimensions tuple
    fixed_dims = ((1, out_idx),)

    # Call specialized function with explicit arguments for each dimension
    return sparse_tensor_marginalize_6d(
        eloga,
        5,                  # out_dim 
        fixed_dims,
        nothing,            # p1 (fixed by out_idx)
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        nothing,            # p5 (output dimension)
        probvec(m_T4),      # p6
    )
end

@rule DiscreteTransition(:T4, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,6}}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))

    # Fixed dimensions tuple
    fixed_dims = ((1, out_idx),)

    # Call specialized function with explicit arguments for each dimension
    return sparse_tensor_marginalize_6d(
        eloga,
        6,                  # out_dim 
        fixed_dims,
        nothing,            # p1 (fixed by out_idx)
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        nothing,            # p6 (output dimension)
    )
end

# Rules for the transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, m_T5::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)

    return sparse_tensor_marginalize_7d(
        eloga,
        1,                  # out_dim 
        (),
        nothing,            # p1 (output dimension)
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        probvec(m_T5)       # p7
    )
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, m_T5::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)

    return sparse_tensor_marginalize_7d(
        eloga,
        2,                  # out_dim 
        (),
        probvec(m_out),     # p1 
        nothing,            # p2 (output dimension)
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        probvec(m_T5)       # p7
    )
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, m_T5::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)

    return sparse_tensor_marginalize_7d(
        eloga,
        3,                  # out_dim 
        (),
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        nothing,            # p3 (output dimension)
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        probvec(m_T5)       # p7
    )
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, m_T5::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)

    return sparse_tensor_marginalize_7d(
        eloga,
        4,                  # out_dim 
        (),
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        nothing,            # p4 (output dimension)
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        probvec(m_T5)       # p7
    )
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T4::DiscreteNonParametric, m_T5::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)

    return sparse_tensor_marginalize_7d(
        eloga,
        5,                  # out_dim 
        (),
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        nothing,            # p5 (output dimension)
        probvec(m_T4),      # p6
        probvec(m_T5)       # p7
    )
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T5::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)

    return sparse_tensor_marginalize_7d(
        eloga,
        6,                  # out_dim 
        (),
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        nothing,            # p6 (output dimension)
        probvec(m_T5)       # p7
    )
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, meta::Any) where {T} = begin
    eloga = mean(q_a)

    return sparse_tensor_marginalize_7d(
        eloga,
        7,                  # out_dim 
        (),
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        nothing             # p7 (output dimension)
    )
end

# Rules for the transition model (7 interfaces, q_T5 is pointmass)
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))

    fixed_dims = ((7, out_idx),)

    return sparse_tensor_marginalize_7d(
        eloga,
        1,                  # out_dim 
        fixed_dims,
        nothing,            # p1 (output dimension)
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        nothing             # p7 (output dimension)
    )
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))

    fixed_dims = ((7, out_idx),)

    return sparse_tensor_marginalize_7d(
        eloga,
        2,                  # out_dim 
        fixed_dims,
        probvec(m_out),     # p1 
        nothing,            # p2 (output dimension)
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        nothing             # p7 (output dimension)
    )
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))

    fixed_dims = ((7, out_idx),)

    return sparse_tensor_marginalize_7d(
        eloga,
        3,                  # out_dim 
        fixed_dims,
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        nothing,            # p3 (output dimension)
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        nothing             # p7 (output dimension)
    )
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T3::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))

    fixed_dims = ((7, out_idx),)

    return sparse_tensor_marginalize_7d(
        eloga,
        4,                  # out_dim 
        fixed_dims,
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        nothing,            # p4 (output dimension)
        probvec(m_T3),      # p5
        probvec(m_T4),      # p6
        nothing             # p7 (output dimension)
    )
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T4::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))

    fixed_dims = ((7, out_idx),)

    return sparse_tensor_marginalize_7d(
        eloga,
        5,                  # out_dim 
        fixed_dims,
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        nothing,            # p5 (output dimension)
        probvec(m_T4),      # p6
        nothing             # p7 (output dimension)
    )
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass{<:SparseArray{T,7}}, q_T5::PointMass{<:AbstractVector}, meta::Any) where {T} = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_T5))

    fixed_dims = ((7, out_idx),)

    return sparse_tensor_marginalize_7d(
        eloga,
        6,                  # out_dim 
        fixed_dims,
        probvec(m_out),     # p1 
        probvec(m_in),      # p2 
        probvec(m_T1),      # p3
        probvec(m_T2),      # p4
        probvec(m_T3),      # p5
        nothing,            # p6 (output dimension)
        nothing             # p7 (output dimension)
    )
end
