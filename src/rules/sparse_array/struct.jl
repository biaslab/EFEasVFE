export SparseArray

"""
    SparseArray{T,N}

A sparse array type that stores only the indices of non-zero elements and provides caching.
The array is assumed to be binary (only contains 0s and 1s).

Fields:
- data: The original tensor dimensions
- nonzero_indices: Vector of CartesianIndex{N} where the tensor has value 1
- dimension_cache: Cache of filtered indices by dimension and value
"""
struct SparseArray{T,N}
    data::NTuple{N,Int}  # Dimensions of the original tensor
    nonzero_indices::Vector{CartesianIndex{N}}  # Indices where tensor is 1
    dimension_cache::Dict{Int,Dict{Int,Vector{CartesianIndex{N}}}}  # Cache by dimension and value

    function SparseArray(tensor::AbstractArray{T,N}) where {T,N}
        # Find all indices where tensor is 1
        indices = findall(x -> x ≈ one(T), tensor)
        new{T,N}(size(tensor), indices, Dict{Int,Dict{Int,Vector{CartesianIndex{N}}}}())
    end
end

# Constructor for empty sparse array of given dimensions
function SparseArray(dims::NTuple{N,Int}) where {N}
    SparseArray{Float64,N}(dims, CartesianIndex{N}[], Dict{Int,Dict{Int,Vector{CartesianIndex{N}}}}())
end

# Get the dimensions of the sparse array
Base.size(sa::SparseArray) = sa.data
Base.size(sa::SparseArray, d::Int) = sa.data[d]

# Get the list of non-zero indices
nonzero_indices(sa::SparseArray) = sa.nonzero_indices

# Get filtered indices for a specific dimension and value, with caching
function nonzero_indices(sa::SparseArray{T,N}, dim::Int, value::Int) where {T,N}
    # Check if we have a cache for this dimension
    if !haskey(sa.dimension_cache, dim)
        sa.dimension_cache[dim] = Dict{Int,Vector{CartesianIndex{N}}}()
    end

    # Check if we have a cache for this value within the dimension
    dim_cache = sa.dimension_cache[dim]
    if !haskey(dim_cache, value)
        # Create and store the filtered indices
        dim_cache[value] = filter(idx -> idx[dim] == value, sa.nonzero_indices)
    end

    return dim_cache[value]
end

# Convert back to dense array if needed
function Base.Array(sa::SparseArray{T,N}) where {T,N}
    result = zeros(T, sa.data...)
    for idx in sa.nonzero_indices
        result[idx] = one(T)
    end
    return result
end

# Show method for better printing
function Base.show(io::IO, sa::SparseArray{T,N}) where {T,N}
    print(io, "SparseArray{$T,$N}($(sa.data), $(length(sa.nonzero_indices)) non-zero elements")
end


"""
Generalized tensor contraction for SparseArray tensors with zero allocations.
Uses a type-stable mask approach instead of Union types for better performance.

Parameters:
- eloga: The sparse tensor
- out_dim: The dimension to marginalize to (output dimension)
- fixed_dims: Tuple of (dimension, index) pairs for PointMass variables
- prob_vecs: Tuple of probability vectors for all dimensions
- dim_mask: BitVector indicating which dimensions have probability vectors

Returns a Categorical distribution for the output dimension.
"""
function sparse_tensor_marginalize(
    eloga::SparseArray{T,N},
    out_dim::Int,
    fixed_dims::NTuple{F,Tuple{Int,Int}}=(),
    prob_vecs::NTuple{N,Vector{T}}=ntuple(i -> Vector{T}(), N),
    dim_mask::NTuple{N,Bool}=ntuple(i -> false, N)
) where {T,N,F}
    # Initialize output
    out = zeros(T, size(eloga, out_dim))

    # Determine indices to process - use the first fixed dimension if available
    indices_to_process = if !isempty(fixed_dims)
        first_fixed_dim, first_fixed_value = fixed_dims[1]
        nonzero_indices(eloga, first_fixed_dim, first_fixed_value)
    else
        nonzero_indices(eloga)
    end

    # Process all indices
    @inbounds for idx in indices_to_process
        # Skip if index doesn't match all fixed dimensions
        skip = false
        for (dim, val) in fixed_dims
            if idx[dim] != val
                skip = true
                break
            end
        end
        if skip
            continue
        end

        # Calculate probability product with type-stable checks
        prob = one(T)
        for dim in 1:N
            if dim != out_dim && dim_mask[dim]
                prob_dim = prob_vecs[dim][idx[dim]]
                if prob_dim < tiny
                    skip = true
                    break
                end
                prob *= prob_dim
            end
        end
        if skip
            continue
        end

        # Accumulate result
        out[idx[out_dim]] += prob
    end

    return Categorical(normalize!(out, 1))
end



"""
Specialized tensor contraction for 7-dimensional SparseArray tensors.
This avoids any type instability or loop overhead.

Parameters as in the general version, but specialized for N=7.
"""
function sparse_tensor_marginalize_7d(
    eloga::SparseArray{T,7},
    out_dim::Int,
    fixed_dims::NTuple{F,Tuple{Int,Int}}=(),
    p1::Union{Vector{T},Nothing}=nothing,
    p2::Union{Vector{T},Nothing}=nothing,
    p3::Union{Vector{T},Nothing}=nothing,
    p4::Union{Vector{T},Nothing}=nothing,
    p5::Union{Vector{T},Nothing}=nothing,
    p6::Union{Vector{T},Nothing}=nothing,
    p7::Union{Vector{T},Nothing}=nothing
) where {T,F}
    # Initialize output
    out = zeros(T, size(eloga, out_dim))

    # Determine indices to process
    indices_to_process = if !isempty(fixed_dims)
        first_fixed_dim, first_fixed_value = fixed_dims[1]
        nonzero_indices(eloga, first_fixed_dim, first_fixed_value)
    else
        nonzero_indices(eloga)
    end

    # Process indices with hand-unrolled dimension checks
    # This completely eliminates the need for a Union type
    @inbounds for idx in indices_to_process
        # Check fixed dimensions
        skip = false
        for (dim, val) in fixed_dims
            if idx[dim] != val
                skip = true
                break
            end
        end
        if skip
            continue
        end

        # Calculate probability product (fully unrolled for all 7 dimensions)
        prob = one(T)

        # Dimension 1
        if out_dim != 1 && p1 !== nothing
            val = p1[idx[1]]
            prob *= val
        end

        # Dimension 2
        if out_dim != 2 && p2 !== nothing
            val = p2[idx[2]]
            prob *= val
        end

        # Dimension 3
        if out_dim != 3 && p3 !== nothing
            val = p3[idx[3]]
            prob *= val
        end

        # Dimension 4
        if out_dim != 4 && p4 !== nothing
            val = p4[idx[4]]
            prob *= val
        end

        # Dimension 5
        if out_dim != 5 && p5 !== nothing
            val = p5[idx[5]]
            prob *= val
        end

        # Dimension 6
        if out_dim != 6 && p6 !== nothing
            val = p6[idx[6]]
            prob *= val
        end

        # Dimension 7
        if out_dim != 7 && p7 !== nothing
            val = p7[idx[7]]
            prob *= val
        end

        # Accumulate result
        out[idx[out_dim]] += prob
    end

    return Categorical(normalize!(out, 1); check_args=false)
end

"""
Specialized tensor contraction for 6-dimensional SparseArray tensors.
This avoids any type instability or loop overhead.

Parameters as in the general version, but specialized for N=6.
"""
function sparse_tensor_marginalize_6d(
    eloga::SparseArray{T,6},
    out_dim::Int,
    fixed_dims::NTuple{F,Tuple{Int,Int}}=(),
    p1::Union{Vector{T},Nothing}=nothing,
    p2::Union{Vector{T},Nothing}=nothing,
    p3::Union{Vector{T},Nothing}=nothing,
    p4::Union{Vector{T},Nothing}=nothing,
    p5::Union{Vector{T},Nothing}=nothing,
    p6::Union{Vector{T},Nothing}=nothing
) where {T,F}
    # Initialize output
    out = zeros(T, size(eloga, out_dim))

    # Determine indices to process
    indices_to_process = if !isempty(fixed_dims)
        first_fixed_dim, first_fixed_value = fixed_dims[1]
        nonzero_indices(eloga, first_fixed_dim, first_fixed_value)
    else
        nonzero_indices(eloga)
    end

    # Process indices with hand-unrolled dimension checks
    @inbounds for idx in indices_to_process
        # Check fixed dimensions
        skip = false
        for (dim, val) in fixed_dims
            if idx[dim] != val
                skip = true
                break
            end
        end
        if skip
            continue
        end

        # Calculate probability product (fully unrolled for all 6 dimensions)
        prob = one(T)

        # Dimension 1
        if out_dim != 1 && p1 !== nothing
            val = p1[idx[1]]
            prob *= val
        end

        # Dimension 2
        if out_dim != 2 && p2 !== nothing
            val = p2[idx[2]]
            prob *= val
        end

        # Dimension 3
        if out_dim != 3 && p3 !== nothing
            val = p3[idx[3]]
            prob *= val
        end

        # Dimension 4
        if out_dim != 4 && p4 !== nothing
            val = p4[idx[4]]
            prob *= val
        end

        # Dimension 5
        if out_dim != 5 && p5 !== nothing
            val = p5[idx[5]]
            prob *= val
        end

        # Dimension 6
        if out_dim != 6 && p6 !== nothing
            val = p6[idx[6]]
            prob *= val
        end

        # Accumulate result
        out[idx[out_dim]] += prob
    end

    return Categorical(normalize!(out, 1))
end