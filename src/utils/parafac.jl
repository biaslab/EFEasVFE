using LinearAlgebra
using StaticArrays
using NPZ  # Add NPZ for reading .npy files

"""
CPTensor represents a tensor in CP/PARAFAC decomposition format.

Fields:
- weights: Vector of weights for each rank-1 component
- factors: Tuple of factor matrices, where each matrix is size (dimension_size, rank)
- dims: Tuple of dimension sizes of the original tensor
- rank: Rank of the decomposition
- temp_products: Vector of temporary products for efficient computation
"""
struct CPTensor{T<:AbstractFloat,N}
    weights::Vector{T}
    factors::NTuple{N,Matrix{T}}
    dims::NTuple{N,Int}
    rank::Int
    temp_products::Vector{T}  # Scratch space for computations

    # Inner constructor with validation
    function CPTensor(weights::Vector{T}, factors::NTuple{N,Matrix{T}}) where {T<:AbstractFloat,N}
        rank = length(weights)
        if any(size(f, 2) ≠ rank for f in factors)
            throw(ArgumentError("All factor matrices must have the same number of columns as the length of weights"))
        end
        dims = ntuple(i -> size(factors[i], 1), N)
        temp_products = Vector{T}(undef, rank)  # Preallocate scratch space
        new{T,N}(weights, factors, dims, rank, temp_products)
    end
end

# Convenience constructor
function CPTensor(dims::NTuple{N,Int}, rank::Int; T::Type=Float64) where {N}
    weights = ones(T, rank)
    factors = ntuple(i -> randn(T, dims[i], rank), N)
    CPTensor(weights, factors)
end

Base.size(cp::CPTensor) = cp.dims
Base.size(cp::CPTensor, d::Int) = cp.dims[d]
rank(cp::CPTensor) = cp.rank

"""
Optimized inner product computation using preallocation and SIMD-friendly operations.
"""
function mode_product_inner!(result::AbstractVector{T},
    cp::CPTensor{T},
    vectors::NTuple{M,AbstractVector{T}},
    modes::NTuple{M,Int}) where {T<:AbstractFloat,M}
    # Preallocate temporary storage for intermediate products
    temp_products = cp.temp_products

    # Precompute all vector-factor products
    @inbounds for r in 1:cp.rank
        temp = cp.weights[r]
        for (mode, vec) in zip(modes, vectors)
            temp *= dot(vec, view(cp.factors[mode], :, r))
        end
        temp_products[r] = temp
    end

    # Find the remaining mode
    remaining_mode = first(setdiff(1:length(cp.dims), modes))

    # Compute final result using matrix-vector product
    mul!(result, cp.factors[remaining_mode], temp_products)
    return result
end

"""
Non-allocating version that takes pre-allocated result vector
"""
function mode_product_inner(cp::CPTensor{T},
    vectors::NTuple{M,AbstractVector{T}},
    modes::NTuple{M,Int}) where {T<:AbstractFloat,M}
    result = zeros(T, cp.dims[first(setdiff(1:length(cp.dims), modes))])
    mode_product_inner!(result, cp, vectors, modes)
end

# For small, fixed-size vectors (e.g., if dimensions are known at compile time)
function mode_product_inner(cp::CPTensor{T},
    vectors::NTuple{M,SVector{D,T}},
    modes::NTuple{M,Int}) where {T<:AbstractFloat,M,D}
    remaining_mode = first(setdiff(1:length(cp.dims), modes))
    result = @MVector zeros(T, cp.dims[remaining_mode])
    mode_product_inner!(result, cp, vectors, modes)
    SVector(result)
end

"""
Reconstruct the full tensor from its CP decomposition.
Note: This is memory intensive and should be used with caution for large tensors.
"""
function full(cp::CPTensor{T}) where {T}
    result = zeros(T, cp.dims...)

    for r in 1:cp.rank
        # Start with the first factor vector
        temp = reshape(cp.factors[1][:, r], cp.dims[1], ones(Int, length(cp.dims) - 1)...)

        # Multiply with each subsequent factor using broadcasting
        for k in 2:length(cp.factors)
            # Reshape the k-th factor vector to align dimensions for broadcasting
            factor_reshape = ones(Int, k - 1)
            factor_reshape = (factor_reshape..., cp.dims[k], ones(Int, length(cp.dims) - k)...)
            factor_vec = reshape(cp.factors[k][:, r], factor_reshape...)

            temp = temp .* factor_vec
        end

        # Add to result with weight
        result .+= cp.weights[r] .* temp
    end

    return result
end

"""
Normalize the factors of the CP decomposition.
This modifies the tensor in-place and returns the modified tensor.
"""
function LinearAlgebra.normalize!(cp::CPTensor)
    for r in 1:cp.rank
        for k in 1:length(cp.factors)
            norm_val = norm(cp.factors[k][:, r])
            cp.factors[k][:, r] ./= norm_val
            cp.weights[r] *= norm_val
        end
    end
    return cp
end

"""
Print a summary of the CP tensor.
"""
function Base.show(io::IO, cp::CPTensor)
    print(io, "CPTensor(dims=$(cp.dims), rank=$(cp.rank))")
end

"""
Load a CP tensor from a directory containing the decomposition files.
The directory should contain:
- weights.npy: Vector of weights
- factor_0.npy through factor_N.npy: Factor matrices
- metadata.npy: Dictionary containing additional information

Parameters:
- dir: Path to the directory containing the decomposition files
- T: Type parameter for the tensor (default: Float64)

Returns:
- CPTensor: The loaded CP tensor
"""
function load_cp_tensor(dir::AbstractString; T::Type=Float64)
    # Load weights
    weights = npzread(joinpath(dir, "weights.npy"))

    # Load all factor matrices
    factors = []
    i = 0
    while true
        factor_path = joinpath(dir, "factor_$(i).npy")
        if !isfile(factor_path)
            break
        end
        push!(factors, npzread(factor_path))
        i += 1
    end

    # Convert to NTuple for better performance
    factors_tuple = NTuple{length(factors),Matrix{T}}(factors)

    # Create and return the CPTensor
    return CPTensor(weights, factors_tuple)
end

"""
Load a CP tensor and its metadata from a directory.

Parameters:
- dir: Path to the directory containing the decomposition files
- T: Type parameter for the tensor (default: Float64)

Returns:
- Tuple{CPTensor, Dict}: The loaded CP tensor and its metadata
"""
function load_cp_tensor_with_metadata(dir::AbstractString; T::Type=Float64)
    # Load the tensor
    tensor = load_cp_tensor(dir; T)

    # Load metadata
    metadata = npzread(joinpath(dir, "metadata.npy"))

    return tensor, metadata
end

"""
    load_observation_tensors(base_dir::String) -> Matrix{CPTensor}

Load all observation tensors from the decomposed directory structure.
Returns a matrix of CPTensor objects.

Parameters:
- base_dir: Base directory containing the observation tensor subdirectories

Returns:
- Matrix{CPTensor}: 7x7 matrix containing all observation tensors
"""
function load_cp_observation_tensors(base_dir::String)
    tensors = Matrix{CPTensor}(undef, 7, 7)

    for x in 1:7, y in 1:7
        dir_name = "observation_tensor_x$(x)_y$(y)"
        dir_path = joinpath(base_dir, dir_name)
        if isdir(dir_path)
            tensors[x, y] = load_cp_tensor(dir_path)
        else
            error("Directory not found: $dir_path")
        end
    end

    return tensors
end

"""
    load_observation_tensors_with_metadata(base_dir::String) -> Tuple{Matrix{CPTensor}, Matrix{Dict}}

Load all observation tensors and their metadata from the decomposed directory structure.
Returns a tuple containing:
1. Matrix of CPTensor objects
2. Matrix of metadata dictionaries

Parameters:
- base_dir: Base directory containing the observation tensor subdirectories

Returns:
- Tuple{Matrix{CPTensor}, Matrix{Dict}}: 7x7 matrices containing all observation tensors and their metadata
"""
function load_cp_observation_tensors_with_metadata(base_dir::String)
    tensors = Matrix{CPTensor}(undef, 7, 7)
    metadata = Matrix{Dict}(undef, 7, 7)

    for x in 1:7, y in 1:7
        dir_name = "observation_tensor_x$(x)_y$(y)"
        dir_path = joinpath(base_dir, dir_name)
        if isdir(dir_path)
            tensors[x, y], metadata[x, y] = load_cp_tensor_with_metadata(dir_path)
        else
            error("Directory not found: $dir_path")
        end
    end

    return tensors, metadata
end

