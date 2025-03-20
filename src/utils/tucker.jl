using NPZ

struct TuckerTensor{T,N}
    core::Array{T,N}
    factors::Vector{Array{T,2}}
end

"""
    load_tucker_tensor(dir::String) -> TuckerTensor

Load a Tucker decomposed tensor from a directory containing core.npy and factor_X.npy files.
"""
function load_tucker_tensor(dir::String)
    core = npzread(dir * "/core.npy")

    # Load all factor matrices
    factors = Matrix{Float64}[]
    i = 0
    while isfile(dir * "/factor_$(i).npy")
        push!(factors, npzread(dir * "/factor_$(i).npy"))
        i += 1
    end

    return TuckerTensor(core, factors)
end

"""
    reconstruct_tensor(tucker::TuckerTensor) -> Array

Reconstruct the full tensor from its Tucker decomposition.
"""
function reconstruct_tensor(tucker::TuckerTensor)
    # Start with the core tensor
    result = tucker.core

    # Apply each factor matrix in sequence
    for factor in tucker.factors
        result = tensorcontract(result, factor, (1, 1))
    end

    return result
end

"""
    load_observation_tensors(base_dir::String) -> Matrix{TuckerTensor}

Load all observation tensors from the decomposed directory structure.
Returns a matrix of TuckerTensor objects.
"""
function load_observation_tensors(base_dir::String)
    tensors = Matrix{TuckerTensor}(undef, 7, 7)

    for x in 1:7, y in 1:7
        dir_name = "observation_tensor_x$(x)_y$(y)"
        if isdir(base_dir * dir_name)
            tensors[x, y] = load_tucker_tensor(base_dir * dir_name)
        else
            error("Directory not found: $(base_dir * dir_name)")
        end
    end

    return tensors
end


