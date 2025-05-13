using Plots
using FileIO

export save_frame

"""
    convert_keys_to_symbols(dict::Dict)

Recursively convert all string keys in a dictionary (and nested dictionaries) to symbols.
"""
function convert_keys_to_symbols(dict::Dict)
    result = Dict{Symbol,Any}()
    for (k, v) in dict
        key = k isa String ? Symbol(k) : k
        if v isa Dict
            result[key] = convert_keys_to_symbols(v)
        else
            result[key] = v
        end
    end
    return result
end

"""
    save_frame(plot, model_name, seed, timestep, output_dir; use_tikz=false, extra_kwargs=nothing)

Save a single frame as a PNG or TikZ file with a specific naming convention to preserve order.

# Arguments
- `plot`: The Plots.jl plot object to save
- `model_name::String`: Name of the model for file naming
- `seed::Int`: Random seed for the episode for file naming
- `timestep::Int`: Current timestep (will be formatted with leading zeros)
- `output_dir::String`: Directory to save the frame in
- `use_tikz::Bool`: Whether to save as a TikZ file instead of PNG
- `extra_kwargs`: Optional backend-specific additional arguments to apply to the plot 
  before saving, useful for configuring PGFPlotsX options

# Returns
- `filepath::String`: The full path to the saved file
"""
function save_frame(plot, model_name, seed, timestep, output_dir; use_tikz=false, extra_kwargs=nothing)
    # Format the timestep with leading zeros for proper sorting
    timestep_str = lpad(timestep, 3, "0")

    # Determine file extension based on use_tikz flag
    extension = use_tikz ? "tikz" : "png"

    # Create the filename with format model_name_episode_seed_frame_NNN.(png|tikz)
    filename = "$(model_name)_episode_$(seed)_frame_$(timestep_str).$(extension)"

    # Full path to save the plot
    filepath = joinpath(output_dir, filename)

    # Apply any extra kwargs to the plot
    if !isnothing(extra_kwargs)
        # Make a copy of the plot to avoid modifying the original
        plot_to_save = deepcopy(plot)

        # Apply extra kwargs to the plot
        for (key, value) in pairs(extra_kwargs)
            if key == :plot
                if value isa Dict
                    # Convert string keys to symbols
                    symbol_dict = convert_keys_to_symbols(value)
                    plot!(plot_to_save; symbol_dict...)
                else
                    plot!(plot_to_save; value...)
                end
            elseif key == :subplot
                if value isa Dict
                    symbol_dict = convert_keys_to_symbols(value)
                    plot!(plot_to_save, extra_kwargs=(:subplot, symbol_dict))
                else
                    plot!(plot_to_save, extra_kwargs=(:subplot, value))
                end
            elseif key == :series
                if value isa Dict
                    symbol_dict = convert_keys_to_symbols(value)
                    plot!(plot_to_save, extra_kwargs=(:series, symbol_dict))
                else
                    plot!(plot_to_save, extra_kwargs=(:series, value))
                end
            end
        end

        # Save the modified plot
        Plots.savefig(plot_to_save, filepath)
    else
        # Save the original plot
        Plots.savefig(plot, filepath)
    end

    return filepath
end