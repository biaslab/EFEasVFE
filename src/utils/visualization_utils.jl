using Plots
using FileIO

export save_frame

"""
    save_frame(plot, model_name, seed, timestep, output_dir)

Save a single frame as a PNG file with a specific naming convention to preserve order.

# Arguments
- `plot`: The Plots.jl plot object to save
- `model_name::String`: Name of the model for file naming
- `seed::Int`: Random seed for the episode for file naming
- `timestep::Int`: Current timestep (will be formatted with leading zeros)
- `output_dir::String`: Directory to save the frame in

# Returns
- `filepath::String`: The full path to the saved file
"""
function save_frame(plot, model_name, seed, timestep, output_dir)
    # Format the timestep with leading zeros for proper sorting
    timestep_str = lpad(timestep, 3, "0")

    # Create the filename with format model_name_episode_seed_frame_NNN.png
    filename = "$(model_name)_episode_$(seed)_frame_$(timestep_str).png"

    # Full path to save the plot
    filepath = joinpath(output_dir, filename)

    # Save the plot as PNG
    Plots.savefig(plot, filepath)

    return filepath
end