using Plots
using LinearAlgebra
using Colors

export plot_belief_grid, plot_inference_results, animate_belief_evolution, animate_trajectory_belief

"""
    plot_belief_grid(belief_vector, grid_size; kwargs...)

Plot a belief vector as a heatmap over a grid.

# Arguments
- `belief_vector`: Vector of probabilities
- `grid_size`: Size of the grid (grid_size × grid_size)
- `kwargs`: Additional arguments passed to heatmap

# Returns
- A Plots.jl plot object
"""
function plot_belief_grid(belief_vector, grid_size; title="Belief Grid", kwargs...)
    belief_matrix = reshape(belief_vector, grid_size, grid_size)
    p = Plots.heatmap(belief_matrix,
        title=title,
        color=:viridis,
        aspect_ratio=1,
        clim=(0, 1);
        kwargs...)

    # Add grid lines at integer positions
    for i in 1:grid_size-1
        vline!(p, [i], color=:white, alpha=0.3, label=nothing)
        hline!(p, [i], color=:white, alpha=0.3, label=nothing)
    end

    # Add coordinate annotations at cell centers
    for i in 1:grid_size, j in 1:grid_size
        # Use i,j directly for cell centers
        annotate!(p, j, i, Plots.text("($(i),$(j))", :white, 8))
    end

    return p
end

"""
    plot_inference_results(inference_result, grid_size; save_path=nothing)

Create a comprehensive visualization of inference results.

# Arguments
- `inference_result`: Result from inference containing posteriors and free energy
- `grid_size`: Size of the environment grid
- `save_path`: Optional path to save the plots

# Returns
- A Plots.jl plot object
"""
function plot_inference_results(inference_result, grid_size; save_path=nothing)
    # Change to the same 2×2 layout used in belief_evolution
    layout = @layout [
        grid(1, 2)  # Top row: Free energy and location belief
        grid(1, 2)  # Bottom row: Orientation and key/door state
    ]

    # 1. Free Energy Plot
    p1 = Plots.plot(inference_result.free_energy,
        xlabel="Iteration",
        ylabel="Free Energy",
        title="Free Energy Progression",
        legend=false,
        linewidth=2)

    # 2. Current Location Belief
    current_loc = probvec(last(inference_result.posteriors[:current_location]))
    p2 = plot_belief_grid(current_loc, grid_size, title="Current Location Belief")

    # 3. Orientation Belief
    orientation = probvec(last(inference_result.posteriors[:current_orientation]))
    p3 = bar(["→", "↓", "←", "↑"],
        orientation,
        title="Orientation Belief",
        legend=false,
        ylim=(0, 1))

    # 4. Key/Door State Belief
    key_door = probvec(last(inference_result.posteriors[:current_key_door_state]))
    p4 = bar(["No key", "Has key", "Door open"],
        key_door,
        title="Key/Door State Belief",
        legend=false,
        ylim=(0, 1))

    # Combine plots using the same approach as belief_evolution
    final_plot = Plots.plot(p1, p2, p3, p4,
        layout=layout,
        size=(1000, 800))

    # Save if path provided
    if !isnothing(save_path)
        savefig(final_plot, save_path)
    end

    return final_plot
end

"""
    animate_belief_evolution(inference_result, grid_size; fps=2, save_path=nothing)

Create an animation showing how beliefs evolve during inference.

# Arguments
- `inference_result`: Result from inference containing posteriors
- `grid_size`: Size of the environment grid
- `fps`: Frames per second for the animation
- `save_path`: Optional path to save the animation

# Returns
- A Plots.jl animation object
"""
function animate_belief_evolution(inference_result, grid_size; fps=2, save_path=nothing)
    n_iterations = length(inference_result.free_energy)

    anim = @animate for i in 1:n_iterations
        # Create subplot layout
        l = @layout [
            grid(1, 2)  # Free energy and location belief
            grid(1, 2)  # Orientation and key/door state
        ]

        # Free energy up to current iteration
        p1 = Plots.plot(inference_result.free_energy[1:i],
            xlabel="Iteration",
            ylabel="Free Energy",
            title="Free Energy - Iteration $i",
            legend=false)

        # Current location belief
        current_loc = probvec(inference_result.posteriors[:current_location][i])
        p2 = plot_belief_grid(current_loc, grid_size, title="Location Belief - Iteration $i")

        # Orientation belief
        orientation = probvec(inference_result.posteriors[:current_orientation][i])
        p3 = bar(["→", "↓", "←", "↑"],
            orientation,
            title="Orientation - Iteration $i",
            legend=false,
            ylim=(0, 1))

        # Key/door state belief
        key_door = probvec(inference_result.posteriors[:current_key_door_state][i])
        p4 = bar(["No key", "Has key", "Door open"],
            key_door,
            title="Key/Door State - Iteration $i",
            legend=false,
            ylim=(0, 1))

        Plots.plot(p1, p2, p3, p4, layout=l, size=(1000, 800))
    end

    # Save if path provided
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
    end

    return anim
end


"""
    animate_trajectory_belief(inference_result, grid_size; fps=2, save_path=nothing)

Creates an animation showing how the agent's beliefs about its future trajectory evolves over time,
including location, orientation, key/door state, and action probabilities for each timestep.
"""
function animate_trajectory_belief(inference_result, grid_size; fps=2, save_path=nothing)
    # Get the final trajectory beliefs
    final_location_beliefs = last(inference_result.posteriors[:location])
    final_orientation_beliefs = last(inference_result.posteriors[:orientation])
    final_key_door_beliefs = last(inference_result.posteriors[:key_door_state])
    action_probabilities = last(inference_result.posteriors[:u])

    n_timesteps = length(final_location_beliefs)

    # Define labels for different states
    action_names = ["Turn Left", "Turn Right", "Forward", "Pickup", "Open Door"]
    orientation_names = ["→", "↓", "←", "↑"]
    key_door_names = ["No key", "Has key", "Door open"]

    # Create animation
    anim = @animate for t in 1:n_timesteps
        # Create a 2×2 layout
        l = @layout [
            grid(1, 2)  # Top row: Location and Orientation
            grid(1, 2)  # Bottom row: Key/Door State and Actions
        ]

        # Top-left: Location belief
        p1 = plot_belief_grid(
            probvec(final_location_beliefs[t]),
            grid_size,
            title="Location t+$(t-1)"
        )

        # Top-right: Orientation belief
        p2 = bar(orientation_names,
            probvec(final_orientation_beliefs[t]),
            title="Orientation t+$(t-1)",
            legend=false,
            ylim=(0, 1))

        # Bottom-left: Key/Door state
        p3 = bar(key_door_names,
            probvec(final_key_door_beliefs[t]),
            title="Key/Door State t+$(t-1)",
            legend=false,
            ylim=(0, 1))

        # Bottom-right: Action probabilities
        p4 = bar(action_names,
            probvec(action_probabilities[t]),
            title="Action Probabilities t+$(t-1)",
            rotation=45,  # Rotate x-axis labels
            legend=false,
            ylim=(0, 1))

        # Combine all plots
        Plots.plot(p1, p2, p3, p4,
            layout=l,
            size=(1000, 800),
            plot_title="Belief Trajectory - Timestep $(t-1)"
        )
    end

    # Save if path provided
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
    end

    return anim
end