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

    # Add grid lines
    for i in 1:grid_size-1
        vline!(p, [i], color=:white, alpha=0.3, label=nothing)
        hline!(p, [i], color=:white, alpha=0.3, label=nothing)
    end

    # Add coordinate annotations
    for i in 1:grid_size, j in 1:grid_size
        annotate!(p, j - 0.5, i - 0.5, Plots.text("($(i),$(j))", :white, 8))
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
    # Create a layout with multiple subplots
    layout = @layout [
        grid(1, 2)     # Top row: Free energy and current location
        grid(1, 3)     # Middle row: Future locations t+1, t+2, t+3
        grid(1, 2)     # Bottom row: Orientation and key/door state
    ]

    plots = []

    # 1. Free Energy Plot
    p_fe = Plots.plot(inference_result.free_energy,
        xlabel="Iteration",
        ylabel="Free Energy",
        title="Free Energy Progression",
        legend=false,
        linewidth=2)
    push!(plots, p_fe)

    # 2. Current Location Belief
    current_loc = last(inference_result.posteriors[:current_location]).p
    p_loc = plot_belief_grid(current_loc, grid_size, title="Current Location Belief")
    push!(plots, p_loc)

    # 3. Future Location Beliefs (if available)
    future_states = [:s_future_1, :s_future_2, :s_future_3]
    for (i, state) in enumerate(future_states)
        if haskey(inference_result.posteriors, state)
            future_loc = last(inference_result.posteriors[state]).p
            p_future = plot_belief_grid(future_loc, grid_size, title="Location Belief t+$i")
            push!(plots, p_future)
        end
    end

    # 4. Orientation Belief
    orientation = last(inference_result.posteriors[:current_orientation]).p
    p_orient = bar(["→", "↓", "←", "↑"],
        orientation,
        title="Orientation Belief",
        legend=false,
        ylim=(0, 1))
    push!(plots, p_orient)

    # 5. Key/Door State Belief
    key_door = last(inference_result.posteriors[:current_key_door_state]).p
    p_state = bar(["No key", "Has key", "Door open"],
        key_door,
        title="Key/Door State Belief",
        legend=false,
        ylim=(0, 1))
    push!(plots, p_state)

    # Combine all plots
    final_plot = Plots.plot(plots...,
        layout=layout,
        size=(1200, 800),
        margin=5Plots.mm)

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
        current_loc = inference_result.posteriors[:current_location][i].p
        p2 = plot_belief_grid(current_loc, grid_size, title="Location Belief - Iteration $i")

        # Orientation belief
        orientation = inference_result.posteriors[:current_orientation][i].p
        p3 = bar(["→", "↓", "←", "↑"],
            orientation,
            title="Orientation - Iteration $i",
            legend=false,
            ylim=(0, 1))

        # Key/door state belief
        key_door = inference_result.posteriors[:current_key_door_state][i].p
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

Creates an animation showing how the agent's belief about its future trajectory evolves over time.
The animation shows a heatmap for each predicted future timestep from the final inference iteration.
"""
function animate_trajectory_belief(inference_result, grid_size; fps=2, save_path=nothing)
    # Get the final trajectory beliefs
    final_location_beliefs = last(inference_result.posteriors[:location])
    n_timesteps = length(final_location_beliefs)

    # Create animation
    anim = @animate for t in 1:n_timesteps
        # Get probability distribution for this timestep
        loc_probs = final_location_beliefs[t].p

        # Create heatmap
        p = plot_belief_grid(
            loc_probs,
            grid_size,
            title="Predicted Location t+$(t-1)"
        )

        Plots.plot!(p, size=(400, 400))
    end

    # Save if path provided
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
    end

    return anim
end
