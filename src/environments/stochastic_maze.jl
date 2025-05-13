using Distributions # For Categorical
using Plots

export StochasticMaze, StochasticMazeAgent, StochasticMazeAction
export step!, reset_maze!, create_stochastic_maze
export generate_maze_tensors, generate_goal_distributions

"""
    StochasticMazeAction

Represents an action in the stochastic maze.

# Fields
- `index::Int`: Action index (1=North, 2=East, 3=South, 4=West)
"""
struct StochasticMazeAction
    index::Int
end

"""
    StochasticMazeAgent

An agent that can move in a stochastic maze environment.

# Fields
- `state::Int`: Current state index
"""
mutable struct StochasticMazeAgent
    state::Int
end

"""
    StochasticMaze

Represents a stochastic maze environment where transitions are governed by probabilities.

# Fields
- `transition_tensor::Array{Float64,3}`: Transition probabilities (next_state × current_state × action)
- `observation_matrix::Matrix{Float64}`: Observation probabilities for each state
- `reward_states::Vector{Tuple{Int,Float64}}`: States and their reward values
- `agent_state::Int`: Current state of the agent
- `grid_size_x::Int`: Grid width
- `grid_size_y::Int`: Grid height
- `sink_states::Vector{Tuple{Int,Int}}`: States where agent can't move (x,y coordinates)
- `stochastic_states::Vector{Tuple{Int,Int}}`: States with randomized movement (x,y coordinates)
"""
mutable struct StochasticMaze
    transition_tensor::Array{Float64,3}
    observation_matrix::Matrix{Float64}
    reward_states::Vector{Tuple{Int,Float64}}
    agent_state::Int
    grid_size_x::Int
    grid_size_y::Int
    sink_states::Vector{Tuple{Int,Int}}
    stochastic_states::Vector{Tuple{Int,Int}}

    function StochasticMaze(
        transition_tensor::Array{Float64,3},
        observation_matrix::Matrix{Float64},
        reward_states::Vector{Tuple{Int,Float64}},
        agent_state::Int,
        grid_size_x::Int,
        grid_size_y::Int,
        sink_states::Vector{Tuple{Int,Int}}=Vector{Tuple{Int,Int}}(),
        stochastic_states::Vector{Tuple{Int,Int}}=Vector{Tuple{Int,Int}}()
    )
        # Validate transition tensor - each slice should be a valid probability distribution
        for a in 1:size(transition_tensor, 3)
            for s in 1:size(transition_tensor, 2)
                probs = transition_tensor[:, s, a]
                if !isapprox(sum(probs), 1.0, atol=1e-10) || any(x -> x < 0, probs)
                    throw(ArgumentError("Invalid transition probabilities for state $s, action $a"))
                end
            end
        end

        # Validate observation matrix - each column should be a valid probability distribution
        for s in 1:size(observation_matrix, 2)
            probs = observation_matrix[:, s]
            if !isapprox(sum(probs), 1.0, atol=1e-10) || any(x -> x < 0, probs)
                throw(ArgumentError("Invalid observation probabilities for state $s"))
            end
        end

        # Validate that agent_state is within bounds
        if agent_state < 1 || agent_state > size(transition_tensor, 2)
            throw(ArgumentError("Initial agent state must be between 1 and $(size(transition_tensor, 2))"))
        end

        new(transition_tensor, observation_matrix, reward_states, agent_state,
            grid_size_x, grid_size_y, sink_states, stochastic_states)
    end
end

"""
    create_stochastic_maze(grid_size_x::Int, grid_size_y::Int, n_actions::Int; 
                          sink_states=[(4, 2), (4, 4)],
                          stochastic_states=[(2, 3), (3, 3), (4, 3)],
                          noisy_observations=[(1, 5, 0.1), (2, 5, 0.1), ...],
                          start_state=1)

Create a stochastic maze environment with specified parameters.

# Arguments
- `grid_size_x::Int`: Width of the grid
- `grid_size_y::Int`: Height of the grid
- `n_actions::Int`: Number of actions (typically 4 for NESW)
- `sink_states::Vector{Tuple{Int,Int}}`: States where agent can't move
- `stochastic_states::Vector{Tuple{Int,Int}}`: States with randomized movement
- `noisy_observations::Vector{Tuple{Int,Int,Float64}}`: States with observation noise (x, y, noise_level)
- `start_state::Int`: Initial state of the agent

# Returns
- `StochasticMaze`: The created environment
"""
function create_stochastic_maze(
    grid_size_x::Int,
    grid_size_y::Int,
    n_actions::Int=4;
    sink_states::Vector{Tuple{Int,Int}}=[(4, 2), (4, 4)],
    stochastic_states::Vector{Tuple{Int,Int}}=[(2, 3), (3, 3), (4, 3)],
    noisy_observations::Vector{Tuple{Int,Int,Float64}}=[(1, 5, 0.1), (2, 5, 0.1), (3, 5, 0.4),
        (4, 5, 0.1), (2, 3, 0.4), (2, 4, 0.4),
        (3, 2, 0.4), (3, 3, 0.4), (4, 3, 0.2),
        (2, 2, 0.3), (3, 2, 0.4), (3, 4, 0.4)],
    start_state::Int=1
)
    # Generate the environment tensors
    observation_matrix, transition_tensor, reward_states = generate_maze_tensors(
        grid_size_x, grid_size_y, n_actions,
        sink_states=sink_states,
        stochastic_states=stochastic_states,
        noisy_observations=noisy_observations
    )

    # Create the environment
    env = StochasticMaze(
        transition_tensor,
        observation_matrix,
        reward_states,
        start_state,
        grid_size_x,
        grid_size_y,
        sink_states,
        stochastic_states
    )

    return env
end

"""
    step!(env::StochasticMaze, action::StochasticMazeAction)

Take a step in the environment with the given action.
Returns a tuple of (observation, reward).

# Arguments
- `env::StochasticMaze`: The environment
- `action::StochasticMazeAction`: The action to take

# Returns
- `Tuple{Int,Float64}`: The resulting observation and reward
"""
function step!(rng, env::StochasticMaze, action::StochasticMazeAction)
    # Get the current state
    current_state = env.agent_state

    # Sample the next state from the transition distribution
    probs = env.transition_tensor[:, current_state, action.index]
    next_state = rand(rng, Categorical(probs))

    # Update the agent's state
    env.agent_state = next_state

    # Sample an observation from the observation matrix
    probs = env.observation_matrix[:, next_state]
    observation = rand(rng, Categorical(probs))

    # Calculate reward
    reward = 0.0
    for (state, value) in env.reward_states
        if next_state == state
            reward = value
            break
        end
    end

    return observation, reward
end

"""
    reset_maze!(env::StochasticMaze, start_state::Int=1)

Reset the agent to the specified starting state.

# Arguments
- `env::StochasticMaze`: The environment
- `start_state::Int=1`: The state to reset to

# Returns
- `StochasticMaze`: The reset environment
"""
function reset_maze!(env::StochasticMaze, start_state::Int=1)
    if start_state < 1 || start_state > size(env.transition_tensor, 2)
        throw(ArgumentError("Start state must be between 1 and $(size(env.transition_tensor, 2))"))
    end

    env.agent_state = start_state
    return env
end

"""
    sample_observation(env::StochasticMaze)

Sample an observation from the current state.

# Arguments
- `env::StochasticMaze`: The environment

# Returns
- `Int`: The sampled observation
"""
function sample_observation(env::StochasticMaze)
    probs = env.observation_matrix[:, env.agent_state]
    return rand(Categorical(probs))
end

"""
    get_current_reward(env::StochasticMaze)

Get the reward for the current state.

# Arguments
- `env::StochasticMaze`: The environment

# Returns
- `Float64`: The reward value
"""
function get_current_reward(env::StochasticMaze)
    for (state, value) in env.reward_states
        if env.agent_state == state
            return value
        end
    end
    return 0.0
end

"""
    state_to_xy(state::Int, grid_size_x::Int)

Convert a linear state index to (x,y) coordinates.

# Arguments
- `state::Int`: The state index
- `grid_size_x::Int`: The width of the grid

# Returns
- `Tuple{Int,Int}`: The (x,y) coordinates
"""
function state_to_xy(state::Int, grid_size_x::Int)
    y = div(state - 1, grid_size_x) + 1
    x = mod(state - 1, grid_size_x) + 1
    return (x, y)
end

"""
    xy_to_state(x::Int, y::Int, grid_size_x::Int)

Convert (x,y) coordinates to a linear state index.

# Arguments
- `x::Int`: The x coordinate
- `y::Int`: The y coordinate
- `grid_size_x::Int`: The width of the grid

# Returns
- `Int`: The state index
"""
function xy_to_state(x::Int, y::Int, grid_size_x::Int)
    return x + (y - 1) * grid_size_x
end

"""
    generate_maze_tensors(grid_size_x::Int, grid_size_y::Int, n_actions::Int;
                         sink_states::Vector{Tuple{Int,Int}}=[(4, 2), (4, 4)],
                         stochastic_states::Vector{Tuple{Int,Int}}=[(2, 3), (3, 3), (4, 3)],
                         noisy_observations::Vector{Tuple{Int,Int,Float64}}=[(1, 5, 0.1), ...])

Generate the observation matrix, transition tensor, and reward states for a stochastic maze.

# Returns
- `Tuple{Matrix{Float64}, Array{Float64,3}, Vector{Tuple{Int,Float64}}}`: 
  The observation matrix, transition tensor, and reward states
"""
function generate_maze_tensors(grid_size_x::Int, grid_size_y::Int, n_actions::Int;
    sink_states::Vector{Tuple{Int,Int}}=[(4, 2), (4, 4)],
    stochastic_states::Vector{Tuple{Int,Int}}=[(2, 3), (3, 3), (4, 3)],
    noisy_observations::Vector{Tuple{Int,Int,Float64}}=[(1, 5, 0.1), (2, 5, 0.1), (3, 5, 0.4), (4, 5, 0.1), (2, 3, 0.4), (2, 4, 0.4), (3, 2, 0.4), (3, 3, 0.4), (4, 3, 0.2), (2, 2, 0.3), (3, 2, 0.4), (3, 4, 0.4),])

    # Create grid transition tensor 
    n_states = grid_size_x * grid_size_y

    # Initialize transition tensor B[s',s,a] - probability of going from s to s' given action a
    B = zeros(n_states, n_states, n_actions)

    # Helper function to convert (x,y) coordinates to state number
    coord_to_state(x, y) = x + (y - 1) * grid_size_x

    # For each state and action, fill in transition probabilities
    for y in 1:grid_size_y
        for x in 1:grid_size_x
            s = coord_to_state(x, y)

            # Skip sink states - all actions keep agent in same state
            if (x, y) in sink_states
                B[s, s, :] .= 1.0
                continue
            end

            # Stochastic states - random up/down movement
            if (x, y) in stochastic_states
                # 40% chance to move up, 40% chance to move down, 20% chance to follow action
                for a in 1:n_actions
                    if y > 1  # Can move up
                        B[coord_to_state(x, y - 1), s, a] = 0.4
                    end
                    if y < grid_size_y  # Can move down
                        B[coord_to_state(x, y + 1), s, a] = 0.4
                    end
                    B[coord_to_state(min(x + 1, grid_size_x), y), s, a] = 0.2
                end
                # Remaining 20% follows normal movement rules
                prob = 0.2
            else
                prob = 1.0
            end

            # Regular movement for non-special states
            # North
            if y < grid_size_y
                B[coord_to_state(x, y + 1), s, 1] = prob
            else
                B[s, s, 1] = prob  # Stay in same state if at boundary
            end

            # East
            if x < grid_size_x
                B[coord_to_state(x + 1, y), s, 2] = prob
            else
                B[s, s, 2] = prob
            end

            # South
            if y > 1
                B[coord_to_state(x, y - 1), s, 3] = prob
            else
                B[s, s, 3] = prob
            end

            # West
            if x > 1
                B[coord_to_state(x - 1, y), s, 4] = prob
            else
                B[s, s, 4] = prob
            end
        end
    end

    # Normalize tensor to ensure proper probability distributions
    for s in 1:n_states, a in 1:n_actions
        total = sum(B[:, s, a])
        if total > 0
            B[:, s, a] ./= total
        end
    end

    # Initialize observation matrix A
    A = zeros(n_states, n_states)
    state_to_coord(s) = (mod(s - 1, grid_size_x) + 1, div(s - 1, grid_size_x) + 1)

    # Fill observation matrix with default perfect observations
    for s in 1:n_states
        A[s, s] = 1.0
    end

    # Add observation noise to specified states
    for (x, y, noise_level) in noisy_observations
        s = coord_to_state(x, y)
        # Reduce the perfect observation probability
        A[s, s] = 1.0 - noise_level

        # Distribute remaining probability to adjacent states
        adjacent_states = Int[]

        # Check all adjacent states including diagonals
        if y > 1
            push!(adjacent_states, coord_to_state(x, y - 1))  # South
            if x > 1
                push!(adjacent_states, coord_to_state(x - 1, y - 1)) # Southwest
            end
            if x < grid_size_x
                push!(adjacent_states, coord_to_state(x + 1, y - 1)) # Southeast
            end
        end
        if y < grid_size_y
            push!(adjacent_states, coord_to_state(x, y + 1))  # North
            if x > 1
                push!(adjacent_states, coord_to_state(x - 1, y + 1)) # Northwest
            end
            if x < grid_size_x
                push!(adjacent_states, coord_to_state(x + 1, y + 1)) # Northeast
            end
        end
        if x > 1
            push!(adjacent_states, coord_to_state(x - 1, y))  # West
        end
        if x < grid_size_x
            push!(adjacent_states, coord_to_state(x + 1, y))  # East
        end

        # Distribute remaining probability equally among adjacent states
        if !isempty(adjacent_states)
            prob_per_adjacent = noise_level / length(adjacent_states)
            for adj_s in adjacent_states
                A[adj_s, s] = prob_per_adjacent
            end
        end
    end

    # Normalize matrix to ensure proper probability distributions
    for s in 1:n_states
        total = sum(A[:, s])
        if total > 0
            A[:, s] ./= total
        end
    end

    # Define reward states
    reward_states = [(9, -1.0), (19, -1.0), (15, 1.0)]

    return A, B, reward_states
end

"""
    generate_goal_distributions(n_states::Int, goal_state::Int, T::Int)

Generate a sequence of categorical distributions that increasingly concentrate 
probability on the goal state as T increases.

# Arguments
- `n_states::Int`: Total number of states
- `goal_state::Int`: The goal state index
- `T::Int`: Time horizon / number of distributions to generate

# Returns
- `Vector{Categorical}`: A vector of T categorical distributions
"""
function generate_goal_distributions(n_states::Int, goal_state::Int, T::Int)
    # Initialize array to store T categorical distributions
    distributions = Vector{Categorical}(undef, T)

    # Create base probability vector
    base_prob = ones(n_states) / n_states

    # Create goal probability vector 
    goal_prob = zeros(n_states)
    goal_prob[goal_state] = 1.0

    # Generate distributions with increasing entropy
    for t in T:-1:1
        # Calculate mixing weight that increases exponentially over time
        α = (exp(10 * t / T) - 1) / (exp(10) - 1)  # Exponential scaling from 0 to 1

        # Mix base and goal probabilities
        mixed_prob = α * goal_prob + (1 - α) * base_prob

        # Create categorical distribution
        distributions[t] = Categorical(mixed_prob ./ sum(mixed_prob))
    end

    return distributions
end

"""
    visualize_stochastic_maze(env::StochasticMaze; show_legend::Bool=false, backend=nothing)

Visualize the StochasticMaze environment using Plots.jl.

# Arguments
- `env::StochasticMaze`: The environment to visualize
- `show_legend::Bool=false`: Whether to show the legend explaining the colors
- `backend`: Optional specification of the backend (e.g., :gr or :pgfplotsx)

# Returns
- `Plots.Plot`: A plot of the maze
"""
function visualize_stochastic_maze(env::StochasticMaze; show_legend::Bool=false, backend=nothing)
    # Switch to specified backend if provided
    if !isnothing(backend)
        if backend == :pgfplotsx
            pgfplotsx()
        elseif backend == :gr
            gr()
        end
    end

    # Get grid dimensions
    grid_size_x = env.grid_size_x
    grid_size_y = env.grid_size_y

    # Different plot settings based on backend
    current_backend = Plots.backend_name()

    # Create a new plot with appropriate size and aspect ratio
    p = plot(
        size=(600, 600),  # Always use square format
        aspect_ratio=:equal,
        xlim=(0, grid_size_x),
        ylim=(0, grid_size_y),
        legend=show_legend,      # Show legend only if requested
        grid=false,
        ticks=false,
        border=:none,
        background=MAZE_THEME.background,
        fontfamily="Computer Modern"  # More paper-like font
    )

    # Use same enhanced scale regardless of legend
    scale = 18

    # Draw border walls with increased linewidth for better visibility
    # Left wall
    plot!(p, [0, 0], [0, grid_size_y], color=MAZE_THEME.wall, linewidth=3, label=nothing)
    # Right wall  
    plot!(p, [grid_size_x, grid_size_x], [0, grid_size_y], color=MAZE_THEME.wall, linewidth=3, label=nothing)
    # Bottom wall
    plot!(p, [0, grid_size_x], [0, 0], color=MAZE_THEME.wall, linewidth=3, label=nothing)
    # Top wall
    plot!(p, [0, grid_size_x], [grid_size_y, grid_size_y], color=MAZE_THEME.wall, linewidth=3, label=nothing)

    # Draw grid lines
    for x in 0:grid_size_x
        plot!(p, [x, x], [0, grid_size_y], color=MAZE_THEME.wall, linewidth=0.7, alpha=0.7, label=nothing)
    end
    for y in 0:grid_size_y
        plot!(p, [0, grid_size_x], [y, y], color=MAZE_THEME.wall, linewidth=0.7, alpha=0.7, label=nothing)
    end

    # Plot sink states
    has_sink = false
    for (x, y) in env.sink_states
        # Plot a filled rectangle for sink states
        x_coords = [x - 1, x, x, x - 1, x - 1]
        y_coords = [grid_size_y - y, grid_size_y - y, grid_size_y - y + 1, grid_size_y - y + 1, grid_size_y - y]
        if show_legend && !has_sink
            plot!(p, x_coords, y_coords, color=MAZE_THEME.sink, alpha=0.6, fill=true, label="Sink state")
            has_sink = true
        else
            plot!(p, x_coords, y_coords, color=MAZE_THEME.sink, alpha=0.6, fill=true, label=nothing)
        end
    end

    # Plot reward states
    has_positive = false
    has_negative = false
    for (state, reward) in env.reward_states
        x, y = state_to_xy(state, grid_size_x)
        color = reward > 0 ? MAZE_THEME.reward_positive : MAZE_THEME.reward_negative
        opacity = min(abs(reward), 1.0) # Use absolute value of reward for opacity, capped at 1.0

        # Add to legend only for the first instance of each reward type if showing legend
        if show_legend
            if reward > 0 && !has_positive
                scatter!(p, [x - 0.5], [grid_size_y - y + 0.5], color=color, alpha=opacity,
                    markersize=ceil(Int, scale), markerstrokewidth=ceil(Int, scale / 12),
                    label="Positive reward")
                has_positive = true
            elseif reward < 0 && !has_negative
                scatter!(p, [x - 0.5], [grid_size_y - y + 0.5], color=color, alpha=opacity,
                    markersize=ceil(Int, scale), markerstrokewidth=ceil(Int, scale / 12),
                    label="Negative reward")
                has_negative = true
            else
                scatter!(p, [x - 0.5], [grid_size_y - y + 0.5], color=color, alpha=opacity,
                    markersize=ceil(Int, scale), markerstrokewidth=ceil(Int, scale / 12),
                    label=nothing)
            end
        else
            scatter!(p, [x - 0.5], [grid_size_y - y + 0.5], color=color, alpha=opacity,
                markersize=ceil(Int, scale), markerstrokewidth=ceil(Int, scale / 12),
                label=nothing)
        end
    end

    has_noisy = false
    # Plot observation noise
    for s in 1:size(env.observation_matrix, 2)
        if env.observation_matrix[s, s] != 1.0
            x, y = state_to_xy(s, grid_size_x)
            noise = 1.0 - env.observation_matrix[s, s]
            # Plot a filled rectangle for noisy states
            x_coords = [x - 1, x, x, x - 1, x - 1]
            y_coords = [grid_size_y - y, grid_size_y - y, grid_size_y - y + 1, grid_size_y - y + 1, grid_size_y - y]
            if show_legend && !has_noisy
                plot!(p, x_coords, y_coords, color=MAZE_THEME.noisy, alpha=noise, fill=true, label="Observation noise")
                has_noisy = true
            else
                plot!(p, x_coords, y_coords, color=MAZE_THEME.noisy, alpha=noise, fill=true, label=nothing)
            end
        end
    end

    # Plot stochastic states (bridge effect)
    has_stochastic = false
    for (x, y) in env.stochastic_states
        # Draw 3 horizontal planks
        for i in 0:2
            x_coords = [x - 1, x, x, x - 1, x - 1]
            y_coords = [
                grid_size_y - y + 0.25 + i * 0.25,
                grid_size_y - y + 0.25 + i * 0.25,
                grid_size_y - y + 0.32 + i * 0.25,
                grid_size_y - y + 0.32 + i * 0.25,
                grid_size_y - y + 0.25 + i * 0.25
            ]

            # Add to legend only for the first plank of the first stochastic state if showing legend
            if show_legend && !has_stochastic && i == 0
                plot!(p, x_coords, y_coords, color=MAZE_THEME.stochastic, alpha=0.6, fill=true,
                    label="Stochastic transition", linewidth=2, markerstrokewidth=2)
                has_stochastic = true
            else
                plot!(p, x_coords, y_coords, color=MAZE_THEME.stochastic, alpha=0.6, fill=true,
                    label=nothing)
            end
        end
    end

    # Plot agent
    x, y = state_to_xy(env.agent_state, grid_size_x)
    scatter!(p, [x - 0.5], [grid_size_y - y + 0.5], color=MAZE_THEME.agent,
        markersize=ceil(Int, (3 / 4) * scale), markerstrokewidth=ceil(Int, scale / 12),
        label=show_legend ? "Agent position" : nothing)

    # Configure legend if showing
    if show_legend
        # Different legend settings based on backend
        if Symbol(current_backend) == :pgfplotsx || Symbol(current_backend) == :pgfplots
            # For PGFPlotsX - position legend to the right
            plot!(p, legend=:outerright, legendfontsize=16, legendtitle="Maze Elements",
                legendtitlefontsize=18, legendtitlealign=:center,
                margin=10Plots.mm, widen=true, foreground_color_legend=:black,
                background_color_legend=:white, framestyle=:box,
                legendmarkersize=16,
                legend_column_gap=10,  # Add gap between marker and text
                legend_cell_align=:left,
                legend_hfactor=1.2,    # Horizontal expansion factor
                legend_vfactor=1.2,    # Vertical expansion factor  
                legend_margin=8        # Margin around the entire legend
            )
        else
            # For other backends like GR
            plot!(p, legend=:outerright, legendfontsize=18, legendtitle="Maze Elements",
                legendtitlefontsize=20, legendtitlealign=:center,
                margin=10Plots.mm, widen=true, foreground_color_legend=:black,
                background_color_legend=:white, framestyle=:box,
                legendmarkersize=18)
        end
    end

    return p
end