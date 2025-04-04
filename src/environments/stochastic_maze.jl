using RxEnvironments
using GLMakie
using Distributions # For Categorical
import Distributions: Categorical

export StochasticMaze, StochasticMazeAgent, StochasticMazeAction
export move!, sample_observation, send_observation_and_reward
export add_agent!, plot_maze_state, reset_agent!, reset!

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
- `transition_tensor::Array{Float64,3}`: Transition probabilities (out × in × action)
- `observation_matrix::Matrix{Float64}`: Observation probabilities for each state
- `reward_states::Vector{Tuple{Int,Float64}}`: States and their reward values
- `agents::Vector{StochasticMazeAgent}`: List of agents in the maze
"""
struct StochasticMaze
    transition_tensor::Array{Float64,3}
    observation_matrix::Matrix{Float64}
    reward_states::Vector{Tuple{Int,Float64}}
    agents::Vector{StochasticMazeAgent}
end

"""
    StochasticMaze(transition_tensor, observation_matrix, reward_states)

Construct a StochasticMaze with specified transition probabilities, observation matrix and reward states.
"""
function StochasticMaze(transition_tensor::Array{Float64,3}, observation_matrix::Matrix{Float64}, reward_states::Vector{Tuple{Int,Float64}})
    # Validate transition tensor - each slice should be a valid probability distribution
    for a in 1:size(transition_tensor, 3)
        for s in 1:size(transition_tensor, 2)  # Changed from 1 to 2 for input state dimension
            probs = transition_tensor[:, s, a]  # Changed from [s, :, a] to [:, s, a]
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

    StochasticMaze(transition_tensor, observation_matrix, reward_states, StochasticMazeAgent[])
end

"""
    StochasticMazeAction

Represents an action in the stochastic maze.

# Fields
- `index::Int`: Action index
"""
struct StochasticMazeAction
    index::Int
end

"""
Move agent to new state based on action by sampling from transition probabilities.
"""
function move!(env::StochasticMaze, agent::StochasticMazeAgent, action::StochasticMazeAction)
    current_state = agent.state
    probs = env.transition_tensor[:, current_state, action.index]
    next_state = rand(Categorical(probs))
    agent.state = next_state
    return next_state
end

# Environment update does nothing
RxEnvironments.update!(::StochasticMaze, dt) = nothing

# Receive action and move agent
RxEnvironments.receive!(maze::StochasticMaze, agent::StochasticMazeAgent, action::StochasticMazeAction) = move!(maze, agent, action)

"""
Sample an observation from the maze's observation matrix at given state.
"""
function sample_observation(env::StochasticMaze, agent::StochasticMazeAgent)
    probs = env.observation_matrix[:, agent.state]
    return rand(Categorical(probs))
end

"""
Get observation and rewards for current state.
"""
function RxEnvironments.what_to_send(agent::StochasticMazeAgent, maze::StochasticMaze, action::StochasticMazeAction)
    # Find if current state has a reward
    reward = nothing
    for (state, value) in maze.reward_states
        if agent.state == state
            reward = value
            break
        end
    end

    observation = sample_observation(maze, agent)
    return (observation, reward)
end

"""
Add agent to environment.
"""
function add_agent!(env::StochasticMaze, initial_state::Int)
    if initial_state < 1 || initial_state > size(env.transition_tensor, 2)
        error("Initial state must be between 1 and $(size(env.transition_tensor, 2))")
    end
    agent = StochasticMazeAgent(initial_state)
    push!(env.agents, agent)
    return agent
end

"""
Reset agent to starting state.
"""
function reset_agent!(agent::StochasticMazeAgent, initial_state::Int)
    agent.state = initial_state
end

"""
Add agent to environment.
"""
function RxEnvironments.add_to_state!(env::StochasticMaze, agent::StochasticMazeAgent)
    push!(env.agents, agent)
end

# Send observation and reward to the agent
function send_observation_and_reward(env::StochasticMaze, agent::StochasticMazeAgent)
    observation = sample_observation(env, agent)
    reward = get_reward(env, agent)
    return observation, reward
end

# Get reward based on current state
function get_reward(env::StochasticMaze, agent::StochasticMazeAgent)
    return agent.state in env.reward_states ? 1.0 : 0.0
end

# Helper function for self-loop visualization
function draw_self_loop!(ax, pos, radius)
    # Draw a circular arrow that loops back to the same position
    θ = range(0, 3π / 2, length=50)  # Leave a gap for arrow head
    center = (pos[1] - 0.1, pos[2] + 0.1)
    circle = [(center[1] + radius * cos(t), center[2] + radius * sin(t)) for t in θ]
    lines!(ax, first.(circle), last.(circle), color=(:black, 0.5))

    # Add arrow head
    arrow_pos = (center[1] + radius * cos(3π / 2), center[2] + radius * sin(3π / 2))
    arrow_dir = (-radius * sin(3π / 2), radius * cos(3π / 2))
    arrows!(ax, [arrow_pos[1]], [arrow_pos[2]],
        [arrow_dir[1] * 0.2], [arrow_dir[2] * 0.2],
        color=(:black, 0.5), arrowsize=10)
end

"""
Reset agent to starting position.
"""
function reset!(env::RxEnvironments.RxEntity{StochasticMaze}, start_pos::Int=1)
    reset_agent!(env.decorated.agents[1], start_pos)
end

function create_environment(::Type{StochasticMaze}, transition_tensor::Array{Float64,3}, observation_matrix::Matrix{Float64}, reward_states::Vector{Tuple{Int,Float64}})
    env = StochasticMaze(transition_tensor, observation_matrix, reward_states)
    rxe = create_entity(env; is_active=true)
    return rxe
end

function RxEnvironments.plot_state(ax, env::StochasticMaze)
    # Get grid dimensions from environment
    grid_size_x = Int(sqrt(size(env.transition_tensor, 1)))
    grid_size_y = grid_size_x

    # Create grid lines
    for x in 0:grid_size_x
        vlines!(ax, x, 0, grid_size_y, color=:black, linewidth=0.5)
    end
    for y in 0:grid_size_y
        hlines!(ax, y, 0, grid_size_x, color=:black, linewidth=0.5)
    end

    # Plot sink states in red
    # Find sink states by checking where all actions point to same state
    sink_states = []
    for s in 1:size(env.transition_tensor, 1)
        if all(env.transition_tensor[s, s, :] .== 1.0)
            x = ((s - 1) % grid_size_x) + 1
            y = div(s - 1, grid_size_x) + 1
            push!(sink_states, (x, y))
        end
    end
    for (x, y) in sink_states
        poly!(ax,
            [Point2f(x - 1, grid_size_y - y), Point2f(x, grid_size_y - y),
                Point2f(x, grid_size_y - y + 1), Point2f(x - 1, grid_size_y - y + 1)],
            color=(:red, 0.3))
    end

    # Plot reward states
    for (state, reward) in env.reward_states
        x = ((state - 1) % grid_size_x) + 1
        y = div(state - 1, grid_size_x) + 1
        color = reward > 0 ? :green : :red
        opacity = abs(reward) # Use absolute value of reward for opacity
        scatter!(ax, [x - 0.5], [grid_size_y - y + 0.5], color=(color, opacity), markersize=20)
    end

    # Plot observation noise
    noisy_obs = []
    for s in 1:size(env.observation_matrix, 1)
        if env.observation_matrix[s, s] != 1.0
            x = ((s - 1) % grid_size_x) + 1
            y = div(s - 1, grid_size_x) + 1
            noise = 1.0 - env.observation_matrix[s, s]
            push!(noisy_obs, (x, y, noise))
        end
    end

    for (x, y, noise) in noisy_obs
        poly!(ax,
            [Point2f(x - 1, grid_size_y - y), Point2f(x, grid_size_y - y),
                Point2f(x, grid_size_y - y + 1), Point2f(x - 1, grid_size_y - y + 1)],
            color=(:lightblue, noise))
    end

    # Plot stochastic states (bridge effect)
    stochastic_states = []
    for s in 1:size(env.transition_tensor, 1)
        # Check if any action has non-1.0 probability for intended direction
        if any(maximum(env.transition_tensor[:, s, a]) < 0.99 for a in 1:size(env.transition_tensor, 3))
            x = ((s - 1) % grid_size_x) + 1
            y = div(s - 1, grid_size_x) + 1
            push!(stochastic_states, (x, y))
        end
    end

    # Draw bridge planks in brown with gaps
    for (x, y) in stochastic_states
        # Draw 3 horizontal planks
        for i in 0:2
            poly!(ax,
                [Point2f(x - 1, grid_size_y - y + 0.25 + i * 0.25),
                    Point2f(x, grid_size_y - y + 0.25 + i * 0.25),
                    Point2f(x, grid_size_y - y + 0.32 + i * 0.25),
                    Point2f(x - 1, grid_size_y - y + 0.32 + i * 0.25)],
                color=(:brown, 0.6))
        end
    end

    for agent in env.agents
        x = ((agent.state - 1) % grid_size_x) + 1
        y = div(agent.state - 1, grid_size_x) + 1
        scatter!(ax, [x - 0.5], [grid_size_y - y + 0.5], color=:blue, markersize=20)
    end

    # Set proper axis limits and remove ticks
    limits!(ax, 0, grid_size_x, 0, grid_size_y)
    hidedecorations!(ax)
end


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
                    B[coord_to_state(x + 1, y), s, a] = 0.2
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
        B ./= sum(B, dims=1)
    end

    # Initialize observation matrix A
    A = zeros(n_states, n_states)
    state_to_coord(s) = (mod(s - 1, grid_size_x) + 1, div(s - 1, grid_size_x) + 1)

    # Fill observation matrix
    for s in 1:n_states
        # Get x,y coordinates of current state
        x, y = state_to_coord(s)

        # Default to perfect observations
        correct_prob = 1.0

        # Check if current state is a noisy observation point
        if (x, y) in [(nx, ny) for (nx, ny, _) in noisy_observations]
            # Find the noise level for this state
            _, _, noise_level = first(filter(p -> p[1] == x && p[2] == y, noisy_observations))
            correct_prob = 1.0 - noise_level
        end

        # Add probability of correct observation
        A[s, s] = correct_prob

        # Distribute remaining probability to adjacent states
        remaining_prob = 1.0 - correct_prob
        adjacent_states = Int[]

        # Check all adjacent states including diagonals
        if y > 1
            push!(adjacent_states, coord_to_state(x, y - 1))  # Up
            if x > 1
                push!(adjacent_states, coord_to_state(x - 1, y - 1)) # Up-Left
            end
            if x < grid_size_x
                push!(adjacent_states, coord_to_state(x + 1, y - 1)) # Up-Right
            end
        end
        if y < grid_size_y
            push!(adjacent_states, coord_to_state(x, y + 1))  # Down
            if x > 1
                push!(adjacent_states, coord_to_state(x - 1, y + 1)) # Down-Left
            end
            if x < grid_size_x
                push!(adjacent_states, coord_to_state(x + 1, y + 1)) # Down-Right
            end
        end
        if x > 1
            push!(adjacent_states, coord_to_state(x - 1, y))  # Left
        end
        if x < grid_size_x
            push!(adjacent_states, coord_to_state(x + 1, y))  # Right
        end

        # Distribute remaining probability equally among adjacent states
        if !isempty(adjacent_states)
            prob_per_adjacent = remaining_prob / length(adjacent_states)
            for adj_s in adjacent_states
                A[adj_s, s] = prob_per_adjacent
            end
        end
    end

    # Normalize matrix to ensure proper probability distributions
    for s in 1:n_states
        A ./= sum(A, dims=1)
    end

    reward_states = [(9, -1.0), (19, -1.0), (15, 1.0)]
    # Add noisy states to reward states with small negative reward
    for s in 1:n_states
        if A[s, s] < 1.0  # If diagonal element is less than 1, it's a noisy state
            push!(reward_states, (s, -0.1))
        end
    end

    return A, B, reward_states
end

