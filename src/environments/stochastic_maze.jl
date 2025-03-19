using RxEnvironments
using RxEnvironmentsZoo.GLMakie

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