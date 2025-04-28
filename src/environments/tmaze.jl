using RxEnvironments
using GLMakie
using Distributions # For Categorical

export North, East, South, West, MazeAction, TMaze
export create_tmaze, reset_tmaze!, step!
export create_reward_observation_tensor, create_location_transition_tensor, create_reward_to_location_mapping

"""
    MazeAgent

An agent that can move in a maze environment.

# Fields
- `pos::Tuple{Int,Int}`: Current position as (x,y) tuple
"""
mutable struct MazeAgent
    pos::Tuple{Int,Int}
end

"""
    Maze

Represents a general maze environment.

# Fields
- `structure::Matrix{UInt8}`: Binary encoding of walls for each cell
- `observation_matrix::Matrix{Float64}`: Observation probabilities for each position
- `reward_pos::NTuple{N,Tuple{Tuple{Int,Int},Int}}`: Positions and values of rewards
- `agents::Vector{MazeAgent}`: List of agents in the maze
"""
struct Maze
    structure::Matrix{UInt8}
    observation_matrix::Matrix{Float64}
    reward_pos::NTuple{N,Tuple{Tuple{Int,Int},Int}} where {N}
    agents::Vector{MazeAgent}
end

"""
    Maze(structure, observation_matrix, rewardpos)

Construct a Maze with specified structure, observation matrix and reward positions.
"""
function Maze(structure::Matrix{UInt8}, observation_matrix::Matrix{Float64}, rewardpos)
    Maze(structure, observation_matrix, rewardpos, MazeAgent[])
end

"""
    Maze(structure, rewardpos)

Construct a Maze with specified structure and reward positions, using default observation matrix.
"""
function Maze(structure::Matrix{UInt8}, rewardpos)
    width, height = size(structure)
    observation_matrix = create_default_observation_matrix(width, height)
    Maze(structure, observation_matrix, rewardpos, MazeAgent[])
end

"""
Get the boundary encoding for a position in the maze.
"""
boundaries(maze::Maze, pos::NTuple{2,Int}) = maze.structure[pos[2], pos[1]]

# Direction types for agent actions
struct North end
struct East end
struct South end
struct West end
struct Stay end

const DIRECTIONS = (North(), East(), South(), West(), Stay())

"""
    MazeAgentAction

Represents a directional action in the maze.

# Fields
- `direction`: One of North(), East(), South(), West(), Stay()
"""
struct MazeAgentAction
    direction::Union{North,East,South,West,Stay}
end

"""
    MazeAction

Represents a directional action in the maze.
"""
struct MazeAction
    direction::Union{North,East,South,West}
end

"""
Move agent to new position based on action.
"""
function move!(agent::MazeAgent, maze::Maze, a::Union{MazeAgentAction,Int})
    agent.pos = next_state(agent.pos, maze, a)
end

"""
Convert numeric action (1-5) to directional action and get next state.
"""
function next_state(agent_pos::Tuple{Int,Int}, maze::Maze, action::Int)
    1 <= action <= 5 || throw(ArgumentError("Action must be between 1 and 5"))
    return next_state(agent_pos, maze, MazeAgentAction(DIRECTIONS[action]))
end

"""
Get next state based on agent position, maze boundaries and action.
"""
function next_state(agent_pos::Tuple{Int,Int}, maze::Maze, action::MazeAgentAction)
    c_bounds = boundaries(maze, agent_pos)
    b_bounds = digits(c_bounds, base=2, pad=4)  # Convert to binary wall encoding
    return next_state(agent_pos, b_bounds, action.direction)
end

"""
Move north if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::North)
    if b_bounds[1] == 0
        return (agent_pos[1], agent_pos[2] + 1)
    else
        return agent_pos
    end
end

"""
Move west if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::West)
    if b_bounds[2] == 0
        return (agent_pos[1] - 1, agent_pos[2])
    else
        return agent_pos
    end
end

"""
Move south if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::South)
    if b_bounds[3] == 0
        return (agent_pos[1], agent_pos[2] - 1)
    else
        return agent_pos
    end
end

"""
Move east if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::East)
    if b_bounds[4] == 0
        return (agent_pos[1] + 1, agent_pos[2])
    else
        return agent_pos
    end
end

"""
Stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::Stay)
    return agent_pos
end

# Environment update does nothing
RxEnvironments.update!(::Maze, dt) = nothing

# Receive action and move agent
RxEnvironments.receive!(maze::Maze, agent::MazeAgent, action::Union{MazeAgentAction,Int}) = move!(agent, maze, action)

"""
Convert linear index to (x,y) coordinates.
"""
function index_to_coord(index::Int, width::Int)::Tuple{Int,Int}
    y = div(index - 1, width) + 1
    x = mod(index - 1, width) + 1
    return (x, y)
end

"""
Convert (x,y) coordinates to linear index.
"""
function coord_to_index(x::Int, y::Int, width::Int)::Int
    return (y - 1) * width + x
end

"""
Sample an observation from the maze's observation matrix at given position.
"""
function sample_observation(maze::Maze, pos::Tuple{Int,Int})
    index = coord_to_index(pos[1], pos[2], size(maze.structure, 2))
    observation_probs = maze.observation_matrix[:, index]
    dist = Categorical(observation_probs)
    sampled_index = rand(dist)
    onehot = zeros(Float64, length(maze.structure))
    onehot[sampled_index] = 1.0
    return onehot
end

"""
Create a default observation matrix with noise increasing with distance from origin.
This is just one possible way to create an observation matrix - users can provide their own.
"""
function create_default_observation_matrix(width::Int, height::Int; noise_scale::Float64=0.1)
    size = width * height
    observation_matrix = zeros(Float64, size, size)
    for i in 1:size
        index = index_to_coord(i, width)
        # Calculate noise based on Manhattan distance from current position
        noise = (abs(width - index[1]) + abs(height - index[2])) * noise_scale
        noise_factor = noise / (size - 1)
        probvec = fill(noise_factor, size)
        probvec[i] = 1 - noise
        # Normalize to ensure probabilities sum to 1
        observation_matrix[:, i] = probvec ./ sum(probvec)
    end
    return observation_matrix
end

"""
Create transition tensor encoding state transitions for each action.
"""
function create_transition_tensor(maze::Maze)
    width, height = size(maze.structure)
    maze_size = width * height
    transition_tensor = zeros(Float64, maze_size, maze_size, 5)

    # Initialize all states to stay in place for the Stay action (index 5)
    transition_tensor[:, :, 5] = diagm(ones(maze_size))

    # Handle directional actions (indices 1-4)
    for i in 1:maze_size
        pos = index_to_coord(i, width)
        for a in 1:4
            new_pos = next_state(pos, maze, a)
            new_pos_index = coord_to_index(new_pos[1], new_pos[2], width)
            transition_tensor[i, new_pos_index, a] = 1
        end
    end
    return transition_tensor
end

"""
Get observation and rewards for current state.
"""
function RxEnvironments.what_to_send(agent::MazeAgent, maze::Maze, action::MazeAgentAction)
    rewards = nothing
    for reward_loc in maze.reward_pos
        if agent.pos == reward_loc[1]
            rewards = reward_loc[2]
            break
        end
    end
    observation = sample_observation(maze, agent.pos)
    return (observation, rewards)
end

"""
Plot cell walls based on boundary encoding.
"""
function plot_cell!(ax, cell, pos, internal=false)
    cell_bounds = digits(cell, base=2, pad=4)
    wall_coords = [
        ((pos[1] - 0.5, pos[2] + 0.5), (pos[1] + 0.5, pos[2] + 0.5)), # North
        ((pos[1] - 0.5, pos[2] - 0.5), (pos[1] - 0.5, pos[2] + 0.5)), # West
        ((pos[1] - 0.5, pos[2] - 0.5), (pos[1] + 0.5, pos[2] - 0.5)), # South
        ((pos[1] + 0.5, pos[2] - 0.5), (pos[1] + 0.5, pos[2] + 0.5))  # East
    ]

    for (i, ((x1, y1), (x2, y2))) in enumerate(wall_coords)
        if cell_bounds[i] == 1
            lines!(ax, [x1, x2], [y1, y2], color="black")
        elseif internal
            lines!(ax, [x1, x2], [y1, y2], color="black", linestyle=:dash)
        end
    end
end

"""
Add agent to environment.
"""
function RxEnvironments.add_to_state!(env::Maze, agent::MazeAgent)
    push!(env.agents, agent)
end

"""
Plot the maze and agents.
"""
function RxEnvironments.plot_state(ax, env::Maze)
    ys, xs = size(env.structure)
    xlims!(ax, -0.5, xs + 0.5)
    ylims!(ax, -0.5, ys + 0.5)

    # Plot maze cells
    for x in 1:xs, y in 1:ys
        plot_cell!(ax, env.structure[y, x], (x - 0.5, (y - 0.5)), false)
    end

    # Plot agents
    for agent in env.agents
        scatter!(ax, [agent.pos[1] - 0.5], [agent.pos[2] - 0.5], color="red")
    end
end

"""
Create environment and agent entities.
"""
function create_environment(::Type{Maze}, structure::Matrix{UInt8}, reward_pos; start_pos::Tuple{Int,Int}=(1, 1))
    maze = Maze(structure, reward_pos)
    agent = MazeAgent(start_pos)
    rx_env = create_entity(maze; is_active=true)
    rx_agent = add!(rx_env, agent)
    return rx_env, rx_agent
end

"""
Reset agent to starting position.
"""
function reset!(env::RxEnvironments.RxEntity{Maze}, start_pos::Tuple{Int,Int}=(1, 1))
    env.decorated.agents[1].pos = start_pos
end

"""
    TMaze

A T-shaped maze environment with 5 cells. The T has a stem cell at the bottom,
a middle junction cell, and three cells at the top (left, middle, right).
The reward can be either at the top-left or top-right cell.

# Fields
- `agent_position::Tuple{Int,Int}`: Current agent position as (x,y) tuple
- `reward_position::Symbol`: Either :left or :right
- `maze_structure::Matrix{UInt8}`: Binary encoding of walls for each cell
"""
mutable struct TMaze
    agent_position::Tuple{Int,Int}
    reward_position::Symbol
    maze_structure::Matrix{UInt8}
    reward_values::Dict{Tuple{Int,Int},Float64}

    function TMaze(reward_position::Symbol=:left)
        reward_position in [:left, :right] || throw(ArgumentError("reward_position must be :left or :right"))

        # Create the T-maze structure with 5 cells in T shape
        # Using a 3×3 grid with T shape
        # Binary encoding: North=1, West=2, South=4, East=8
        structure = [
            0x0B 0x09 0x0A; # Top row of T: left (0x0B), middle (0x09), right (0x0A)
            0x0F 0x05 0x0F; # Middle row: only middle cell accessible (0x05)
            0x0F 0x01 0x0F  # Bottom row: only middle cell accessible (0x01) - entry point
        ]

        # Define reward values
        reward_values = Dict{Tuple{Int,Int},Float64}()
        if reward_position == :left
            reward_values[(1, 3)] = 1.0  # Top left with positive reward
            reward_values[(3, 3)] = -1.0 # Top right with negative reward
        else
            reward_values[(1, 3)] = -1.0 # Top left with negative reward
            reward_values[(3, 3)] = 1.0  # Top right with positive reward
        end

        # Start at the bottom of the T
        agent_position = (2, 1)

        return new(agent_position, reward_position, structure, reward_values)
    end

    function TMaze(reward_position::Symbol, start_position::Tuple{Int,Int})
        reward_position in [:left, :right] || throw(ArgumentError("reward_position must be :left or :right"))

        # Validate start position
        valid_positions = [(2, 1), (2, 2), (1, 3), (2, 3), (3, 3)]
        start_position in valid_positions || throw(ArgumentError("Invalid start position. Must be one of: $valid_positions"))

        # Create the T-maze structure with 5 cells in T shape
        structure = [
            0x0B 0x09 0x0A; # Top row of T: left (0x0B), middle (0x09), right (0x0A)
            0x0F 0x05 0x0F; # Middle row: only middle cell accessible (0x05)
            0x0F 0x01 0x0F  # Bottom row: only middle cell accessible (0x01) - entry point
        ]

        # Define reward values
        reward_values = Dict{Tuple{Int,Int},Float64}()
        if reward_position == :left
            reward_values[(1, 3)] = 1.0  # Top left with positive reward
            reward_values[(3, 3)] = -1.0 # Top right with negative reward
        else
            reward_values[(1, 3)] = -1.0 # Top left with negative reward
            reward_values[(3, 3)] = 1.0  # Top right with positive reward
        end

        return new(start_position, reward_position, structure, reward_values)
    end
end

"""
    create_tmaze(reward_position::Symbol=rand([:left, :right]))

Create a T-maze environment with a reward at the specified position.
"""
function create_tmaze(reward_position::Symbol=rand([:left, :right]))
    return TMaze(reward_position)
end

"""
    create_tmaze(reward_position::Symbol, start_position::Tuple{Int,Int})

Create a T-maze environment with a reward at the specified position and the agent
starting at the specified position.
"""
function create_tmaze(reward_position::Symbol, start_position::Tuple{Int,Int})
    return TMaze(reward_position, start_position)
end

"""
    reset_tmaze!(env::TMaze, reward_position::Symbol=rand([:left, :right]))

Reset the T-maze environment with a new or specified reward position.
"""
function reset_tmaze!(env::TMaze, reward_position::Symbol=rand([:left, :right]))
    # Check reward position is valid
    reward_position in [:left, :right] || throw(ArgumentError("reward_position must be :left or :right"))

    # Reset agent position
    env.agent_position = (2, 1)

    # Update reward position
    env.reward_position = reward_position

    # Update reward values
    if reward_position == :left
        env.reward_values[(1, 3)] = 1.0  # Top left with positive reward
        env.reward_values[(3, 3)] = -1.0 # Top right with negative reward
    else
        env.reward_values[(1, 3)] = -1.0 # Top left with negative reward
        env.reward_values[(3, 3)] = 1.0  # Top right with positive reward
    end

    return env
end

"""
    reset_tmaze!(env::TMaze, reward_position::Symbol, start_position::Tuple{Int,Int})

Reset the T-maze environment with a new reward position and starting position.
"""
function reset_tmaze!(env::TMaze, reward_position::Symbol, start_position::Tuple{Int,Int})
    # Check reward position is valid
    reward_position in [:left, :right] || throw(ArgumentError("reward_position must be :left or :right"))

    # Validate start position
    valid_positions = [(2, 1), (2, 2), (1, 3), (2, 3), (3, 3)]
    start_position in valid_positions || throw(ArgumentError("Invalid start position. Must be one of: $valid_positions"))

    # Reset agent position
    env.agent_position = start_position

    # Update reward position
    env.reward_position = reward_position

    # Update reward values
    if reward_position == :left
        env.reward_values[(1, 3)] = 1.0  # Top left with positive reward
        env.reward_values[(3, 3)] = -1.0 # Top right with negative reward
    else
        env.reward_values[(1, 3)] = -1.0 # Top left with negative reward
        env.reward_values[(3, 3)] = 1.0  # Top right with positive reward
    end

    return env
end

"""
    boundaries(env::TMaze, pos::Tuple{Int,Int})

Get the boundary encoding for a position in the maze.
"""
function boundaries(env::TMaze, pos::Tuple{Int,Int})
    return env.maze_structure[pos[2], pos[1]]
end

"""
    step!(env::TMaze, action::MazeAction)

Take a step in the T-maze environment with the given action.
Returns a tuple of (position_obs, reward_cue, reward).
"""
function step!(env::TMaze, action::MazeAction)
    # Update agent position based on action
    env.agent_position = next_position(env, env.agent_position, action)

    # Create observations
    position_obs = create_position_observation(env)
    reward_cue = create_reward_cue(env)
    reward = get_reward(env)

    return position_obs, reward_cue, reward
end

"""
    create_position_observation(env::TMaze)

Create a one-hot encoded vector representing the agent's position.
"""
function create_position_observation(env::TMaze)
    # Create vector for 5 positions
    position_obs = zeros(Float64, 5)
    position_idx = position_to_index(env.agent_position)
    position_obs[position_idx] = 1.0
    return position_obs
end

"""
    create_reward_cue(env::TMaze)

Create a vector encoding information about the reward location.
At the bottom position, it reveals the true reward location.
At other positions, it provides uniform uncertainty.
"""
function create_reward_cue(env::TMaze)
    reward_cue = zeros(Float64, 2)

    # Only provide informative cue at the bottom position
    if env.agent_position == (2, 1)
        if env.reward_position == :left
            reward_cue = [1.0, 0.0]  # Left reward
        else
            reward_cue = [0.0, 1.0]  # Right reward
        end
    else
        # At other positions, provide uniform uncertainty
        reward_cue = [0.5, 0.5]
    end

    return reward_cue
end

"""
    get_reward(env::TMaze)

Get the reward at the current position, or 0 if no reward.
"""
function get_reward(env::TMaze)
    # Return the reward at the current position, or 0 if there's no reward here
    return get(env.reward_values, env.agent_position, 0.0)
end

"""
    next_position(env::TMaze, pos::Tuple{Int,Int}, action::MazeAction)

Calculate the next position based on current position and action.
"""
function next_position(env::TMaze, pos::Tuple{Int,Int}, action::MazeAction)
    # Handle each of the 5 valid positions and their possible movements

    # Bottom of T (2,1)
    if pos == (2, 1)
        if action.direction isa North
            return (2, 2)  # Move to middle junction
        else
            return pos     # Stay in place for all other directions (hitting walls)
        end

        # Middle junction (2,2)
    elseif pos == (2, 2)
        if action.direction isa North
            return (2, 3)  # Move to top middle
        elseif action.direction isa East
            return (3, 3)  # Move to top right
        elseif action.direction isa South
            return (2, 1)  # Move to bottom
        elseif action.direction isa West
            return (1, 3)  # Move to top left
        end

        # Top left (1,3)
    elseif pos == (1, 3)
        if action.direction isa East
            return (2, 3)  # Move to top middle
        elseif action.direction isa South
            return (2, 2)  # Move to middle junction
        else
            return pos     # Stay in place for other directions (hitting walls)
        end

        # Top middle (2,3)
    elseif pos == (2, 3)
        if action.direction isa East
            return (3, 3)  # Move to top right
        elseif action.direction isa South
            return (2, 2)  # Move to middle junction
        elseif action.direction isa West
            return (1, 3)  # Move to top left
        else
            return pos     # Stay in place for North (hitting wall)
        end

        # Top right (3,3)
    elseif pos == (3, 3)
        if action.direction isa South
            return (2, 2)  # Move to middle junction
        elseif action.direction isa West
            return (2, 3)  # Move to top middle
        else
            return pos     # Stay in place for other directions (hitting walls)
        end

        # Should not reach here with valid positions
    else
        error("Invalid position in T-maze: $pos")
    end
end

"""
    position_to_index(pos::Tuple{Int,Int})

Convert position coordinates to state index (1-5).
"""
function position_to_index(pos::Tuple{Int,Int})
    # Valid positions in the T-maze
    position_mapping = Dict(
        (2, 1) => 1,  # Bottom of T
        (2, 2) => 2,  # Middle junction
        (1, 3) => 3,  # Top left
        (2, 3) => 4,  # Top middle
        (3, 3) => 5   # Top right
    )

    if haskey(position_mapping, pos)
        return position_mapping[pos]
    else
        error("Invalid T-maze position: $pos")
    end
end

"""
    index_to_position(idx::Int)

Convert state index (1-5) to position coordinates.
"""
function index_to_position(idx::Int)
    if idx == 1
        return (2, 1)  # Bottom of T
    elseif idx == 2
        return (2, 2)  # Middle junction
    elseif idx == 3
        return (1, 3)  # Top left
    elseif idx == 4
        return (2, 3)  # Top middle
    elseif idx == 5
        return (3, 3)  # Top right
    else
        error("Invalid T-maze index: $idx")
    end
end

"""
    create_reward_observation_tensor()

Create the reward observation tensor for the T-maze environment.
This tensor has dimensions (2×5×2) representing:
- First dimension (2): Observation values [left_prob, right_prob]
- Second dimension (5): Agent location states (1-5)
- Third dimension (2): Reward location states (1=left, 2=right)

Returns a 2×5×2 Float64 tensor.
"""
function create_reward_observation_tensor()
    # Create reward observation tensor (2×5×2)
    # Dimensions: (observation_values, agent_location, reward_location)
    reward_obs_tensor = zeros(Float64, 2, 5, 2)

    # Fill with default uncertainty [0.5, 0.5] at all non-bottom positions
    for loc in 2:5, reward_loc in 1:2
        reward_obs_tensor[:, loc, reward_loc] .= 0.5
    end

    # At bottom position (state 1), reveal true reward location
    reward_obs_tensor[:, 1, 1] = [1.0, 0.0]  # Left reward
    reward_obs_tensor[:, 1, 2] = [0.0, 1.0]  # Right reward

    return reward_obs_tensor
end

"""
    create_location_transition_tensor()

Create the location transition tensor for the T-maze environment.
This tensor has dimensions (5×5×4) representing:
- First dimension (5): Next location state
- Second dimension (5): Current location state
- Third dimension (4): Action (1=North, 2=East, 3=South, 4=West)

Returns a 5×5×4 Float64 tensor.
"""
function create_location_transition_tensor()
    # Create location transition tensor (5×5×4)
    # Dimensions: (next_location, current_location, action)
    transition_tensor = zeros(Float64, 5, 5, 4)

    # Bottom of T (state 1)
    transition_tensor[2, 1, 1] = 1.0  # North -> Middle junction
    transition_tensor[1, 1, 2] = 1.0  # East -> Stay (wall)
    transition_tensor[1, 1, 3] = 1.0  # South -> Stay (wall)
    transition_tensor[1, 1, 4] = 1.0  # West -> Stay (wall)

    # Middle junction (state 2)
    transition_tensor[4, 2, 1] = 1.0  # North -> Top middle
    transition_tensor[5, 2, 2] = 1.0  # East -> Top right
    transition_tensor[1, 2, 3] = 1.0  # South -> Bottom
    transition_tensor[3, 2, 4] = 1.0  # West -> Top left

    # Top left (state 3)
    transition_tensor[3, 3, 1] = 1.0  # North -> Stay (wall)
    transition_tensor[4, 3, 2] = 1.0  # East -> Top middle
    transition_tensor[2, 3, 3] = 1.0  # South -> Middle junction
    transition_tensor[3, 3, 4] = 1.0  # West -> Stay (wall)

    # Top middle (state 4)
    transition_tensor[4, 4, 1] = 1.0  # North -> Stay (wall)
    transition_tensor[5, 4, 2] = 1.0  # East -> Top right
    transition_tensor[2, 4, 3] = 1.0  # South -> Middle junction
    transition_tensor[3, 4, 4] = 1.0  # West -> Top left

    # Top right (state 5)
    transition_tensor[5, 5, 1] = 1.0  # North -> Stay (wall)
    transition_tensor[5, 5, 2] = 1.0  # East -> Stay (wall)
    transition_tensor[2, 5, 3] = 1.0  # South -> Middle junction
    transition_tensor[4, 5, 4] = 1.0  # West -> Top middle

    return transition_tensor
end

"""
    create_reward_to_location_mapping()

Create the reward-to-location mapping tensor for the T-maze environment.
This tensor has dimensions (5×2) representing:
- First dimension (5): Location states
- Second dimension (2): Reward location states (1=left, 2=right)

Returns a 5×2 Float64 tensor.
"""
function create_reward_to_location_mapping()
    # Create reward-to-location mapping tensor (5×2)
    # Dimensions: (location, reward_location)
    reward_mapping = zeros(Float64, 5, 2)

    # Left reward (reward_location=1) is at top-left position (state 3)
    reward_mapping[3, 1] = 1.0

    # Right reward (reward_location=2) is at top-right position (state 5)
    reward_mapping[5, 2] = 1.0

    return reward_mapping
end