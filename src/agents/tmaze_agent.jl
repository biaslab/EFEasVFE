using DrWatson
using ReactiveMP
using RxInfer
using ProgressMeter
using VideoIO
using Colors
using StableRNGs
import RxInfer: Categorical

export TMazeConfig, run_tmaze_agent, run_tmaze_single_episode, tmaze_convert_action

"""
    TMazeConfig

Configuration for TMaze agent experiments.

# Fields
- `time_horizon::Int`: Planning horizon for the agent
- `n_episodes::Int`: Number of episodes to run
- `n_iterations::Int`: Number of inference iterations per step
- `wait_time::Float64`: Time to wait between steps (for visualization)
- `number_type::Type{T}`: Numeric type for computations
- `visualize::Bool`: Whether to visualize the environment
- `seed::Int`: Random seed
- `record_episode::Bool`: Whether to record the last episode as video
- `experiment_name::String`: Name of the experiment (for saving results)
- `parallel::Bool`: Whether to run episodes in parallel
"""
Base.@kwdef struct TMazeConfig{T<:AbstractFloat}
    time_horizon::Int
    n_episodes::Int
    n_iterations::Int
    wait_time::Float64
    number_type::Type{T}
    visualize::Bool
    seed::Int
    record_episode::Bool = false
    experiment_name::String
    parallel::Bool = false
end

"""
    TMazeBeliefs

Container for agent's beliefs about the TMaze environment.

# Fields
- `location::Categorical{T}`: Belief about current location (5 possible states)
- `reward_location::Categorical{T}`: Belief about reward location (left or right)
"""
Base.@kwdef mutable struct TMazeBeliefs{T<:AbstractFloat}
    location::Categorical{T}
    reward_location::Categorical{T}
end

"""
    validate_config(config::TMazeConfig)

Validate that the TMaze configuration has valid values.
"""
function validate_config(config::TMazeConfig)
    config.time_horizon > 0 || throw(ArgumentError("time_horizon must be positive"))
    config.n_episodes > 0 || throw(ArgumentError("n_episodes must be positive"))
    config.n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    config.wait_time >= 0 || throw(ArgumentError("wait_time must be non-negative"))
end

"""
    initialize_beliefs(T::Type{<:AbstractFloat})

Initialize agent beliefs for the TMaze environment.
"""
function initialize_beliefs(T::Type{<:AbstractFloat})
    # Initialize with uniform beliefs over states
    return TMazeBeliefs(
        location=Categorical(fill(T(1 / 5), 5)),
        reward_location=Categorical([T(0.5), T(0.5)])
    )
end

"""
    tmaze_convert_action(next_action::Int)

Convert model action index to environment action.
Action mapping: 1=North, 2=East, 3=South, 4=West
"""
function tmaze_convert_action(next_action::Int)
    action_map = Dict(
        1 => MazeAction(North()),  # North
        2 => MazeAction(East()),   # East
        3 => MazeAction(South()),  # South
        4 => MazeAction(West())    # West
    )
    return get(action_map, next_action) do
        error("Invalid action: $next_action")
    end
end

tmaze_convert_action(next_action::AbstractVector) = tmaze_convert_action(argmax(next_action))

"""
    get_initialization_tmaze(initialization_fn, beliefs, previous_result)

Get initialization for the inference procedure based on previous results.
"""
function get_initialization_tmaze(initialization_fn, beliefs, previous_result::Nothing)
    future_location_beliefs = vague(Categorical, 5)
    return initialization_fn(beliefs.location, beliefs.reward_location, future_location_beliefs)
end

function get_initialization_tmaze(initialization_fn, beliefs, previous_result)
    current_location_belief = first(last(previous_result.posteriors[:location]))
    future_location_beliefs = last(previous_result.posteriors[:location])[2:end]
    reward_location_belief = last(previous_result.posteriors[:reward_location])

    return initialization_fn(current_location_belief, reward_location_belief, future_location_beliefs)
end

"""
    execute_step(env, position_obs, reward_cue, beliefs, model, tensors, config, goal, callbacks, time_remaining, previous_result, previous_action;
                constraints_fn, initialization_fn, inference_kwargs...)

Execute a single step of inference to determine the next action in the TMaze environment.
Takes current observations and returns the next planned action.
"""
function execute_step(env, position_obs, reward_cue, beliefs, model, tensors, config, goal, callbacks, time_remaining, previous_result, previous_action;
    constraints_fn, initialization_fn, inference_kwargs...)
    # Convert previous action to one-hot encoding
    previous_action_vec = zeros(config.number_type, 4)
    if previous_action.direction isa North
        previous_action_vec[1] = one(config.number_type)
    elseif previous_action.direction isa East
        previous_action_vec[2] = one(config.number_type)
    elseif previous_action.direction isa South
        previous_action_vec[3] = one(config.number_type)
    elseif previous_action.direction isa West
        previous_action_vec[4] = one(config.number_type)
    end

    # Get initialization from previous results or initialize fresh
    initialization = get_initialization_tmaze(initialization_fn, beliefs, previous_result)

    # Run inference
    result = infer(
        model=model(
            reward_observation_tensor=tensors.reward_observation,
            location_transition_tensor=tensors.location_transition,
            prior_location=beliefs.location,
            prior_reward_location=beliefs.reward_location,
            reward_to_location_mapping=tensors.reward_to_location,
            u_prev=previous_action_vec,
            T=time_remaining
        ),
        data=(
            location_observation=position_obs,
            reward_observation=reward_cue
        ),
        constraints=constraints_fn(),
        callbacks=callbacks,
        iterations=config.n_iterations,
        initialization=initialization;
        inference_kwargs...
    )

    # Select next action based on posterior
    next_action_idx = Int(mode(first(last(result.posteriors[:u]))))
    next_action = tmaze_convert_action(next_action_idx)

    # Update beliefs
    beliefs.location = last(result.posteriors[:current_location])
    beliefs.reward_location = last(result.posteriors[:reward_location])

    return next_action_idx, next_action, result
end

"""
    convert_to_rgb(env)

Convert TMaze state to RGB visualization for recording.
"""
function convert_to_rgb(env)
    height, width = 300, 300
    img = zeros(UInt8, height, width, 3)

    # Background color - light gray
    img .= UInt8(240)

    # Draw the T shape
    # Vertical stem
    img[100:250, 140:160, :] .= UInt8(200)
    # Horizontal top
    img[100:120, 60:240, :] .= UInt8(200)

    # Draw agent (red circle)
    agent_pos = env.agent_position

    # Convert position to pixel coordinates
    agent_x, agent_y = 0, 0
    if agent_pos == (2, 1)  # Bottom
        agent_x, agent_y = 150, 230
    elseif agent_pos == (2, 2)  # Middle
        agent_x, agent_y = 150, 150
    elseif agent_pos == (1, 3)  # Top left
        agent_x, agent_y = 75, 110
    elseif agent_pos == (2, 3)  # Top middle
        agent_x, agent_y = 150, 110
    elseif agent_pos == (3, 3)  # Top right
        agent_x, agent_y = 225, 110
    end

    # Draw agent
    for dx in -10:10, dy in -10:10
        if dx^2 + dy^2 <= 100  # Circle with radius 10
            x, y = agent_x + dx, agent_y + dy
            if 1 <= x <= width && 1 <= y <= height
                img[y, x, 1] = 255  # Red component
                img[y, x, 2] = 0
                img[y, x, 3] = 0
            end
        end
    end

    # Draw reward locations
    reward_position = env.reward_position

    # Left reward
    left_x, left_y = 75, 110
    left_color = reward_position == :left ? [0, 255, 0] : [150, 150, 150]

    # Right reward
    right_x, right_y = 225, 110
    right_color = reward_position == :right ? [0, 255, 0] : [150, 150, 150]

    # Draw reward markers
    for dx in -5:5, dy in -5:5
        if dx^2 + dy^2 <= 25  # Circle with radius 5
            # Left reward
            x, y = left_x + dx, left_y + dy
            if 1 <= x <= width && 1 <= y <= height
                img[y, x, 1] = left_color[1]
                img[y, x, 2] = left_color[2]
                img[y, x, 3] = left_color[3]
            end

            # Right reward
            x, y = right_x + dx, right_y + dy
            if 1 <= x <= width && 1 <= y <= height
                img[y, x, 1] = right_color[1]
                img[y, x, 2] = right_color[2]
                img[y, x, 3] = right_color[3]
            end
        end
    end

    return img
end

"""
    visualize_tmaze(env::TMaze)

Simple visualization of the TMaze environment.
"""
function visualize_tmaze(env::TMaze)
    # Define symbols for visualization
    symbols = Dict(
        "agent" => 'A',
        "wall" => '█',
        "empty" => ' ',
        "reward_left" => 'L',
        "reward_right" => 'R'
    )

    # Create a 3x3 grid for visualization
    grid = fill(symbols["wall"], 3, 3)

    # Mark the T shape
    grid[3, 2] = symbols["empty"]  # Bottom
    grid[2, 2] = symbols["empty"]  # Middle
    grid[1, 1:3] .= symbols["empty"]  # Top

    # Mark rewards
    grid[1, 1] = symbols["reward_left"]
    grid[1, 3] = symbols["reward_right"]

    # Highlight the active reward
    if env.reward_position == :left
        grid[1, 1] = '*'
    else
        grid[1, 3] = '*'
    end

    # Place agent
    agent_pos = env.agent_position
    grid_x = agent_pos[2]  # Convert to grid coordinates
    grid_y = agent_pos[1]

    # Store original value
    original = grid[grid_x, grid_y]
    grid[grid_x, grid_y] = symbols["agent"]

    # Print the grid
    println("TMaze Environment:")
    println("Reward position: $(env.reward_position)")
    for i in 1:3
        println(join(grid[i, :]))
    end
    println()

    # Restore original value
    grid[grid_x, grid_y] = original
end

"""
    record_episode_to_video_tmaze(frames::Vector{Array{UInt8, 3}}, video_path::String)

Save a sequence of frames to a video file.
"""
function record_episode_to_video_tmaze(frames::Vector{Array{UInt8,3}}, video_path::String="tmaze_episode.mp4")
    if isempty(frames)
        @warn "No frames to record"
        return
    end

    # Ensure directory exists
    video_dir = dirname(video_path)
    if !isdir(video_dir)
        mkpath(video_dir)
    end

    # Convert UInt8 arrays to RGB{N0f8} arrays
    height, width, _ = size(frames[1])
    rgb_frames = [RGB{Colors.N0f8}.(frames[i][:, :, 1] ./ 255, frames[i][:, :, 2] ./ 255, frames[i][:, :, 3] ./ 255)
                  for i in 1:length(frames)]

    # Set up encoder options
    encoder_options = (crf=23, preset="medium")
    framerate = 5

    # Save video using VideoIO.save
    VideoIO.save(video_path, rgb_frames;
        framerate=framerate,
        encoder_options=encoder_options
    )

    @info "Episode recorded to $video_path"
end

"""
    get_position_observation(env::TMaze)

Get the position observation from the TMaze environment. 
Delegates to create_position_observation function.
"""
function get_position_observation(env::TMaze)
    return create_position_observation(env)
end

"""
    get_reward_cue(env::TMaze)

Get the reward cue observation from the TMaze environment.
Delegates to create_reward_cue function.
"""
function get_reward_cue(env::TMaze)
    return create_reward_cue(env)
end

"""
    run_tmaze_single_episode(model, tensors, config, goal, callbacks, seed;
                      constraints_fn, initialization_fn, record, inference_kwargs...)

Run a single episode in the TMaze environment.
"""
function run_tmaze_single_episode(model, tensors, config, goal, callbacks, seed;
    constraints_fn, initialization_fn, record=false, debug_mode=false, inference_kwargs...)
    # Set up RNG
    rng = StableRNG(seed)

    # Create environment with random reward position and fixed start position (2,2)
    reward_position = rand(rng, [:left, :right])
    env = create_tmaze(reward_position, (2, 2))  # Start at middle junction (2,2)

    # Initialize beliefs
    beliefs = initialize_beliefs(config.number_type)

    # Initialize tracking variables
    total_reward = 0.0
    previous_result = nothing

    # Initial action (placeholder)
    next_action = MazeAction(East())
    next_action_idx = 2

    # Initialize frames collection if recording
    frames = record ? Vector{Array{UInt8,3}}() : nothing

    # Tracking data for detailed logging
    episode_data = Dict(
        "reward_position" => string(reward_position),
        "trajectory" => [],
        "actions" => [],
        "action_names" => [],
        "rewards" => [],
        "positions" => [],
        "timestamps" => []
    )

    # Initial position observation and reward cue
    position_obs = convert.(config.number_type, get_position_observation(env))
    reward_cue = convert.(config.number_type, get_reward_cue(env))

    # Record initial state
    push!(episode_data["positions"], [env.agent_position...])
    push!(episode_data["timestamps"], 0)

    # Record initial state if recording
    if record
        push!(frames, convert_to_rgb(env))
    end

    # Visualization
    if config.visualize
        visualize_tmaze(env)
    end

    # Log initial state if in debug mode
    if debug_mode
        @info "Episode $(seed): Starting at $(env.agent_position), reward at :$(reward_position)"
    end

    # Run episode with corrected action execution flow
    for t in config.time_horizon:-1:1
        # Plan the next action based on current observations
        next_action_idx, next_action, result = execute_step(
            env, position_obs, reward_cue, beliefs, model, tensors, config, goal,
            callbacks, t, previous_result, next_action;
            constraints_fn=constraints_fn,
            initialization_fn=initialization_fn,
            inference_kwargs...
        )

        # Update the previous result for the next iteration
        previous_result = result

        # Execute the planned action and get observations
        position_obs, reward_cue, reward = step!(env, next_action)

        # Convert to the required numeric type
        position_obs = convert.(config.number_type, position_obs)
        reward_cue = convert.(config.number_type, reward_cue)

        # Update total reward
        episode_reward = reward isa Number ? reward : 0
        total_reward += episode_reward

        # Update tracking data for the action just executed
        push!(episode_data["actions"], next_action_idx)
        push!(episode_data["action_names"], action_to_string(next_action_idx))
        push!(episode_data["rewards"], episode_reward)
        push!(episode_data["positions"], [env.agent_position...])
        push!(episode_data["timestamps"], config.time_horizon - t + 1)

        # Record frame if recording
        if record
            push!(frames, convert_to_rgb(env))
        end

        # Visualization update
        if config.visualize
            visualize_tmaze(env)
        end

        # Log step information if in debug mode
        if debug_mode
            action_str = action_to_string(next_action_idx)
            reward_str = isnothing(reward) ? "None" : reward
            @info "Episode $(seed): t=$t, Position=$(env.agent_position), Action=$action_str, Reward=$reward_str"
        end

        # Check if goal reached
        if reward == 1
            if debug_mode
                @info "Episode $(seed): Goal reached at t=$(t)!"
            end
            break
        end

        # Delay for visualization
        sleep(config.wait_time)
    end

    # Add final trajectory information
    episode_data["total_reward"] = total_reward
    episode_data["final_position"] = [env.agent_position...]
    episode_data["seed"] = seed

    # Save video if recording
    if record && !isnothing(frames)
        # Get model name from function object if possible
        model_name = try
            string(nameof(model))
        catch
            "unknown_model"
        end

        # Create the full path and ensure directory exists
        video_dir = datadir("results", "tmaze", config.experiment_name)
        mkpath(video_dir)
        video_path = joinpath(video_dir, "$(model_name)_episode_$(seed).mp4")

        record_episode_to_video_tmaze(frames, video_path)
    end

    return total_reward, episode_data
end

"""
    action_to_string(action_idx::Int)

Convert action index to string representation.
"""
function action_to_string(action_idx::Int)
    if action_idx == 1
        return "North"
    elseif action_idx == 2
        return "East"
    elseif action_idx == 3
        return "South"
    elseif action_idx == 4
        return "West"
    else
        return "Unknown"
    end
end

"""
    run_tmaze_agent(model, config, goal;
                   callbacks, constraints_fn, initialization_fn, parallel, inference_kwargs...)

Run a TMaze agent experiment with the given model and configuration.
"""
function run_tmaze_agent(
    model::Function,
    config::TMazeConfig,
    goal::Categorical;
    callbacks=nothing,
    constraints_fn=klcontrol_tmaze_agent_constraints,
    initialization_fn=klcontrol_tmaze_agent_initialization,
    parallel::Union{Nothing,Bool}=nothing,
    inference_kwargs...
)
    # Validate configuration
    validate_config(config)

    # Create tensors
    tensors = (
        reward_observation=create_reward_observation_tensor(),
        location_transition=create_location_transition_tensor(),
        reward_to_location=create_reward_to_location_mapping()
    )

    # Initialize rewards array
    rewards = zeros(config.n_episodes)

    # Determine if we should use parallel execution
    use_parallel = isnothing(parallel) ? config.parallel : parallel

    # Set up RNG
    rng = StableRNG(config.seed)

    # Create directory for results if recording
    if config.record_episode
        mkpath(datadir("results", "tmaze", config.experiment_name))
    end

    if use_parallel
        thread_count = Threads.nthreads()
        @info "Running with parallelization using $thread_count threads"
        episode_seeds = rand(rng, UInt32, config.n_episodes)

        # Use Threads.@threads for parallelization
        progress = Progress(config.n_episodes; desc="Running episodes: ")

        Threads.@threads for i in 1:config.n_episodes
            # Get the RNG for the current thread
            thread_id = Threads.threadid()
            episode_seed = episode_seeds[i]

            # Record only the last episode if record_episode is true
            should_record = config.record_episode && i == config.n_episodes

            # Turn off visualization for parallel execution except for the recording episode
            local_config = config
            if should_record && config.visualize
                # Keep visualization for recording
            elseif use_parallel && config.visualize
                # Create a copy with visualization turned off
                local_config = TMazeConfig(
                    time_horizon=config.time_horizon,
                    n_episodes=config.n_episodes,
                    n_iterations=config.n_iterations,
                    wait_time=config.wait_time,
                    number_type=config.number_type,
                    visualize=false,  # Turn off visualization for parallel execution
                    seed=config.seed,
                    record_episode=config.record_episode,
                    experiment_name=config.experiment_name,
                    parallel=use_parallel
                )
            end

            rewards[i] = run_tmaze_single_episode(
                model, tensors, local_config, goal, callbacks, episode_seed;
                constraints_fn=constraints_fn,
                initialization_fn=initialization_fn,
                record=should_record,
                inference_kwargs...
            )

            # Update progress atomically
            ProgressMeter.next!(progress)
        end
    else
        # Sequential execution
        @info "Running sequentially"
        episode_seeds = rand(rng, UInt32, config.n_episodes)

        @showprogress for i in 1:config.n_episodes
            episode_seed = episode_seeds[i]
            # Record only the last episode if record_episode is true
            should_record = config.record_episode && i == config.n_episodes
            rewards[i] = run_tmaze_single_episode(
                model, tensors, config, goal, callbacks, episode_seed;
                constraints_fn=constraints_fn,
                initialization_fn=initialization_fn,
                record=should_record,
                inference_kwargs...
            )
        end
    end

    return mean(rewards), std(rewards)
end