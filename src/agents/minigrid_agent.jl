using DrWatson
using ReactiveMP
using HTTP
using JSON
using ProgressMeter
using NPZ
import RxInfer: Categorical
using StableRNGs
using VideoIO
using Colors  # Add Colors.jl for RGB{N0f8} support
using FileIO  # Add FileIO for saving individual frames

export MinigridConfig, run_minigrid_agent, create_observation_tensor, convert_action, save_minigrid_frame

Base.@kwdef struct MinigridConfig{T<:AbstractFloat}
    grid_size::Int
    time_horizon::Int
    n_episodes::Int
    n_iterations::Int
    wait_time::Float64
    number_type::Type{T}
    visualize::Bool
    seed::Int
    record_episode::Bool = false  # Default to false
    experiment_name::String
    parallel::Bool = false  # Default to sequential execution
end

Base.@kwdef mutable struct MinigridBeliefs{T<:AbstractFloat}
    location::Categorical{T}
    orientation::Categorical{T}
    key_door_state::Categorical{T}
    key_location::Categorical{T}
    door_location::Categorical{T}
end

function validate_config(config::MinigridConfig)
    config.grid_size > 0 || throw(ArgumentError("grid_size must be positive"))
    config.time_horizon > 0 || throw(ArgumentError("time_horizon must be positive"))
    config.n_episodes > 0 || throw(ArgumentError("n_episodes must be positive"))
    config.n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    config.wait_time >= 0 || throw(ArgumentError("wait_time must be non-negative"))
end

function create_cell_observation(cell_value, T::Type{<:AbstractFloat})
    cell_idx = match_cell_type(cell_value)
    cell_obs = zeros(T, 5)
    cell_obs[cell_idx] = one(T)
    return cell_obs
end

function match_cell_type(value)
    return if value == 0
        Int(INVISIBLE)
    elseif value == 1
        Int(EMPTY)
    elseif value == 2
        Int(WALL)
    elseif value == 4
        Int(DOOR)
    elseif value == 5
        Int(KEY)
    else
        Int(EMPTY)
    end
end

function create_observation_tensor(current_obs, T::Type{<:AbstractFloat})
    obs_tensor = fill(zeros(T, 5), 7, 7)
    for x in 1:7, y in 1:7
        obs_tensor[x, y] = create_cell_observation(current_obs["image"][x][y][1], T)
    end
    return obs_tensor
end

function convert_action(next_action::Int)
    action_map = Dict(
        Int(TURN_LEFT) => 0,   # left
        Int(TURN_RIGHT) => 1,  # right
        Int(FORWARD) => 2,     # forward
        Int(PICKUP) => 3,      # pickup
        Int(OPEN_DOOR) => 5    # toggle/open
    )
    return get(action_map, next_action) do
        error("Invalid action: $next_action")
    end
end

convert_action(next_action::AbstractVector) = convert_action(argmax(next_action))

function initialize_beliefs_minigrid(grid_size, T::Type{<:AbstractFloat})
    return MinigridBeliefs(
        location=Categorical([i <= grid_size^2 - 2 * grid_size ? T(1 / (grid_size^2 - 2 * grid_size)) : tiny(T) for i in 1:grid_size^2]),
        orientation=Categorical(fill(T(1 / 4), 4)),
        key_location=Categorical(fill(T(1 / (grid_size^2 - 2 * grid_size)), grid_size^2 - 2 * grid_size)),
        door_location=Categorical(fill(T(1 / (grid_size^2 - 2 * grid_size)), grid_size^2 - 2 * grid_size)),
        key_door_state=Categorical(T[1-2*tiny, tiny, tiny])
    )
end

function execute_initial_action(grid_size::Int, session_id::String)
    next_action = Int(TURN_LEFT)
    env_state = step_environment(next_action, session_id)
    return env_state
end

function get_initialization(initialization_fn, beliefs, previous_result::Nothing)
    return initialization_fn(beliefs.location, beliefs.orientation, beliefs.key_door_state, beliefs.location, beliefs.orientation, beliefs.key_door_state, beliefs.door_location, beliefs.key_location)
end

function get_initialization(initialization_fn, beliefs, previous_result)
    current_location_belief = first(last(previous_result.posteriors[:location]))
    future_location_beliefs = last(previous_result.posteriors[:location])[2:end]
    current_orientation_belief = first(last(previous_result.posteriors[:orientation]))
    future_orientation_beliefs = last(previous_result.posteriors[:orientation])[2:end]
    current_key_door_state_belief = first(last(previous_result.posteriors[:key_door_state]))
    future_key_door_state_beliefs = last(previous_result.posteriors[:key_door_state])[2:end]
    door_location_belief = beliefs.door_location
    key_location_belief = beliefs.key_location
    return initialization_fn(current_location_belief, current_orientation_belief, current_key_door_state_belief, future_location_beliefs, future_orientation_beliefs, future_key_door_state_beliefs, door_location_belief, key_location_belief)
end

"""
    execute_step(env_state, executed_action, beliefs, model, tensors, config, goal, callbacks, time_remaining, session_id::String; 
                constraints_fn=klcontrol_minigrid_agent_constraints, 
                initialization_fn=klcontrol_minigrid_agent_initialization,
                inference_kwargs...)

Execute a single step in the Minigrid environment using the given model and beliefs.

# Arguments
- `env_state`: The current state of the environment
- `executed_action`: The last executed action
- `beliefs`: Current agent beliefs
- `model`: The agent model to use
- `tensors`: Named tuple of required transition tensors
- `config`: Configuration parameters
- `goal`: Goal distribution
- `callbacks`: Optional callback functions for inference
- `time_remaining`: How many time steps remain in the episode
- `session_id`: The environment session ID
- `constraints_fn`: Function that returns the constraints for inference (default: klcontrol_minigrid_agent_constraints)
- `initialization_fn`: Function that returns the initialization for inference (default: klcontrol_minigrid_agent_initialization)
- `inference_kwargs...`: Additional keyword arguments to pass to the inference process

# Returns
- Tuple of (next_action, new_env_state, inference_result)
"""
function execute_step(env_state, executed_action, beliefs, model, tensors, config, goal, callbacks, time_remaining, previous_result, session_id::String;
    constraints_fn=klcontrol_minigrid_agent_constraints,
    initialization_fn=klcontrol_minigrid_agent_initialization,
    inference_kwargs...)
    current_obs = env_state["observation"]
    obs_tensor = create_observation_tensor(current_obs, config.number_type)

    initialization = get_initialization(initialization_fn, beliefs, previous_result)

    orientation = zeros(config.number_type, 4)
    orientation[current_obs["direction"]+1] = one(config.number_type)

    previous_action = zeros(config.number_type, 5)
    previous_action[executed_action] = one(config.number_type)

    # Run inference with additional kwargs
    result = infer(
        model=model(
            p_old_location=beliefs.location,
            p_old_orientation=beliefs.orientation,
            p_key_location=beliefs.key_location,
            p_door_location=beliefs.door_location,
            p_old_key_door_state=beliefs.key_door_state,
            location_transition_tensor=tensors.location,
            orientation_transition_tensor=tensors.orientation,
            key_door_transition_tensor=tensors.door_key,
            observation_tensors=tensors.observation,
            T=time_remaining,
            goal=goal,
            number_type=config.number_type
        ),
        data=(
            observations=obs_tensor,
            action=previous_action,
            orientation_observation=orientation
        ),
        constraints=constraints_fn(),
        callbacks=callbacks,
        iterations=config.n_iterations,
        initialization=initialization;
        inference_kwargs...  # Pass through any additional inference arguments
    )

    next_action = mode(first(last(result.posteriors[:u])))
    env_action = convert_action(next_action)
    @debug "Executing action: $next_action with environment encoding $env_action"
    env_state = step_environment(env_action, session_id)
    @debug "Received reward: $(env_state["reward"])"

    # Update beliefs
    beliefs.location = last(result.posteriors[:current_location])
    beliefs.orientation = last(result.posteriors[:current_orientation])
    beliefs.key_door_state = last(result.posteriors[:current_key_door_state])
    beliefs.key_location = last(result.posteriors[:key_location])
    beliefs.door_location = last(result.posteriors[:door_location])

    return next_action, env_state, result  # Return the inference result as well
end

"""
    convert_frame(frame_list)

Convert a frame from a nested list structure [[[r,g,b]]] to a 3D UInt8 array.
The input frame_list has the structure:
- Outer list: height
- Middle list: width
- Inner list: RGB values

Returns a (height, width, 3) UInt8 array.
"""
function convert_frame(frame_list)
    isempty(frame_list) && throw(ArgumentError("Empty frame"))

    height = length(frame_list)
    width = length(frame_list[1])

    # Verify consistent dimensions
    all(length(row) == width for row in frame_list) ||
        throw(ArgumentError("Inconsistent row lengths"))
    all(all(length(pixel) == 3 for pixel in row) for row in frame_list) ||
        throw(ArgumentError("Invalid pixel format"))

    # Create output array
    result = Array{UInt8}(undef, height, width, 3)

    # Copy values
    for i in 1:height, j in 1:width, k in 1:3
        result[i, j, k] = UInt8(frame_list[i][j][k])
    end

    return result
end

"""
    record_episode_to_video(frames::Vector{Array{UInt8, 3}}, video_path::String="episode_recording.mp4")

Save a sequence of frames to a video file.
"""
function record_episode_to_video(frames::Vector{Array{UInt8,3}}, video_path::String="episode_recording.mp4")
    if isempty(frames)
        @warn "No frames to record"
        return
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
    save_minigrid_frame(frame::Array{UInt8,3}, model_name::String, seed::Int, timestep::Int, output_dir::String)

Save a single Minigrid frame as a PNG file.

# Arguments
- `frame::Array{UInt8,3}`: The RGB frame to save (height, width, 3) UInt8 array
- `model_name::String`: Name of the model for file naming
- `seed::Int`: Random seed for the episode for file naming
- `timestep::Int`: Current timestep (will be formatted with leading zeros)
- `output_dir::String`: Directory to save the frame in

# Returns
- `filepath::String`: The full path to the saved file
"""
function save_minigrid_frame(frame::Array{UInt8,3}, model_name::String, seed, timestep::Int, output_dir::String)
    # Format the timestep with leading zeros for proper sorting
    timestep_str = lpad(timestep, 3, "0")

    # Create the filename with format model_name_episode_seed_frame_NNN.png
    filename = "$(model_name)_episode_$(seed)_frame_$(timestep_str).png"

    # Full path to save the frame
    filepath = joinpath(output_dir, filename)

    # Convert UInt8 array to RGB{N0f8} for saving with FileIO
    height, width, _ = size(frame)
    rgb_frame = RGB{Colors.N0f8}.(frame[:, :, 1] ./ 255, frame[:, :, 2] ./ 255, frame[:, :, 3] ./ 255)

    # Save the frame as PNG
    FileIO.save(filepath, rgb_frame)

    return filepath
end

"""
    run_single_episode(model, tensors, config, goal, callbacks, seed; 
                        constraints_fn=klcontrol_minigrid_agent_constraints,
                        initialization_fn=klcontrol_minigrid_agent_initialization,
                        record=false,
                        inference_kwargs...)

Run a single episode of the minigrid environment.

# Arguments
- `model`: The agent model to use
- `tensors`: Named tuple of required transition tensors
- `config`: Configuration parameters
- `goal`: Goal distribution
- `callbacks`: Optional callback functions
- `seed`: Random seed for the episode
- `constraints_fn`: Function that returns the constraints for inference (default: klcontrol_minigrid_agent_constraints)
- `initialization_fn`: Function that returns the initialization for inference (default: klcontrol_minigrid_agent_initialization)
- `record`: Whether to record the episode to video
- `inference_kwargs...`: Additional keyword arguments to pass to the inference process

# Returns
- The total reward for the episode
"""
function run_single_episode(model, tensors, config, goal, callbacks, seed;
    constraints_fn=klcontrol_minigrid_agent_constraints,
    initialization_fn=klcontrol_minigrid_agent_initialization,
    record=false,
    inference_kwargs...)
    # Ensure we use rgb_array render mode if recording
    render_mode = if record
        "rgb_array"
    else
        config.visualize ? "human" : "rgb_array"
    end
    env_response = create_environment(config.grid_size + 2, render_mode=render_mode, seed=seed)
    session_id = env_response["session_id"]

    try
        # Initialize frames collection if recording
        frames = record ? Vector{Array{UInt8,3}}() : nothing

        # Get model name for frame saving
        model_name = try
            string(nameof(model))
        catch
            "unknown_model"
        end

        # Set up the directory for saving individual frames if requested
        frames_dir = nothing
        if record
            # Create directory structure: results/minigrid/experiment_name/model_name/episode_seed
            frames_dir = datadir("results", "minigrid", config.experiment_name, model_name, "episode_$(seed)")
            mkpath(frames_dir)
        end

        beliefs = initialize_beliefs_minigrid(config.grid_size, config.number_type)
        reward = 0
        env_state = execute_initial_action(config.grid_size, session_id)
        action = 1
        previous_result = nothing

        # Store initial frame if recording
        if record
            initial_frame = convert_frame(env_state["frame"])
            push!(frames, initial_frame)

            # Save individual frame as PNG
            if !isnothing(frames_dir)
                save_minigrid_frame(initial_frame, model_name, seed, 0, frames_dir)
            end
        end

        for t in config.time_horizon:-1:1
            action, env_state, result = execute_step(
                env_state, action, beliefs, model, tensors, config, goal,
                callbacks, t, previous_result, session_id;
                constraints_fn=constraints_fn,
                initialization_fn=initialization_fn,
                inference_kwargs...  # Forward any additional inference arguments
            )
            reward += env_state["reward"]
            previous_result = result

            if record
                current_frame = convert_frame(env_state["frame"])
                push!(frames, current_frame)

                # Save individual frame as PNG
                if !isnothing(frames_dir)
                    current_timestep = config.time_horizon - t + 1
                    save_minigrid_frame(current_frame, model_name, seed, current_timestep, frames_dir)
                end
            end

            env_state["terminated"] && break
            sleep(config.wait_time)
        end

        # Save video if recording
        if record && !isnothing(frames)
            video_path = datadir("results", "minigrid", config.experiment_name, "$(model_name)_episode_$(seed).mp4")
            record_episode_to_video(frames, video_path)
        end

        return reward
    finally
        # Always clean up the environment session
        try
            close_environment(session_id)
        catch e
            @warn "Failed to close environment session: $e"
        end
    end
end

"""
    run_minigrid_agent(model, tensors, config, goal; 
                      callbacks=nothing,
                      constraints_fn=klcontrol_minigrid_agent_constraints,
                      initialization_fn=klcontrol_minigrid_agent_initialization,
                      parallel=nothing,
                      inference_kwargs...)

Run a minigrid agent experiment with the given model and configuration.

# Arguments
- `model`: The agent model to use (e.g., klcontrol_minigrid_agent)
- `tensors`: Named tuple of required transition tensors
- `config::MinigridConfig`: Configuration parameters
- `goal`: Goal distribution
- `callbacks`: Optional callback functions
- `constraints_fn`: Function that returns the constraints for inference (default: klcontrol_minigrid_agent_constraints)
- `initialization_fn`: Function that returns the initialization for inference (default: klcontrol_minigrid_agent_initialization)
- `parallel`: Whether to run episodes in parallel (defaults to config.parallel if nothing)
- `inference_kwargs...`: Additional keyword arguments to pass to the inference process

# Returns
- Tuple of (mean_reward, std_reward)

# Throws
- `EnvironmentError` if environment communication fails
"""
function run_minigrid_agent(
    model::Function,
    tensors::NamedTuple,
    config::MinigridConfig,
    goal::Categorical;
    callbacks=nothing,
    constraints_fn=klcontrol_minigrid_agent_constraints,
    initialization_fn=klcontrol_minigrid_agent_initialization,
    parallel::Union{Nothing,Bool}=nothing,  # New keyword argument, defaults to config value if nothing
    inference_kwargs...  # Additional inference kwargs to pass through
)
    validate_config(config)
    rewards = zeros(config.n_episodes)

    # Determine if we should use parallel execution
    # If parallel keyword is provided, it overrides the config setting
    use_parallel = isnothing(parallel) ? config.parallel : parallel

    rng = StableRNG(config.seed)

    # Create a thread-safe RNG for each thread if running in parallel
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
                local_config = MinigridConfig(
                    grid_size=config.grid_size,
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

            rewards[i] = run_single_episode(
                model, tensors, local_config, goal, callbacks, episode_seed;
                constraints_fn=constraints_fn,
                initialization_fn=initialization_fn,
                record=should_record,
                inference_kwargs...  # Forward any additional inference arguments
            )

            # Update progress atomically
            ProgressMeter.next!(progress)
        end
    else
        # Sequential execution (old behavior)
        @info "Running sequentially"
        episode_seeds = rand(rng, UInt32, config.n_episodes)

        @showprogress for i in 1:config.n_episodes
            episode_seed = episode_seeds[i]
            # Record only the last episode if record_episode is true
            should_record = config.record_episode && i == config.n_episodes
            rewards[i] = run_single_episode(
                model, tensors, config, goal, callbacks, episode_seed;
                constraints_fn=constraints_fn,
                initialization_fn=initialization_fn,
                record=should_record,
                inference_kwargs...  # Forward any additional inference arguments
            )
        end
    end

    return mean(rewards), std(rewards)
end