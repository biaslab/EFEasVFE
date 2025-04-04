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

export MinigridConfig, run_minigrid_agent, create_observation_tensor, convert_action

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

function convert_action(next_action)
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

function initialize_beliefs(grid_size, T::Type{<:AbstractFloat})
    return MinigridBeliefs(
        location=Categorical(fill(T(1 / grid_size^2), grid_size^2)),
        orientation=Categorical(fill(T(1 / 4), 4)),
        key_location=Categorical(fill(T(1 / (grid_size^2 - 2 * grid_size)), grid_size^2 - 2 * grid_size)),
        door_location=Categorical(fill(T(1 / (grid_size^2 - 2 * grid_size)), grid_size^2 - 2 * grid_size)),
        key_door_state=Categorical(T[1-2*tiny, tiny, tiny])
    )
end

function execute_initial_action(grid_size::Int)
    next_action = Int(TURN_LEFT)
    env_state = step_environment(next_action)
    return env_state
end

function execute_step(env_state, executed_action, beliefs, model, tensors, config, goal, callbacks, time_remaining)
    current_obs = env_state["observation"]
    obs_tensor = create_observation_tensor(current_obs, config.number_type)

    orientation = zeros(config.number_type, 4)
    orientation[current_obs["direction"]+1] = one(config.number_type)

    previous_action = zeros(config.number_type, 5)
    previous_action[executed_action] = one(config.number_type)

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
            goal=goal
        ),
        data=(
            observations=obs_tensor,
            action=previous_action,
            orientation_observation=orientation
        ),
        callbacks=callbacks,
        iterations=config.n_iterations,
        initialization=klcontrol_minigrid_agent_initialization(
            config.grid_size,
            beliefs.location,
            beliefs.orientation,
            beliefs.key_door_state,
            beliefs.door_location,
            beliefs.key_location
        )
    )

    next_action = mode(first(last(result.posteriors[:u])))
    env_action = convert_action(next_action)
    @debug "Executing action: $next_action with environment encoding $env_action"
    env_state = step_environment(env_action)
    @debug "Received reward: $(env_state["reward"])"

    # Update beliefs
    beliefs.location = last(result.posteriors[:current_location])
    beliefs.orientation = last(result.posteriors[:current_orientation])
    beliefs.key_door_state = last(result.posteriors[:current_key_door_state])
    beliefs.key_location = last(result.posteriors[:key_location])
    beliefs.door_location = last(result.posteriors[:door_location])

    return next_action, env_state
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
    run_single_episode(model, tensors, config, goal, callbacks, rng; record=false)

Run a single episode of the minigrid environment.

# Arguments
- `model`: The agent model to use
- `tensors`: Named tuple of required transition tensors
- `config`: Configuration parameters
- `goal`: Goal distribution
- `callbacks`: Optional callback functions
- `rng`: Random number generator
- `record`: Whether to record the episode to video

# Returns
- The total reward for the episode
"""
function run_single_episode(model, tensors, config, goal, callbacks, rng; record=false)
    # Ensure we use rgb_array render mode if recording
    render_mode = if record
        "rgb_array"
    else
        config.visualize ? "human" : "rgb_array"
    end

    episode_seed = rand(rng, UInt32)
    env_state = reinitialize_environment(config.grid_size + 2, render_mode=render_mode, seed=episode_seed)

    # Initialize frames collection if recording
    frames = record ? Vector{Array{UInt8,3}}() : nothing

    beliefs = initialize_beliefs(config.grid_size, config.number_type)
    reward = 0
    env_state = execute_initial_action(config.grid_size)
    action = 1

    # Store initial frame if recording
    if record
        push!(frames, convert_frame(env_state["frame"]))
    end

    for t in config.time_horizon:-1:1
        action, env_state = execute_step(env_state, action, beliefs, model, tensors, config, goal, callbacks, t)
        reward += env_state["reward"]

        if record
            push!(frames, convert_frame(env_state["frame"]))
        end

        env_state["terminated"] && break
        sleep(config.wait_time)
    end

    # Save video if recording
    if record && !isnothing(frames)
        record_episode_to_video(frames, datadir("results", config.experiment_name, "episode_$(config.n_episodes).mp4"))
    end

    return reward
end

"""
    run_minigrid_agent(model, tensors, config, goal; callbacks=nothing)

Run a minigrid agent experiment with the given model and configuration.

# Arguments
- `model`: The agent model to use (e.g., klcontrol_minigrid_agent)
- `tensors`: Named tuple of required transition tensors
- `config::MinigridConfig`: Configuration parameters
- `goal`: Goal distribution
- `callbacks`: Optional callback functions

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
    callbacks=nothing
)
    validate_config(config)
    rewards = zeros(config.n_episodes)
    rng = StableRNG(config.seed)

    @showprogress for i in 1:config.n_episodes
        # Record only the last episode if record_episode is true
        should_record = config.record_episode && i == config.n_episodes
        rewards[i] = run_single_episode(
            model, tensors, config, goal, callbacks, rng;
            record=should_record
        )
    end

    return mean(rewards), std(rewards)
end