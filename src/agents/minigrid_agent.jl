using DrWatson
using ReactiveMP
using HTTP
using JSON
using ProgressMeter
using NPZ
import RxInfer: Categorical

export MinigridConfig, run_minigrid_agent, create_observation_tensor, convert_action

Base.@kwdef struct MinigridConfig{T<:AbstractFloat}
    grid_size::Int
    time_horizon::Int
    n_episodes::Int
    n_iterations::Int
    wait_time::Float64
    number_type::Type{T}
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
    cell_obs = zeros(T, 5) .+ tiny
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
    return env_state["reward"]
end

function execute_step(env_state, executed_action, beliefs, model, tensors, config, goal, callbacks, T)
    current_obs = env_state["observation"]
    obs_tensor = create_observation_tensor(current_obs, config.number_type)

    orientation = zeros(config.number_type, 4) .+ tiny
    orientation[current_obs["direction"]+1] = one(config.number_type)

    previous_action = zeros(config.number_type, 5) .+ tiny
    previous_action[executed_action] = one(config.number_type)

    @debug "Running inference with location $(beliefs.location) orientation $(beliefs.orientation) key_location $(beliefs.key_location) door_location $(beliefs.door_location) key_door_state $(beliefs.key_door_state)"

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
            T=T,
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

    @debug "Found policy: $(mode.(last(result.posteriors[:u])))"

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

    return next_action, env_state["reward"]
end

function run_single_episode(model, tensors, config, goal, callbacks)
    # Reinitialize environment with correct grid size
    env_state = reinitialize_environment(config.grid_size + 2)
    beliefs = initialize_beliefs(config.grid_size, config.number_type)
    reward = execute_initial_action(config.grid_size)
    action = 1

    for t in config.time_horizon:-1:1
        action, step_reward = execute_step(env_state, action, beliefs, model, tensors, config, goal, callbacks, t)
        reward += step_reward
        step_reward > 0 && break
        sleep(config.wait_time)
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

    @showprogress for i in 1:config.n_episodes
        rewards[i] = run_single_episode(model, tensors, config, goal, callbacks)
    end

    return mean(rewards), std(rewards)
end