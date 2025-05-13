using DrWatson
using ReactiveMP
using RxInfer
using ProgressMeter
using Distributions
using StableRNGs
using Plots
using FileIO
using VideoIO
using Colors  # Add Colors.jl for RGB conversion
import RxInfer: Categorical

export StochasticMazeConfig, run_stochastic_maze_agent, run_stochastic_maze_single_episode
export stochastic_maze_convert_action, action_to_string
export convert_frames_to_video, create_video_from_experiment

"""
    StochasticMazeConfig

Configuration for StochasticMaze agent experiments.

# Fields
- `time_horizon::Int`: Planning horizon for the agent
- `n_episodes::Int`: Number of episodes to run
- `n_iterations::Int`: Number of inference iterations per step
- `wait_time::Float64`: Time to wait between steps (for visualization)
- `seed::Int`: Random seed
- `record_episode::Bool`: Whether to record episode frames as individual PNG files
- `experiment_name::String`: Name of the experiment (for saving results)
- `parallel::Bool`: Whether to run episodes in parallel
"""
Base.@kwdef struct StochasticMazeConfig
    time_horizon::Int
    n_episodes::Int
    n_iterations::Int
    wait_time::Float64
    seed::Int
    record_episode::Bool = false
    experiment_name::String
    parallel::Bool = false
end

"""
    StochasticMazeBeliefs

Container for agent's beliefs about the StochasticMaze environment.

# Fields
- `state::Categorical{Float64}`: Belief about current state
"""
Base.@kwdef mutable struct StochasticMazeBeliefs
    state::Categorical{Float64}
end

"""
    validate_config(config::StochasticMazeConfig)

Validate that the StochasticMaze configuration has valid values.
"""
function validate_config(config::StochasticMazeConfig)
    config.time_horizon > 0 || throw(ArgumentError("time_horizon must be positive"))
    config.n_episodes > 0 || throw(ArgumentError("n_episodes must be positive"))
    config.n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    config.wait_time >= 0 || throw(ArgumentError("wait_time must be non-negative"))
end

"""
    initialize_beliefs_stochastic_maze(n_states::Int)

Initialize agent beliefs for the StochasticMaze environment.
"""
function initialize_beliefs_stochastic_maze(n_states::Int)
    # Initialize with uniform beliefs over states
    return StochasticMazeBeliefs(
        state=Categorical(fill(1.0 / n_states, n_states))
    )
end

"""
    stochastic_maze_convert_action(next_action::Int)

Convert model action index to environment action.
Action mapping: 1=North, 2=East, 3=South, 4=West
"""
function stochastic_maze_convert_action(next_action::Int)
    1 <= next_action <= 4 || throw(ArgumentError("Invalid action: $next_action"))
    return StochasticMazeAction(next_action)
end

stochastic_maze_convert_action(next_action::AbstractVector) = stochastic_maze_convert_action(argmax(next_action))

"""
    execute_step(env, observation, beliefs, model, tensors, config, goal, callbacks, time_remaining, previous_result, previous_action;
                constraints_fn, initialization_fn, options, ...)

Execute a single step of inference to determine the next action in the StochasticMaze environment.
Takes current observations and returns the next planned action.
"""
function execute_step(env, observation, beliefs, model, tensors, config, goal, callbacks, time_remaining, previous_result, previous_action;
    constraints_fn, initialization_fn, options=NamedTuple(), inference_kwargs...)

    # Convert previous action to one-hot encoding
    n_actions = 4
    previous_action_vec = zeros(Float64, n_actions)
    if !isnothing(previous_action)
        previous_action_vec[previous_action.index] = one(Float64)
    end

    # Get initialization from previous results or initialize fresh
    n_states = size(tensors.transition_tensor, 1)
    initialization = initialization_fn(n_states)

    # Create observation vector
    observation_vec = zeros(Float64, n_states)
    observation_vec[observation] = one(Float64)

    # Run inference
    result = infer(;
        model=model(
            A=tensors.observation_matrix,
            B=tensors.transition_tensor,
            p_s_0=beliefs.state,
            T=time_remaining,
            n_states=n_states,
            n_actions=n_actions,
            goal=goal
        ),
        data=(
            y_current=observation_vec,
            u_prev=previous_action_vec
        ),
        constraints=constraints_fn(),
        callbacks=callbacks,
        iterations=config.n_iterations,
        initialization=initialization,
        options=options,
        inference_kwargs...
    )

    # Select next action based on posterior
    next_action_idx = Int(mode(first(last(result.posteriors[:u]))))
    next_action = stochastic_maze_convert_action(next_action_idx)

    # Update beliefs
    beliefs.state = last(result.posteriors[:s_current])

    return next_action_idx, next_action, result
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
    elseif action_idx == 5
        return "Stay"
    else
        return "Unknown"
    end
end

"""
    run_stochastic_maze_single_episode(model, tensors, config, goal, callbacks, seed;
                                      constraints_fn, initialization_fn, record, options, inference_kwargs...)

Run a single episode in the StochasticMaze environment.
"""
function run_stochastic_maze_single_episode(model, tensors, config, goal, callbacks, seed;
    constraints_fn, initialization_fn, record=false, debug_mode=false, options=NamedTuple(), use_tikz=false,
    show_legend=false, inference_kwargs...)

    # Set up RNG
    rng = StableRNG(seed)

    # Create environment
    env = create_stochastic_maze(
        5, 5, 4,  # Default grid size and actions
        start_state=11  # Start in the middle of the grid
    )

    # Initialize beliefs
    n_states = size(tensors.transition_tensor, 1)
    beliefs = initialize_beliefs_stochastic_maze(n_states)

    # Initialize tracking variables
    total_reward = 0.0
    previous_result = nothing

    # Initial action (placeholder)
    next_action = nothing
    next_action_idx = 0

    # Get model name for frame saving
    model_name = try
        string(nameof(model))
    catch
        "unknown_model"
    end

    # Set up the directory for saving frames if requested
    frames_dir = nothing
    if record
        # Create directory structure: results/stochastic_maze/experiment_name/model_name/episode_seed
        frames_dir = datadir("results", "stochastic_maze", config.experiment_name, model_name, "episode_$(seed)")
        mkpath(frames_dir)
    end

    # Tracking data for detailed logging
    episode_data = Dict(
        "trajectory" => [],
        "actions" => [],
        "action_names" => [],
        "rewards" => [],
        "states" => [],
        "observations" => [],
        "timestamps" => [],
        "total_reward" => 0.0,
        "seed" => seed
    )

    # Sample initial observation
    observation = sample_observation(env)

    # Record initial state
    push!(episode_data["states"], env.agent_state)
    push!(episode_data["observations"], observation)
    push!(episode_data["timestamps"], 0)

    # Choose appropriate backend for plotting
    backend = use_tikz ? :pgfplotsx : :gr

    # Define extra kwargs for tikz if needed
    tikz_extra_kwargs = nothing
    if use_tikz
        # Define PGFPlotsX-specific options to improve appearance
        tikz_extra_kwargs = Dict(
            :plot => Dict(
                :scale_only_axis => true,
                :width => "\\textwidth",
                :legend_style => Dict(
                    :font => "\\footnotesize",
                    :row_sep => "3pt",
                    :legend_columns => 1
                )
            )
        )
    end

    # Save initial frame if requested
    if record
        initial_plot = visualize_stochastic_maze(env; show_legend=show_legend, backend=backend)
        save_frame(initial_plot, model_name, seed, 0, frames_dir;
            use_tikz=use_tikz, extra_kwargs=tikz_extra_kwargs)
    end

    # Log initial state if in debug mode
    if debug_mode
        @debug "Episode $(seed): Starting at state $(env.agent_state)"
    end

    # Run episode
    for t in config.time_horizon:-1:1
        # Plan the next action based on current observations
        next_action_idx, next_action, result = execute_step(
            env, observation, beliefs, model, tensors, config, goal,
            callbacks, t, previous_result, next_action;
            constraints_fn=constraints_fn,
            initialization_fn=initialization_fn,
            options=options,
            inference_kwargs...
        )

        # Update the previous result for the next iteration
        previous_result = result

        # Execute the planned action and get observations and reward
        observation, reward = step!(rng, env, next_action)

        # Update total reward
        total_reward += reward

        # Update tracking data for the action just executed
        push!(episode_data["actions"], next_action_idx)
        push!(episode_data["action_names"], action_to_string(next_action_idx))
        push!(episode_data["rewards"], reward)
        push!(episode_data["states"], env.agent_state)
        push!(episode_data["observations"], observation)

        # Current timestep (for frame numbering)
        current_timestep = config.time_horizon - t + 1
        push!(episode_data["timestamps"], current_timestep)

        # Save current frame if requested
        if record
            current_plot = visualize_stochastic_maze(env; show_legend=show_legend, backend=backend)
            save_frame(current_plot, model_name, seed, current_timestep, frames_dir;
                use_tikz=use_tikz, extra_kwargs=tikz_extra_kwargs)
        end

        # Log step information if in debug mode
        if debug_mode
            action_str = action_to_string(next_action_idx)
            @debug "Episode $(seed): t=$t, State=$(env.agent_state), Action=$action_str, Reward=$reward"
        end

        # Check if reached positive reward
        if reward >= 1.0
            if debug_mode
                @debug "Episode $(seed): Goal reached at t=$(t)!"
            end
            break
        end

        if reward == -1.0
            if debug_mode
                @debug "Episode $(seed): Negative reward at t=$(t)!"
            end
            break
        end

        # Delay for visualization
        sleep(config.wait_time)
    end

    # Compile the frames into a video if recording and not using tikz
    if record && !use_tikz
        # Compile the saved frames into a video
        video_path = datadir("results", "stochastic_maze", config.experiment_name, model_name, "episode_$(seed).mp4")
        convert_frames_to_video(frames_dir, video_path)
    end

    # Add final trajectory information
    episode_data["total_reward"] = total_reward
    episode_data["final_state"] = env.agent_state
    episode_data["seed"] = seed
    @debug "Episode $(seed) finished with total reward $(total_reward)"
    return total_reward, episode_data
end

"""
    convert_frames_to_video(frames_dir::String, video_path::String; framerate::Int=5)

Convert a directory of PNG frame images to a video file using the existing record_episode_to_video function.

# Arguments
- `frames_dir::String`: Directory containing the PNG frames
- `video_path::String`: Path where the video file should be saved
- `framerate::Int=5`: Frame rate for the video (default: 5 fps)

# Returns
- Nothing
"""
function convert_frames_to_video(frames_dir::String, video_path::String; framerate::Int=5)
    # Find all PNG files in the directory
    frame_files = filter(file -> endswith(file, ".png"), readdir(frames_dir, sort=true))

    if isempty(frame_files)
        @warn "No frames found in $frames_dir"
        return
    end

    # Sort frames by their filenames to ensure correct order
    sort!(frame_files, by=file -> match(r"frame_(\d+)\.png$", file).captures[1])

    # Load all frames and convert them to the format expected by record_episode_to_video
    frames = Vector{Array{UInt8,3}}()

    for file in frame_files
        # Load the image
        img = FileIO.load(joinpath(frames_dir, file))

        # Convert RGB image to UInt8 array with dimensions (height, width, 3)
        height, width = size(img)
        frame = Array{UInt8}(undef, height, width, 3)

        for i in 1:height, j in 1:width
            pixel = img[i, j]
            frame[i, j, 1] = round(UInt8, 255 * Float64(pixel.r))
            frame[i, j, 2] = round(UInt8, 255 * Float64(pixel.g))
            frame[i, j, 3] = round(UInt8, 255 * Float64(pixel.b))
        end

        push!(frames, frame)
    end

    # Use the existing record_episode_to_video function to create the video
    record_episode_to_video(frames, video_path)

    @info "Converted $(length(frames)) frames to video at $video_path"
end

"""
    run_stochastic_maze_agent(model, tensors, config, goal;
                             callbacks, constraints_fn, initialization_fn, parallel, options, inference_kwargs...)

Run a StochasticMaze agent experiment with the given model and configuration.
"""
function run_stochastic_maze_agent(
    model::Function,
    tensors,
    config::StochasticMazeConfig,
    goal::Categorical;
    callbacks=nothing,
    constraints_fn=() -> NamedTuple(),
    initialization_fn=(s, sf) -> NamedTuple(),
    parallel::Union{Nothing,Bool}=nothing,
    options=NamedTuple(),
    inference_kwargs...
)
    # Validate configuration
    validate_config(config)

    # Initialize rewards array
    rewards = zeros(config.n_episodes)

    # Determine if we should use parallel execution
    use_parallel = isnothing(parallel) ? config.parallel : parallel

    # Set up RNG
    rng = StableRNG(config.seed)

    # Create directory for results if recording episodes
    if config.record_episode
        # Base directory for all experiment results
        mkpath(datadir("results", "stochastic_maze", config.experiment_name))
    end

    if use_parallel
        thread_count = Threads.nthreads()
        @info "Running with parallelization using $thread_count threads"
        episode_seeds = rand(rng, UInt32, config.n_episodes)

        # Use Threads.@threads for parallelization
        progress = Progress(config.n_episodes; desc="Running episodes: ")

        Threads.@threads for i in 1:config.n_episodes
            episode_seed = episode_seeds[i]

            # Record only the last episode or specific episodes if record_episode is true
            should_record = config.record_episode && (i == config.n_episodes || config.record_episode == :all)

            rewards[i], _ = run_stochastic_maze_single_episode(
                model, tensors, config, goal, callbacks, episode_seed;
                constraints_fn=constraints_fn,
                initialization_fn=initialization_fn,
                record=should_record,
                options=options,
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
            # Record only the last episode or specific episodes if record_episode is true
            should_record = config.record_episode && (i == config.n_episodes || config.record_episode == :all)

            rewards[i], _ = run_stochastic_maze_single_episode(
                model, tensors, config, goal, callbacks, episode_seed;
                constraints_fn=constraints_fn,
                initialization_fn=initialization_fn,
                record=should_record,
                options=options,
                inference_kwargs...
            )
        end
    end

    return mean(rewards), std(rewards)
end