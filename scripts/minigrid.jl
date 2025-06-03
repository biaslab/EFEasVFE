using DrWatson
@quickactivate "EFEasVFE"

using Logging
using LoggingExtras
using JLD2
using Dates
using Statistics
using ArgParse
using TinyHugeNumbers
using RxInfer.GraphPPL
import RxInfer: Categorical
using JSON
using HTTP

# Import the main package
using EFEasVFE

function not_HTTP_message_filter(log)
    # HTTP.jl utilizes internal modules so call parentmodule(...)
    log._module !== HTTP && parentmodule(log._module) !== HTTP
end

# Configure logging
function setup_logging(verbosity::Symbol)
    level = if verbosity == :debug
        Logging.Debug
    elseif verbosity == :info
        Logging.Info
    else
        Logging.Warn
    end

    # Create a timestamped log file
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    log_file = datadir("logs", "minigrid_$(timestamp).log")
    mkpath(dirname(log_file))

    # Configure logging to both file and console
    logger = SimpleLogger(stdout, level)
    file_logger = SimpleLogger(open(log_file, "w"), level)

    # Create a filtered logger that excludes HTTP debug messages
    http_filtered_logger = EarlyFilteredLogger(not_HTTP_message_filter, TeeLogger(logger, file_logger))

    global_logger(http_filtered_logger)
    @info "Logging initialized" level timestamp log_file
end

"""
    ExperimentConfig

Configuration struct for the minigrid experiment.
"""
Base.@kwdef struct ExperimentConfig
    grid_size::Int
    time_horizon::Int
    n_episodes::Int
    n_iterations::Int
    wait_time::Float64
    number_type::Type{<:AbstractFloat}
    verbosity::Symbol
    visualize::Bool
    save_results::Bool
    seed::Int
    experiment_name::String = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")  # Default to timestamp
    save_video::Bool = false  # Default to false
    sparse_tensor::Bool = false
    parallel::Bool = false    # Whether to use parallel execution
end

function Base.show(io::IO, config::ExperimentConfig)
    println(io, "ExperimentConfig:")
    println(io, "grid_size=$(config.grid_size), ")
    println(io, "time_horizon=$(config.time_horizon), ")
    println(io, "n_episodes=$(config.n_episodes), ")
    println(io, "n_iterations=$(config.n_iterations), ")
    println(io, "wait_time=$(config.wait_time), ")
    println(io, "number_type=$(config.number_type), ")
    println(io, "verbosity=$(config.verbosity), ")
    println(io, "visualize=$(config.visualize), ")
    println(io, "save_results=$(config.save_results), ")
    println(io, "seed=$(config.seed), ")
    println(io, "experiment_name=$(config.experiment_name), ")
    println(io, "save_video=$(config.save_video)")
    println(io, "sparse_tensor=$(config.sparse_tensor)")
    println(io, "parallel=$(config.parallel)")
    if config.parallel
        println(io, "threads=", Threads.nthreads())
    end
end

"""
    validate_parameters(grid_size, time_horizon, n_episodes)

Validate the experiment parameters and throw an error if they are invalid.
"""
function validate_parameters(grid_size, time_horizon, n_episodes)
    grid_size > 0 || throw(ArgumentError("grid_size must be positive"))
    time_horizon > 0 || throw(ArgumentError("time_horizon must be positive"))
    n_episodes > 0 || throw(ArgumentError("n_episodes must be positive"))
end

"""
    load_tensors(grid_size)

Load the required tensors for the experiment.
"""
function load_tensors(grid_size, number_type; sparse_tensor=false)
    @info "Loading tensors for grid size $grid_size"
    if sparse_tensor
        observation_tensors = SparseArray.(eachslice(EFEasVFE.generate_observation_tensor(grid_size, number_type), dims=(1, 2)))
        door_key_transition_tensor = SparseArray(EFEasVFE.get_key_door_state_transition_tensor(grid_size, number_type))
        location_transition_tensor = SparseArray(EFEasVFE.get_self_transition_tensor(grid_size, number_type))
    else
        observation_tensors = collect(eachslice(EFEasVFE.generate_observation_tensor(grid_size, number_type), dims=(1, 2)))
        door_key_transition_tensor = EFEasVFE.get_key_door_state_transition_tensor(grid_size, number_type)
        location_transition_tensor = EFEasVFE.get_self_transition_tensor(grid_size, number_type)
    end
    orientation_transition_tensor = EFEasVFE.get_orientation_transition_tensor(number_type)
    @debug "Tensors loaded successfully"

    return (
        observation=observation_tensors,
        door_key=door_key_transition_tensor,
        location=location_transition_tensor,
        orientation=orientation_transition_tensor
    )
end

"""
    create_goal(grid_size, T::Type{<:AbstractFloat})

Create the goal distribution for the experiment.
"""
function create_goal(grid_size, T::Type{<:AbstractFloat})
    @debug "Creating goal distribution for grid size $grid_size"
    goal = zeros(T, grid_size^2) .+ tiny
    goal[grid_size^2-grid_size+1] = one(T)
    return Categorical(goal ./ sum(goal))
end

"""
    run_experiment(config::ExperimentConfig)

Run the minigrid experiment with the given configuration.
"""
function run_experiment(config::ExperimentConfig)
    @info "Starting experiment" config

    if config.parallel
        @info "Using parallel execution with $(Threads.nthreads()) threads"
    end

    # Load tensors
    tensors = load_tensors(config.grid_size, config.number_type; sparse_tensor=config.sparse_tensor)

    # Create results directory
    mkpath(datadir("results", "minigrid", config.experiment_name))

    # Create goal
    goal = create_goal(config.grid_size, config.number_type)

    # Create agent configuration
    agent_config = MinigridConfig(
        grid_size=config.grid_size,
        time_horizon=config.time_horizon,
        n_episodes=config.n_episodes,
        n_iterations=config.n_iterations,
        wait_time=config.wait_time,
        number_type=config.number_type,
        visualize=config.visualize,
        seed=config.seed,
        record_episode=config.save_video,
        experiment_name=config.experiment_name,
        parallel=config.parallel  # Pass through parallel option
    )

    # Run KL control agent
    @info "Running KL control agent"
    kl_stats = run_minigrid_agent(
        klcontrol_minigrid_agent,
        tensors,
        agent_config,
        goal;
        parallel=config.parallel            # Explicitly set parallel execution
    )

    # @info "Running EFE agent"
    # efe_stats = run_minigrid_agent(
    #     efe_minigrid_agent,
    #     tensors,
    #     agent_config,
    #     goal,
    #     constraints_fn=efe_minigrid_agent_constraints,
    #     initialization_fn=efe_minigrid_agent_initialization;
    #     parallel=config.parallel,            # Explicitly set parallel execution
    #     options=(force_marginal_computation=true,
    #         limit_stack_depth=500), # Force marginal computation
    # )

    # Log detailed statistics
    @info "KL Control agent results" mean_reward = kl_stats.mean_reward std_reward = kl_stats.std_reward success_rate = kl_stats.success_rate mean_key_visible_time = kl_stats.mean_key_visible_time mean_door_visible_time = kl_stats.mean_door_visible_time key_never_visible = kl_stats.key_never_visible key_visible_at_start = kl_stats.key_visible_at_start door_never_visible = kl_stats.door_never_visible door_visible_at_start = kl_stats.door_visible_at_start

    # @info "EFE agent results" mean_reward = efe_stats.mean_reward std_reward = efe_stats.std_reward success_rate = efe_stats.success_rate mean_key_visible_time = efe_stats.mean_key_visible_time mean_door_visible_time = efe_stats.mean_door_visible_time key_never_visible = efe_stats.key_never_visible key_visible_at_start = efe_stats.key_visible_at_start door_never_visible = efe_stats.door_never_visible door_visible_at_start = efe_stats.door_visible_at_start
    efe_stats = kl_stats
    @info "Experiment completed"

    # Save results if requested
    if config.save_results
        save_results(config, kl_stats, efe_stats)
    end

    return kl_stats, efe_stats
end

"""
    save_results(config::ExperimentConfig, kl_stats::NamedTuple, efe_stats::NamedTuple)

Save experiment results to disk.
"""
function save_results(config::ExperimentConfig, kl_stats::NamedTuple, efe_stats::NamedTuple)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

    # Create base result dictionary with experiment configuration
    results = Dict(
        "timestamp" => timestamp,
        "grid_size" => config.grid_size,
        "time_horizon" => config.time_horizon,
        "n_episodes" => config.n_episodes,
        "n_iterations" => config.n_iterations,
        "seed" => config.seed,
        "experiment_name" => config.experiment_name,
        "parallel" => config.parallel,
        "thread_count" => config.parallel ? Threads.nthreads() : 1
    )

    # Add KL control results
    results["kl_control"] = Dict(
        "mean_reward" => kl_stats.mean_reward,
        "std_reward" => kl_stats.std_reward,
        "success_rate" => kl_stats.success_rate,
        "mean_key_visible_time" => kl_stats.mean_key_visible_time,
        "std_key_visible_time" => kl_stats.std_key_visible_time,
        "mean_door_visible_time" => kl_stats.mean_door_visible_time,
        "std_door_visible_time" => kl_stats.std_door_visible_time,
        "key_never_visible" => kl_stats.key_never_visible,
        "key_visible_at_start" => kl_stats.key_visible_at_start,
        "door_never_visible" => kl_stats.door_never_visible,
        "door_visible_at_start" => kl_stats.door_visible_at_start,
        "model_source" => GraphPPL.getsource(klcontrol_minigrid_agent())
    )

    # Add EFE results
    results["efe"] = Dict(
        "mean_reward" => efe_stats.mean_reward,
        "std_reward" => efe_stats.std_reward,
        "success_rate" => efe_stats.success_rate,
        "mean_key_visible_time" => efe_stats.mean_key_visible_time,
        "std_key_visible_time" => efe_stats.std_key_visible_time,
        "mean_door_visible_time" => efe_stats.mean_door_visible_time,
        "std_door_visible_time" => efe_stats.std_door_visible_time,
        "key_never_visible" => efe_stats.key_never_visible,
        "key_visible_at_start" => efe_stats.key_visible_at_start,
        "door_never_visible" => efe_stats.door_never_visible,
        "door_visible_at_start" => efe_stats.door_visible_at_start,
        "model_source" => GraphPPL.getsource(efe_minigrid_agent())
    )

    # Save results in multiple formats
    base_filename = config.experiment_name
    results_file = datadir("results", "minigrid", base_filename, base_filename * ".jld2")
    results_json = datadir("results", "minigrid", base_filename, base_filename * ".json")
    results_md = datadir("results", "minigrid", base_filename, base_filename * ".md")

    # Create JSON string with formatted results
    json_str = JSON.json(results, 2)

    # Create markdown report
    md_content = """
    # Minigrid Experiment Results

    ## Experiment Configuration
    - Grid Size: $(config.grid_size)
    - Time Horizon: $(config.time_horizon) 
    - Number of Episodes: $(config.n_episodes)
    - Number of Iterations: $(config.n_iterations)
    - Seed: $(config.seed)
    - Experiment Name: $(config.experiment_name)
    - Parallel Execution: $(config.parallel)
    $(config.parallel ? "- Thread Count: $(Threads.nthreads())" : "")

    ## KL Control Results
    - Mean Reward: $(round(kl_stats.mean_reward, digits=3))
    - Standard Deviation: $(round(kl_stats.std_reward, digits=3))
    - Success Rate: $(round(kl_stats.success_rate * 100, digits=1))%
    - Mean Key Visible Time: $(kl_stats.mean_key_visible_time == -1.0 ? "N/A" : round(kl_stats.mean_key_visible_time, digits=2)) $(kl_stats.std_key_visible_time == -1.0 ? "" : "±$(round(kl_stats.std_key_visible_time, digits=2))")
    - Mean Door Visible Time: $(kl_stats.mean_door_visible_time == -1.0 ? "N/A" : round(kl_stats.mean_door_visible_time, digits=2)) $(kl_stats.std_door_visible_time == -1.0 ? "" : "±$(round(kl_stats.std_door_visible_time, digits=2))")
    - Key Never Visible: $(kl_stats.key_never_visible) episodes ($(round(kl_stats.key_never_visible / config.n_episodes * 100, digits=1))%)
    - Key Visible at Start: $(kl_stats.key_visible_at_start) episodes ($(round(kl_stats.key_visible_at_start / config.n_episodes * 100, digits=1))%)
    - Door Never Visible: $(kl_stats.door_never_visible) episodes ($(round(kl_stats.door_never_visible / config.n_episodes * 100, digits=1))%)
    - Door Visible at Start: $(kl_stats.door_visible_at_start) episodes ($(round(kl_stats.door_visible_at_start / config.n_episodes * 100, digits=1))%)

    ## EFE Results
    - Mean Reward: $(round(efe_stats.mean_reward, digits=3))
    - Standard Deviation: $(round(efe_stats.std_reward, digits=3))
    - Success Rate: $(round(efe_stats.success_rate * 100, digits=1))%
    - Mean Key Visible Time: $(efe_stats.mean_key_visible_time == -1.0 ? "N/A" : round(efe_stats.mean_key_visible_time, digits=2)) $(efe_stats.std_key_visible_time == -1.0 ? "" : "±$(round(efe_stats.std_key_visible_time, digits=2))")
    - Mean Door Visible Time: $(efe_stats.mean_door_visible_time == -1.0 ? "N/A" : round(efe_stats.mean_door_visible_time, digits=2)) $(efe_stats.std_door_visible_time == -1.0 ? "" : "±$(round(efe_stats.std_door_visible_time, digits=2))")
    - Key Never Visible: $(efe_stats.key_never_visible) episodes ($(round(efe_stats.key_never_visible / config.n_episodes * 100, digits=1))%)
    - Key Visible at Start: $(efe_stats.key_visible_at_start) episodes ($(round(efe_stats.key_visible_at_start / config.n_episodes * 100, digits=1))%)
    - Door Never Visible: $(efe_stats.door_never_visible) episodes ($(round(efe_stats.door_never_visible / config.n_episodes * 100, digits=1))%)
    - Door Visible at Start: $(efe_stats.door_visible_at_start) episodes ($(round(efe_stats.door_visible_at_start / config.n_episodes * 100, digits=1))%)

    ## Timestamp
    Experiment conducted at: $timestamp

    ## KL Control Model
    ```
    $(results["kl_control"]["model_source"])
    ```

    ## EFE Model
    ```
    $(results["efe"]["model_source"])
    ```
    """

    mkpath(dirname(results_file))
    # Write additional formats
    write(results_json, json_str)
    write(results_md, md_content)

    @save results_file results

    @info "Results saved to $results_file"
end

"""
    parse_command_line()

Parse command line arguments for the experiment.
"""
function parse_command_line()
    s = ArgParseSettings()
    s.description = "Run minigrid experiment with specified parameters"
    s.version = "1.0.0"

    @add_arg_table! s begin
        "--grid-size"
        help = "Size of the grid (default: 3)"
        arg_type = Int
        default = 3
        "--time-horizon"
        help = "Maximum number of steps per episode (default: 15)"
        arg_type = Int
        default = 15
        "--n-episodes"
        help = "Number of episodes to run (default: 10)"
        arg_type = Int
        default = 10
        "--n-iterations"
        help = "Number of iterations per step (default: 70)"
        arg_type = Int
        default = 70
        "--wait-time"
        help = "Time to wait between steps in seconds (default: 0.0)"
        arg_type = Float64
        default = 0.0
        "--number-type"
        help = "Number type to use (default: Float32)"
        arg_type = Symbol
        default = :Float32
        "--visualize"
        help = "Enable visualization"
        action = :store_true
        "--save-results"
        help = "Save experiment results"
        action = :store_true
        "--verbosity"
        help = "Logging verbosity level (debug, info, warn) (default: info)"
        arg_type = Symbol
        default = :info
        "--seed"
        help = "Random seed for the experiment"
        arg_type = Int
        default = 42
        "--experiment-name"
        help = "Name for the experiment (default: current timestamp)"
        arg_type = String
        default = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        "--save-video"
        help = "Save video of the last episode"
        action = :store_true
        "--sparse-tensor"
        help = "Use sparse representation for transition and observation tensors"
        action = :store_true
        "--parallel"
        help = "Enable parallel execution of episodes using $(Threads.nthreads()) threads"
        action = :store_true
    end

    args = parse_args(s)

    # Validate parameters
    validate_parameters(args["grid-size"], args["time-horizon"], args["n-episodes"])

    # Convert number type string to actual type
    number_type = if args["number-type"] == :Float32
        Float32
    elseif args["number-type"] == :Float64
        Float64
    elseif args["number-type"] == :Float16
        Float16
    else
        throw(ArgumentError("Unsupported number type: $(args["number-type"])"))
    end

    # Check for incompatible visualization and video saving options
    if args["save-video"] && args["visualize"]
        @warn "Video saving is not compatible with visualization mode. No video will be saved."
        args["save-video"] = false
    end

    return (
        grid_size=args["grid-size"],
        time_horizon=args["time-horizon"],
        n_episodes=args["n-episodes"],
        n_iterations=args["n-iterations"],
        wait_time=args["wait-time"],
        number_type=number_type,
        verbosity=args["verbosity"],
        visualize=args["visualize"],
        save_results=args["save-results"],
        seed=args["seed"],
        experiment_name=args["experiment-name"],
        save_video=args["save-video"],
        sparse_tensor=args["sparse-tensor"],
        parallel=args["parallel"]
    )
end

function main()
    # Parse command line arguments
    args = parse_command_line()

    # Set up logging
    setup_logging(args.verbosity)

    # Set up experiment parameters
    config = ExperimentConfig(
        grid_size=args.grid_size,
        time_horizon=args.time_horizon,
        n_episodes=args.n_episodes,
        n_iterations=args.n_iterations,
        wait_time=args.wait_time,
        number_type=args.number_type,
        verbosity=args.verbosity,
        visualize=args.visualize,
        save_results=args.save_results,
        seed=args.seed,
        experiment_name=args.experiment_name,
        save_video=args.save_video,
        sparse_tensor=args.sparse_tensor,
        parallel=args.parallel
    )

    # Run experiment
    kl_stats, efe_stats = run_experiment(config)

    return kl_stats, efe_stats
end

# Run main function if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
