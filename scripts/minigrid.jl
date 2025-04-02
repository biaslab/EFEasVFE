using DrWatson
@quickactivate "EFEasVFE"

using Logging
using LoggingExtras
using JLD2
using Dates
using Statistics
using ArgParse
using TinyHugeNumbers
import RxInfer: Categorical
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
end

"""
    validate_parameters(grid_size, time_horizon, n_episodes)

Validate the experiment parameters and throw an error if they are invalid.
"""
function validate_parameters(grid_size, time_horizon, n_episodes)
    grid_size > 0 || throw(ArgumentError("grid_size must be positive"))
    grid_size <= 10 || throw(ArgumentError("grid_size must be <= 10 (current implementation limit)"))
    time_horizon > 0 || throw(ArgumentError("time_horizon must be positive"))
    n_episodes > 0 || throw(ArgumentError("n_episodes must be positive"))
end

"""
    load_tensors(grid_size)

Load the required tensors for the experiment.
"""
function load_tensors(grid_size)
    @info "Loading tensors for grid size $grid_size"

    observation_tensors = EFEasVFE.load_cp_observation_tensors("data/parafac_decomposed_tensors/grid_size$(grid_size)/")
    door_key_transition_tensor = EFEasVFE.load_cp_tensor("data/parafac_decomposed_tensors/grid_size$(grid_size)/door_key_transition_tensor")
    location_transition_tensor = EFEasVFE.load_cp_tensor("data/parafac_decomposed_tensors/grid_size$(grid_size)/location_transition_tensor")
    orientation_transition_tensor = EFEasVFE.get_orientation_transition_tensor()

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
    # Load tensors
    tensors = load_tensors(config.grid_size)

    # Create goal
    goal = create_goal(config.grid_size, config.number_type)

    # Create agent configuration
    agent_config = MinigridConfig(
        grid_size=config.grid_size,
        time_horizon=config.time_horizon,
        n_episodes=config.n_episodes,
        n_iterations=config.n_iterations,
        wait_time=config.wait_time,
        number_type=config.number_type
    )

    # Run KL control agent
    @info "Running KL control agent"
    m_kl, s_kl = run_minigrid_agent(
        klcontrol_minigrid_agent,
        tensors,
        agent_config,
        goal;
    )

    @info "Experiment completed" mean_reward = m_kl std_reward = s_kl

    # Save results if requested
    if config.save_results
        save_results(config, m_kl, s_kl)
    end

    return m_kl, s_kl
end

"""
    save_results(config::ExperimentConfig, mean_reward::Float64, std_reward::Float64)

Save experiment results to disk.
"""
function save_results(config::ExperimentConfig, mean_reward::Float64, std_reward::Float64)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results = Dict(
        "timestamp" => timestamp,
        "grid_size" => config.grid_size,
        "time_horizon" => config.time_horizon,
        "n_episodes" => config.n_episodes,
        "n_iterations" => config.n_iterations,
        "mean_reward" => mean_reward,
        "std_reward" => std_reward
    )

    results_file = datadir("results", "minigrid_$(timestamp).jld2")
    mkpath(dirname(results_file))
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
    end

    args = parse_args(s)

    # Validate parameters
    validate_parameters(args["grid-size"], args["time-horizon"], args["n-episodes"])

    # Convert number type string to actual type
    number_type = if args["number-type"] == :Float32
        Float32
    elseif args["number-type"] == :Float64
        Float64
    else
        throw(ArgumentError("Unsupported number type: $(args["number-type"])"))
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
        save_results=args["save-results"]
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
        save_results=args.save_results
    )

    # Run experiment
    mean_reward, std_reward = run_experiment(config)

    # Visualize if requested
    if args.visualize
        @info "Running visualization"
        visualize_episode(config)
    end
end

function visualize_episode(config::ExperimentConfig)
    # Create visualization config
    viz_config = MinigridConfig(
        grid_size=config.grid_size,
        time_horizon=config.time_horizon,
        n_episodes=1,
        n_iterations=3,
        wait_time=1.0,
        number_type=config.number_type
    )

    # Load required data
    tensors = load_tensors(config.grid_size)
    goal = create_goal(config.grid_size, config.number_type)
    callbacks = RxInferBenchmarkCallbacks()

    # Run visualization
    run_minigrid_agent(
        klcontrol_minigrid_agent,
        tensors,
        viz_config,
        goal;
        callbacks=callbacks
    )
end

# Run main function if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


