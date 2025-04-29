using DrWatson
@quickactivate "EFEasVFE"

using RxInfer
using Distributions
using ProgressMeter
using Statistics
using StableRNGs
using Dates
using JSON
using FileIO
using ArgParse
import RxInfer: Categorical
using EFEasVFE

"""
    parse_command_line()

Parse command line arguments for the stochastic maze experiment.
"""
function parse_command_line()
    s = ArgParseSettings()
    s.description = "Run stochastic maze experiment with specified parameters"
    s.version = "1.0.0"

    @add_arg_table! s begin
        "--time-horizon", "-t"
        help = "Maximum number of steps per episode (default: 11)"
        arg_type = Int
        default = 11
        "--n-episodes", "-e"
        help = "Number of episodes to run (default: 100)"
        arg_type = Int
        default = 100
        "--n-iterations", "-i"
        help = "Number of iterations per step (default: 10)"
        arg_type = Int
        default = 10
        "--wait-time", "-w"
        help = "Time to wait between steps in seconds (default: 0.0)"
        arg_type = Float64
        default = 0.0
        "--number-type", "-n"
        help = "Number type to use (default: Float64)"
        arg_type = Symbol
        default = :Float64
        "--seed", "-s"
        help = "Random seed for the experiment"
        arg_type = Int
        default = 123
        "--record-episode", "-r"
        help = "Record episode frames for visualization"
        action = :store_true
        "--experiment-name", "-x"
        help = "Name for the experiment (default: timestamp-based)"
        arg_type = String
        default = "stochastic_maze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
        "--debug", "-d"
        help = "Enable debug mode"
        action = :store_true
        "--save-results"
        help = "Save experiment results"
        action = :store_true
        "--no-save-results"
        help = "Don't save experiment results"
        action = :store_true
    end

    args = parse_args(s)

    # Validate parameters
    args["time-horizon"] > 0 || throw(ArgumentError("time-horizon must be positive"))
    args["n-episodes"] > 0 || throw(ArgumentError("n-episodes must be positive"))
    args["n-iterations"] > 0 || throw(ArgumentError("n-iterations must be positive"))
    args["wait-time"] >= 0 || throw(ArgumentError("wait-time must be non-negative"))

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

    # Handle save_results argument logic
    save_results = true
    if args["no-save-results"]
        save_results = false
    elseif args["save-results"]
        save_results = true
    end

    return (
        time_horizon=args["time-horizon"],
        n_episodes=args["n-episodes"],
        n_iterations=args["n-iterations"],
        wait_time=args["wait-time"],
        number_type=number_type,
        seed=args["seed"],
        record_episode=args["record-episode"],
        experiment_name=args["experiment-name"],
        debug_mode=args["debug"],
        save_results=save_results
    )
end

# Set up experiment parameters
function run_stochastic_maze_experiment(;
    T::Int=11,
    n_episodes::Int=100,
    n_iterations::Int=10,
    wait_time::Float64=0.0,
    record_episode::Bool=false,
    seed::Int=123,
    number_type::Type{<:AbstractFloat}=Float64,
    experiment_name::String="stochastic_maze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    log_dir::String=datadir("logs", "stochastic_maze"),
    save_results::Bool=true,
    debug_mode::Bool=false
)
    # Create directory for logs
    results_dir = datadir("results", "stochastic_maze", experiment_name)

    if save_results
        mkpath(results_dir)
        log_file = joinpath(results_dir, "logs.log")
        results_file = joinpath(results_dir, "results.json")
        episodic_results_file = joinpath(results_dir, "episodes.json")
    end

    # Set up goal state in the middle of the top row (state 15)
    goal_state = 15
    n_states = 25  # 5x5 grid
    n_actions = 4  # NESW

    # Create goal distribution focused on the goal state
    p_goal = zeros(number_type, n_states)
    p_goal[goal_state] = 1.0
    goal_distribution = Categorical(p_goal)

    # Create config
    config = StochasticMazeConfig(
        time_horizon=T,
        n_episodes=n_episodes,
        n_iterations=n_iterations,
        wait_time=wait_time,
        number_type=number_type,
        seed=seed,
        record_episode=record_episode,
        experiment_name=experiment_name
    )

    # Generate environment tensors
    A, B, reward_states = generate_maze_tensors(5, 5, n_actions)

    # Create tensors object
    tensors = (
        observation_matrix=A,
        transition_tensor=B,
        reward_states=reward_states
    )

    # Create benchmark callbacks
    callbacks = RxInferBenchmarkCallbacks()

    # Initialize result metrics
    experiment_metrics = Dict{String,Any}(
        "experiment_name" => experiment_name,
        "date" => string(now()),
        "time_horizon" => T,
        "n_episodes" => n_episodes,
        "n_iterations" => n_iterations,
        "seed" => seed,
        "models" => Dict{String,Any}()
    )

    # Function to log information to both console and log file
    function log_info(message)
        println(message)
        if save_results
            open(log_file, "a") do io
                println(io, "$(now()) - $message")
            end
        end
    end

    # Run KL-Control agent episodes
    log_info("Running experiments with KL-Control agent...")
    kl_episodic_data = []
    kl_rewards = zeros(config.n_episodes)

    episode_seeds = rand(StableRNG(seed), UInt32, config.n_episodes)

    @showprogress desc = "Running KL-Control episodes: " for i in 1:config.n_episodes
        episode_seed = episode_seeds[i]

        reward, episode_data = run_stochastic_maze_single_episode(
            klcontrol_stochastic_maze_agent,
            tensors,
            config,
            goal_distribution,
            callbacks,
            episode_seed;
            constraints_fn=klcontrol_stochastic_maze_agent_constraints,
            initialization_fn=klcontrol_stochastic_maze_agent_initialization,
            record=i == config.n_episodes && config.record_episode,
            debug_mode=debug_mode
        )

        kl_rewards[i] = reward
        push!(kl_episodic_data, episode_data)
    end
    # Calculate KL-Control statistics
    kl_mean = mean(kl_rewards)
    kl_std = std(kl_rewards)
    log_info("KL-Control agent results: mean reward = $kl_mean, std = $kl_std")

    # Initialize storage for episodic data
    log_info("Running experiments with EFE agent...")
    efe_episodic_data = []
    efe_rewards = zeros(config.n_episodes)

    # Run EFE episodes
    episode_seeds = rand(StableRNG(seed), UInt32, config.n_episodes)

    @showprogress desc = "Running EFE episodes: " for i in 1:config.n_episodes
        episode_seed = episode_seeds[i]

        reward, episode_data = run_stochastic_maze_single_episode(
            efe_stochastic_maze_agent,
            tensors,
            config,
            goal_distribution,
            callbacks,
            episode_seed;
            constraints_fn=efe_stochastic_maze_agent_constraints,
            initialization_fn=efe_stochastic_maze_agent_initialization,
            record=i == config.n_episodes && config.record_episode,
            options=(force_marginal_computation=true,),
            debug_mode=debug_mode
        )

        efe_rewards[i] = reward
        push!(efe_episodic_data, episode_data)
    end

    # Calculate EFE statistics
    efe_mean = mean(efe_rewards)
    efe_std = std(efe_rewards)
    log_info("EFE agent results: mean reward = $efe_mean, std = $efe_std")

    # Record results
    experiment_metrics["models"]["efe"] = Dict(
        "mean_reward" => efe_mean,
        "std_reward" => efe_std,
        "rewards" => efe_rewards,
    )
    experiment_metrics["models"]["klcontrol"] = Dict(
        "mean_reward" => kl_mean,
        "std_reward" => kl_std,
        "rewards" => kl_rewards,
    )

    # Save episodic data
    if save_results
        open(episodic_results_file, "w") do io
            JSON.print(io, Dict(
                    "efe" => efe_episodic_data,
                    "klcontrol" => kl_episodic_data
                ), 2)
        end

        open(results_file, "w") do io
            JSON.print(io, experiment_metrics, 2)
        end

        log_info("Results saved to $results_file")
        log_info("Episode data saved to $episodic_results_file")
    end

    return experiment_metrics
end

function main()
    # Parse command line arguments
    args = parse_command_line()

    # Run experiment with parsed arguments
    run_stochastic_maze_experiment(
        T=args.time_horizon,
        n_episodes=args.n_episodes,
        n_iterations=args.n_iterations,
        wait_time=args.wait_time,
        record_episode=args.record_episode,
        seed=args.seed,
        number_type=args.number_type,
        experiment_name=args.experiment_name,
        save_results=args.save_results,
        debug_mode=args.debug_mode
    )
end

# Run the experiment with default parameters if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end