using DrWatson
@quickactivate "EFEasVFE"

using RxInfer
using ReactiveMP
using ProgressMeter
using Statistics
using Distributions
using StableRNGs
using Dates
using JSON
using FileIO
using ArgParse
using Plots
pgfplotsx()
import RxInfer: Categorical
using EFEasVFE

@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))
    result = eloga[out_idx, :]
    return Categorical(normalize!(guard(result), 1); check_args=false)
end

"""
    parse_command_line()

Parse command line arguments for the TMaze experiment.
"""
function parse_command_line()
    s = ArgParseSettings()
    s.description = "Run TMaze experiment with specified parameters"
    s.version = "1.0.0"

    @add_arg_table! s begin
        "--time-horizon"
        help = "Maximum number of steps per episode (default: 6)"
        arg_type = Int
        default = 6
        "--n-episodes", "-e"
        help = "Number of episodes to run (default: 50)"
        arg_type = Int
        default = 50
        "--n-iterations", "-i"
        help = "Number of iterations per step (default: 20)"
        arg_type = Int
        default = 20
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
        default = "tmaze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
        "--tikz"
        help = "Use PGFPlotsX backend and save frames as TikZ files instead of PNG"
        action = :store_true
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
        save_results=save_results,
        use_tikz=args["tikz"]
    )
end

function run_tmaze_experiment(;
    T::Int=6,
    n_episodes::Int=50,
    n_iterations::Int=20,
    wait_time::Float64=0.0,
    record_episode::Bool=false,
    seed::Int=123,
    number_type::Type{<:AbstractFloat}=Float64,
    experiment_name::String="tmaze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    log_dir::String=datadir("logs", "tmaze"),
    save_results::Bool=true,
    debug_mode::Bool=false,
    use_tikz::Bool=false
)
    # Set the appropriate plotting backend
    if use_tikz
        pgfplotsx()
    else
        gr()
    end

    # Create directory for logs
    results_dir = datadir("results", "tmaze", experiment_name)
    if save_results
        mkpath(results_dir)
        log_file = joinpath(results_dir, "logs.log")
        results_file = joinpath(results_dir, "results.json")
        episodic_results_file = joinpath(results_dir, "episodes.json")
    end

    # Create goal distribution - prefer the left arm location (state 3)
    left_goal = zeros(number_type, 5)
    left_goal[3] = 1.0
    left_goal_distribution = Categorical(left_goal)

    # Create config
    config = TMazeConfig(
        time_horizon=T,
        n_episodes=n_episodes,
        n_iterations=n_iterations,
        wait_time=wait_time,
        number_type=number_type,
        seed=seed,
        record_episode=record_episode,
        experiment_name=experiment_name
    )

    # Create tensors
    tensors = (
        reward_observation=create_reward_observation_tensor(),
        location_transition=create_location_transition_tensor(),
        reward_to_location=create_reward_to_location_mapping()
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

    # Run experiments with KL control model
    log_info("Running experiments with KL Control agent...")

    # Initialize storage for episodic data
    kl_episodic_data = []
    kl_rewards = zeros(config.n_episodes)

    efe_episodic_data = []
    efe_rewards = zeros(config.n_episodes)

    # Sequential execution
    log_info("Running KL Control episodes")
    episode_seeds = rand(StableRNG(seed), UInt32, config.n_episodes)

    @showprogress desc = "Running KL Control episodes: " for i in 1:config.n_episodes
        episode_seed = episode_seeds[i]

        reward, episode_data = run_tmaze_single_episode(
            klcontrol_tmaze_agent,
            tensors,
            config,
            left_goal_distribution,
            callbacks,
            episode_seed;
            constraints_fn=klcontrol_tmaze_agent_constraints,
            initialization_fn=klcontrol_tmaze_agent_initialization,
            record=i == config.n_episodes && config.record_episode,
            debug_mode=debug_mode,
            use_tikz=use_tikz
        )

        kl_rewards[i] = reward
        push!(kl_episodic_data, episode_data)
    end

    log_info("KL Control episodes completed")

    log_info("Running EFE episodes")
    @showprogress desc = "Running EFE episodes: " for i in 1:config.n_episodes
        episode_seed = episode_seeds[i]

        reward, episode_data = run_tmaze_single_episode(
            efe_tmaze_agent,
            tensors,
            config,
            left_goal_distribution,
            callbacks,
            episode_seed;
            constraints_fn=efe_tmaze_agent_constraints,
            initialization_fn=efe_tmaze_agent_initialization,
            record=i == config.n_episodes && config.record_episode,
            debug_mode=debug_mode,
            use_tikz=use_tikz,
            options=(force_marginal_computation=true,
                limit_stack_depth=500), # Force marginal computation
        )

        efe_rewards[i] = reward
        push!(efe_episodic_data, episode_data)
    end

    # Calculate statistics
    kl_mean = mean(kl_rewards)
    kl_std = std(kl_rewards)
    efe_mean = mean(efe_rewards)
    efe_std = std(efe_rewards)

    # Record results
    log_info("KL Control results: mean reward = $kl_mean, std = $kl_std")
    experiment_metrics["models"]["klcontrol"] = Dict(
        "mean_reward" => kl_mean,
        "std_reward" => kl_std,
        "rewards" => kl_rewards,
    )
    log_info("EFE results: mean reward = $efe_mean, std = $efe_std")
    experiment_metrics["models"]["efe"] = Dict(
        "mean_reward" => efe_mean,
        "std_reward" => efe_std,
        "rewards" => efe_rewards,
    )

    # Save episodic data
    if save_results
        open(episodic_results_file, "w") do io
            JSON.print(io, Dict(
                    "klcontrol" => kl_episodic_data,
                    "efe" => efe_episodic_data
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
    run_tmaze_experiment(
        T=args.time_horizon,
        n_episodes=args.n_episodes,
        n_iterations=args.n_iterations,
        wait_time=args.wait_time,
        record_episode=args.record_episode,
        seed=args.seed,
        number_type=args.number_type,
        experiment_name=args.experiment_name,
        save_results=args.save_results,
        debug_mode=args.debug_mode,
        use_tikz=args.use_tikz
    )
end

# Run the experiment with command line arguments if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end