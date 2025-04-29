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
import RxInfer: Categorical
using EFEasVFE

# Set up experiment parameters
function run_stochastic_maze_experiment(;
    T::Int=9,
    n_episodes::Int=100,
    n_iterations::Int=3,
    visualize::Bool=false,
    wait_time::Float64=0.0,
    record_episode::Bool=false,
    seed::Int=123,
    number_type::Type{<:AbstractFloat}=Float64,
    experiment_name::String="stochastic_maze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    parallel::Bool=false,
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
        visualize=visualize,
        seed=seed,
        record_episode=record_episode,
        experiment_name=experiment_name,
        parallel=parallel
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

    if parallel
        thread_count = Threads.nthreads()
        log_info("Running with parallelization using $thread_count threads")
        episode_seeds = rand(StableRNG(seed), UInt32, config.n_episodes)

        episodic_data_lock = ReentrantLock()

        Threads.@threads for i in 1:config.n_episodes
            episode_seed = episode_seeds[i]
            local_config = config

            if local_config.visualize && parallel && i != config.n_episodes
                # Turn off visualization for all but last episode in parallel mode
                local_config = StochasticMazeConfig(
                    time_horizon=config.time_horizon,
                    n_episodes=config.n_episodes,
                    n_iterations=config.n_iterations,
                    wait_time=config.wait_time,
                    number_type=config.number_type,
                    visualize=false,
                    seed=config.seed,
                    record_episode=i == config.n_episodes && config.record_episode,
                    experiment_name=config.experiment_name,
                    parallel=parallel
                )
            end

            reward, episode_data = run_stochastic_maze_single_episode(
                klcontrol_stochastic_maze_agent,
                tensors,
                local_config,
                goal_distribution,
                callbacks,
                episode_seed;
                constraints_fn=klcontrol_stochastic_maze_agent_constraints,
                initialization_fn=klcontrol_stochastic_maze_agent_initialization,
                record=i == config.n_episodes && config.record_episode,
                debug_mode=debug_mode
            )

            kl_rewards[i] = reward

            # Thread-safe update of episodic data
            lock(episodic_data_lock) do
                push!(kl_episodic_data, episode_data)
            end
        end
    else
        # Sequential execution
        log_info("Running sequentially")
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
    if parallel
        thread_count = Threads.nthreads()
        log_info("Running with parallelization using $thread_count threads")
        episode_seeds = rand(StableRNG(seed), UInt32, config.n_episodes)

        episodic_data_lock = ReentrantLock()

        Threads.@threads for i in 1:config.n_episodes
            episode_seed = episode_seeds[i]
            local_config = config

            if local_config.visualize && parallel && i != config.n_episodes
                # Turn off visualization for all but last episode in parallel mode
                local_config = StochasticMazeConfig(
                    time_horizon=config.time_horizon,
                    n_episodes=config.n_episodes,
                    n_iterations=config.n_iterations,
                    wait_time=config.wait_time,
                    number_type=config.number_type,
                    visualize=false,
                    seed=config.seed,
                    record_episode=i == config.n_episodes && config.record_episode,
                    experiment_name=config.experiment_name,
                    parallel=parallel
                )
            end

            reward, episode_data = run_stochastic_maze_single_episode(
                efe_stochastic_maze_agent,
                tensors,
                local_config,
                goal_distribution,
                callbacks,
                episode_seed;
                constraints_fn=efe_stochastic_maze_agent_constraints,
                initialization_fn=efe_stochastic_maze_agent_initialization,
                record=i == config.n_episodes && config.record_episode,
                options=(force_marginal_computation=true,
                    limit_stack_depth=500), # Force marginal computation
                debug_mode=debug_mode
            )

            efe_rewards[i] = reward

            # Thread-safe update of episodic data
            lock(episodic_data_lock) do
                push!(efe_episodic_data, episode_data)
            end
        end
    else
        # Sequential execution
        log_info("Running sequentially")
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

# Run the experiment with default parameters if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_stochastic_maze_experiment(
        T=11,
        n_episodes=100,
        n_iterations=10,
        visualize=false,
        wait_time=0.0,
        record_episode=true,
        seed=123,
        number_type=Float64,
        experiment_name="stochastic_maze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
        parallel=false,
        debug_mode=true
    )
end