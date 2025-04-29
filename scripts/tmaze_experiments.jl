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
import RxInfer: Categorical
using EFEasVFE

@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    out_idx = findfirst(isone, probvec(q_out))
    result = eloga[out_idx, :]
    return Categorical(normalize!(guard(result), 1); check_args=false)
end

function run_tmaze_experiment(;
    T::Int=10,
    n_episodes::Int=10,
    n_iterations::Int=10,
    visualize::Bool=false,
    wait_time::Float64=0.0,
    record_episode::Bool=false,
    seed::Int=123,
    number_type::Type{<:AbstractFloat}=Float64,
    experiment_name::String="tmaze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    parallel::Bool=false,
    log_dir::String=datadir("logs", "tmaze"),
    save_results::Bool=true,
    debug_mode::Bool=false
)
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
        visualize=visualize,
        seed=seed,
        record_episode=record_episode,
        experiment_name=experiment_name,
        parallel=parallel
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

    # Run episodes
    if parallel
        thread_count = Threads.nthreads()
        log_info("Running with parallelization using $thread_count threads")
        episode_seeds = rand(StableRNG(seed), UInt32, config.n_episodes)

        # Use Threads.@threads for parallelization
        progress = Progress(config.n_episodes; desc="Running KL Control episodes: ")

        episodic_data_lock = ReentrantLock()

        Threads.@threads for i in 1:config.n_episodes
            episode_seed = episode_seeds[i]
            local_config = config

            if local_config.visualize && parallel && i != config.n_episodes
                # Turn off visualization for all but last episode in parallel mode
                local_config = TMazeConfig(
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

            reward, episode_data = run_tmaze_single_episode(
                klcontrol_tmaze_agent,
                tensors,
                local_config,
                left_goal_distribution,
                callbacks,
                episode_seed;
                constraints_fn=klcontrol_tmaze_agent_constraints,
                initialization_fn=klcontrol_tmaze_agent_initialization,
                record=i == config.n_episodes && config.record_episode,
                debug_mode=debug_mode
            )

            kl_rewards[i] = reward

            # Thread-safe update of episodic data
            lock(episodic_data_lock) do
                push!(kl_episodic_data, episode_data)
            end

            # Update progress
            ProgressMeter.next!(progress)
        end
    else
        # Sequential execution
        log_info("Running sequentially")
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
                debug_mode=debug_mode
            )

            kl_rewards[i] = reward
            push!(kl_episodic_data, episode_data)
        end
        log_info("KL Control episodes completed")
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
                options=(force_marginal_computation=true,
                    limit_stack_depth=500), # Force marginal computation
            )

            efe_rewards[i] = reward
            push!(efe_episodic_data, episode_data)
        end
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
            JSON.print(io, Dict("klcontrol" => kl_episodic_data), 2)
            JSON.print(io, Dict("efe" => efe_episodic_data), 2)
        end

        open(results_file, "w") do io
            JSON.print(io, experiment_metrics, 2)
        end

        log_info("Results saved to $results_file")
        log_info("Episode data saved to $episodic_results_file")
    end

    return experiment_metrics
end

# Run the experiment with default parameters
if abspath(PROGRAM_FILE) == @__FILE__
    run_tmaze_experiment(
        T=6,
        n_episodes=50,
        n_iterations=20,
        visualize=false,
        wait_time=0.0,
        record_episode=true,
        seed=123,
        number_type=Float64,
        experiment_name="tmaze_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
        parallel=false,
        debug_mode=false
    )
end