using DrWatson
@quickactivate "EFEasVFE"

using Logging
using LoggingExtras
using JLD2
using Dates
using Statistics
using ArgParse
using RxInfer.GraphPPL
import RxInfer: Categorical, mode
using StableRNGs
using JSON
using VideoIO
using Colors
import Colors: N0f8
using EFEasVFE
using Plots
using TinyHugeNumbers
pgfplotsx()

import EFEasVFE: create_stochastic_maze, step!, generate_maze_tensors, sample_observation
import EFEasVFE: initialize_beliefs_stochastic_maze, stochastic_maze_convert_action, execute_step, action_to_string
import EFEasVFE: efe_stochastic_maze_agent, efe_stochastic_maze_agent_constraints, efe_stochastic_maze_agent_initialization
import EFEasVFE: klcontrol_stochastic_maze_agent, klcontrol_stochastic_maze_agent_constraints, klcontrol_stochastic_maze_agent_initialization
import EFEasVFE: plot_inference_results_stochastic_maze, visualize_stochastic_maze

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--time-horizon"
        help = "Time horizon for planning"
        arg_type = Int
        default = 10
        "--iterations"
        help = "Number of iterations for inference"
        arg_type = Int
        default = 10
        "--visualize"
        help = "Whether to visualize the environment"
        action = :store_true
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 42
        "--save-frame"
        help = "Save the initial frame as an image"
        action = :store_true
        "--klcontrol"
        help = "Use KL control agent instead of EFE agent"
        action = :store_true
    end

    return parse_args(s)
end

"""
Print detailed information about the agent's beliefs and state
"""
function print_debug_info(beliefs, env_state, action)
    println("\n=== Debug Information ===")

    # Print environment state
    println("\nEnvironment State:")
    println("Agent state: ", env_state.agent_state)

    # Print reward states
    println("\nReward States:")
    for (state, reward) in env_state.reward_states
        println("  State $state: reward = $reward")
    end

    # Print beliefs
    println("\nBeliefs:")
    println("State probabilities:")
    state_probs = beliefs.state.p
    for i in 1:length(state_probs)
        if state_probs[i] > 0.01  # Only show significant probabilities
            println("  State $i: $(round(state_probs[i], digits=3))")
        end
    end
end

function main()
    @info "Starting stochastic maze debug script"
    # Parse command line arguments
    args = parse_command_line()
    @info "Parsed arguments"

    # Create configuration
    config = StochasticMazeConfig(
        time_horizon=args["time-horizon"],
        n_episodes=1,  # We only need one episode for debugging
        n_iterations=args["iterations"],
        wait_time=0.0,
        seed=args["seed"],
        record_episode=false,
        experiment_name="debug"
    )
    @info "Created configuration"

    # Set up RNG
    rng = StableRNG(config.seed)

    # Create environment
    env = create_stochastic_maze(
        5, 5, 4,  # Default grid size and actions
        start_state=11  # Start in the middle of the grid
    )
    @info "Created environment"

    # Create results directory
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_dir = mkpath(datadir("debug",
        timestamp * "_" *
        "seed_$(config.seed)_" *
        "iterations_$(config.n_iterations)_" *
        "agent_$(args["klcontrol"] ? "klcontrol" : "efe")"
    ))
    @info "Created results directory at $results_dir"

    # Generate tensors
    A, B, reward_states = generate_maze_tensors(5, 5, 4)
    tensors = (
        observation_matrix=A,
        transition_tensor=B,
        reward_states=reward_states
    )
    @info "Generated tensors"

    # Set up goal
    goal_state = 15  # Center of top row
    p_goal = fill(tiny(Float64), 25)
    p_goal[goal_state] = 1.0
    goal_distribution = Categorical(p_goal)
    @info "Set up goal distribution"

    # Initialize beliefs
    n_states = size(tensors.transition_tensor, 1)
    beliefs = initialize_beliefs_stochastic_maze(n_states)
    @info "Initialized beliefs"

    # Initial observation
    observation = sample_observation(env)
    @info "Sampled initial observation: $observation"

    # Save initial frame if requested
    if args["save-frame"]
        gr()
        initial_plot = visualize_stochastic_maze(env)
        savefig(initial_plot, joinpath(results_dir, "initial_state.png"))
        pgfplotsx()
        initial_plot = visualize_stochastic_maze(env)
        savefig(initial_plot, joinpath(results_dir, "initial_state.tikz"))
        @info "Saved initial frame"
    end

    # Select agent model
    agent_model = args["klcontrol"] ? klcontrol_stochastic_maze_agent : efe_stochastic_maze_agent
    agent_constraints = args["klcontrol"] ? klcontrol_stochastic_maze_agent_constraints : efe_stochastic_maze_agent_constraints
    agent_initialization = args["klcontrol"] ? klcontrol_stochastic_maze_agent_initialization : efe_stochastic_maze_agent_initialization

    @info "Starting inference..."
    callbacks = nothing

    # Execute a single step
    next_action_idx, next_action, inference_result = execute_step(
        env,
        observation,
        beliefs,
        agent_model,
        tensors,
        config,
        goal_distribution,
        callbacks,
        config.time_horizon,
        nothing,  # no previous result 
        nothing;  # no previous action
        constraints_fn=agent_constraints,
        initialization_fn=agent_initialization,
        free_energy=true,  # Enable free energy tracking
        showprogress=true,  # Show inference progress
        options=(force_marginal_computation=true,
            limit_stack_depth=500), # Force marginal computation
    )
    @info "Inference completed"

    # Store the initial environment state for visualization
    initial_env_state = deepcopy(env)

    # Execute the action to see the result
    observation, reward = step!(rng, env, next_action)
    @info "Executed action $(next_action_idx) ($(action_to_string(next_action_idx))), received observation $observation and reward $reward"

    # Plot and save inference results
    @info "Plotting inference results..."
    plot_inference_results_stochastic_maze(
        inference_result,
        initial_env_state,  # Use initial state for visualization
        save_path=joinpath(results_dir, "inference_results")
    )

    # Print debug information
    print_debug_info(beliefs, env, next_action_idx)

    @info "Debug run completed. Results saved in: $results_dir"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end