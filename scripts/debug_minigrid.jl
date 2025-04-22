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
import RxInfer: Categorical, mode
using JSON
using HTTP
using VideoIO
using Colors
import Colors: N0f8
using EFEasVFE
using Plots

import EFEasVFE: reinitialize_environment, execute_initial_action, step_environment, initialize_beliefs, convert_frame, execute_step

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--grid-size"
        help = "Size of the grid"
        arg_type = Int
        default = 3
        "--time-horizon"
        help = "Time horizon for planning"
        arg_type = Int
        default = 10
        "--iterations"
        help = "Number of iterations for inference"
        arg_type = Int
        default = 3
        "--number-type"
        help = "Number type to use (Float32 or Float64)"
        arg_type = String
        default = "Float32"
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
        "--save-animation"
        help = "Save an animation of belief evolution"
        action = :store_true
        "--sparse-tensor"
        help = "Use sparse representation for transition and observation tensors"
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
    println("Direction: ", env_state["observation"]["direction"])
    println("Reward: ", env_state["reward"])
    println("Terminated: ", get(env_state, "terminated", false))

    # Print beliefs
    println("\nBeliefs:")
    println("Location probabilities:")
    location_probs = beliefs.location.p
    for i in 1:length(location_probs)
        if location_probs[i] > 0.01  # Only show significant probabilities
            println("  State $i: $(round(location_probs[i], digits=3))")
        end
    end

    println("\nOrientation probabilities:")
    orientations = ["RIGHT", "DOWN", "LEFT", "UP"]
    for (i, p) in enumerate(beliefs.orientation.p)
        println("  $(orientations[i]): $(round(p, digits=3))")
    end

    println("\nKey/Door State probabilities:")
    states = ["No key", "Has key", "Door open"]
    for (i, p) in enumerate(beliefs.key_door_state.p)
        println("  $(states[i]): $(round(p, digits=3))")
    end
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

function main()
    @info "Starting debug script"
    # Parse command line arguments
    args = parse_command_line()
    @info "Parsed arguments"
    # Convert number type
    number_type = if args["number-type"] == "Float32"
        Float32
    else
        Float64
    end

    # Create configuration
    config = MinigridConfig(
        grid_size=args["grid-size"],
        time_horizon=args["time-horizon"],
        n_episodes=1,  # We only need one episode for debugging
        n_iterations=args["iterations"],
        wait_time=0.0,
        number_type=number_type,
        visualize=args["visualize"],
        seed=args["seed"],
        record_episode=false,
        experiment_name="debug"
    )
    @info "Created configuration"
    # Initialize environment

    env_response = EFEasVFE.create_environment(
        config.grid_size + 2,
        render_mode=args["visualize"] ? "human" : "rgb_array",
        seed=UInt32(config.seed)
    )
    session_id = env_response["session_id"]

    # Create results directory with grid size, seed, iterations and sparse-tensor info
    results_dir = mkpath(datadir("debug",
        "grid$(config.grid_size)_seed$(config.seed)_iter$(config.n_iterations)_sparsetensor$(args["sparse-tensor"])"
    ))
    @info "Initialized environment"
    # Initialize beliefs and tensors
    beliefs = initialize_beliefs(config.grid_size, config.number_type)
    tensors = load_tensors(
        config.grid_size,
        config.number_type;
        sparse_tensor=args["sparse-tensor"]  # Pass the flag value
    )
    goal = create_goal(config.grid_size, config.number_type)
    @info "Initialized beliefs and tensors"
    # Execute initial action
    env_state = execute_initial_action(config.grid_size, session_id)
    action = 1
    @info "Executed initial action"
    # Save initial frame if requested
    if args["save-frame"]
        frame = convert_frame(env_state["frame"])
        rgb_frame = RGB{N0f8}.(frame[:, :, 1] ./ 255, frame[:, :, 2] ./ 255, frame[:, :, 3] ./ 255)
        save(joinpath(results_dir, "initial_frame.png"), rgb_frame)
        @info "Saved initial frame"
    end
    @info "Starting inference..."
    # Execute a single step with debug options
    action, new_env_state, inference_result = execute_step(
        env_state,
        action,
        beliefs,
        efe_minigrid_agent,
        tensors,
        config,
        goal,
        nothing,  # no callbacks
        config.time_horizon,
        session_id;
        constraints_fn=efe_minigrid_agent_constraints,
        initialization_fn=efe_minigrid_agent_initialization,
        free_energy=true,  # Enable free energy tracking
        showprogress=true,  # Show inference progress,
        options=(force_marginal_computation=true,
            limit_stack_depth=500), # Force marginal computation
        # Add any other inference kwargs as needed
    )
    @info "Inference completed"
    next_action = mode(first(last(inference_result.posteriors[:u])))
    env_action = EFEasVFE.convert_action(next_action)

    # Plot and save inference results
    @info "Plotting inference results..."
    plot_inference_results(
        inference_result,
        config.grid_size,
        save_path=joinpath(results_dir, "inference_results.png")
    )

    # Create and save animation if requested
    if args["save-animation"]
        @info "Creating belief evolution animation..."
        animate_belief_evolution(
            inference_result,
            config.grid_size,
            fps=2,
            save_path=joinpath(results_dir, "belief_evolution.gif")
        )
        animate_trajectory_belief(
            inference_result,
            config.grid_size,
            save_path=joinpath(results_dir, "trajectory_belief.gif")
        )
    end

    # Print debug information
    print_debug_info(beliefs, new_env_state, env_action)

    @info "Debug run completed. Results saved in: $results_dir"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end