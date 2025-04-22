using DrWatson
@quickactivate "EFEasVFE"

using Logging
using LoggingExtras
using ArgParse
using TinyHugeNumbers
using RxInfer.GraphPPL
import RxInfer: Categorical
using EFEasVFE
using SparseArrays
using BenchmarkTools
using Statistics

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
        default = 5
        "--iterations"
        help = "Number of iterations for inference"
        arg_type = Int
        default = 5
        "--number-type"
        help = "Number type to use (Float32 or Float64)"
        arg_type = String
        default = "Float32"
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 42
        "--full-tensor"
        help = "Use full tensors instead of sparse tensors"
        action = :store_true
        "--compute-free-energy"
        help = "Compute free energy during inference"
        action = :store_true
        "--benchmark"
        help = "Run benchmarks"
        action = :store_true
        "--warmup"
        help = "Number of warmup runs"
        arg_type = Int
        default = 1
    end

    return parse_args(s)
end

"""
    load_tensors(grid_size, number_type; full_tensor=false)

Load the required tensors for the experiment.
"""
function load_tensors(grid_size, number_type; full_tensor=false)
    @info "Loading tensors for grid size $grid_size"
    if full_tensor
        observation_tensors = collect(eachslice(EFEasVFE.generate_observation_tensor(grid_size, number_type), dims=(1, 2)))
        door_key_transition_tensor = EFEasVFE.get_key_door_state_transition_tensor(grid_size, number_type)
        location_transition_tensor = EFEasVFE.get_self_transition_tensor(grid_size, number_type)
    else
        observation_tensors = SparseArray.(eachslice(EFEasVFE.generate_observation_tensor(grid_size, number_type), dims=(1, 2)))
        door_key_transition_tensor = SparseArray(EFEasVFE.get_key_door_state_transition_tensor(grid_size, number_type))
        location_transition_tensor = SparseArray(EFEasVFE.get_self_transition_tensor(grid_size, number_type))
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
    run_benchmark(config, tensors, beliefs, goal)

Run a single benchmark iteration of execute_step.
"""
function run_benchmark(config, tensors, beliefs, goal, compute_free_energy)
    # Initialize environment and state
    env_state = reinitialize_environment(
        config.grid_size + 2,
        render_mode="rgb_array",
        seed=UInt32(config.seed)
    )
    env_state = execute_initial_action(config.grid_size)
    action = 1

    # Execute a single step and measure performance
    action, env_action, inference_result = execute_step(
        env_state,
        action,
        beliefs,
        klcontrol_minigrid_agent,
        tensors,
        config,
        goal,
        nothing,  # no callbacks
        config.time_horizon;
        free_energy=compute_free_energy,
        showprogress=false
    )

    return action, env_action, inference_result
end

function main()
    # Parse command line arguments
    args = parse_command_line()

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
        n_episodes=1,
        n_iterations=args["iterations"],
        wait_time=0.0,
        number_type=number_type,
        visualize=false,
        seed=args["seed"],
        record_episode=false,
        experiment_name="benchmark"
    )

    # Initialize beliefs and tensors
    beliefs = initialize_beliefs(config.grid_size, config.number_type)
    tensors = load_tensors(
        config.grid_size,
        config.number_type;
        full_tensor=args["full-tensor"]
    )
    goal = create_goal(config.grid_size, config.number_type)

    # Print configuration
    println("\nBenchmark Configuration:")
    println("------------------------")
    println("Grid Size: $(config.grid_size)")
    println("Time Horizon: $(config.time_horizon)")
    println("Iterations: $(config.n_iterations)")
    println("Number Type: $(config.number_type)")
    println("Full Tensor: $(args["full-tensor"])")
    println("Compute Free Energy: $(args["compute-free-energy"])")
    println("------------------------\n")

    # Run warmup if benchmarking
    if args["benchmark"]
        println("Running warmup iterations...")
        for _ in 1:args["warmup"]
            run_benchmark(config, tensors, beliefs, goal, args["compute-free-energy"])
        end
    end

    # Run benchmarks if requested
    if args["benchmark"]
        println("\nRunning benchmarks...")
        result = @benchmark run_benchmark($config, $tensors, $beliefs, $goal, $(args["compute-free-energy"]))

        println("\nBenchmark Results:")
        println("------------------")
        show(stdout, MIME("text/plain"), result)
        println("\n")

        # Print summary statistics
        println("\nSummary Statistics:")
        println("------------------")
        println("Minimum time: $(minimum(result.times) / 1e9) seconds")
        println("Median time: $(median(result.times) / 1e9) seconds")
        println("Mean time: $(mean(result.times) / 1e9) seconds")
        println("Maximum time: $(maximum(result.times) / 1e9) seconds")
        println("Standard deviation: $(std(result.times) / 1e9) seconds")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end