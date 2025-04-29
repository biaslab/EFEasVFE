module EFEasVFE

using DrWatson

# Include all submodules
include("meta/jointmarginalmeta.jl")
include("environments/minigrid.jl")
include("environments/tmaze.jl")
include("environments/stochastic_maze.jl")
include("nodes/oneway/oneway.jl")
include("nodes/Exploration/node.jl")
include("nodes/Exploration/out.jl")
include("nodes/Ambiguity/node.jl")
include("nodes/Ambiguity/out.jl")
include("agents/minigrid_agent.jl")
include("agents/tmaze_agent.jl")
include("agents/stochastic_maze_agent.jl")
include("models/minigrid/klcontrol.jl")
include("models/minigrid/efe.jl")
include("models/stochastic_maze/klcontrol.jl")
include("models/stochastic_maze/efe.jl")
include("models/tmaze/klcontrol.jl")
include("models/tmaze/efe.jl")
include("utils/environment_communication.jl")
include("utils/visualization_utils.jl")
include("rules/rules.jl")
include("visualization/belief_visualization.jl")

RxInfer.ReactiveMP.sdtype(any::RxInfer.ReactiveMP.StandaloneDistributionNode) = ReactiveMP.Stochastic()

# Re-export commonly used functions
export run_minigrid_agent, MinigridConfig, plot_free_energy, plot_posteriors,
    create_environment, step_environment, close_environment, EnvironmentError

# Export TMaze types and functions
export North, East, South, West, MazeAction, TMaze
export create_tmaze, reset_tmaze!, step!
export create_reward_observation_tensor, create_location_transition_tensor, create_reward_to_location_mapping
export run_tmaze_agent, TMazeConfig
export klcontrol_tmaze_agent, klcontrol_tmaze_agent_constraints, klcontrol_tmaze_agent_initialization
export efe_tmaze_agent, efe_tmaze_agent_constraints, efe_tmaze_agent_initialization

# Export visualization functions
export plot_belief_grid, plot_inference_results, animate_belief_evolution, save_frame

export klcontrol_minigrid_agent, klcontrol_minigrid_agent_constraints, klcontrol_minigrid_agent_initialization
export efe_minigrid_agent, efe_minigrid_agent_constraints, efe_minigrid_agent_initialization
export klcontrol_stochastic_maze_agent, klcontrol_stochastic_maze_agent_constraints, klcontrol_stochastic_maze_agent_initialization
export efe_stochastic_maze_agent, efe_stochastic_maze_agent_constraints, efe_stochastic_maze_agent_initialization

end # module