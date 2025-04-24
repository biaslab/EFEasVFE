module EFEasVFE

using DrWatson

# Include all submodules
include("meta/cachingmeta.jl")
include("meta/jointmarginalmeta.jl")
include("environments/minigrid.jl")
include("nodes/oneway/oneway.jl")
include("nodes/Exploration/node.jl")
include("nodes/Exploration/out.jl")
include("nodes/Ambiguity/node.jl")
include("nodes/Ambiguity/out.jl")
include("agents/minigrid_agent.jl")
include("environments/stochastic_maze.jl")
include("models/minigrid/klcontrol.jl")
include("models/minigrid/efe.jl")
include("models/stochastic_maze/klcontrol.jl")
include("models/stochastic_maze/efe.jl")
include("utils/parafac.jl")
include("utils/environment_communication.jl")
include("rules/rules.jl")
include("visualization/belief_visualization.jl")
include("rules/sparse_array/struct.jl")
include("rules/sparse_array/rules.jl")

# Re-export commonly used functions
export run_minigrid_agent, MinigridConfig, plot_free_energy, plot_posteriors,
    create_environment, step_environment, close_environment, EnvironmentError

# Export visualization functions
export plot_belief_grid, plot_inference_results, animate_belief_evolution

export klcontrol_minigrid_agent, klcontrol_minigrid_agent_constraints, klcontrol_minigrid_agent_initialization
export efe_minigrid_agent, efe_minigrid_agent_constraints, efe_minigrid_agent_initialization
export klcontrol_stochastic_maze_agent, klcontrol_stochastic_maze_agent_constraints, klcontrol_stochastic_maze_agent_initialization
export efe_stochastic_maze_agent, efe_stochastic_maze_agent_constraints, efe_stochastic_maze_agent_initialization

end # module