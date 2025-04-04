module EFEasVFE

using DrWatson

# Include all submodules
include("meta/cachingmeta.jl")
include("environments/minigrid.jl")
include("agents/minigrid_agent.jl")
include("environments/stochastic_maze.jl")
include("models/minigrid/klcontrol.jl")
include("models/stochastic_maze/klcontrol.jl")
include("models/stochastic_maze/efe.jl")
include("utils/parafac.jl")
include("utils/environment_communication.jl")
include("rules/rules.jl")
include("visualization/belief_visualization.jl")

# Re-export commonly used functions
export run_minigrid_agent, MinigridConfig, plot_free_energy, plot_posteriors,
    reset_environment, step_environment, get_action_space, EnvironmentError

# Export visualization functions
export plot_belief_grid, plot_inference_results, animate_belief_evolution

export klcontrol_minigrid_agent, klcontrol_stochastic_maze_agent, efe_stochastic_maze_agent, klcontrol_minigrid_agent_initialization, klcontrol_stochastic_maze_agent_initialization, efe_stochastic_maze_agent_initialization, klcontrol_minigrid_agent_initialization, klcontrol_stochastic_maze_agent_initialization, efe_stochastic_maze_agent_initialization

end # module