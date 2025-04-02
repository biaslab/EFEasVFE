using DrWatson
@quickactivate "EFEasVFE"

using RxEnvironmentsZoo
using RxEnvironments
using RxInfer
using RxAgents
using ProgressMeter
import RxInfer: Categorical

include(srcdir("environments/stochastic_maze.jl"))
include(srcdir("models/klcontrol.jl"))
include(srcdir("models/efe.jl"))

function RxEnvironments.plot_state(ax, env::StochasticMaze)
    # Get grid dimensions from environment
    grid_size_x = Int(sqrt(size(env.transition_tensor, 1)))
    grid_size_y = grid_size_x

    # Create grid lines
    for x in 0:grid_size_x
        vlines!(ax, x, 0, grid_size_y, color=:black, linewidth=0.5)
    end
    for y in 0:grid_size_y
        hlines!(ax, y, 0, grid_size_x, color=:black, linewidth=0.5)
    end

    # Plot sink states in red
    # Find sink states by checking where all actions point to same state
    sink_states = []
    for s in 1:size(env.transition_tensor, 1)
        if all(env.transition_tensor[s, s, :] .== 1.0)
            x = ((s - 1) % grid_size_x) + 1
            y = div(s - 1, grid_size_x) + 1
            push!(sink_states, (x, y))
        end
    end
    for (x, y) in sink_states
        poly!(ax,
            [Point2f(x - 1, grid_size_y - y), Point2f(x, grid_size_y - y),
                Point2f(x, grid_size_y - y + 1), Point2f(x - 1, grid_size_y - y + 1)],
            color=(:red, 0.3))
    end

    # Plot reward states
    for (state, reward) in env.reward_states
        x = ((state - 1) % grid_size_x) + 1
        y = div(state - 1, grid_size_x) + 1
        color = reward > 0 ? :green : :red
        opacity = abs(reward) # Use absolute value of reward for opacity
        scatter!(ax, [x - 0.5], [grid_size_y - y + 0.5], color=(color, opacity), markersize=20)
    end

    # Plot observation noise
    noisy_obs = []
    for s in 1:size(env.observation_matrix, 1)
        if env.observation_matrix[s, s] != 1.0
            x = ((s - 1) % grid_size_x) + 1
            y = div(s - 1, grid_size_x) + 1
            noise = 1.0 - env.observation_matrix[s, s]
            push!(noisy_obs, (x, y, noise))
        end
    end

    for (x, y, noise) in noisy_obs
        poly!(ax,
            [Point2f(x - 1, grid_size_y - y), Point2f(x, grid_size_y - y),
                Point2f(x, grid_size_y - y + 1), Point2f(x - 1, grid_size_y - y + 1)],
            color=(:lightblue, noise))
    end

    # Plot stochastic states (bridge effect)
    stochastic_states = []
    for s in 1:size(env.transition_tensor, 1)
        # Check if any action has non-1.0 probability for intended direction
        if any(maximum(env.transition_tensor[:, s, a]) < 0.99 for a in 1:size(env.transition_tensor, 3))
            x = ((s - 1) % grid_size_x) + 1
            y = div(s - 1, grid_size_x) + 1
            push!(stochastic_states, (x, y))
        end
    end

    # Draw bridge planks in brown with gaps
    for (x, y) in stochastic_states
        # Draw 3 horizontal planks
        for i in 0:2
            poly!(ax,
                [Point2f(x - 1, grid_size_y - y + 0.25 + i * 0.25),
                    Point2f(x, grid_size_y - y + 0.25 + i * 0.25),
                    Point2f(x, grid_size_y - y + 0.32 + i * 0.25),
                    Point2f(x - 1, grid_size_y - y + 0.32 + i * 0.25)],
                color=(:brown, 0.6))
        end
    end

    for agent in env.agents
        x = ((agent.state - 1) % grid_size_x) + 1
        y = div(agent.state - 1, grid_size_x) + 1
        scatter!(ax, [x - 0.5], [grid_size_y - y + 0.5], color=:blue, markersize=20)
    end

    # Set proper axis limits and remove ticks
    limits!(ax, 0, grid_size_x, 0, grid_size_y)
    hidedecorations!(ax)
end

T = 9
goal_state = 15

states_x = 5
states_y = 5
actions = 4

A, B, reward_states = generate_maze_tensors(states_x, states_y, actions)
# Choose the environment
environment = create_environment(StochasticMaze, B, A, reward_states)
agent = RxEnvironments.add!(environment, StochasticMazeAgent(11))
observations = keep(Any)
subscribe_to_observations!(agent, observations)

n_states = states_x * states_y
# Create goal distribution
p_goal = zeros(n_states)
p_goal[goal_state] = 1.0
p_goal = Categorical(p_goal ./ sum(p_goal))
callbacks = RxInferBenchmarkCallbacks()

function run_maze_agent(environment, agent, model, A, B, goal, n_states, n_actions; n_episodes=1000, n_iterations=10, callbacks=(after_iteration=RxAgents.after_iteration_callback,), T=8, wait_time=0.0)
    observation_storage = zeros(n_states)
    previous_action = zeros(n_actions)
    rewards = zeros(n_episodes)
    observations = keep(Any)
    subscribe_to_observations!(agent, observations)



    @showprogress for i in 1:n_episodes
        reset!(environment, 11)
        p_s_0 = vague(Categorical, n_states)

        # Initial action
        send!(environment, agent, StochasticMazeAction(4))
        next_action = 4
        reward = 0.0
        sleep(wait_time)
        # Run episode
        for t in 1:T-1
            # Get current state and create observation vector
            current_state = first(RxEnvironments.data(last(observations)))
            observation_storage .= 0.0
            observation_storage[current_state] = 1.0

            # Create previous action vector
            previous_action .= 0.0
            previous_action[next_action] = 1.0
            # Run inference
            result = infer(
                model=model(A=A, B=B, p_s_0=p_s_0, T=T - t, n_states=n_states, n_actions=n_actions, goal=goal),
                data=(
                    y_current=observation_storage,
                    y_future=UnfactorizedData(fill(missing, T - t)),
                    u_prev=UnfactorizedData(previous_action)
                ),
                callbacks=callbacks,
                iterations=n_iterations
            )

            # Take action
            next_action = mode(first(last(result.posteriors[:u])))
            send!(environment, agent, StochasticMazeAction(next_action))
            p_s_0 = last(result.posteriors[:s_current])
            sleep(wait_time)
            current_reward = last(RxEnvironments.data(last(observations)))
            reward += current_reward
            # @show reward
        end
        rewards[i] = reward
    end
    return mean(rewards), std(rewards)
end

m, s = run_maze_agent(environment, agent, efe_stochastic_maze_agent, A, B, p_goal, n_states, actions; n_episodes=1000, n_iterations=3, T=T,)
@show m, s
m, s = run_maze_agent(environment, agent, klcontrol_stochastic_maze_agent, A, B, p_goal, n_states, actions; n_episodes=1000, n_iterations=1, T=T,)
@show m, s


environment = create_environment(StochasticMaze, B, A, reward_states)
agent = RxEnvironments.add!(environment, StochasticMazeAgent(11))
animate_state(environment)
run_maze_agent(environment, agent, efe_stochastic_maze_agent, A, B, p_goal, n_states, actions; n_episodes=1, n_iterations=3, T=T, wait_time=1.0)