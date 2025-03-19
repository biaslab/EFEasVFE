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