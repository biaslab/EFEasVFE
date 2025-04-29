
# Define the model and constraints for the maze RxEnvironmentsZoo
@model function efe_stochastic_maze_agent(A, B, p_s_0, y_current, u_prev, T, n_states, n_actions, goal)
    # Prior initialization
    s_0 ~ p_s_0

    # State inference (filtering)
    s_current ~ DiscreteTransition(s_0, B, u_prev)
    y_current ~ DiscreteTransition(s_current, A)

    # Planning (Active Inference)
    s_prev = s_current
    for t in 1:T
        state_marginalstorage = JointMarginalStorage(Contingency(ones(n_states, n_states, n_actions)))
        state_marginalcomponent = JointMarginalMetaComponent(state_marginalstorage, 1, 3)
        u[t] ~ Exploration(y_current) where {meta=JointMarginalMeta([state_marginalcomponent])}
        s[t] ~ DiscreteTransition(s_prev, B, u[t]) where {meta=state_marginalstorage}
        observation_marginalstorage = JointMarginalStorage(Contingency(ones(size(A))))
        observation_marginalcomponent = JointMarginalMetaComponent(observation_marginalstorage, 1, 2)
        s[t] ~ Ambiguity(y_current) where {meta=JointMarginalMeta([observation_marginalcomponent])}
        y_future[t] ~ DiscreteTransition(s[t], A) where {meta=observation_marginalstorage}
        y_future[t] ~ Categorical(fill(1 / n_states, n_states))
        s_prev = s[t]
    end
    s[end] ~ goal
end

@constraints function efe_stochastic_maze_agent_constraints()

end

@initialization function efe_stochastic_maze_agent_initialization(n_states)
    μ(s) = vague(Categorical, n_states)
end