
# Define the model and constraints for the maze RxEnvironmentsZoo
@model function efe_stochastic_maze_agent(A, B, p_s_0, y_current, u_prev, y_future, T, n_states, n_actions, goal)
    # Prior initialization
    s_0 ~ p_s_0

    # State inference (filtering)
    s_current ~ DiscreteTransition(s_0, B, u_prev)
    y_current ~ DiscreteTransition(s_current, A)

    # Planning (Active Inference)
    s_prev = s_current
    for t in 1:T
        meta = JointMarginalMeta(Contingency(ones(n_states, n_states, n_actions)))
        u[t] ~ Exploration(y_current) where {meta=meta}
        s[t] ~ DiscreteTransition(s_prev, B, u[t]) where {meta=meta}
        s[t] ~ Ambiguity(A)
        y_future[t] ~ DiscreteTransition(s[t], A)
        s_prev = s[t]
    end
    s[end] ~ goal
end

@constraints function efe_stochastic_maze_agent_constraints()

end

@initialization function efe_stochastic_maze_agent_initialization(n_states, n_actions)
    q(s) ~ vague(Categorical, n_states)
end