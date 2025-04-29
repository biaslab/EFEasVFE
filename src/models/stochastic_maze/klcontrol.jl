
# Define the model and constraints for the maze RxEnvironmentsZoo
@model function klcontrol_stochastic_maze_agent(A, B, p_s_0, y_current, u_prev, T, n_states, n_actions, goal)
    # Prior initialization
    s_0 ~ p_s_0

    # State inference (filtering)
    s_current ~ DiscreteTransition(s_0, B, u_prev)
    y_current ~ DiscreteTransition(s_current, A)

    # Planning 
    s_prev = s_current
    for t in 1:T
        u[t] ~ Categorical([0.25, 0.25, 0.25, 0.25])
        s[t] ~ DiscreteTransition(s_prev, B, u[t])
        y_future[t] ~ DiscreteTransition(s[t], A)
        y_future[t] ~ Categorical(fill(1 / n_states, n_states))
        s_prev = s[t]
    end
    s[end] ~ goal
end

@constraints function klcontrol_stochastic_maze_agent_constraints()

end

@initialization function klcontrol_stochastic_maze_agent_initialization(n_states)
    μ(s) = vague(Categorical, n_states)
end