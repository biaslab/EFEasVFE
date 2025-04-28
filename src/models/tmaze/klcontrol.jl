using RxInfer

@model function klcontrol_tmaze_agent(reward_observation_tensor, location_transition_tensor, prior_location, prior_reward_location, reward_to_location_mapping, u_prev, T, reward_observation, location_observation)
    old_location ~ prior_location
    reward_location ~ prior_reward_location

    current_location ~ DiscreteTransition(old_location, location_transition_tensor, u_prev)
    location_observation ~ DiscreteTransition(current_location, diageye(5))
    reward_observation ~ DiscreteTransition(current_location, reward_observation_tensor, reward_location) # Reward observation tensor is 2x5x2 (reward location observation x agent location x reward location state)

    previous_location = current_location
    for t in 1:T
        u[t] ~ Categorical([0.25, 0.25, 0.25, 0.25])
        location[t] ~ DiscreteTransition(previous_location, location_transition_tensor, u[t])
        previous_location = location[t]
    end
    location[end] ~ DiscreteTransition(reward_location, reward_to_location_mapping) # Reward tensor is 5x2 mapping reward possibilities (2) to locations (5)
end

@constraints function klcontrol_tmaze_agent_constraints()
end

@initialization function klcontrol_tmaze_agent_initialization(prior_location, prior_reward_location, prior_future_locations)
    μ(old_location) = prior_location
    μ(reward_location) = prior_reward_location
    μ(location) = prior_future_locations
end