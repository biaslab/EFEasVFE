using RxInfer

@model function efe_tmaze_agent(reward_observation_tensor, location_transition_tensor, prior_location, prior_reward_location, reward_to_location_mapping, u_prev, T, reward_observation, location_observation)
    old_location ~ prior_location
    reward_location ~ prior_reward_location

    current_location ~ DiscreteTransition(old_location, location_transition_tensor, u_prev)
    location_observation ~ DiscreteTransition(current_location, diageye(5))
    reward_observation ~ DiscreteTransition(current_location, reward_observation_tensor, reward_location) # Reward observation tensor is 2x5x2 (reward location observation x agent location x reward location state)

    previous_location = current_location
    for t in 1:T

        loc_marginalstorage = JointMarginalStorage(Contingency(ones(size(location_transition_tensor))))
        loc_marginalcomponent = JointMarginalMetaComponent(loc_marginalstorage, 1, 3)
        u[t] ~ Exploration(reward_observation) where {meta=JointMarginalMeta([loc_marginalcomponent])}
        location[t] ~ DiscreteTransition(previous_location, location_transition_tensor, u[t]) where {meta=loc_marginalstorage}

        observation_marginalstorage = JointMarginalStorage(Contingency(ones(size(reward_observation_tensor))))
        observation_marginalcomponent = JointMarginalMetaComponent(observation_marginalstorage, 1, 2)

        future_observation[t] ~ DiscreteTransition(location[t], reward_observation_tensor, reward_location) where {meta=observation_marginalstorage}
        future_observation[t] ~ Categorical([0.5, 0.5])
        location[t] ~ Ambiguity(reward_observation) where {meta=JointMarginalMeta([observation_marginalcomponent])}
        previous_location = location[t]
    end
    location[end] ~ DiscreteTransition(reward_location, reward_to_location_mapping) # Reward tensor is 5x2 mapping reward possibilities (2) to locations (5)
end

@constraints function efe_tmaze_agent_constraints()
end

@initialization function efe_tmaze_agent_initialization(prior_location, prior_reward_location, prior_future_locations)
    μ(old_location) = prior_location
    μ(reward_location) = prior_reward_location
    μ(location) = prior_future_locations
end