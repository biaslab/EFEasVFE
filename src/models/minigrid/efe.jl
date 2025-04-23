using RxInfer
using TensorOperations
using Tullio
import RxInfer: Categorical


# Define the model and constraints for the maze RxEnvironmentsZoo
@model function efe_minigrid_agent(p_old_location, p_old_orientation, p_key_location, p_door_location,
    p_old_key_door_state, location_transition_tensor, orientation_transition_tensor, key_door_transition_tensor,
    observation_tensors, T, goal, observations, action, orientation_observation, number_type)
    # Prior initialization
    old_location ~ p_old_location
    old_orientation ~ p_old_orientation
    old_key_door_state ~ p_old_key_door_state

    door_location ~ p_door_location
    key_location ~ p_key_location

    # State inference
    current_location ~ DiscreteTransition(old_location, location_transition_tensor, old_orientation, key_location, door_location, old_key_door_state, action)
    current_key_door_state ~ DiscreteTransition(old_key_door_state, key_door_transition_tensor, old_location, old_orientation, key_location, door_location, action)

    # Observation model with Parafac decomposed tensors
    for x in 1:7, y in 1:7
        decomposed_tensor = observation_tensors[x, y]
        observations[x, y] ~ DiscreteTransition(current_location, decomposed_tensor, orientation_observation, key_location, door_location, current_key_door_state)
    end
    current_orientation ~ DiscreteTransition(orientation_observation, diageye(number_type, 4))

    # Planning (Active Inference)
    previous_location = current_location
    previous_orientation = current_orientation
    previous_key_door_state = current_key_door_state
    for t in 1:T
        location_marginalstorage = JointMarginalStorage(Contingency(ones(number_type, size(location_transition_tensor))))
        location_marginalcomponent = JointMarginalMetaComponent(location_marginalstorage, 1, 7)
        orientation_marginalstorage = JointMarginalStorage(Contingency(ones(number_type, size(orientation_transition_tensor))))
        orientation_marginalcomponent = JointMarginalMetaComponent(orientation_marginalstorage, 1, 3)
        key_door_state_marginalstorage = JointMarginalStorage(Contingency(ones(number_type, size(key_door_transition_tensor))))
        key_door_state_marginalcomponent = JointMarginalMetaComponent(key_door_state_marginalstorage, 1, 7)
        u[t] ~ Exploration(observations[1, 1]) where {meta=JointMarginalMeta([location_marginalcomponent, orientation_marginalcomponent, key_door_state_marginalcomponent])}
        location[t] ~ DiscreteTransition(previous_location, location_transition_tensor, previous_orientation, key_location, door_location, previous_key_door_state, u[t]) where {meta=location_marginalstorage}
        orientation[t] ~ DiscreteTransition(previous_orientation, orientation_transition_tensor, u[t]) where {meta=orientation_marginalstorage}
        key_door_state[t] ~ DiscreteTransition(previous_key_door_state, key_door_transition_tensor, previous_location, previous_orientation, key_location, door_location, u[t]) where {meta=key_door_state_marginalstorage}
        previous_location = location[t]
        previous_orientation = orientation[t]
        previous_key_door_state = key_door_state[t]
        location_observation_marginalcomponents = JointMarginalMetaComponent[]
        orientation_observation_marginalcomponents = JointMarginalMetaComponent[]
        key_door_state_observation_marginalcomponents = JointMarginalMetaComponent[]

        for x in 1:7, y in 1:7
            marginalstorage = JointMarginalStorage(Contingency(ones(number_type, size(observation_tensors[x, y]))))
            location_observation_marginalcomponent = JointMarginalMetaComponent(marginalstorage, 1, 2)
            push!(location_observation_marginalcomponents, location_observation_marginalcomponent)
            orientation_observation_marginalcomponent = JointMarginalMetaComponent(marginalstorage, 1, 3)
            push!(orientation_observation_marginalcomponents, orientation_observation_marginalcomponent)
            key_door_state_observation_marginalcomponent = JointMarginalMetaComponent(marginalstorage, 1, 6)
            push!(key_door_state_observation_marginalcomponents, key_door_state_observation_marginalcomponent)
            decomposed_tensor = observation_tensors[x, y]
            future_observations[x, y] ~ DiscreteTransition(current_location, decomposed_tensor, current_orientation, key_location, door_location, current_key_door_state) where {meta=marginalstorage}
            future_observations[x, y] ~ Categorical(fill(number_type(1 / 5), 5))
        end
        location[t] ~ Ambiguity(observations[1, 1]) where {meta=JointMarginalMeta(location_observation_marginalcomponents)}
        orientation[t] ~ Ambiguity(observations[1, 1]) where {meta=JointMarginalMeta(orientation_observation_marginalcomponents)}
        key_door_state[t] ~ Ambiguity(observations[1, 1]) where {meta=JointMarginalMeta(key_door_state_observation_marginalcomponents)}
    end
    location[end] ~ goal
    orientation[end] ~ Categorical(fill(number_type(1 / 4), 4))
    key_door_state[end] ~ Categorical(number_type[tiny, tiny, 1.0-2*tiny])
end

@constraints function efe_minigrid_agent_constraints()

end

RxInfer.GraphPPL.default_constraints(::typeof(efe_minigrid_agent)) = efe_minigrid_agent_constraints()

@initialization function efe_minigrid_agent_initialization(p_current_location, p_current_orientation, p_current_key_door_state, p_future_locations, p_future_orientations, p_future_key_door_states, p_door_location, p_key_location)
    μ(current_location) = p_current_location
    μ(current_orientation) = p_current_orientation
    μ(current_key_door_state) = p_current_key_door_state

    μ(location) = p_future_locations
    μ(orientation) = p_future_orientations
    μ(key_door_state) = p_future_key_door_states
    μ(door_location) = p_door_location
    μ(key_location) = p_key_location

    μ(old_location) = p_current_location
    μ(old_orientation) = p_current_orientation
    μ(old_key_door_state) = p_current_key_door_state
end
