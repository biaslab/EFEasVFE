using RxInfer
using TensorOperations
using Tullio
import RxInfer: Categorical


# Define the model and constraints for the maze RxEnvironmentsZoo
@model function klcontrol_minigrid_agent(p_old_location, p_old_orientation, p_key_location, p_door_location,
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
    current_orientation ~ DiscreteTransition(old_orientation, orientation_transition_tensor, action)
    current_key_door_state ~ DiscreteTransition(old_key_door_state, key_door_transition_tensor, old_location, old_orientation, key_location, door_location, action)

    # Observation model with Parafac decomposed tensors
    for x in 1:7, y in 1:7
        decomposed_tensor = observation_tensors[x, y]
        observations[x, y] ~ DiscreteTransition(current_location, decomposed_tensor, current_orientation, key_location, door_location, current_key_door_state)
    end
    orientation_observation ~ DiscreteTransition(current_orientation, diageye(number_type, 4))

    # Planning (Active Inference)
    previous_location = current_location
    previous_orientation = current_orientation
    previous_key_door_state = current_key_door_state
    for t in 1:T
        u[t] ~ Categorical(fill(number_type(1 / 5), 5))
        location[t] ~ DiscreteTransition(previous_location, location_transition_tensor, previous_orientation, key_location, door_location, previous_key_door_state, u[t])
        orientation[t] ~ DiscreteTransition(previous_orientation, orientation_transition_tensor, u[t])
        key_door_state[t] ~ DiscreteTransition(previous_key_door_state, key_door_transition_tensor, previous_location, previous_orientation, key_location, door_location, u[t])
        previous_location = location[t]
        previous_orientation = orientation[t]
        previous_key_door_state = key_door_state[t]
    end
    location[end] ~ goal
    orientation[end] ~ Categorical(fill(number_type(1 / 4), 4))
    key_door_state[end] ~ Categorical(number_type[tiny, tiny, 1.0-2*tiny])
end


@constraints function klcontrol_minigrid_agent_constraints()
    # q(u, location, orientation, key_door_state, key_location, door_location) = q(u)q(location, orientation, key_door_state, key_location, door_location)
    # q(u, current_location, current_orientation, current_key_door_state, key_location, door_location) = q(u)q(current_location, current_orientation, current_key_door_state, key_location, door_location)
    # q(u)::PointMassFormConstraint()
end

RxInfer.GraphPPL.default_constraints(::typeof(klcontrol_minigrid_agent)) = klcontrol_minigrid_agent_constraints()

@initialization function klcontrol_minigrid_agent_initialization(size, p_current_location, p_current_orientation, p_current_key_door_state, p_door_location, p_key_location, number_type)
    μ(current_location) = p_current_location
    μ(current_orientation) = p_current_orientation
    μ(current_key_door_state) = p_current_key_door_state

    μ(location) = Categorical(fill(number_type(1 / size^2), size^2))
    μ(orientation) = Categorical(fill(number_type(1 / 4), 4))
    μ(key_door_state) = p_current_key_door_state
    μ(door_location) = p_door_location
    μ(key_location) = p_key_location

    μ(old_location) = p_current_location
    μ(old_orientation) = p_current_orientation
    μ(old_key_door_state) = p_current_key_door_state
end
