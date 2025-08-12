using RxInfer
using TensorOperations
using Tullio
import RxInfer: Categorical


@model function klcontrol_minigrid_agent(p_location, p_orientation, p_key_door_state, p_key_location, p_door_location,
    location_transition_tensor, orientation_transition_tensor, key_door_transition_tensor,
    observation_tensors, T, goal, number_type)
    # Prior initialization
    current_location ~ p_location
    current_orientation ~ p_orientation
    current_key_door_state ~ p_key_door_state

    door_location ~ p_door_location
    key_location ~ p_key_location

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

@initialization function klcontrol_minigrid_agent_initialization(p_current_location, p_current_orientation, p_current_key_door_state, p_future_locations, p_future_orientations, p_future_key_door_states, p_door_location, p_key_location)
    μ(current_location) = p_current_location
    μ(current_orientation) = p_current_orientation
    μ(current_key_door_state) = p_current_key_door_state

    μ(location) = p_future_locations
    μ(orientation) = p_future_orientations
    μ(key_door_state) = p_future_key_door_states
    μ(door_location) = p_door_location
    μ(key_location) = p_key_location
end
