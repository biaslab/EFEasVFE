using RxInfer
using TensorOperations
using Tullio
using RxEnvironmentsZoo.GLMakie
import RxInfer: Categorical


# Define the model and constraints for the maze RxEnvironmentsZoo
@model function klcontrol_minigrid_agent(p_old_location, p_old_orientation, p_key_location, p_door_location, p_old_key_state,
    p_old_door_state, loc_t_tensor, ori_t_tensor, door_t_tensor, key_t_tensor, observation_tensors, T, goal, observations, action, orientation_observation)
    # Prior initialization
    old_location ~ p_old_location
    old_orientation ~ p_old_orientation
    old_door_state ~ p_old_door_state
    old_key_state ~ p_old_key_state

    door_location ~ p_door_location
    key_location ~ p_key_location

    # State inference
    current_location ~ DiscreteTransition(old_location, loc_t_tensor, old_orientation, key_location, door_location, old_key_state, old_door_state, action)
    current_orientation ~ DiscreteTransition(old_orientation, ori_t_tensor, action)
    current_door_state ~ DiscreteTransition(old_door_state, door_t_tensor, old_location, old_orientation, door_location, old_key_state, action)
    current_key_state ~ DiscreteTransition(old_key_state, key_t_tensor, old_location, old_orientation, key_location, action)

    # Observation model with Tucker decomposed tensors
    for x in 1:7, y in 1:7
        tucker_tensor = observation_tensors[x, y]
        observations[x, y] ~ DiscreteTransition(current_location, tucker_tensor, current_orientation, key_location, door_location, current_key_state, current_door_state)
    end
    orientation_observation ~ DiscreteTransition(current_orientation, diageye(4))

    # Planning (Active Inference)
    previous_location = current_location
    previous_orientation = current_orientation
    previous_door_state = current_door_state
    previous_key_state = current_key_state
    for t in 1:T
        u[t] ~ Categorical([0.2, 0.2, 0.2, 0.2, 0.2])
        location[t] ~ DiscreteTransition(previous_location, loc_t_tensor, previous_orientation, key_location, door_location, previous_key_state, previous_door_state, u[t])
        orientation[t] ~ DiscreteTransition(previous_orientation, ori_t_tensor, u[t])
        door_state[t] ~ DiscreteTransition(previous_door_state, door_t_tensor, previous_location, previous_orientation, door_location, previous_key_state, u[t])
        key_state[t] ~ DiscreteTransition(previous_key_state, key_t_tensor, previous_location, previous_orientation, key_location, u[t])
        previous_location = location[t]
        previous_orientation = orientation[t]
        previous_door_state = door_state[t]
        previous_key_state = key_state[t]
    end
    location[end] ~ goal
    orientation[end] ~ Categorical([0.25, 0.25, 0.25, 0.25])
    door_state[end] ~ Categorical([tiny, tiny, 1.0 - 2 * tiny])
    key_state[end] ~ Categorical([tiny, 1.0 - tiny])
end

@constraints function klcontrol_minigrid_agent_constraints()
end

@initialization function klcontrol_minigrid_agent_initialization(size, p_current_location, p_current_orientation, p_current_door_state, p_current_key_state, p_door_location, p_key_location)
    μ(current_location) = p_current_location
    μ(current_orientation) = p_current_orientation
    μ(current_door_state) = p_current_door_state
    μ(current_key_state) = p_current_key_state

    μ(location) = vague(Categorical, size^2)
    μ(orientation) = vague(Categorical, 4)
    μ(door_state) = vague(Categorical, 3)
    μ(key_state) = vague(Categorical, 2)
    μ(door_location) = p_door_location
    μ(key_location) = p_key_location

    μ(old_location) = p_current_location
    μ(old_orientation) = p_current_orientation
    μ(old_door_state) = p_current_door_state
    μ(old_key_state) = p_current_key_state
end
