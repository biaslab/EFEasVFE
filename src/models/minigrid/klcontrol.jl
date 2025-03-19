using RxInfer
using TensorOperations
using Tullio
using RxEnvironmentsZoo.GLMakie
import RxInfer: Categorical

# Rules for observation model (q_out is pointmass)
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, i, c, d, e, f, g] * probvec(q_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, i, c, d, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, i, d, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, i, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (q_out::PointMass, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, i, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T5)[f]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (8 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[i, a, c, d, e, f, g, h] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g] * probvec(m_T6)[h]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, i, c, d, e, f, g, h] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g] * probvec(m_T6)[h]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, i, c, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, i, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[d] * probvec(m_T4)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, i, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, i, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T5)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, f, i, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T6)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T6, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, f, g, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (6 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[i, a, c, d, e, f] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, i, c, d, e, f] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, i, c, d, e] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, c, i, d, e] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, c, d, i, e] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[e]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, c, d, e, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e]
    return Categorical(normalize!(out, 1))
end

# Rules for transition model (7 interfaces)
@rule DiscreteTransition(:out, Marginalisation) (m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[i, a, c, d, e, f, g] * probvec(m_in)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, i, c, d, e, f, g] * probvec(m_out)[a] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, i, d, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, i, e, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, i, f, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T4)[f] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T4, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, i, g] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T5)[g]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T5, Marginalisation) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tensor out[i] := eloga[a, b, c, d, e, f, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Categorical(normalize!(out, 1))
end

# Marginal rule for the transition model (8 interfaces)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4_T5_T6) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, m_T6::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tullio eloga[a, b, c, d, e, f, g, h] = eloga[a, b, c, d, e, f, g, h] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f] * probvec(m_T5)[g] * probvec(m_T6)[h]
    return Contingency(eloga)
end

# Marginal rule for the transition model (6 interfaces)
@marginalrule DiscreteTransition(:out_in_T1_T2_T3_T4) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tullio eloga[a, b, c, d, e, f] = eloga[a, b, c, d, e, f] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Contingency(eloga)
end

# Marginal rule for the transition model (6 interfaces)
@marginalrule DiscreteTransition(:in_T1_T2_T3_T4) (q_out::Categorical, m_in::Categorical, m_T1::Categorical, m_T2::Categorical, m_T3::Categorical, m_T4::Categorical, m_T5::Categorical, q_a::PointMass) = begin
    eloga = mean(q_a) .+ tiny
    @tullio eloga[a, b, c, d, e, f] = eloga[a, b, c, d, e, f] * probvec(q_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d] * probvec(m_T3)[e] * probvec(m_T4)[f]
    return Contingency(eloga)
end


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


    for x in 1:7, y in 1:7
        slice = observation_tensors[x, y, :, :, :, :, :, :, :]
        observations[x, y] ~ DiscreteTransition(current_location, slice, current_orientation, key_location, door_location, current_key_state, current_door_state)
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


