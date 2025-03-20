using DrWatson
@quickactivate "EFEasVFE"

using ReactiveMP
using HTTP
using JSON
using ProgressMeter
using NPZ

ReactiveMP.sdtype(::ReactiveMP.StandaloneDistributionNode) = ReactiveMP.Stochastic()

include(srcdir("meta/cachingmeta.jl"))
include(srcdir("environments/minigrid.jl"))
include(srcdir("models/minigrid/klcontrol.jl"))
include(srcdir("utils/tucker.jl"))
include(srcdir("rules/rules.jl"))

BayesBase.mean(q_a::PointMass) = q_a.point

# FastAPI server configuration
const API_URL = "http://localhost:8000"

function reset_environment()
    response = HTTP.request("GET", "$API_URL/reset")
    return JSON.parse(String(response.body))
end

function step_environment(action::Int)
    response = HTTP.request("POST", "$API_URL/step",
        ["Content-Type" => "application/json"],
        JSON.json(Dict("action" => action))
    )
    return JSON.parse(String(response.body))
end

function get_action_space()
    response = HTTP.request("GET", "$API_URL/action_space")
    return JSON.parse(String(response.body))
end

function run_minigrid_agent(model, loc_t_tensor, ori_t_tensor, door_t_tensor, key_t_tensor, observation_tensors, T, goal; n_episodes=1000, n_iterations=10, callbacks=nothing, wait_time=0.0)
    rewards = zeros(n_episodes)
    observations = keep(Any)

    @showprogress for i in 1:n_episodes
        # Reset environment
        env_state = reset_environment()

        # Initialize beliefs
        p_old_location = vague(Categorical, grid_size^2)
        p_old_orientation = vague(Categorical, 4)
        p_key_location = vague(Categorical, grid_size^2 - 2 * grid_size)
        p_door_location = vague(Categorical, grid_size^2 - 2 * grid_size)
        p_old_key_state = Categorical([1 - tiny, tiny])
        p_old_door_state = Categorical([1 - 2 * tiny, tiny, tiny])

        reward = 0.0

        # Initial action (Turn left)
        next_action = Int(TURN_LEFT)
        env_state = step_environment(next_action)
        reward += env_state["reward"]
        sleep(wait_time)

        # Run episode
        for t in 1:T-1
            # Get current observation
            current_obs = env_state["observation"]

            # Create observation tensor
            obs_tensor = fill(zeros(5), 7, 7)
            for x in 1:7, y in 1:7
                # Map from environment indices to our CellType indices
                cell_idx = if current_obs["image"][x][y][1] == 0  # unseen
                    Int(INVISIBLE)
                elseif current_obs["image"][x][y][1] == 1  # empty
                    Int(EMPTY)
                elseif current_obs["image"][x][y][1] == 2  # wall
                    Int(WALL)
                elseif current_obs["image"][x][y][1] == 4  # door
                    Int(DOOR)
                elseif current_obs["image"][x][y][1] == 5  # key
                    Int(KEY)
                else  # Default to empty for other cases
                    Int(EMPTY)
                end
                # Create a fresh zero vector for each cell
                obs_tensor[x, y] = zeros(5)
                # Set only the corresponding index to 1.0
                obs_tensor[x, y][cell_idx] = 1.0
            end

            orientation = zeros(4)
            orientation[current_obs["direction"]+1] = 1.0

            # Create previous action vector
            previous_action = zeros(5)
            previous_action[next_action] = 1.0

            # Run inference
            result = infer(
                model=model(
                    p_old_location=p_old_location,
                    p_old_orientation=p_old_orientation,
                    p_key_location=p_key_location,
                    p_door_location=p_door_location,
                    p_old_key_state=p_old_key_state,
                    p_old_door_state=p_old_door_state,
                    loc_t_tensor=loc_t_tensor,
                    ori_t_tensor=ori_t_tensor,
                    door_t_tensor=door_t_tensor,
                    key_t_tensor=key_t_tensor,
                    observation_tensors=observation_tensors,
                    T=T - t,
                    goal=goal
                ),
                data=(
                    observations=obs_tensor,
                    action=previous_action,
                    orientation_observation=orientation
                ),
                callbacks=(after_iteration=after_iteration_callback,),
                iterations=n_iterations,
                initialization=klcontrol_minigrid_agent_initialization(grid_size, p_old_location, p_old_orientation, p_old_door_state, p_old_key_state, p_door_location, p_key_location)
            )

            # Take action
            next_action = mode(first(last(result.posteriors[:u])))
            # Transform action from our enum (1-5) to environment action (0-6)
            env_action = if next_action == Int(TURN_LEFT)
                0  # left
            elseif next_action == Int(TURN_RIGHT)
                1  # right
            elseif next_action == Int(FORWARD)
                2  # forward
            elseif next_action == Int(PICKUP)
                3  # pickup
            elseif next_action == Int(OPEN_DOOR)
                5  # toggle/open
            else
                error("Invalid action")
            end
            @show env_action, next_action
            env_state = step_environment(env_action)
            reward += env_state["reward"]
            if env_state["reward"] > 0
                break
            end
            sleep(wait_time)

            # Update beliefs for next step
            p_old_location = last(result.posteriors[:current_location])
            p_old_orientation = last(result.posteriors[:current_orientation])
            p_old_key_state = last(result.posteriors[:current_key_state])
            p_old_door_state = last(result.posteriors[:current_door_state])
            p_key_location = last(result.posteriors[:key_location])
            p_door_location = last(result.posteriors[:door_location])
        end
        rewards[i] = reward
    end
    return mean(rewards), std(rewards)
end

# Set up environment parameters
grid_size = 4
T = 17

# Load Tucker decomposed tensors
observation_tensors = load_observation_tensors("data/tucker_decomposed_tensors/grid_size$(grid_size)/");
door_transition_tensor = load_tucker_tensor("data/tucker_decomposed_tensors/grid_size$(grid_size)/door_transition_tensor/");
orientation_transition_tensor = get_orientation_transition_tensor();
location_transition_tensor = load_tucker_tensor("data/tucker_decomposed_tensors/grid_size$(grid_size)/location_transition_tensor/");
key_transition_tensor = load_tucker_tensor("data/tucker_decomposed_tensors/grid_size$(grid_size)/key_transition_tensor/");

# Generate other tensors
# door_st_t = get_door_state_transition_tensor(grid_size) .+ tiny;
# ori_st_t = get_orientation_transition_tensor() .+ tiny;
# loc_st_t = get_self_transition_tensor(grid_size) .+ tiny;
# key_st_t = get_key_state_transition_tensor(grid_size) .+ tiny;

# Set goal (bottom right corner)
goal = zeros(grid_size^2) .+ tiny
goal[grid_size^2-grid_size+1] = 1.0
goal = Categorical(goal ./ sum(goal))

# Run experiments
callbacks = RxInferBenchmarkCallbacks()

# Run KL control agent
m_kl, s_kl = run_minigrid_agent(
    klcontrol_minigrid_agent,
    location_transition_tensor, orientation_transition_tensor, door_transition_tensor, key_transition_tensor, observation_tensors, T, goal;
    n_episodes=3, n_iterations=100, wait_time=0.0, callbacks=callbacks
)
@show m_kl, s_kl

# Run EFE agent (if implemented)
# m_efe, s_efe = run_minigrid_agent(
#     efe_minigrid_agent,
#     p_old_location, p_old_orientation, p_key_location, p_door_location,
#     p_old_key_state, p_old_door_state, loc_st_t, ori_st_t,
#     door_st_t, key_st_t, observation_tensors, T, goal;
#     n_episodes=1000, n_iterations=3, wait_time=0.0
# )
# @show m_efe, s_efe

# Run visualization
# env_state = reset_environment()
# run_minigrid_agent(
#     klcontrol_minigrid_agent,
#     p_old_location, p_old_orientation, p_key_location, p_door_location,
#     p_old_key_state, p_old_door_state, loc_st_t, ori_st_t,
#     door_st_t, key_st_t, observation_tensors, T, goal;
#     n_episodes=1, n_iterations=3, wait_time=1.0
# )



# Test single inference step
env_state = reset_environment()

# Initial action (Turn left) 
next_action = Int(TURN_LEFT)
env_state = step_environment(0)

# Get current observation
current_obs = env_state["observation"]

# Create observation tensor
obs_tensor = fill(zeros(5), 7, 7)
for x in 1:7, y in 1:7
    # Map from environment indices to our CellType indices
    cell_idx = if current_obs["image"][x][y][1] == 0  # unseen
        Int(INVISIBLE)
    elseif current_obs["image"][x][y][1] == 1  # empty
        Int(EMPTY)
    elseif current_obs["image"][x][y][1] == 2  # wall
        Int(WALL)
    elseif current_obs["image"][x][y][1] == 4  # door
        Int(DOOR)
    elseif current_obs["image"][x][y][1] == 5  # key
        Int(KEY)
    else  # Default to empty for other cases
        Int(EMPTY)
    end
    # Create a fresh zero vector for each cell
    obs_tensor[x, y] = zeros(5) .+ tiny
    # Set only the corresponding index to 1.0
    obs_tensor[x, y][cell_idx] = 1.0
end

orientation = zeros(4) .+ tiny
orientation[current_obs["direction"]+1] = 1.0

# Create previous action vector
previous_action = zeros(5) .+ tiny
previous_action[next_action] = 1.0

p_old_location = vague(Categorical, grid_size^2)
p_old_orientation = vague(Categorical, 4)
p_key_location = vague(Categorical, grid_size^2 - 2 * grid_size)
p_door_location = vague(Categorical, grid_size^2 - 2 * grid_size)
p_old_key_state = Categorical([1 - tiny, tiny])
p_old_door_state = Categorical([1 - 2 * tiny, tiny, tiny])

T = 20

# Run single inference step
# Run inference
result = infer(
    model=klcontrol_minigrid_agent(
        p_old_location=p_old_location,
        p_old_orientation=p_old_orientation,
        p_key_location=p_key_location,
        p_door_location=p_door_location,
        p_old_key_state=p_old_key_state,
        p_old_door_state=p_old_door_state,
        loc_t_tensor=location_transition_tensor,
        ori_t_tensor=orientation_transition_tensor,
        door_t_tensor=door_transition_tensor,
        key_t_tensor=key_transition_tensor,
        observation_tensors=observation_tensors,
        T=T,
        goal=goal
    ),
    data=(
        observations=obs_tensor,
        action=previous_action,
        orientation_observation=orientation
    ),
    callbacks=callbacks,
    iterations=50,
    initialization=klcontrol_minigrid_agent_initialization(grid_size, p_old_location, p_old_orientation, p_old_door_state, p_old_key_state, p_door_location, p_key_location),
    showprogress=true,
    free_energy=false
)

# Print posteriors
@show result.posteriors[:current_location]
@show result.posteriors[:current_orientation]
@show result.posteriors[:current_key_state]
@show result.posteriors[:current_door_state]
@show result.posteriors[:key_location]
@show result.posteriors[:door_location]

using RxEnvironmentsZoo.GLMakie

# Extract free energy values
fe_values = result.free_energy

# Create figure and plot
fig = Figure()
ax = Axis(fig[1, 1],
    xlabel="Iteration",
    ylabel="Free Energy",
    title="Free Energy During Inference")

# Plot free energy curve
lines!(ax, 1:(length(fe_values)-5), fe_values[6:end])

# Display the figure
display(fig)

m_T1 = vague(Categorical, 4)
m_T2 = p_key_location
m_T3 = p_door_location
m_T4 = p_old_key_state
m_T5 = p_old_door_state
q_out = PointMass(obs_tensor[1, 1])
B = generate_observation_tensor(7);
q_a_ts = PointMass(B[1, 1, :, :, :, :, :, :, :]);
q_a_tucker = PointMass(observation_tensors[1, 1]);

@benchmark @call_rule DiscreteTransition(:in, Marginalisation) (q_out=$q_out, m_T1=$m_T1, m_T2=$m_T2, m_T3=$m_T3, m_T4=$m_T4, m_T5=$m_T5, q_a=$q_a_ts)
@benchmark @call_rule DiscreteTransition(:in, Marginalisation) (q_out=$q_out, m_T1=$m_T1, m_T2=$m_T2, m_T3=$m_T3, m_T4=$m_T4, m_T5=$m_T5, q_a=$q_a_tucker)


@call_rule DiscreteTransition(:in, Marginalisation) (q_out=q_out, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a_ts)
@call_rule DiscreteTransition(:in, Marginalisation) (q_out=q_out, m_T1=m_T1, m_T2=m_T2, m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a_tucker)