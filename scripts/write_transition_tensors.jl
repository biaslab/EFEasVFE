using NPZ

using DrWatson
@quickactivate "EFEasVFE"

include(srcdir("environments", "minigrid.jl"))
function write_all_tensors(grid_size)

    observation_tensors = generate_observation_tensor(grid_size)
    door_key_transition_tensor = get_key_door_state_transition_tensor(grid_size)
    orientation_transition_tensor = get_orientation_transition_tensor()
    location_transition_tensor = get_self_transition_tensor(grid_size)

    # Create output directory if it doesn't exist
    mkpath("data/raw_tensors/grid_size$(grid_size)")


    for x in 1:7, y in 1:7
        npzwrite("data/raw_tensors/grid_size$(grid_size)/observation_tensor_x$(x)_y$(y).npy", observation_tensors[x, y, :, :, :, :, :, :])
    end

    npzwrite("data/raw_tensors/grid_size$(grid_size)/door_key_transition_tensor.npy", door_key_transition_tensor)
    npzwrite("data/raw_tensors/grid_size$(grid_size)/orientation_transition_tensor.npy", orientation_transition_tensor)
    npzwrite("data/raw_tensors/grid_size$(grid_size)/location_transition_tensor.npy", location_transition_tensor)
end

grid_size = 5

write_all_tensors(grid_size)



