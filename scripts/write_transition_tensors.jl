using NPZ

using DrWatson
@quickactivate "EFEasVFE"

include(srcdir("environments", "minigrid.jl"))
function write_all_tensors(grid_size)
    println("Generating tensors for grid_size = $(grid_size)...")

    # Generate tensors
    println("Generating observation tensor...")
    observation_tensors = generate_observation_tensor(grid_size)
    println("Observation tensor shape: $(size(observation_tensors))")
    
    # Check for NaN values in observation tensors
    nan_count = count(isnan, observation_tensors)
    inf_count = count(isinf, observation_tensors)
    
    if nan_count > 0 || inf_count > 0
        println("WARNING: Observation tensor contains $(nan_count) NaN and $(inf_count) Inf values!")
        println("Fixing by replacing with zeros...")
        observation_tensors = replace(observation_tensors, NaN => 0.0, Inf => 0.0)
    else
        println("Observation tensor check: No NaN/Inf values detected")
    end
    
    println("Generating door/key transition tensor...")
    door_key_transition_tensor = get_key_door_state_transition_tensor(grid_size)
    println("Door/key tensor shape: $(size(door_key_transition_tensor))")
    
    # Check for NaN values
    nan_count = count(isnan, door_key_transition_tensor)
    inf_count = count(isinf, door_key_transition_tensor)
    
    if nan_count > 0 || inf_count > 0
        println("WARNING: Door/key transition tensor contains $(nan_count) NaN and $(inf_count) Inf values!")
        println("Fixing by replacing with zeros...")
        door_key_transition_tensor = replace(door_key_transition_tensor, NaN => 0.0, Inf => 0.0)
    else
        println("Door/key tensor check: No NaN/Inf values detected")
    end
    
    println("Generating orientation transition tensor...")
    orientation_transition_tensor = get_orientation_transition_tensor()
    println("Orientation tensor shape: $(size(orientation_transition_tensor))")
    
    # Check for NaN values
    nan_count = count(isnan, orientation_transition_tensor)
    inf_count = count(isinf, orientation_transition_tensor)
    
    if nan_count > 0 || inf_count > 0
        println("WARNING: Orientation transition tensor contains $(nan_count) NaN and $(inf_count) Inf values!")
        println("Fixing by replacing with zeros...")
        orientation_transition_tensor = replace(orientation_transition_tensor, NaN => 0.0, Inf => 0.0)
    else
        println("Orientation tensor check: No NaN/Inf values detected")
    end
    
    println("Generating location transition tensor...")
    location_transition_tensor = get_self_transition_tensor(grid_size)
    println("Location tensor shape: $(size(location_transition_tensor))")
    
    # Check for NaN values
    nan_count = count(isnan, location_transition_tensor)
    inf_count = count(isinf, location_transition_tensor)
    
    if nan_count > 0 || inf_count > 0
        println("WARNING: Location transition tensor contains $(nan_count) NaN and $(inf_count) Inf values!")
        println("Fixing by replacing with zeros...")
        location_transition_tensor = replace(location_transition_tensor, NaN => 0.0, Inf => 0.0)
    else
        println("Location tensor check: No NaN/Inf values detected")
    end

    # Create output directory if it doesn't exist
    mkpath("data/raw_tensors/grid_size$(grid_size)")

    # Save observation tensors - check each slice for any remaining NaN values
    println("\nSaving tensors...")
    for x in 1:7, y in 1:7
        tensor_slice = observation_tensors[x, y, :, :, :, :, :, :]
        
        # Final safety check
        if any(isnan, tensor_slice) || any(isinf, tensor_slice)
            println("WARNING: Still found NaN/Inf in tensor slice ($x,$y)! Making final fix...")
            tensor_slice = replace(tensor_slice, NaN => 0.0, Inf => 0.0)
        end
        
        npzwrite("data/raw_tensors/grid_size$(grid_size)/observation_tensor_x$(x)_y$(y).npy", tensor_slice)
    end

    # Final safety check for other tensors
    if any(isnan, door_key_transition_tensor) || any(isinf, door_key_transition_tensor)
        println("WARNING: Still found NaN/Inf in door_key_transition_tensor! Making final fix...")
        door_key_transition_tensor = replace(door_key_transition_tensor, NaN => 0.0, Inf => 0.0)
    end
    
    if any(isnan, orientation_transition_tensor) || any(isinf, orientation_transition_tensor)
        println("WARNING: Still found NaN/Inf in orientation_transition_tensor! Making final fix...")
        orientation_transition_tensor = replace(orientation_transition_tensor, NaN => 0.0, Inf => 0.0)
    end
    
    if any(isnan, location_transition_tensor) || any(isinf, location_transition_tensor)
        println("WARNING: Still found NaN/Inf in location_transition_tensor! Making final fix...")
        location_transition_tensor = replace(location_transition_tensor, NaN => 0.0, Inf => 0.0)
    end

    npzwrite("data/raw_tensors/grid_size$(grid_size)/door_key_transition_tensor.npy", door_key_transition_tensor)
    npzwrite("data/raw_tensors/grid_size$(grid_size)/orientation_transition_tensor.npy", orientation_transition_tensor)
    npzwrite("data/raw_tensors/grid_size$(grid_size)/location_transition_tensor.npy", location_transition_tensor)
    
    println("Tensor processing complete!")
end

grid_size = 5

write_all_tensors(grid_size)


