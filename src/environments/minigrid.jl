using LinearAlgebra

@enum ActionType begin
    TURN_LEFT = 1
    TURN_RIGHT = 2
    FORWARD = 3
    PICKUP = 4
    OPEN_DOOR = 5
end

@enum CellType begin
    INVISIBLE = 1
    EMPTY = 2
    WALL = 3
    DOOR = 4
    KEY = 5
end

@enum Orientation begin
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4
end

"""
Convert state number to (x,y) coordinates in a grid of size n×n.
"""
function state_to_coords(s::Int, n::Int)
    return (div(s - 1, n) + 1, mod(s - 1, n) + 1)
end

"""
Convert (x,y) coordinates to state number in a grid of size n×n.
"""
function coords_to_state(x::Int, y::Int, n::Int)
    return (x - 1) * n + y
end

"""
Transform coordinates from world space to agent's relative frame.

Args:
    agent_x, agent_y: Agent's position in world coordinates
    orientation: Agent's orientation (1=RIGHT, 2=DOWN, 3=LEFT, 4=UP)
    target_x, target_y: Target position in world coordinates

Returns:
    Tuple(Int, Int): Relative (x,y) coordinates from agent's perspective
    where:
    - x: positive is right, negative is left
    - y: positive is forward, negative is backward
    relative to the agent's current orientation
"""
function get_relative_coords(agent_x::Int, agent_y::Int, orientation::Int, target_x::Int, target_y::Int)
    # Translate to agent-centered coordinates
    dx = target_x - agent_x
    dy = target_y - agent_y

    # Rotate based on orientation
    if orientation == Int(RIGHT)
        return (-dy, dx)  # Right is forward, up is left
    elseif orientation == Int(DOWN)
        return (-dx, -dy)  # Down is forward, right is right
    elseif orientation == Int(LEFT)
        return (dy, -dx)  # Left is forward, down is left
    else # UP
        return (dx, dy)  # Up is forward, left is left
    end
end

"""
Check if a position is within the 7×7 field of view.
"""
function in_fov(rel_x::Int, rel_y::Int)
    return -3 ≤ rel_x ≤ 3 && 0 ≤ rel_y ≤ 6
end

"""
Convert relative coordinates to field of view coordinates.
"""
function relative_to_fov_coords(rel_x::Int, rel_y::Int)
    # Agent is at (4, 7) in FOV coordinates
    fov_x = 4 + rel_x
    fov_y = 7 - rel_y  # Flip y-axis since FOV is counted from top
    return (fov_x, fov_y)
end

"""
Generate a visibility mask for the grid based on the agent's position.
This implements a flood-fill visibility algorithm similar to the one used in MiniGrid.

Args:
    agent_x, agent_y: Agent's position in world coordinates
    width, height: Dimensions of the grid
    walls: Set of (x,y) tuples representing wall positions that block vision

Returns:
    Array{Bool, 2}: A boolean mask where true indicates a visible cell
"""
function generate_visibility_mask(agent_x::Int, agent_y::Int, width::Int, height::Int, walls::Set{Tuple{Int,Int}})
    # Initialize visibility mask
    mask = zeros(Bool, width, height)

    # Agent's position is always visible
    mask[agent_x, agent_y] = true

    # Process visibility in a top-down manner (from higher y to lower)
    for j in height:-1:1
        # Left to right pass
        for i in 1:width-1
            if !mask[i, j]
                continue
            end

            # If current cell is a wall, don't spread visibility
            if (i, j) in walls
                continue
            end

            # Spread visibility to adjacent cells
            mask[i+1, j] = true
            if j > 1
                mask[i+1, j-1] = true
                mask[i, j-1] = true
            end
        end

        # Right to left pass
        for i in width:-1:2
            if !mask[i, j]
                continue
            end

            # If current cell is a wall, don't spread visibility
            if (i, j) in walls
                continue
            end

            # Spread visibility to adjacent cells
            mask[i-1, j] = true
            if j > 1
                mask[i-1, j-1] = true
                mask[i, j-1] = true
            end
        end
    end

    return mask
end

"""
Convert a key position index to (x,y) coordinates.

Args:
    key_pos: Key position index
    n: Grid size

Returns:
    Tuple(Int, Int): (x,y) coordinates of the key
"""
function key_position(key_pos::Int, n::Int)
    return div(key_pos - 1, n) + 1, mod(key_pos - 1, n) + 1
end

"""
Convert a door position index to (x,y) coordinates.

Args:
    door_pos: Door position index
    n: Grid size

Returns:
    Tuple(Int, Int): (x,y) coordinates of the door
"""
function door_position(door_pos::Int, n::Int)
    return div(door_pos - 1, n) + 2, mod(door_pos - 1, n) + 1
end

"""
Create the set of wall positions for a given door position.
Walls only occur in the same column as the door.

Args:
    door_x: x-coordinate of the door
    door_y: y-coordinate of the door
    n: Grid size

Returns:
    Set{Tuple{Int,Int}}: Set of wall coordinates
"""
function create_wall_set(door_x::Int, door_y::Int, n::Int)
    walls = Set{Tuple{Int,Int}}()
    # Add all positions in the door's column except the door position itself
    for y in 1:n
        if y != door_y
            push!(walls, (door_x, y))
        end
    end
    for x in 1:n
        push!(walls, (x, 0))
        push!(walls, (x, n + 1))
    end
    for y in 1:n
        push!(walls, (0, y))
        push!(walls, (n + 1, y))
    end
    # Add corners
    push!(walls, (0, 0))     # Bottom-left corner
    push!(walls, (n + 1, 0)) # Bottom-right corner 
    push!(walls, (0, n + 1)) # Top-left corner
    push!(walls, (n + 1, n + 1)) # Top-right corner
    return walls
end

"""
Transform coordinates from agent's relative frame to world space coordinates.

Args:
    agent_x, agent_y: Agent's position in world coordinates
    orientation: Agent's orientation (1=RIGHT, 2=DOWN, 3=LEFT, 4=UP)
    rel_x, rel_y: Relative coordinates from agent's perspective
        where:
        - x: positive is right, negative is left
        - y: positive is forward, negative is backward
        relative to the agent's current orientation

Returns:
    Tuple(Int, Int): Absolute (x,y) coordinates in world space
"""
function relative_to_absolute_coords(agent_x::Int, agent_y::Int, orientation::Int, rel_x::Int, rel_y::Int)
    # Rotate based on orientation
    if orientation == Int(RIGHT)
        # Convert from (right/left, forward/back) to (x,y)
        dx = rel_y
        dy = -rel_x
    elseif orientation == Int(DOWN)
        dx = -rel_x
        dy = -rel_y
    elseif orientation == Int(LEFT)
        dx = -rel_y
        dy = rel_x
    else # UP
        dx = rel_x
        dy = rel_y
    end

    # Translate from agent-centered to world coordinates
    return (agent_x + dx, agent_y + dy)
end

function get_fov(agent_x::Int, agent_y::Int, orientation::Int, key_x::Int, key_y::Int, door_x::Int, door_y::Int, has_key::Int, door_state::Int, n::Int)
    fov = fill(Int(EMPTY), 7, 7)
    # Create wall set for occlusion checking
    walls = create_wall_set(door_x, door_y, n)

    # Process walls first
    for (wall_x, wall_y) in walls
        rel_wall = get_relative_coords(agent_x, agent_y, orientation, wall_x, wall_y)
        if in_fov(rel_wall...)
            fov_x, fov_y = relative_to_fov_coords(rel_wall...)
            fov[fov_x, fov_y] = Int(WALL)
        end
    end

    # Process key
    if has_key == 0 # If we don't have the key
        rel_key = get_relative_coords(agent_x, agent_y, orientation, key_x, key_y)
        if in_fov(rel_key...)
            fov_x, fov_y = relative_to_fov_coords(rel_key...)
            fov[fov_x, fov_y] = Int(KEY)
        end
    else # If we have the key, the key location will be the agent positon
        fov[4, 7] = Int(KEY)
    end


    # Process door
    rel_door = get_relative_coords(agent_x, agent_y, orientation, door_x, door_y)
    if in_fov(rel_door...)
        fov_x, fov_y = relative_to_fov_coords(rel_door...)
        fov[fov_x, fov_y] = Int(DOOR)
    end

    if door_state != 3 # If the door is not open, the door is not see-through and blocks visibility
        push!(walls, (door_x, door_y))
    end
    relative_walls = map(wall -> relative_to_fov_coords(get_relative_coords(agent_x, agent_y, orientation, wall...)...), collect(walls))

    visibility_mask = generate_visibility_mask(4, 7, 7, 7, Set(relative_walls))
    for x in -3:3, y in 0:6
        fov_x, fov_y = relative_to_fov_coords(x, y)
        if !visibility_mask[fov_x, fov_y]
            fov[fov_x, fov_y] = Int(INVISIBLE)
        end
    end
    return fov
end

"""
Generate the observation tensor for a Minigrid environment.

The observation tensor maps from:
- Agent position (n×n grid, numbered left-to-right, top-to-bottom)
- Agent orientation (4 directions: right, down, left, up)
- Key position (can be in leftmost 2 columns)
- Door position (can be in middle 2 columns, in a column with only walls)
to:
- A 7×7 field of view where each cell contains probabilities over cell types

Args:
    n::Int: Size of the grid (n×n)

Returns:
    Array{Float64, 6}: Observation tensor of shape (7, 7, 5, n^2, 4, n^2 - 2n, n^2 - 2n)
"""
function generate_observation_tensor(n::Int)
    # Calculate dimensions
    n_states = n * n
    n_key_positions = n_states - 2n  # key cannot be in two rightmost columns
    n_door_positions = n_states - 2n  # door cannot be in leftmost or rightmost columns

    # Initialize observation tensor
    B = zeros(Float64, 7, 7, 5, n_states, 4, n_key_positions, n_door_positions, 2, 3)

    # For each agent state, orientation, key position, and door position
    for agent_state in 1:n_states
        agent_x, agent_y = state_to_coords(agent_state, n)

        for orientation in 1:4
            # For each key position
            for key_pos in 1:n_key_positions
                key_x, key_y = key_position(key_pos, n)

                # For each door position
                for door_pos in 1:n_door_positions
                    door_x, door_y = door_position(door_pos, n)
                    for has_key in 0:1
                        for door_state in 1:3
                            fov = get_fov(agent_x, agent_y, orientation, key_x, key_y, door_x, door_y, has_key, door_state, n)

                            # Convert FOV to one-hot encoding in observation tensor
                            for i in 1:7, j in 1:7
                                B[i, j, fov[i, j], agent_state, orientation, key_pos, door_pos, has_key+1, door_state] = 1.0
                            end
                        end
                    end
                end
            end
        end
    end

    return B
end

function generate_flat_observation_tensor(n::Int)
    n_states = n * n
    n_key_positions = n_states - 2n  # key cannot be in two rightmost columns
    n_door_positions = n_states - 2n  # door cannot be in leftmost or rightmost columns
    B = zeros(Float64, 7, 7, 5, 4 * n_states, n_key_positions * n_door_positions, 2 * 3)

    for agent_state in 1:n_states
        agent_x, agent_y = state_to_coords(agent_state, n)

        for orientation in 1:4
            # For each key position
            for key_pos in 1:n_key_positions
                key_x, key_y = key_position(key_pos, n)

                # For each door position
                for door_pos in 1:n_door_positions
                    door_x, door_y = door_position(door_pos, n)
                    for has_key in 0:1
                        for door_state in 1:3
                            fov = get_fov(agent_x, agent_y, orientation, key_x, key_y, door_x, door_y, has_key, door_state, n)

                            # Convert FOV to one-hot encoding in observation tensor
                            for i in 1:7, j in 1:7
                                B[i, j, fov[i, j], (n_states*(orientation-1))+agent_state, (n_key_positions*(door_pos-1))+key_pos, has_key*3+door_state] = 1.0
                            end
                        end
                    end
                end
            end
        end
    end

    return B
end

"""
Helper function to get a specific observation from the tensor.
"""
function get_observation(B, agent_state, orientation, key_pos, door_pos, has_key, door_state)
    return B[:, :, :, agent_state, orientation, key_pos, door_pos, has_key, door_state]
end


function get_orientation_transition_tensor()
    T = zeros(Float64, 4, 4, 5)
    T[:, :, 1] = [
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
        1.0 0.0 0.0 0.0
    ]
    T[:, :, 2] = [
        0.0 0.0 0.0 1.0
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
    ]
    T[:, :, 3] = [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ]
    T[:, :, 4] = [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ]
    T[:, :, 5] = [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ]
    return T
end

function merge_states(dim1, dim2)
    n_states = dim1 * dim2
    T = zeros(Float64, n_states, dim1, dim2)
    for i in 1:dim1
        for j in 1:dim2
            T[(dim2*(i-1))+j, i, j] = 1.0
        end
    end
    return T
end

function get_next_agent_position(agent_x::Int, agent_y::Int, orientation::Int, door_x::Int, door_y::Int, key_x::Int, key_y::Int, door_state::Int, key_state::Int, action::Int, n::Int)
    if action == 1 # rotate right
        return coords_to_state(agent_x, agent_y, n)
    elseif action == 2 # rotate left
        return coords_to_state(agent_x, agent_y, n)
    elseif action == 3
        # Calculate new position based on orientation
        new_x, new_y = agent_x, agent_y
        if orientation == Int(RIGHT)
            new_x += 1
        elseif orientation == Int(DOWN)
            new_y -= 1
        elseif orientation == Int(LEFT)
            new_x -= 1
        else # UP
            new_y += 1
        end

        # Check for collisions with walls, key, or unopened door
        if new_x < 1 || new_x > n || new_y < 1 || new_y > n # Grid boundaries
            return coords_to_state(agent_x, agent_y, n)
        elseif new_x == key_x && new_y == key_y && key_state == 1 # Key present
            return coords_to_state(agent_x, agent_y, n)
        elseif new_x == door_x && door_state != 3 # Closed door
            return coords_to_state(agent_x, agent_y, n)
        else
            return coords_to_state(new_x, new_y, n)
        end
    elseif action == 4
        return coords_to_state(agent_x, agent_y, n)
    elseif action == 5
        return coords_to_state(agent_x, agent_y, n)
    end
end

function get_self_transition_tensor(n::Int)
    n_orientations = 4
    n_actions = 5
    n_states = n * n
    n_key_positions = n_states - 2n
    n_door_positions = n_states - 2n
    n_key_states = 2
    n_door_states = 3
    T = zeros(Float64, n_states, n_states, n_orientations, n_key_positions, n_door_positions, n_key_states, n_door_states, n_actions)
    for old_agent_state in 1:n_states
        agent_x, agent_y = state_to_coords(old_agent_state, n)
        for orientation in 1:n_orientations
            for key_pos in 1:n_key_positions
                key_x, key_y = key_position(key_pos, n)
                for door_pos in 1:n_door_positions
                    door_x, door_y = door_position(door_pos, n)
                    for key_state in 1:n_key_states
                        for door_state in 1:n_door_states
                            for action in 1:n_actions
                                if agent_x == door_x && agent_y != door_y
                                    T[old_agent_state, old_agent_state, orientation, key_pos, door_pos, key_state, door_state, action] = 1.0
                                    continue
                                end
                                if key_x == door_x
                                    T[old_agent_state, old_agent_state, orientation, key_pos, door_pos, key_state, door_state, action] = 1.0
                                    continue
                                end
                                new_agent_state = get_next_agent_position(agent_x, agent_y, orientation, door_x, door_y, key_x, key_y, door_state, key_state, action, n)
                                T[new_agent_state, old_agent_state, orientation, key_pos, door_pos, key_state, door_state, action] = 1.0
                            end
                        end
                    end
                end
            end
        end
    end
    return T
end

function get_new_door_state(agent_x::Int, agent_y::Int, orientation::Int, door_x::Int, door_y::Int, action::Int, door_state::Int, key_state::Int)
    # Only handle door opening action (5), all other actions don't change door state
    if action != 5
        return door_state
    end

    # Can't open door without key
    if key_state == 1
        return door_state
    end

    # Get relative coordinates of door from agent's perspective
    rel_x, rel_y = get_relative_coords(agent_x, agent_y, orientation, door_x, door_y)

    # Check if agent is directly in front of door and facing it
    if rel_x == 0 && rel_y == 1
        return 3 # Door is opened
    end

    return door_state # Door state remains unchanged
end

function get_door_state_transition_tensor(n::Int)
    n_states = n * n
    n_door_states = 3
    n_key_states = 2
    n_orientations = 4
    n_door_positions = n_states - 2n
    T = zeros(Float64, n_door_states, n_door_states, n_states, n_orientations, n_door_positions, n_key_states, 5)
    for old_agent_state in 1:n_states
        agent_x, agent_y = state_to_coords(old_agent_state, n)
        for orientation in 1:n_orientations
            for door_pos in 1:n_door_positions
                door_x, door_y = door_position(door_pos, n)
                for key_state in 1:n_key_states
                    for door_state in 1:n_door_states
                        for action in 1:5
                            new_door_state = get_new_door_state(agent_x, agent_y, orientation, door_x, door_y, action, door_state, key_state)
                            T[new_door_state, door_state, old_agent_state, orientation, door_pos, key_state, action] = 1.0
                        end
                    end
                end
            end
        end
    end
    return T
end

function get_new_key_state(agent_x::Int, agent_y::Int, orientation::Int, key_x::Int, key_y::Int, action::Int, key_state::Int)
    if action == 4  # Pickup action
        # If key is already picked up, state remains the same
        if key_state == 2
            return key_state
        end

        # Get relative coordinates of key from agent's perspective
        rel_x, rel_y = get_relative_coords(agent_x, agent_y, orientation, key_x, key_y)

        # Check if agent is directly in front of key and facing it
        if rel_x == 0 && rel_y == 1
            return 2  # Key is picked up
        else
            return key_state  # Key state remains unchanged
        end
    else
        # For all other actions, key state remains unchanged
        return key_state
    end
end

function get_key_state_transition_tensor(n::Int)
    n_states = n * n
    n_key_states = 2
    n_orientations = 4
    n_key_positions = n_states - 2n
    T = zeros(Float64, n_key_states, n_key_states, n_states, n_orientations, n_key_positions, 5)

    for old_agent_state in 1:n_states
        agent_x, agent_y = state_to_coords(old_agent_state, n)
        for orientation in 1:n_orientations
            for key_pos in 1:n_key_positions
                key_x, key_y = key_position(key_pos, n)
                for key_state in 1:n_key_states
                    for action in 1:5
                        new_key_state = get_new_key_state(agent_x, agent_y, orientation, key_x, key_y, action, key_state)
                        T[new_key_state, key_state, old_agent_state, orientation, key_pos, action] = 1.0
                    end
                end
            end
        end
    end

    return T
end
