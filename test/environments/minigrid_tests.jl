@testitem "Minigrid Environment Tests" begin
    using Test
    using LinearAlgebra
    using EFEasVFE

    @testset "Coordinate Conversion" begin
        import EFEasVFE: state_to_coords, coords_to_state
        # Test state_to_coords
        @test state_to_coords(1, 4) == (1, 1)  # Top-left corner
        @test state_to_coords(4, 4) == (1, 4)  # Top-right corner
        @test state_to_coords(13, 4) == (4, 1)  # Bottom-left corner
        @test state_to_coords(16, 4) == (4, 4)  # Bottom-right corner

        # Test coords_to_state
        @test coords_to_state(1, 1, 4) == 1
        @test coords_to_state(1, 4, 4) == 4
        @test coords_to_state(4, 1, 4) == 13
        @test coords_to_state(4, 4, 4) == 16

        # Test roundtrip conversion
        for s in 1:16
            @test coords_to_state(state_to_coords(s, 4)..., 4) == s
        end
    end

    @testset "Relative Coordinates" begin
        import EFEasVFE: get_relative_coords, relative_to_absolute_coords, RIGHT, DOWN, LEFT, UP
        # Test facing right
        @test get_relative_coords(2, 2, Int(RIGHT), 2, 3) == (-1, 0)  # Target to left
        @test get_relative_coords(2, 2, Int(RIGHT), 1, 2) == (0, -1)  # Target below
        @test get_relative_coords(2, 2, Int(RIGHT), 2, 1) == (1, 0)   # Target to right
        @test get_relative_coords(2, 2, Int(RIGHT), 3, 2) == (0, 1)   # Target above
        @test get_relative_coords(2, 2, Int(RIGHT), 2, 2) == (0, 0)   # Same position
        @test get_relative_coords(1, 1, Int(RIGHT), 1, 2) == (-1, 0)

        # Test facing down
        @test get_relative_coords(2, 2, Int(DOWN), 2, 3) == (0, -1)   # Target behind
        @test get_relative_coords(2, 2, Int(DOWN), 3, 2) == (-1, 0)   # Target to left
        @test get_relative_coords(2, 2, Int(DOWN), 2, 1) == (0, 1)    # Target in front
        @test get_relative_coords(2, 2, Int(DOWN), 1, 2) == (1, 0)    # Target to right
        @test get_relative_coords(2, 2, Int(DOWN), 2, 2) == (0, 0)    # Same position

        # Test facing left
        @test get_relative_coords(2, 2, Int(LEFT), 2, 3) == (1, 0)    # Target to right
        @test get_relative_coords(2, 2, Int(LEFT), 1, 2) == (0, 1)    # Target above
        @test get_relative_coords(2, 2, Int(LEFT), 2, 1) == (-1, 0)   # Target to left
        @test get_relative_coords(2, 2, Int(LEFT), 3, 2) == (0, -1)   # Target below
        @test get_relative_coords(2, 2, Int(LEFT), 2, 2) == (0, 0)    # Same position

        # Test facing up
        @test get_relative_coords(2, 2, Int(UP), 2, 3) == (0, 1)      # Target in front
        @test get_relative_coords(2, 2, Int(UP), 3, 2) == (1, 0)      # Target to right
        @test get_relative_coords(2, 2, Int(UP), 2, 1) == (0, -1)     # Target behind
        @test get_relative_coords(2, 2, Int(UP), 1, 2) == (-1, 0)     # Target to left
        @test get_relative_coords(2, 2, Int(UP), 2, 2) == (0, 0)      # Same position

        # Test relative_to_absolute_coords
        # Test facing right
        @test relative_to_absolute_coords(2, 2, Int(RIGHT), -1, 0) == (2, 3)  # Left becomes forward
        @test relative_to_absolute_coords(2, 2, Int(RIGHT), 0, 1) == (3, 2)   # Forward becomes up
        @test relative_to_absolute_coords(2, 2, Int(RIGHT), 1, 0) == (2, 1)   # Right becomes back
        @test relative_to_absolute_coords(2, 2, Int(RIGHT), 0, -1) == (1, 2)  # Back becomes down

        # Test facing down
        @test relative_to_absolute_coords(2, 2, Int(DOWN), -1, 0) == (3, 2)   # Left becomes left
        @test relative_to_absolute_coords(2, 2, Int(DOWN), 0, 1) == (2, 1)    # Forward becomes forward
        @test relative_to_absolute_coords(2, 2, Int(DOWN), 1, 0) == (1, 2)    # Right becomes right
        @test relative_to_absolute_coords(2, 2, Int(DOWN), 0, -1) == (2, 3)   # Back becomes back

        # Test facing left
        @test relative_to_absolute_coords(2, 2, Int(LEFT), -1, 0) == (2, 1)   # Left becomes back
        @test relative_to_absolute_coords(2, 2, Int(LEFT), 0, 1) == (1, 2)    # Forward becomes down
        @test relative_to_absolute_coords(2, 2, Int(LEFT), 1, 0) == (2, 3)    # Right becomes forward
        @test relative_to_absolute_coords(2, 2, Int(LEFT), 0, -1) == (3, 2)   # Back becomes up

        # Test facing up
        @test relative_to_absolute_coords(2, 2, Int(UP), -1, 0) == (1, 2)     # Left becomes left
        @test relative_to_absolute_coords(2, 2, Int(UP), 0, 1) == (2, 3)      # Forward becomes forward
        @test relative_to_absolute_coords(2, 2, Int(UP), 1, 0) == (3, 2)      # Right becomes right
        @test relative_to_absolute_coords(2, 2, Int(UP), 0, -1) == (2, 1)     # Back becomes back

        # Test inverse relationship
        # Test multiple positions and orientations
        for agent_x in 1:4, agent_y in 1:4, orientation in 1:4, target_x in 1:4, target_y in 1:4
            rel_coords = get_relative_coords(agent_x, agent_y, orientation, target_x, target_y)
            abs_coords = relative_to_absolute_coords(agent_x, agent_y, orientation, rel_coords...)
            @test abs_coords == (target_x, target_y)
        end
    end

    @testset "Field of View" begin
        import EFEasVFE: in_fov, relative_to_fov_coords
        # Test in_fov
        @test in_fov(0, 0) == true     # Agent's position
        @test in_fov(3, 6) == true     # Top-right corner of FOV
        @test in_fov(-3, 6) == true    # Top-left corner of FOV
        @test in_fov(3, 0) == true    # Bottom-right corner of FOV
        @test in_fov(-3, 0) == true   # Bottom-left corner of FOV
        @test in_fov(4, 0) == false    # Outside FOV (right)
        @test in_fov(0, 7) == false    # Outside FOV (top)

        # Test relative_to_fov_coords
        @test relative_to_fov_coords(0, 0) == (4, 7)    # Agent's position
        @test relative_to_fov_coords(1, 1) == (5, 6)    # One step up-right
        @test relative_to_fov_coords(-1, -1) == (3, 8)  # One step down-left
    end


    @testset "Visibility Mask" begin
        import EFEasVFE: generate_visibility_mask
        # Test simple visibility mask with no walls
        width, height = 7, 7
        agent_x, agent_y = 4, 7
        walls = Set{Tuple{Int,Int}}()

        mask = generate_visibility_mask(agent_x, agent_y, width, height, walls)

        # In an empty grid, all cells should be visible
        @test all(mask)

        # Test with a single wall
        walls = Set([(3, 3)])
        mask = generate_visibility_mask(agent_x, agent_y, width, height, walls)

        # Wall itself should be visible
        @test mask[3, 3] == true

        # Cell directly behind wall should not be visible
        @test mask[3, 2] == true

        # Test with a wall to the right
        walls = Set([(3, 2)])
        mask = generate_visibility_mask(agent_x, agent_y, width, height, walls)

        # Wall itself should be visible
        @test mask[3, 2] == true

        # Cell directly to the right of wall should visible
        @test mask[4, 2] == true

        # Cell diagonally down- behind from wall should not be visible
        @test mask[4, 1] == true

        # Test with a corridor of walls
        walls = Set([(3, 7), (3, 6), (3, 5), (3, 4), (3, 3)])
        mask = generate_visibility_mask(agent_x, agent_y, width, height, walls)

        # Corridor at y=2 should be visible
        @test mask[3, 2] == true
        @test mask[2, 4] == false


        # Areas behind walls should not be visible
        @test mask[2, 7] == false
        @test mask[2, 6] == false
    end

    @testset "Key and Door Position" begin
        import EFEasVFE: key_position, door_position
        @test key_position(1, 4) == (1, 1)
        @test key_position(2, 4) == (1, 2)
        @test key_position(3, 4) == (1, 3)
        @test key_position(4, 4) == (1, 4)
        @test key_position(5, 4) == (2, 1)
        @test key_position(6, 4) == (2, 2)
        @test key_position(7, 4) == (2, 3)
        @test key_position(8, 4) == (2, 4)

        @test door_position(1, 4) == (2, 1)
        @test door_position(2, 4) == (2, 2)
        @test door_position(3, 4) == (2, 3)
        @test door_position(4, 4) == (2, 4)
        @test door_position(5, 4) == (3, 1)
        @test door_position(6, 4) == (3, 2)
        @test door_position(7, 4) == (3, 3)
        @test door_position(8, 4) == (3, 4)
    end

    @testset "Observation Tensor" begin
        import EFEasVFE: generate_observation_tensor, get_observation, EMPTY, WALL, INVISIBLE, KEY, DOOR
        for n in 4:7
            B = generate_observation_tensor(n, Float64)

            # Test tensor dimensions
            @test size(B) == (7, 7, 5, n^2, 4, n^2 - 2n, n^2 - 2n, 3)

            # Test that agent is always visible at (4, 7)
            for state in 1:n^2, orient in 1:4, key in 1:(n^2-2n), door in 1:(n^2-2n)
                door_column = ((door - 1) ÷ n) + 1
                key_column = ((key - 1) ÷ n)
                agent_column = ((state - 1) ÷ n)
                if door_column == key_column # Door and key cannot be in the same column
                    continue
                elseif door_column == agent_column && state - door ≠ n # user cannot be in a wall in the column with the door
                    continue
                elseif state - key == 0 # Should check what happens if key is in the same cell with the user
                    continue
                elseif state - door == n # Should check what happens if door is in the same cell with the user
                    continue
                else
                    obs = get_observation(B, state, orient, key, door, 1, 1)
                    @test obs[4, 7, Int(EMPTY)] == 1.0
                end
            end

            # Test that all observations are one-hot encoded
            for i in 1:7, j in 1:7
                @test sum(B[i, j, :, 1, 1, 1, 1, 1, 1]) == 1.0
            end

            # Test wall occlusion
            # Choose a state where we know walls should block view
            agent_state = 3
            obs = get_observation(B, agent_state, Int(RIGHT), 1, 1, 1, 1)
            # Cells behind walls should be INVISIBLE
            @test obs[4, 6, Int(WALL)] == 1.0
            @test obs[4, 5, Int(INVISIBLE)] == 1.0

            # Test that the key will be visible at (4, 6) and the door will be visible at (4, 5)
            obs = get_observation(B, agent_state, Int(RIGHT), 3 + n, 3 + n, 1, 1)
            @test obs[4, 6, Int(KEY)] == 1.0
            @test obs[4, 5, Int(DOOR)] == 1.0
        end
    end

    @testset "Wall Set Creation" begin
        import EFEasVFE: create_wall_set
        # Test wall set for a 4×4 grid with door at (2, 2)
        walls = create_wall_set(2, 2, 4)

        # Test wall set size
        # Should include:
        # - 3 walls in door column (4 positions - 1 door)
        # - 4 walls on each side (including corners)
        # - 4 walls on top and bottom (including corners)
        @test length(walls) == 3 + 6 + 6 + 4 + 4

        # Test door column walls
        @test (2, 1) in walls  # Wall above door
        @test (2, 3) in walls  # Wall below door
        @test (2, 4) in walls  # Wall at bottom
        @test !((2, 2) in walls) # Door position should not be a wall
        @test (0, 0) in walls  # Bottom-left corner
        @test (5, 0) in walls  # Bottom-right corner
        @test (0, 5) in walls  # Top-left corner
        @test (5, 5) in walls  # Top-right corner

        # Test boundary walls
        # Bottom boundary
        @test (2, 0) in walls  # Bottom middle
        @test (4, 0) in walls  # Bottom-right

        # Top boundary
        @test (2, 5) in walls  # Top middle
        @test (4, 5) in walls  # Top-right

        # Left boundary
        @test (0, 2) in walls  # Left middle

        # Right boundary
        @test (5, 2) in walls  # Right middle

        # Test wall set for door at edge
        walls_edge = create_wall_set(2, 1, 4)
        @test length(walls_edge) == 3 + 6 + 6 + 4 + 4  # Same total as before
        @test (2, 2) in walls_edge
        @test (2, 3) in walls_edge
        @test (2, 4) in walls_edge
        @test !((2, 1) in walls_edge)  # Door position

        # Test boundary walls still present
    end

    @testset "get_fov" begin
        import EFEasVFE: get_fov, EMPTY, RIGHT, DOWN, INVISIBLE, WALL, DOOR, KEY, UP, LEFT
        fov = get_fov(1, 4, Int(RIGHT), 1, 2, 3, 2, 1, 4)
        @test fov == [
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 3 to the left is completely invisible
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 2 to the left is completely invisible
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(WALL) Int(WALL) # Column 1 to the left has some walls
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(EMPTY) Int(EMPTY) # Column with agent has empty cells and 1 wall
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(EMPTY) Int(EMPTY) # Column 1 to the right has empty cells and 1 wall
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(DOOR) Int(EMPTY) Int(KEY) # Column 2 to the right has a door and key
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(EMPTY) Int(EMPTY) # Column has empty cells and a wall
        ]

        fov = get_fov(2, 4, Int(DOWN), 1, 2, 3, 2, 1, 4)
        @test fov == [
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 3 to the right is completely invisible
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 2 to the right is completely invisible
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(WALL) Int(DOOR) Int(WALL) Int(WALL) # Column 1 to the right has a wall and a door
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(EMPTY) Int(EMPTY) Int(EMPTY) Int(EMPTY) # Column with agent has empty cells and 1 wall
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(EMPTY) Int(KEY) Int(EMPTY) Int(EMPTY) # Column 1 to the left has a wall and a key
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(WALL) Int(WALL) Int(WALL) Int(WALL) # Column 2 to the left has some walls
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 3 to the left is completely invisible
        ]
        fov = get_fov(2, 4, Int(DOWN), 1, 2, 3, 2, 2, 4)
        @test fov == [
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 3 to the right is completely invisible
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 2 to the right is completely invisible
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(WALL) Int(DOOR) Int(WALL) Int(WALL) # Column 1 to the right has a wall and a door
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(EMPTY) Int(EMPTY) Int(EMPTY) Int(KEY) # Column with agent has empty cells and 1 wall
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(EMPTY) Int(EMPTY) Int(EMPTY) Int(EMPTY) # Column 1 to the left has a wall and a key
            Int(INVISIBLE) Int(INVISIBLE) Int(WALL) Int(WALL) Int(WALL) Int(WALL) Int(WALL) # Column 2 to the left has some walls
            Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) Int(INVISIBLE) # Column 3 to the left is completely invisible
        ]
    end

    @testset "Movement transition tensor" begin
        import EFEasVFE: get_self_transition_tensor, EMPTY, RIGHT, DOWN, INVISIBLE, WALL, DOOR, KEY, UP, LEFT
        for n in 4:7
            T = get_self_transition_tensor(n, Float64)
            @test size(T) == (n^2, n^2, 4, n^2 - 2n, n^2 - 2n, 3, 5)
            for old_state in 1:n^2, orientation in 1:4, key_pos in 1:(n^2-2n), door_pos in 1:(n^2-2n), door_key_state in 1:3, action in 1:5
                if action != 3
                    @test T[old_state, old_state, orientation, key_pos, door_pos, door_key_state, action] == 1.0
                end
            end
            for door_key_state in 1:3
                @test T[n+1, 1, Int(RIGHT), 2, n+1, door_key_state, 3] == 1.0
            end
            for door_loc in (n+1):(n^2-2n)
                for old_state in 1:(n^2-3n)
                    @test T[old_state, old_state, Int(RIGHT), old_state+n, door_loc, 1, 3] == 1.0
                end
            end
        end
    end

    @testset "Door state transition tensor" begin
        import EFEasVFE: get_key_door_state_transition_tensor, EMPTY, RIGHT, DOWN, INVISIBLE, WALL, DOOR, KEY, UP, LEFT
        for n in 4:7
            T = get_key_door_state_transition_tensor(n, Float64)
            @test size(T) == (3, 3, n^2, 4, n^2 - 2n, n^2 - 2n, 5)

            # Test that door state remains unchanged for actions other than toggle (5) or pickup (4)
            for door_key_state in 1:3, agent_state in 1:n^2, orientation in 1:4, key_pos in 1:(n^2-2n), door_pos in 1:(n^2-2n), action in 1:3

                @test T[door_key_state, door_key_state, agent_state, orientation, key_pos, door_pos, action] == 1.0

            end

            # Test door opening when agent is in front of door with key
            # Agent at (2,1) facing right, door at (3,1)
            agent_state = coords_to_state(1, 1, n)
            door_pos = 1  # Door at (2,1) in door position coordinates

            for key_pos in 1:(n^2-3n)
                # Door should open when agent has key, is in front of door, and uses toggle action
                @test T[3, 2, agent_state, Int(RIGHT), key_pos, door_pos, 5] == 1.0

                # Door should not open when agent doesn't have key
                @test T[1, 1, agent_state, Int(RIGHT), key_pos, door_pos, 5] == 1.0

                # Door should not open when agent is not facing door
                @test T[1, 1, agent_state, Int(LEFT), key_pos, door_pos, 5] == 1.0
                @test T[1, 1, agent_state, Int(UP), key_pos, door_pos, 5] == 1.0
                @test T[1, 1, agent_state, Int(DOWN), key_pos, door_pos, 5] == 1.0

                # Door should not open when agent is not in front of door
                agent_state_far = coords_to_state(4, 4, n)
                @test T[1, 1, agent_state_far, Int(RIGHT), key_pos, door_pos, 5] == 1.0

                # Door state should remain unchanged if already open
                @test T[3, 3, agent_state, Int(RIGHT), key_pos, door_pos, 5] == 1.0
            end
        end
    end

    @testset "Key door state transition tensor" begin
        import EFEasVFE: get_key_door_state_transition_tensor, EMPTY, RIGHT, DOWN, INVISIBLE, WALL, DOOR, KEY, UP, LEFT
        for n in 4:7
            T = get_key_door_state_transition_tensor(n, Float64)
            @test size(T) == (3, 3, n^2, 4, n^2 - 2n, n^2 - 2n, 5)

            # Test that key state remains unchanged for actions other than pickup (4)
            for door_key_state in 1:3, agent_state in 1:n^2, orientation in 1:4, key_pos in 1:(n^2-2n), door_pos in 1:(n^2-2n), action in 1:3
                @test T[door_key_state, door_key_state, agent_state, orientation, key_pos, door_pos, action] == 1.0
            end

            # Test key pickup when agent is in front of key
            # Agent at (1,1) facing right, key at (2,1)
            agent_state = coords_to_state(1, 1, n)
            key_pos = n + 1  # Key at (2,1) in key position coordinates

            for door_pos in n:(n^2-2n)
                # Key should be picked up when agent is in front of key and uses pickup action
                @test T[2, 1, agent_state, Int(RIGHT), key_pos, door_pos, 4] == 1.0

                # Key should not be picked up when agent is not facing key
                @test T[1, 1, agent_state, Int(LEFT), key_pos, door_pos, 4] == 1.0
                @test T[1, 1, agent_state, Int(UP), key_pos, door_pos, 4] == 1.0
                @test T[1, 1, agent_state, Int(DOWN), key_pos, door_pos, 4] == 1.0

                # Key should not be picked up when agent is not in front of key
                agent_state_far = coords_to_state(3, 3, n)
                @test T[1, 1, agent_state_far, Int(RIGHT), key_pos, door_pos, 4] == 1.0

                # Key state should remain unchanged if already picked up
                @test T[2, 2, agent_state, Int(RIGHT), key_pos, door_pos, 4] == 1.0
            end
        end
    end
end

