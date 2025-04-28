@testitem "TMaze Tensors" begin
    using EFEasVFE
    import EFEasVFE: create_reward_observation_tensor, create_location_transition_tensor, create_reward_to_location_mapping
    @testset "Reward Observation Tensor" begin
        reward_obs = create_reward_observation_tensor()

        # Check dimensions
        @test size(reward_obs) == (2, 5, 2)

        # Check bottom state reveals true reward location
        @test reward_obs[:, 1, 1] == [1.0, 0.0] # At bottom state with left reward
        @test reward_obs[:, 1, 2] == [0.0, 1.0] # At bottom state with right reward

        # Check other states show uncertainty
        for loc in 2:5, reward_loc in 1:2
            @test reward_obs[:, loc, reward_loc] == [0.5, 0.5]
        end
    end

    @testset "Location Transition Tensor" begin
        transition = create_location_transition_tensor()

        # Check dimensions
        @test size(transition) == (5, 5, 4)

        # Sample key transitions
        # Bottom of T (state 1)
        @test transition[2, 1, 1] == 1.0 # North -> Middle junction
        @test transition[1, 1, 2] == 1.0 # East -> Stay (wall)
        @test transition[1, 1, 3] == 1.0 # South -> Stay (wall)
        @test transition[1, 1, 4] == 1.0 # West -> Stay (wall)

        # Middle junction (state 2)
        @test transition[4, 2, 1] == 1.0 # North -> Top middle
        @test transition[5, 2, 2] == 1.0 # East -> Top right
        @test transition[1, 2, 3] == 1.0 # South -> Bottom
        @test transition[3, 2, 4] == 1.0 # West -> Top left

        # Top left (state 3)
        @test transition[3, 3, 1] == 1.0 # North -> Stay (wall)
        @test transition[4, 3, 2] == 1.0 # East -> Top middle
        @test transition[2, 3, 3] == 1.0 # South -> Middle junction
        @test transition[3, 3, 4] == 1.0 # West -> Stay (wall)

        # Top middle (state 4)
        @test transition[4, 4, 1] == 1.0 # North -> Stay (wall)
        @test transition[5, 4, 2] == 1.0 # East -> Top right
        @test transition[2, 4, 3] == 1.0 # South -> Middle junction
        @test transition[3, 4, 4] == 1.0 # West -> Top left

        # Top right (state 5)
        @test transition[5, 5, 1] == 1.0 # North -> Stay (wall)
        @test transition[5, 5, 2] == 1.0 # East -> Stay (wall)
        @test transition[2, 5, 3] == 1.0 # South -> Middle junction
        @test transition[4, 5, 4] == 1.0 # West -> Top middle

        # Check that each slice is a valid transition matrix (rows sum to 1)
        for loc in 1:5, action in 1:4
            @test sum(transition[:, loc, action]) ≈ 1.0
        end
    end

    @testset "Reward-to-Location Mapping" begin
        reward_mapping = create_reward_to_location_mapping()

        # Check dimensions
        @test size(reward_mapping) == (5, 2)

        # Check mappings
        @test reward_mapping[3, 1] == 1.0 # Left reward -> top-left location
        @test reward_mapping[5, 2] == 1.0 # Right reward -> top-right location

        # Check all other entries are zero
        for i in 1:5, j in 1:2
            if (i == 3 && j == 1) || (i == 5 && j == 2)
                continue
            end
            @test reward_mapping[i, j] == 0.0
        end

        # Check that the total sum is 2.0 (only two non-zero entries)
        @test sum(reward_mapping) == 2.0
    end

    @testset "Integration with Model" begin
        # Create tensors
        reward_obs_tensor = create_reward_observation_tensor()
        location_transition_tensor = create_location_transition_tensor()
        reward_mapping = create_reward_to_location_mapping()

        # Check shapes match what's expected in the model
        @test size(reward_obs_tensor) == (2, 5, 2) # reward_observation_tensor in model
        @test size(location_transition_tensor) == (5, 5, 4) # location_transition_tensor in model
        @test size(reward_mapping) == (5, 2) # reward_to_location_mapping in model
    end
end

@testitem "TMaze Environment" begin
    using EFEasVFE
    import EFEasVFE: position_to_index, index_to_position

    @testset "Initial State" begin
        # Test with left reward
        env_left = create_tmaze(:left)
        @test env_left.agent_position == (2, 1) # Starting at bottom of T
        @test position_to_index(env_left.agent_position) == 1
        @test env_left.reward_position == :left

        # Test with right reward
        env_right = create_tmaze(:right)
        @test env_right.agent_position == (2, 1) # Starting at bottom of T
        @test position_to_index(env_right.agent_position) == 1
        @test env_right.reward_position == :right

        # Test with custom start position - middle junction
        env_middle = create_tmaze(:left, (2, 2))
        @test env_middle.agent_position == (2, 2) # Starting at middle junction
        @test position_to_index(env_middle.agent_position) == 2

        # Test with custom start position - top left
        env_top_left = create_tmaze(:right, (1, 3))
        @test env_top_left.agent_position == (1, 3) # Starting at top left
        @test position_to_index(env_top_left.agent_position) == 3

        # Test invalid start position should throw an error
        @test_throws ArgumentError create_tmaze(:left, (4, 4))
    end

    @testset "Reset Function" begin
        # Create and move agent
        env = create_tmaze()
        env.agent_position = (2, 2) # Move to junction

        # Reset and check position returns to start
        reset_tmaze!(env, :left)
        @test env.agent_position == (2, 1) # Back to bottom of T
        @test env.reward_position == :left

        # Reset with different reward position
        reset_tmaze!(env, :right)
        @test env.agent_position == (2, 1) # Still at bottom of T
        @test env.reward_position == :right

        # Reset with custom start position
        reset_tmaze!(env, :left, (2, 2))
        @test env.agent_position == (2, 2) # Now at middle junction
        @test env.reward_position == :left

        # Reset with invalid position should throw an error
        @test_throws ArgumentError reset_tmaze!(env, :left, (5, 5))
    end

    @testset "Step Function - Navigation" begin
        env = create_tmaze(:left)

        # Test starting position
        @test env.agent_position == (2, 1) # Bottom of T
        pos_obs, reward_cue, reward = step!(env, MazeAction(North()))

        # Should move to junction
        @test env.agent_position == (2, 2) # Junction
        @test position_to_index(env.agent_position) == 2
        @test pos_obs == [0.0, 1.0, 0.0, 0.0, 0.0] # One-hot for position 2
        @test reward_cue == [0.5, 0.5] # Uncertain at junction
        @test reward === 0.0 # No reward at junction

        # Move west to left arm
        pos_obs, reward_cue, reward = step!(env, MazeAction(West()))
        @test env.agent_position == (1, 3) # Top left
        @test position_to_index(env.agent_position) == 3
        @test pos_obs == [0.0, 0.0, 1.0, 0.0, 0.0] # One-hot for position 3
        @test reward_cue == [0.5, 0.5] # Uncertain at arms
        @test reward == 1.0 # Positive reward at left arm with :left setting

        # Attempt to move further left (should hit wall and stay in place)
        pos_obs, reward_cue, reward = step!(env, MazeAction(West()))
        @test env.agent_position == (1, 3) # Still at top left
        @test position_to_index(env.agent_position) == 3
        @test reward == 1.0 # Still get reward

        # Move east to top middle
        pos_obs, reward_cue, reward = step!(env, MazeAction(East()))
        @test env.agent_position == (2, 3) # Top middle
        @test position_to_index(env.agent_position) == 4
        @test pos_obs == [0.0, 0.0, 0.0, 1.0, 0.0] # One-hot for position 4
        @test reward === 0.0 # No reward at top middle
    end

    @testset "Starting at Middle Junction" begin
        # Create environment with agent starting at middle junction
        env = create_tmaze(:left, (2, 2))

        # Verify starting position
        @test env.agent_position == (2, 2) # Middle junction
        @test position_to_index(env.agent_position) == 2

        # Test reward cue at junction is uncertain
        pos_obs, reward_cue, reward = step!(env, MazeAction(East())) # Try any action to get observations
        @test reward_cue == [0.5, 0.5] # Should be uncertain at junction

        # Test navigation from middle junction - move north
        env = create_tmaze(:left, (2, 2)) # Reset
        pos_obs, reward_cue, reward = step!(env, MazeAction(North()))
        @test env.agent_position == (2, 3) # Top middle
        @test position_to_index(env.agent_position) == 4

        # Test navigation from middle junction - move east
        env = create_tmaze(:left, (2, 2)) # Reset
        pos_obs, reward_cue, reward = step!(env, MazeAction(East()))
        @test env.agent_position == (3, 3) # Top right
        @test position_to_index(env.agent_position) == 5
        @test reward == -1.0 # Negative reward with :left setting

        # Test navigation from middle junction - move west
        env = create_tmaze(:left, (2, 2)) # Reset
        pos_obs, reward_cue, reward = step!(env, MazeAction(West()))
        @test env.agent_position == (1, 3) # Top left
        @test position_to_index(env.agent_position) == 3
        @test reward == 1.0 # Positive reward with :left setting

        # Test navigation from middle junction - move south
        env = create_tmaze(:left, (2, 2)) # Reset
        pos_obs, reward_cue, reward = step!(env, MazeAction(South()))
        @test env.agent_position == (2, 1) # Bottom
        @test position_to_index(env.agent_position) == 1

        # Bottom position should reveal reward location
        pos_obs, reward_cue, reward = step!(env, MazeAction(South())) # Try any action to get observations
        @test reward_cue == [1.0, 0.0] # Should indicate left reward
    end

    @testset "Step Function - Reward Structure" begin
        # Test left reward configuration
        env_left = create_tmaze(:left)

        # Navigate to left arm
        step!(env_left, MazeAction(North())) # To junction
        pos_obs, reward_cue, reward = step!(env_left, MazeAction(West())) # To left arm
        @test reward == 1.0 # Positive reward at left with :left setting

        # Navigate to right arm
        step!(env_left, MazeAction(East())) # To middle
        step!(env_left, MazeAction(East())) # To right arm
        @test env_left.agent_position == (3, 3) # Top right
        @test position_to_index(env_left.agent_position) == 5
        pos_obs, reward_cue, reward = step!(env_left, MazeAction(North())) # Try move north (wall)
        @test reward == -1.0 # Negative reward at right with :left setting

        # Verify staying in the same position with a wall gives the reward
        pos_obs2, reward_cue2, reward2 = step!(env_left, MazeAction(East())) # Try hitting east wall (stays in place)
        @test reward2 == -1.0 # Should still get negative reward at right

        # Test right reward configuration
        env_right = create_tmaze(:right)

        # Navigate to left arm
        step!(env_right, MazeAction(North())) # To junction
        pos_obs, reward_cue, reward = step!(env_right, MazeAction(West())) # To left arm
        @test reward == -1.0 # Negative reward at left with :right setting

        # Navigate to right arm
        pos_obs, reward_cue, reward = step!(env_right, MazeAction(East())) # To middle
        @test env_right.agent_position == (2, 3) # At top middle
        @test reward === 0.0 # No reward in middle position
        pos_obs, reward_cue, reward = step!(env_right, MazeAction(East())) # To right arm
        @test env_right.agent_position == (3, 3) # At top right
        @test reward == 1.0 # Negative reward at right with :right setting
        pos_obs, reward_cue, reward = step!(env_right, MazeAction(North())) # Try a movement that hits wall (stays in place)
        @test reward == 1.0 # Positive reward at right with :right setting when staying in place
    end

    @testset "Reward Cue Information" begin
        env = create_tmaze(:left)

        # At bottom position, should reveal true reward location
        pos_obs, reward_cue, reward = step!(env, MazeAction(South())) # Try move south (wall, stays in place)
        @test env.agent_position == (2, 1) # Still at bottom
        @test reward_cue == [1.0, 0.0] # Indicates left reward

        # Reset with right reward
        reset_tmaze!(env, :right)
        pos_obs, reward_cue, reward = step!(env, MazeAction(South())) # Try move south (wall)
        @test reward_cue == [0.0, 1.0] # Indicates right reward

        # Move to junction, cue should be uncertain
        pos_obs, reward_cue, reward = step!(env, MazeAction(North()))
        @test env.agent_position == (2, 2) # At junction
        @test reward_cue == [0.5, 0.5] # Uncertain at junction
    end
end
