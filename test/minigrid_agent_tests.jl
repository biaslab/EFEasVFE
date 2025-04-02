@testitem "MinigridAgent" begin
    using Test
    using EFEasVFE
    @testset "MinigridConfig" begin
        # Test valid config
        config = MinigridConfig(3, 10, 5, 3, 0.0, Float32)
        @test config.grid_size == 3
        @test config.time_horizon == 10
        @test config.n_episodes == 5
        @test config.n_iterations == 3
        @test config.wait_time == 0.0
        @test config.number_type == Float32

        # Test invalid configs
        @test_throws ArgumentError MinigridConfig(0, 10, 5, 3, 0.0, Float32)  # grid_size <= 0
        @test_throws ArgumentError MinigridConfig(3, 0, 5, 3, 0.0, Float32)   # time_horizon <= 0
        @test_throws ArgumentError MinigridConfig(3, 10, 0, 3, 0.0, Float32)  # n_episodes <= 0
        @test_throws ArgumentError MinigridConfig(3, 10, 5, 0, 0.0, Float32)  # n_iterations <= 0
        @test_throws ArgumentError MinigridConfig(3, 10, 5, 3, -1.0, Float32) # wait_time < 0
    end

    @testset "Cell Observation" begin
        # Test cell observation creation
        obs = create_cell_observation(0, Float32)  # INVISIBLE
        @test length(obs) == 5
        @test obs[Int(INVISIBLE)] ≈ 1.0f0
        @test all(obs[i] ≈ tiny for i in 1:5 if i != Int(INVISIBLE))

        # Test different number types
        obs_double = create_cell_observation(1, Float64)  # EMPTY
        @test length(obs_double) == 5
        @test obs_double[Int(EMPTY)] ≈ 1.0
        @test all(obs_double[i] ≈ tiny for i in 1:5 if i != Int(EMPTY))
    end

    @testset "Action Conversion" begin
        # Test valid actions
        @test convert_action(Int(TURN_LEFT)) == 0
        @test convert_action(Int(TURN_RIGHT)) == 1
        @test convert_action(Int(FORWARD)) == 2
        @test convert_action(Int(PICKUP)) == 3
        @test convert_action(Int(OPEN_DOOR)) == 5

        # Test invalid action
        @test_throws ErrorException convert_action(99)
    end

    @testset "Belief Initialization" begin
        grid_size = 3
        beliefs = initialize_beliefs(grid_size, Float32)

        # Test location belief
        @test length(beliefs.location.p) == grid_size^2
        @test all(beliefs.location.p .≈ Float32(1 / grid_size^2))

        # Test orientation belief
        @test length(beliefs.orientation.p) == 4
        @test all(beliefs.orientation.p .≈ Float32(1 / 4))

        # Test key and door location beliefs
        valid_positions = grid_size^2 - 2 * grid_size
        @test length(beliefs.key_location.p) == valid_positions
        @test length(beliefs.door_location.p) == valid_positions
        @test all(beliefs.key_location.p .≈ Float32(1 / valid_positions))
        @test all(beliefs.door_location.p .≈ Float32(1 / valid_positions))

        # Test key_door_state belief
        @test length(beliefs.key_door_state.p) == 3
        @test beliefs.key_door_state.p[1] ≈ 1.0f0 - 2 * tiny
        @test beliefs.key_door_state.p[2] ≈ tiny
        @test beliefs.key_door_state.p[3] ≈ tiny
    end

    @testset "Observation Tensor" begin
        # Create a mock observation
        mock_obs = Dict(
            "image" => fill(zeros(3), 7, 7),
            "direction" => 0
        )
        mock_obs["image"][4, 4] = [1, 0, 0]  # EMPTY cell
        mock_obs["image"][3, 3] = [2, 0, 0]  # WALL cell
        mock_obs["image"][5, 5] = [5, 0, 0]  # KEY cell

        # Create observation tensor
        obs_tensor = create_observation_tensor(mock_obs, Float32)

        # Test tensor dimensions
        @test size(obs_tensor) == (7, 7)
        @test all(size(obs_tensor[i, j]) == (5,) for i in 1:7, j in 1:7)

        # Test specific cell values
        @test obs_tensor[4, 4][Int(EMPTY)] ≈ 1.0f0
        @test obs_tensor[3, 3][Int(WALL)] ≈ 1.0f0
        @test obs_tensor[5, 5][Int(KEY)] ≈ 1.0f0
    end
end