@testitem "MinigridAgent" begin
    using Test
    using EFEasVFE
    @testset "MinigridConfig" begin
        import EFEasVFE: validate_config
        # Test valid config
        config = MinigridConfig(3, 10, 5, 3, 0.0, Float32)
        @test config.grid_size == 3
        @test config.time_horizon == 10
        @test config.n_episodes == 5
        @test config.n_iterations == 3
        @test config.wait_time == 0.0
        @test config.number_type == Float32

        # Test invalid configs
        @test_throws ArgumentError validate_config(MinigridConfig(0, 10, 5, 3, 0.0, Float32))  # grid_size <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 0, 5, 3, 0.0, Float32))   # time_horizon <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 10, 0, 3, 0.0, Float32))  # n_episodes <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 10, 5, 0, 0.0, Float32))  # n_iterations <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 10, 5, 3, -1.0, Float32)) # wait_time < 0
    end

    @testset "Cell Observation" begin
        import EFEasVFE: INVISIBLE, EMPTY, WALL, KEY, DOOR, create_cell_observation
        using TinyHugeNumbers
        # Test cell observation creation
        obs = create_cell_observation(0, Float32)  # INVISIBLE
        @test length(obs) == 5
        @test obs[Int(INVISIBLE)] == 1.0f0
        @test all(obs[i] == 0.0 for i in 1:5 if i != Int(INVISIBLE))

        # Test different number types
        obs_double = create_cell_observation(1, Float64)  # EMPTY
        @test length(obs_double) == 5
        @test obs_double[Int(EMPTY)] ≈ 1.0
        @test all(obs_double[i] == 0.0 for i in 1:5 if i != Int(EMPTY))
    end

    @testset "Action Conversion" begin
        import EFEasVFE: TURN_LEFT, TURN_RIGHT, FORWARD, PICKUP, OPEN_DOOR, convert_action
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
        import EFEasVFE: initialize_beliefs

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


    @testset "observation_tensor" begin
        import EFEasVFE: create_observation_tensor, INVISIBLE, EMPTY, WALL, DOOR, KEY

        @testset "Basic cell type encoding" begin
            for T in [Float32, Float64]
                # Create a simple 7x7 observation with one of each type
                basic_obs = Dict(
                    "image" => [[[1] for _ in 1:7] for _ in 1:7]  # All empty cells
                )
                # Set different cell types
                basic_obs["image"][1][1][1] = 0  # INVISIBLE
                basic_obs["image"][2][2][1] = 2  # WALL
                basic_obs["image"][3][3][1] = 4  # DOOR
                basic_obs["image"][4][4][1] = 5  # KEY

                result = create_observation_tensor(basic_obs, T)

                # Test dimensions
                @test size(result) == (7, 7)
                @test size(result[1, 1]) == (5,)

                # Test specific cell encodings
                @test result[1, 1][Int(INVISIBLE)] == 1.0  # INVISIBLE cell
                @test result[2, 2][Int(WALL)] == 1.0      # WALL cell
                @test result[3, 3][Int(DOOR)] == 1.0      # DOOR cell
                @test result[4, 4][Int(KEY)] == 1.0       # KEY cell
                @test result[5, 5][Int(EMPTY)] == 1.0     # EMPTY cell (default)

                # Test that other values in each vector are 0
                @test sum(result[1, 1]) ≈ 1.0
                @test sum(result[2, 2]) ≈ 1.0
                @test sum(result[3, 3]) ≈ 1.0
                @test sum(result[4, 4]) ≈ 1.0
            end
        end

        @testset "Invalid cell values" begin
            for T in [Float32, Float64]
                invalid_obs = Dict(
                    "image" => [[[1] for _ in 1:7] for _ in 1:7]
                )
                invalid_obs["image"][1][1][1] = 99  # Invalid value

                result = create_observation_tensor(invalid_obs, T)
                # Should default to EMPTY for invalid values
                @test result[1, 1][Int(EMPTY)] == 1.0
            end
        end

        @testset "Full grid consistency" begin
            for T in [Float32, Float64]
                # Test a full grid of each type
                cell_types = [0, 1, 2, 4, 5]
                for cell_type in cell_types
                    obs = Dict(
                        "image" => [[[cell_type] for _ in 1:7] for _ in 1:7]
                    )
                    result = create_observation_tensor(obs, T)

                    # Check every cell
                    for x in 1:7, y in 1:7
                        expected_idx = if cell_type == 0
                            Int(INVISIBLE)
                        elseif cell_type == 1
                            Int(EMPTY)
                        elseif cell_type == 2
                            Int(WALL)
                        elseif cell_type == 4
                            Int(DOOR)
                        elseif cell_type == 5
                            Int(KEY)
                        end

                        @test result[x, y][expected_idx] ≈ 1.0
                        @test sum(result[x, y]) ≈ 1.0
                    end
                end
            end
        end
    end
end