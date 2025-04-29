@testitem "MinigridAgent" begin
    using Test
    using EFEasVFE
    using VideoIO
    using FileIO
    using RxInfer
    using Colors  # Add Colors.jl for RGB{N0f8} support

    @testset "MinigridConfig" begin
        import EFEasVFE: validate_config
        # Test valid config
        config = MinigridConfig(3, 10, 5, 3, 0.0, Float32, false, 42, false, "test", false)
        @test config.grid_size == 3
        @test config.time_horizon == 10
        @test config.n_episodes == 5
        @test config.n_iterations == 3
        @test config.wait_time == 0.0
        @test config.number_type == Float32

        # Test invalid configs
        @test_throws ArgumentError validate_config(MinigridConfig(0, 10, 5, 3, 0.0, Float32, false, 42, false, "test", false))  # grid_size <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 0, 5, 3, 0.0, Float32, false, 42, false, "test", false))   # time_horizon <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 10, 0, 3, 0.0, Float32, false, 42, false, "test", false))  # n_episodes <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 10, 5, 0, 0.0, Float32, false, 42, false, "test", false))  # n_iterations <= 0
        @test_throws ArgumentError validate_config(MinigridConfig(3, 10, 5, 3, -1.0, Float32, false, 42, false, "test", false)) # wait_time < 0
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
        import EFEasVFE: initialize_beliefs_minigrid
        using TinyHugeNumbers

        grid_size = 3
        beliefs = initialize_beliefs_minigrid(grid_size, Float32)

        # Calculate valid positions
        valid_positions = grid_size^2 - 2 * grid_size

        # Test location belief
        @test length(probvec(beliefs.location)) == grid_size^2
        # The location belief should assign equal probabilities to valid positions and tiny to invalid ones
        expected_value = Float32(1 / valid_positions)
        @test count(x -> x ≈ expected_value, probvec(beliefs.location)) == valid_positions
        @test count(x -> x ≈ tiny(Float32), probvec(beliefs.location)) == 2 * grid_size

        # Test orientation belief
        @test length(probvec(beliefs.orientation)) == 4
        @test all(probvec(beliefs.orientation) .≈ Float32(1 / 4))

        # Test key and door location beliefs
        @test length(probvec(beliefs.key_location)) == valid_positions
        @test length(probvec(beliefs.door_location)) == valid_positions
        @test all(probvec(beliefs.key_location) .≈ Float32(1 / valid_positions))
        @test all(probvec(beliefs.door_location) .≈ Float32(1 / valid_positions))

        # Test key_door_state belief
        @test length(probvec(beliefs.key_door_state)) == 3
        @test probvec(beliefs.key_door_state)[1] ≈ 1.0f0 - 2 * tiny
        @test probvec(beliefs.key_door_state)[2] ≈ tiny
        @test probvec(beliefs.key_door_state)[3] ≈ tiny
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

    @testset "Frame Conversion" begin
        import EFEasVFE: convert_frame

        @testset "Basic frame conversion" begin
            # Test conversion of a simple 3x3 RGB frame
            frame_list = [[[1, 2, 3] for _ in 1:3] for _ in 1:3]
            result = convert_frame(frame_list)

            @test size(result) == (3, 3, 3)
            @test eltype(result) == UInt8

            # Verify the values are correctly placed
            for i in 1:3, j in 1:3
                @test result[i, j, :] == UInt8[1, 2, 3]
            end
        end

        @testset "Different frame sizes" begin
            # Test various frame sizes
            sizes = [(2, 2), (4, 4), (7, 7), (10, 10)]
            for (h, w) in sizes
                frame_list = [[[1, 2, 3] for _ in 1:w] for _ in 1:h]
                result = convert_frame(frame_list)

                @test size(result) == (h, w, 3)
                @test eltype(result) == UInt8
            end
        end

        @testset "Edge cases" begin
            # Test empty frame
            @test_throws ArgumentError convert_frame([])

            # Test frame with empty rows
            @test_throws ArgumentError convert_frame([[[]], [[]], [[]]])

            # Test frame with inconsistent row lengths
            @test_throws ArgumentError convert_frame([[[1, 2, 3]], [[1, 2, 3], [1, 2, 3]], [[1, 2, 3]]])
        end

        @testset "Value range" begin
            # Test conversion of values at boundaries
            frame_list = [[[0, 127, 255] for _ in 1:3] for _ in 1:3]
            result = convert_frame(frame_list)

            @test all(result[:, :, 1] .== UInt8(0))
            @test all(result[:, :, 2] .== UInt8(127))
            @test all(result[:, :, 3] .== UInt8(255))
        end
    end

    @testset "Video Recording" begin
        import EFEasVFE: record_episode_to_video

        # Create a temporary directory for test videos
        test_dir = mktempdir()

        @testset "Basic video recording" begin
            # Create a sequence of test frames
            frames = [rand(UInt8, 1080, 1920, 3) for _ in 1:5]
            video_path = joinpath(test_dir, "test_video.mp4")

            # Record the video
            record_episode_to_video(frames, video_path)

            # Verify the video file was created
            @test isfile(video_path)

            # Clean up
            rm(video_path)
        end

        @testset "Empty frames" begin
            # Test with empty frames vector
            video_path = joinpath(test_dir, "empty_video.mp4")
            record_episode_to_video(Array{UInt8,3}[], video_path)

            # Verify no file was created
            @test !isfile(video_path)
        end

        @testset "Different frame sizes" begin
            # Test recording frames of different sizes
            frames = [
                rand(UInt8, 3, 3, 3),
                rand(UInt8, 4, 4, 3),
                rand(UInt8, 5, 5, 3)
            ]
            video_path = joinpath(test_dir, "varying_sizes.mp4")

            # Should throw an error for inconsistent frame sizes
            @test_throws ArgumentError record_episode_to_video(frames, video_path)
        end

        @testset "Invalid frame data" begin
            # Test with invalid frame data
            frames = [rand(Float32, 1080, 1920, 3) for _ in 1:3]  # Wrong element type
            video_path = joinpath(test_dir, "invalid_frames.mp4")

            @test_throws MethodError record_episode_to_video(frames, video_path)
        end

        @testset "Video quality settings" begin
            # Test with different quality settings
            frames = [rand(UInt8, 1080, 1920, 3) for _ in 1:3]
            video_path = joinpath(test_dir, "high_quality.mp4")

            # Record with high quality settings
            record_episode_to_video(frames, video_path)

            # Verify the video file was created
            @test isfile(video_path)

            # Clean up
            rm(video_path)
        end

        # Clean up test directory
        rm(test_dir, recursive=true)
    end
end