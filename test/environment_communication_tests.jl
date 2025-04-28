@testitem "EnvironmentCommunication" begin
    using Test
    using EFEasVFE
    using HTTP
    using JSON

    @testset "EnvironmentError" begin
        # Test error creation
        error = EnvironmentError("Test error", 404, "Not found")
        @test error.message == "Test error"
        @test error.status == 404
        @test error.response == "Not found"

        # Test error display
        io = IOBuffer()
        showerror(io, error)
        error_str = String(take!(io))
        @test contains(error_str, "EnvironmentError")
        @test contains(error_str, "Status: 404")
        @test contains(error_str, "Response: Not found")
    end

    @testset "Environment API" begin
        # Test create environment
        env_response = create_environment(5, render_mode="rgb_array", seed=UInt32(42))
        @test haskey(env_response, "observation")
        @test haskey(env_response, "session_id")
        session_id = env_response["session_id"]

        # Test step
        action = 0  # Turn left
        next_state = step_environment(action, session_id)
        @test haskey(next_state, "observation")
        @test haskey(next_state, "reward")
        @test haskey(next_state, "terminated")
        @test haskey(next_state, "info")

        # Test close
        close_result = close_environment(session_id)
        @test haskey(close_result, "success")
        @test close_result["success"] == true
    end

    @testset "Error Handling" begin
        # Create a test environment
        env_response = create_environment(5, render_mode="rgb_array", seed=UInt32(123))
        session_id = env_response["session_id"]

        # Test invalid action
        @test_throws EnvironmentError step_environment(999, session_id)

        # Clean up
        close_environment(session_id)
    end
end