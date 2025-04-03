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

    @testset "check_response" begin
        # Test successful response
        response = HTTP.Response(200, "{\"test\": \"success\"}")
        result = check_response(response)
        @test result == Dict("test" => "success")

        # Test error response
        response = HTTP.Response(404, "Not found")
        @test_throws EnvironmentError check_response(response)
    end

    @testset "Environment API" begin
        # Test action space
        action_space = get_action_space()
        @test haskey(action_space, "n")
        @test haskey(action_space, "actions")
        @test action_space["n"] == 7  # Expected number of actions

        # Test observation space
        obs_space = get_observation_space()
        @test haskey(obs_space, "shape")
        @test haskey(obs_space, "dtype")

        # Test reset
        state = reset_environment()
        @test haskey(state, "observation")
        @test haskey(state, "reward")
        @test haskey(state, "done")
        @test haskey(state, "info")

        # Test step
        action = 0  # Turn left
        next_state = step_environment(action)
        @test haskey(next_state, "observation")
        @test haskey(next_state, "reward")
        @test haskey(next_state, "done")
        @test haskey(next_state, "info")

        # Test render
        render_info = render_environment()
        @test haskey(render_info, "success")

        # Test close
        close_result = close_environment()
        @test haskey(close_result, "success")
    end

    @testset "Error Handling" begin
        # Test invalid action
        @test_throws EnvironmentError step_environment(999)

        # Test server not running
        # Note: This test might need to be adjusted based on how the server is managed
        # original_url = API_URL
        # global API_URL = "http://localhost:9999"  # Non-existent port
        # @test_throws EnvironmentError reset_environment()
        # global API_URL = original_url
    end
end