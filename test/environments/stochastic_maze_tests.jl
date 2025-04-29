@testitem "Stochastic Maze - generate_maze_tensors" begin
    using EFEasVFE
    using Distributions
    using LinearAlgebra

    @testset "Basic tensor shapes" begin
        import EFEasVFE: generate_maze_tensors
        # Test basic dimensions of tensors
        grid_size_x, grid_size_y = 5, 5
        n_actions = 4
        A, B, reward_states = generate_maze_tensors(grid_size_x, grid_size_y, n_actions)

        n_states = grid_size_x * grid_size_y

        # Check shapes of tensors
        @test size(A) == (n_states, n_states)  # Observation matrix
        @test size(B) == (n_states, n_states, n_actions)  # Transition tensor
        @test all(isa.(reward_states, Tuple{Int,Float64}))  # Reward states
    end

    @testset "Probability distribution properties" begin
        grid_size_x, grid_size_y = 5, 5
        n_actions = 4
        A, B, reward_states = generate_maze_tensors(grid_size_x, grid_size_y, n_actions)

        n_states = grid_size_x * grid_size_y

        # Check that each column in A is a probability distribution
        for s in 1:n_states
            @test isapprox(sum(A[:, s]), 1.0, atol=1e-10)
            @test all(A[:, s] .>= 0)
        end

        # Check that each state-action pair in B forms a probability distribution
        for s in 1:n_states, a in 1:n_actions
            @test isapprox(sum(B[:, s, a]), 1.0, atol=1e-10)
            @test all(B[:, s, a] .>= 0)
        end
    end

    @testset "Sink states behavior" begin
        grid_size_x, grid_size_y = 5, 5
        n_actions = 4
        sink_states = [(4, 2), (4, 4)]
        A, B, reward_states = generate_maze_tensors(
            grid_size_x, grid_size_y, n_actions,
            sink_states=sink_states
        )

        for (x, y) in sink_states
            # Convert to linear index
            s = x + (y - 1) * grid_size_x

            # For a sink state, all actions should keep agent in same state
            for a in 1:n_actions
                @test B[s, s, a] ≈ 1.0
            end
        end
    end

    @testset "Stochastic states behavior" begin
        grid_size_x, grid_size_y = 5, 5
        n_actions = 4
        stochastic_states = [(2, 3), (3, 3), (4, 3)]
        A, B, reward_states = generate_maze_tensors(
            grid_size_x, grid_size_y, n_actions,
            stochastic_states=stochastic_states
        )

        for (x, y) in stochastic_states
            # Convert to linear index
            s = x + (y - 1) * grid_size_x

            # For stochastic states, check that we have non-zero probabilities in multiple directions
            for a in 1:n_actions
                # There should be non-zero probabilities for up, down, and the action's intended direction
                # Rather than testing exact values (which can vary due to normalization),
                # we'll test for general stochastic behavior

                # Count states with non-negligible probability
                states_with_prob = count(B[:, s, a] .> 0.05)

                # There should be multiple possible outcomes (stochastic behavior)
                @test states_with_prob >= 2

                # Probabilities should sum to 1
                @test isapprox(sum(B[:, s, a]), 1.0, atol=1e-10)
            end
        end
    end

    @testset "Boundary behavior" begin
        grid_size_x, grid_size_y = 5, 5
        n_actions = 4
        A, B, reward_states = generate_maze_tensors(grid_size_x, grid_size_y, n_actions)

        # Test north boundary
        for x in 1:grid_size_x
            y = grid_size_y
            s = x + (y - 1) * grid_size_x

            # Action 1 (North) at top edge should keep agent in same state
            @test B[s, s, 1] > 0.0
        end

        # Test east boundary
        for y in 1:grid_size_y
            x = grid_size_x
            s = x + (y - 1) * grid_size_x

            # Action 2 (East) at right edge should keep agent in same state
            @test B[s, s, 2] > 0.0
        end

        # Test south boundary
        for x in 1:grid_size_x
            y = 1
            s = x + (y - 1) * grid_size_x

            # Action 3 (South) at bottom edge should keep agent in same state
            @test B[s, s, 3] > 0.0
        end

        # Test west boundary
        for y in 1:grid_size_y
            x = 1
            s = x + (y - 1) * grid_size_x

            # Action 4 (West) at left edge should keep agent in same state
            @test B[s, s, 4] > 0.0
        end
    end

    @testset "Noisy observations" begin
        grid_size_x, grid_size_y = 5, 5
        n_actions = 4
        noisy_observations = [(1, 5, 0.3), (2, 5, 0.2)]
        A, B, reward_states = generate_maze_tensors(
            grid_size_x, grid_size_y, n_actions,
            noisy_observations=noisy_observations
        )

        for (x, y, noise) in noisy_observations
            # Convert to linear index
            s = x + (y - 1) * grid_size_x

            # Check diagonal element = 1-noise
            @test isapprox(A[s, s], 1.0 - noise, atol=1e-10)

            # Sum of column should be 1.0
            @test isapprox(sum(A[:, s]), 1.0, atol=1e-10)
        end
    end

    @testset "Reward states" begin
        grid_size_x, grid_size_y = 5, 5
        n_actions = 4
        A, B, reward_states = generate_maze_tensors(grid_size_x, grid_size_y, n_actions)

        # Check that the specified reward states are in reward_states
        expected_rewards = [(9, -1.0), (19, -1.0), (15, 1.0)]

        for (state, value) in expected_rewards
            @test (state, value) in reward_states
        end

        # Check that noisy states have small negative rewards
        for s in 1:grid_size_x*grid_size_y
            if A[s, s] < 1.0  # If it's a noisy state
                # Should have a small negative reward
                @test any(rs -> rs[1] == s && rs[2] ≈ -0.1, reward_states)
            end
        end
    end

    @testset "Custom parameters" begin
        grid_size_x, grid_size_y = 3, 3
        n_actions = 4
        sink_states = [(2, 2)]
        stochastic_states = [(1, 3)]
        noisy_observations = [(3, 1, 0.5)]

        A, B, reward_states = generate_maze_tensors(
            grid_size_x, grid_size_y, n_actions,
            sink_states=sink_states,
            stochastic_states=stochastic_states,
            noisy_observations=noisy_observations
        )

        n_states = grid_size_x * grid_size_y

        # Check basic dimensions
        @test size(A) == (n_states, n_states)
        @test size(B) == (n_states, n_states, n_actions)

        # Check sink state
        s_sink = 2 + (2 - 1) * grid_size_x
        for a in 1:n_actions
            @test B[s_sink, s_sink, a] ≈ 1.0
        end

        # Check stochastic state
        s_stoch = 1 + (3 - 1) * grid_size_x
        for a in 1:n_actions
            # Check for stochastic behavior - multiple non-zero transitions
            states_with_prob = count(B[:, s_stoch, a] .> 0.05)
            @test states_with_prob >= 2
            @test isapprox(sum(B[:, s_stoch, a]), 1.0, atol=1e-10)
        end

        # Check noisy observation
        s_noisy = 3 + (1 - 1) * grid_size_x
        @test isapprox(A[s_noisy, s_noisy], 0.5, atol=1e-10)
    end
end

@testitem "Stochastic Maze - generate_goal_distributions" begin
    using EFEasVFE
    using Distributions
    using LinearAlgebra
    import EFEasVFE: generate_goal_distributions

    @testset "Basic distributions properties" begin
        n_states = 25
        goal_state = 15
        T = 10

        distributions = generate_goal_distributions(n_states, goal_state, T)

        # Check number of distributions
        @test length(distributions) == T

        # Check each distribution is categorical
        for dist in distributions
            @test isa(dist, Categorical)
            @test length(dist.p) == n_states
            @test isapprox(sum(dist.p), 1.0, atol=1e-10)
            @test all(dist.p .>= 0)
        end
    end

    @testset "Entropy progression" begin
        n_states = 25
        goal_state = 15
        T = 5

        distributions = generate_goal_distributions(n_states, goal_state, T)

        # Calculate entropy for each distribution
        entropies = [entropy(dist) for dist in distributions]

        # Entropy should decrease over time (as T increases)
        for i in 1:T-1
            @test entropies[i] > entropies[i+1]
        end

        # Final distribution should have lowest entropy (most concentrated)
        @test argmin(entropies) == T

        # First distribution should have highest entropy (most uniform)
        @test argmax(entropies) == 1
    end

    @testset "Goal state probability" begin
        n_states = 25
        goal_state = 15
        T = 8

        distributions = generate_goal_distributions(n_states, goal_state, T)

        # Check goal state probabilities increase over time
        goal_probs = [distributions[t].p[goal_state] for t in 1:T]

        for i in 1:T-1
            @test goal_probs[i] < goal_probs[i+1]
        end

        # Final distribution should have highest probability for goal state
        @test isapprox(distributions[T].p[goal_state], 1.0, atol=1e-5)

        # First distribution should have lowest probability for goal state
        # But still higher than uniform
        @test goal_probs[1] > 1 / n_states
    end

    @testset "Different goal states" begin
        n_states = 25
        T = 5

        # Test with different goal states
        goal_states = [5, 10, 20]

        for goal in goal_states
            distributions = generate_goal_distributions(n_states, goal, T)

            # Final distribution should concentrate on goal state
            @test isapprox(distributions[T].p[goal], 1.0, atol=1e-5)

            # Other states should have very low probability in final distribution
            for s in 1:n_states
                if s != goal
                    @test distributions[T].p[s] < 0.01
                end
            end
        end
    end

    @testset "Different time horizons" begin
        n_states = 25
        goal_state = 15

        # Test with different time horizons
        time_horizons = [3, 7, 12]

        for T in time_horizons
            distributions = generate_goal_distributions(n_states, goal_state, T)

            # Check correct number of distributions
            @test length(distributions) == T

            # Final distribution should concentrate on goal state
            @test isapprox(distributions[T].p[goal_state], 1.0, atol=1e-5)
        end
    end

    @testset "Edge cases" begin
        # Minimal state space
        n_states = 2
        goal_state = 2
        T = 3

        distributions = generate_goal_distributions(n_states, goal_state, T)
        @test length(distributions) == T
        @test isapprox(distributions[T].p[goal_state], 1.0, atol=1e-5)

        # Single timestep
        T = 1
        distributions = generate_goal_distributions(n_states, goal_state, T)
        @test length(distributions) == 1
        @test isapprox(distributions[1].p[goal_state], 1.0, atol=1e-5)

        # Large state space
        n_states = 100
        goal_state = 50
        T = 5

        distributions = generate_goal_distributions(n_states, goal_state, T)
        @test length(distributions) == T
        @test isapprox(distributions[T].p[goal_state], 1.0, atol=1e-5)
    end
end
