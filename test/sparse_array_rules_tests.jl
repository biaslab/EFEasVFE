@testitem "SparseArray rules match full tensor rules" begin
    import EFEasVFE: get_self_transition_tensor, SparseArray, probvec
    using Test, ReTestItems
    using EFEasVFE
    using ReactiveMP: Categorical
    using Random
    using RxInfer
    # Helper function to ensure messages are properly formatted
    function create_random_categorical(dim, rng)
        probs = rand(rng, dim)
        return Categorical(probs ./ sum(probs))
    end

    function create_one_hot_vector(dim, idx)
        v = zeros(dim)
        v[idx] = 1.0
        return v
    end

    # Setup test parameters
    rng = Random.MersenneTwister(123)
    n = 4  # Small grid size for faster tests
    dims = (n * n, n * n, 4, n * n - 2 * n, n * n - 2 * n, 3, 5)  # Dimensions of transition tensor

    # Generate 6D and 7D tensors for testing
    tensor7d = get_self_transition_tensor(n, Float64)
    # Create a 6D tensor by selecting a specific value in the first dimension
    tensor6d = tensor7d[1, :, :, :, :, :, :]

    # Create sparse representations
    sparse_tensor7d = SparseArray(tensor7d)
    sparse_tensor6d = SparseArray(tensor6d)

    # Create PointMass distributions for tensors
    q_a7d_sparse = PointMass(sparse_tensor7d)
    q_a7d_full = PointMass(tensor7d)
    q_a6d_sparse = PointMass(sparse_tensor6d)
    q_a6d_full = PointMass(tensor6d)

    # Test case 1: 7D tensor with all categorical messages
    @testset "7D tensor - all categoricals" begin
        for i in 1:10
            # Create random categorical messages
            m_out = create_random_categorical(dims[1], rng)
            m_in = create_random_categorical(dims[2], rng)
            m_T1 = create_random_categorical(dims[3], rng)
            m_T2 = create_random_categorical(dims[4], rng)
            m_T3 = create_random_categorical(dims[5], rng)
            m_T4 = create_random_categorical(dims[6], rng)
            m_T5 = create_random_categorical(dims[7], rng)

            # Test for each output interface
            # out interface
            sparse_result = @call_rule DiscreteTransition(:out, Marginalisation) (
                m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
            )

            full_result = @call_rule DiscreteTransition(:out, Marginalisation) (
                m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # in interface
            sparse_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                m_out=m_out, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
            )

            full_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                m_out=m_out, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T1 interface
            sparse_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T2 interface
            sparse_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1,
                m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T3 interface
            sparse_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T4 interface
            sparse_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T5=m_T5, q_a=q_a7d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T5=m_T5, q_a=q_a7d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T5 interface
            sparse_result = @call_rule DiscreteTransition(:T5, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T5, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12
        end
    end

    # Test case 2: 7D tensor with one PointMass message (q_T5)
    @testset "7D tensor - with PointMass q_T5" begin
        for i in 1:10
            # Create random categorical messages
            m_out = create_random_categorical(dims[1], rng)
            m_in = create_random_categorical(dims[2], rng)
            m_T1 = create_random_categorical(dims[3], rng)
            m_T2 = create_random_categorical(dims[4], rng)
            m_T3 = create_random_categorical(dims[5], rng)
            m_T4 = create_random_categorical(dims[6], rng)

            # Create point mass for q_T5
            q_T5_idx = rand(rng, 1:dims[7])
            q_T5 = PointMass(create_one_hot_vector(dims[7], q_T5_idx))

            # out interface
            sparse_result = @call_rule DiscreteTransition(:out, Marginalisation) (
                m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_sparse, q_T5=q_T5
            )

            full_result = @call_rule DiscreteTransition(:out, Marginalisation) (
                m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_full, q_T5=q_T5
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # in interface
            sparse_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                m_out=m_out, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_sparse, q_T5=q_T5
            )

            full_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                m_out=m_out, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_full, q_T5=q_T5
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T1 interface
            sparse_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_sparse, q_T5=q_T5
            )

            full_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_full, q_T5=q_T5
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T2 interface
            sparse_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_sparse, q_T5=q_T5
            )

            full_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_full, q_T5=q_T5
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T3 interface
            sparse_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T4=m_T4, q_a=q_a7d_sparse, q_T5=q_T5
            )

            full_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T4=m_T4, q_a=q_a7d_full, q_T5=q_T5
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12

            # T4 interface
            sparse_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, q_a=q_a7d_sparse, q_T5=q_T5
            )

            full_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, q_a=q_a7d_full, q_T5=q_T5
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12
        end
    end

    # Test case 3: 6D tensor with PointMass q_out
    @testset "6D tensor - with PointMass q_out" begin
        for i in 1:100
            # Create random categorical messages
            m_in = create_random_categorical(dims[3], rng)
            m_T1 = create_random_categorical(dims[4], rng)
            m_T2 = create_random_categorical(dims[5], rng)
            m_T3 = create_random_categorical(dims[6], rng)
            m_T4 = create_random_categorical(dims[7], rng)

            # Create point mass for q_out
            q_out_idx = rand(rng, 1:dims[1])
            q_out = PointMass(create_one_hot_vector(dims[1], q_out_idx))

            # in interface
            sparse_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                q_out=q_out, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a6d_sparse
            )

            full_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                q_out=q_out, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a6d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-7

            # T1 interface
            sparse_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a6d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T2=m_T2,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a6d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-7

            # T2 interface
            sparse_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T1=m_T1,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a6d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T1=m_T1,
                m_T3=m_T3, m_T4=m_T4, q_a=q_a6d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-7

            # T3 interface
            sparse_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T4=m_T4, q_a=q_a6d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T4=m_T4, q_a=q_a6d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-7

            # T4 interface
            sparse_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, q_a=q_a6d_sparse
            )

            full_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                q_out=q_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                m_T3=m_T3, q_a=q_a6d_full
            )

            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-5
        end
    end

    # Test case 4: Multiple random configurations
    @testset "Multiple random configurations" begin
        for test_iter in 1:100  # Run 5 random tests
            # Create random categorical messages for 7D tensor
            m_out = create_random_categorical(dims[1], rng)
            m_in = create_random_categorical(dims[2], rng)
            m_T1 = create_random_categorical(dims[3], rng)
            m_T2 = create_random_categorical(dims[4], rng)
            m_T3 = create_random_categorical(dims[5], rng)
            m_T4 = create_random_categorical(dims[6], rng)
            m_T5 = create_random_categorical(dims[7], rng)

            # Pick a random interface to test
            interfaces = [:out, :in, :T1, :T2, :T3, :T4, :T5]
            interface_idx = rand(rng, 1:length(interfaces))
            interface = interfaces[interface_idx]

            # Test the selected interface
            if interface == :out
                sparse_result = @call_rule DiscreteTransition(:out, Marginalisation) (
                    m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
                )

                full_result = @call_rule DiscreteTransition(:out, Marginalisation) (
                    m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
                )
            elseif interface == :in
                sparse_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                    m_out=m_out, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
                )

                full_result = @call_rule DiscreteTransition(:in, Marginalisation) (
                    m_out=m_out, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
                )
            elseif interface == :T1
                sparse_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
                )

                full_result = @call_rule DiscreteTransition(:T1, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
                )
            elseif interface == :T2
                sparse_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
                )

                full_result = @call_rule DiscreteTransition(:T2, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1,
                    m_T3=m_T3, m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
                )
            elseif interface == :T3
                sparse_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_sparse
                )

                full_result = @call_rule DiscreteTransition(:T3, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T4=m_T4, m_T5=m_T5, q_a=q_a7d_full
                )
            elseif interface == :T4
                sparse_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T5=m_T5, q_a=q_a7d_sparse
                )

                full_result = @call_rule DiscreteTransition(:T4, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T5=m_T5, q_a=q_a7d_full
                )
            elseif interface == :T5
                sparse_result = @call_rule DiscreteTransition(:T5, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_sparse
                )

                full_result = @call_rule DiscreteTransition(:T5, Marginalisation) (
                    m_out=m_out, m_in=m_in, m_T1=m_T1, m_T2=m_T2,
                    m_T3=m_T3, m_T4=m_T4, q_a=q_a7d_full
                )
            end

            # Compare results
            @test sparse_result isa Categorical
            @test full_result isa Categorical
            @test probvec(sparse_result) ≈ probvec(full_result) atol = 1e-12
        end
    end
end