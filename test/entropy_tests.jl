@testitem "Conditional entropy" begin
    using EFEasVFE
    using RxInfer
    @testset "Conditional entropy, basics" begin
        import EFEasVFE: conditional_entropy
        d = Contingency(rand(5, 5, 5))
        mat = components(d)
        @test size(conditional_entropy(mat, 1, 2)) == (5,)
    end

    @testset "3D distribution - independent case" begin
        import EFEasVFE: conditional_entropy

        # Create a 3D joint distribution where all variables are independent
        # p(x,y,z) = p(x)p(y)p(z)
        nx, ny, nz = 3, 4, 5
        px = fill(1.0 / nx, nx)
        py = fill(1.0 / ny, ny)
        pz = fill(1.0 / nz, nz)

        # Create joint distribution
        joint_3d = zeros(nx, ny, nz)
        for x in 1:nx, y in 1:ny, z in 1:nz
            joint_3d[x, y, z] = px[x] * py[y] * pz[z]
        end

        # For independent variables, H(Y|X,Z=z) = H(Y) for all z
        h_y = -sum(py .* log.(py))
        expected = fill(h_y, nz)

        # Test H(Y|X,Z=z)
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)

        @test size(h_y_given_xz) == (nz,)
        @test all(isapprox.(h_y_given_xz, expected, atol=1e-6))
    end

    @testset "3D distribution - Z determines Y" begin
        import EFEasVFE: conditional_entropy

        # Create a 3D joint distribution where Z completely determines Y
        # regardless of X
        nx, ny, nz = 3, 4, 4  # ny = nz since Z determines Y

        joint_3d = zeros(nx, ny, nz)

        # For each z, y is deterministically related to z
        for x in 1:nx, z in 1:nz
            # When z=i, y=i with probability 1
            y = z
            if y <= ny  # ensure y is within bounds
                joint_3d[x, y, z] = 1.0 / (nx * nz)
            end
        end

        # For deterministic Y given Z, H(Y|X,Z=z) = 0 for all z
        expected = zeros(nz)

        # Test H(Y|X,Z=z)
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)

        @test size(h_y_given_xz) == (nz,)
        @test all(isapprox.(h_y_given_xz, expected, atol=1e-6))
    end

    @testset "3D distribution - Y depends on X for each Z" begin
        import EFEasVFE: conditional_entropy

        # Create a 3D joint distribution where Y depends on X differently for each Z
        nx, ny, nz = 3, 3, 4

        joint_3d = zeros(nx, ny, nz)

        # Fill the distribution
        for z in 1:nz
            for x in 1:nx
                # For each z, create a different dependency pattern between X and Y
                for y in 1:ny
                    if z == 1
                        # When z=1, Y is strongly dependent on X (Y=X with high probability)
                        joint_3d[x, y, z] = x == y ? 0.8 / (nx) : 0.2 / (nx * (ny - 1))
                    elseif z == 2
                        # When z=2, Y is moderately dependent on X
                        joint_3d[x, y, z] = x == y ? 0.6 / (nx) : 0.4 / (nx * (ny - 1))
                    elseif z == 3
                        # When z=3, Y is weakly dependent on X
                        joint_3d[x, y, z] = x == y ? 0.4 / (nx) : 0.6 / (nx * (ny - 1))
                    else
                        # When z=4, Y is independent of X
                        joint_3d[x, y, z] = 1.0 / (nx * ny)
                    end
                end
            end
        end

        # Normalize to ensure it's a valid probability distribution
        joint_3d ./= sum(joint_3d)

        # Calculate the expected conditional entropy for each z
        expected = zeros(nz)
        for z in 1:nz
            # Extract the joint distribution p(x,y|z)
            joint_xy_given_z = joint_3d[:, :, z]
            joint_xy_given_z ./= sum(joint_xy_given_z)

            # Calculate p(x|z)
            p_x_given_z = dropdims(sum(joint_xy_given_z, dims=2), dims=2)

            # Calculate entropy for each p(y|x,z)
            h_y_given_x_z = zeros(nx)
            for x in 1:nx
                if p_x_given_z[x] > 0
                    p_y_given_xz = joint_xy_given_z[x, :] ./ p_x_given_z[x]
                    h_y_given_x_z[x] = -sum(p_y_given_xz .* log.(p_y_given_xz))
                end
            end

            # Calculate H(Y|X,Z=z) = sum_x p(x|z) * H(Y|X=x,Z=z)
            expected[z] = sum(p_x_given_z .* h_y_given_x_z)
        end

        # Test H(Y|X,Z=z)
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)

        @test size(h_y_given_xz) == (nz,)
        @test all(isapprox.(h_y_given_xz, expected, atol=1e-6))
    end

    @testset "3D distribution - testing different dimension orders" begin
        import EFEasVFE: conditional_entropy

        # Create a 3D joint distribution
        nx, ny, nz = 3, 4, 5
        joint_3d = rand(nx, ny, nz)
        joint_3d ./= sum(joint_3d)

        # Test different combinations of out_dim and in_dim
        # H(X|Y,Z=z)
        h_x_given_yz = conditional_entropy(joint_3d, 1, 3)
        @test size(h_x_given_yz) == (nz,)

        # H(Y|X,Z=z)
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)
        @test size(h_y_given_xz) == (nz,)

        # H(Z|X,Y=y)
        h_z_given_xy = conditional_entropy(joint_3d, 3, 2)
        @test size(h_z_given_xy) == (ny,)

        # Manually calculate expected values for one case
        expected_h_x_given_yz = zeros(nz)
        for z in 1:nz
            # Extract joint distribution p(x,y|z)
            joint_xy_given_z = joint_3d[:, :, z]
            joint_xy_given_z ./= sum(joint_xy_given_z)

            # Calculate p(y|z)
            p_y_given_z = dropdims(sum(joint_xy_given_z, dims=1), dims=1)

            # Calculate entropy for each p(x|y,z)
            h_x_given_y_z = zeros(ny)
            for y in 1:ny
                if p_y_given_z[y] > 0
                    p_x_given_yz = joint_xy_given_z[:, y] ./ p_y_given_z[y]
                    h_x_given_y_z[y] = -sum(p_x_given_yz .* log.(max.(p_x_given_yz, 1e-10)))
                end
            end

            # Calculate H(X|Y,Z=z) = sum_y p(y|z) * H(X|Y=y,Z=z)
            expected_h_x_given_yz[z] = sum(p_y_given_z .* h_x_given_y_z)
        end

        @test all(isapprox.(h_x_given_yz, expected_h_x_given_yz, atol=1e-6))
    end

    @testset "Higher dimensional distributions (4D)" begin
        import EFEasVFE: conditional_entropy

        # Create a 4D joint distribution
        nx, ny, nz, nw = 2, 3, 4, 2
        joint_4d = rand(nx, ny, nz, nw)
        joint_4d ./= sum(joint_4d)

        # Test H(Y|X,Z,W=w)
        h_y_given_xzw = conditional_entropy(joint_4d, 2, 4)
        @test size(h_y_given_xzw) == (nw,)

        # Test H(X|Y,Z,W=w)
        h_x_given_yzw = conditional_entropy(joint_4d, 1, 4)
        @test size(h_x_given_yzw) == (nw,)

        # Manually calculate expected value for one case
        expected_h_y_given_xzw = zeros(nw)
        for w in 1:nw
            # Extract joint distribution p(x,y,z|w)
            joint_xyz_given_w = joint_4d[:, :, :, w]
            joint_xyz_given_w ./= sum(joint_xyz_given_w)

            # Calculate p(x,z|w)
            p_xz_given_w = dropdims(sum(joint_xyz_given_w, dims=2), dims=2)

            # Calculate entropy for each p(y|x,z,w)
            h_y_given_xz_w = zeros(nx, nz)
            for x in 1:nx, z in 1:nz
                if p_xz_given_w[x, z] > 0
                    p_y_given_xzw = joint_xyz_given_w[x, :, z] ./ p_xz_given_w[x, z]
                    h_y_given_xz_w[x, z] = -sum(p_y_given_xzw .* log.(max.(p_y_given_xzw, 1e-10)))
                end
            end

            # Calculate H(Y|X,Z,W=w) = sum_{x,z} p(x,z|w) * H(Y|X=x,Z=z,W=w)
            expected_h_y_given_xzw[w] = sum(p_xz_given_w .* h_y_given_xz_w)
        end

        @test all(isapprox.(h_y_given_xzw, expected_h_y_given_xzw, atol=1e-6))
    end

    @testset "Performance with large arrays" begin
        import EFEasVFE: conditional_entropy

        # Create a large 3D joint distribution
        nx, ny, nz = 10, 15, 20
        joint_3d = rand(nx, ny, nz)
        joint_3d ./= sum(joint_3d)

        # Measure time for calculating H(Y|X,Z=z)
        t_start = time()
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)
        t_end = time()

        @test size(h_y_given_xz) == (nz,)
        @test t_end - t_start < 1.0  # This threshold may need adjustment
    end

    @testset "Edge cases - uniform distribution" begin
        import EFEasVFE: conditional_entropy

        # Create a uniform 3D joint distribution
        nx, ny, nz = 3, 4, 5
        joint_3d = ones(nx, ny, nz) / (nx * ny * nz)

        # For uniform distribution, H(Y|X,Z=z) = log(ny) for all z
        expected = fill(log(ny), nz)

        # Test H(Y|X,Z=z)
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)

        @test size(h_y_given_xz) == (nz,)
        @test all(isapprox.(h_y_given_xz, expected, atol=1e-6))
    end

    @testset "Edge cases - degenerate distribution" begin
        import EFEasVFE: conditional_entropy

        # Create a degenerate 3D joint distribution where only one entry is non-zero
        nx, ny, nz = 3, 4, 5
        joint_3d = zeros(nx, ny, nz)
        joint_3d[1, 1, 1] = 1.0

        # For this distribution, H(Y|X,Z=z) should be 0 for z=1, and undefined (NaN) for z>1
        # We'll just test z=1 since the others have zero probability
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)

        @test size(h_y_given_xz) == (nz,)
        @test isapprox(h_y_given_xz[1], 0.0, atol=1e-6)
    end

    @testset "Numerical stability - very small probabilities" begin
        import EFEasVFE: conditional_entropy

        # Create a 3D joint distribution with some very small probabilities
        nx, ny, nz = 3, 4, 5
        joint_3d = rand(nx, ny, nz) * 1e-10
        joint_3d ./= sum(joint_3d)

        # Calculate H(Y|X,Z=z)
        h_y_given_xz = conditional_entropy(joint_3d, 2, 3)

        @test size(h_y_given_xz) == (nz,)
        @test all(isfinite.(h_y_given_xz))  # Ensure no NaN or Inf values
    end

    @testset "Test with different floating-point types" begin
        import EFEasVFE: conditional_entropy

        # Create 3D joint distributions with different floating-point types
        nx, ny, nz = 3, 4, 5

        # Float32
        joint_3d_f32 = rand(Float32, nx, ny, nz)
        joint_3d_f32 ./= sum(joint_3d_f32)

        # Float64
        joint_3d_f64 = Float64.(joint_3d_f32)

        # Calculate H(Y|X,Z=z) for both types
        h_y_given_xz_f32 = conditional_entropy(joint_3d_f32, 2, 3)
        h_y_given_xz_f64 = conditional_entropy(joint_3d_f64, 2, 3)

        @test size(h_y_given_xz_f32) == (nz,)
        @test size(h_y_given_xz_f64) == (nz,)
        @test all(isapprox.(Float64.(h_y_given_xz_f32), h_y_given_xz_f64, atol=1e-5))
    end
end