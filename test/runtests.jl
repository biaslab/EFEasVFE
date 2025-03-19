using DrWatson, Test, ReTestItems
@quickactivate "EFEasVFE"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

include("minigrid_tests.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds")
