using DrWatson, Test, ReTestItems, EFEasVFE

# Run test suite
println("Starting tests")
ti = time()

runtests(EFEasVFE)

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds")
