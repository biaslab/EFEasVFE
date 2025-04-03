using RxEnvironmentsZoo.GLMakie

export plot_free_energy, plot_posteriors

"""
    plot_free_energy(fe_values)

Create a plot showing the free energy values during inference.

# Arguments
- `fe_values`: Vector of free energy values

# Returns
- A GLMakie figure with the free energy plot
"""
function plot_free_energy(fe_values)
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="Iteration",
        ylabel="Free Energy",
        title="Free Energy During Inference")

    # Plot free energy curve
    lines!(ax, 1:(length(fe_values)-5), fe_values[6:end])

    return fig
end

"""
    plot_posteriors(posteriors)

Create plots showing the posterior distributions.

# Arguments
- `posteriors`: Named tuple of posterior distributions

# Returns
- A GLMakie figure with the posterior plots
"""
function plot_posteriors(posteriors)
    fig = Figure()

    # Plot each posterior in a separate subplot
    for (i, (key, value)) in enumerate(pairs(posteriors))
        ax = Axis(fig[i, 1], title=string(key))
        barplot!(ax, probvec(value))
    end

    return fig
end