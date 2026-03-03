using BlackBoxOptim
using Random
using PyPlot

# Define the objective function for Problem02
function f(x)
    noise = 0.0 * randn()  # Add Gaussian noise (set to 0 for now)
    return sin(x[1]) + sin(10/3 * x[1]) + noise
end

# Function to plot the solution space
function plot_solution_space(x_range, f_range, optimal_x, optimal_f)
    figure(1)
    clf()  # Clear figure before plotting
    plot(x_range, f_range, color="black", linestyle="--", linewidth=2, label="f(x)", alpha=0.5)
    scatter([optimal_x], [optimal_f], color="red", s=100, label="Optimal Solution")
    xlabel("x", fontsize=14)  # Set fontsize for xlabel
    ylabel("f(x)", fontsize=14)  # Set fontsize for ylabel
    title("Solution Space", fontsize=16)  # Set fontsize for title
    legend(loc="upper left", fontsize=12)  # Position legend at the top-left and set fontsize
    tick_params(axis="both", labelsize=12)  # Set fontsize for axis ticks
end


# Main function to perform optimization and plotting
function main()

    # Define the lower and upper bounds
    lb = [0.0]  # Lower bound
    ub = [10.0]  # Upper bound

    # Perform optimization 
    result = bboptimize(f; 
        SearchRange = [(lb[1], ub[1])],     # Search range for the single dimension
        NumDimensions = 1,                  # Number of dimensions (1D problem)
        Method = :adaptive_de_rand_1_bin,   # A variant of Differential Evolution
        MaxSteps = 100,                     # Maximum number of iterations
        PopulationSize = 50,                # Population Size
        FitnessTolerance = 1e-3,            # Stop if the change in fitness is < 1e-6
        SolutionTolerance = 1e-3            # Stop if the change in solution is < 1e-6
        )

    # Extract the results
    best_solution = best_candidate(result)
    best_value = best_fitness(result)

    println("Best solution: ", best_solution)  # Should be close to the global minimum
    println("Best value: ", best_value)        # Should be close to the minimum value of f(x)

    # Generate solution space data for plotting
    x_range = range(lb[1], ub[1], length=1000)
    f_range = f.(x_range)

    # Plot the solution space
    plot_solution_space(x_range, f_range, best_solution[1], best_value)
end

# Run the main function
main()