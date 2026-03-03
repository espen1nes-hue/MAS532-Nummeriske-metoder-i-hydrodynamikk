using Optim
using Random
using PyPlot

# Define the objective function for Problem02
function f(x)
    noise = 0.0 * randn()  # Add Gaussian noise
    return sin(x[1]) + sin(10/3 * x[1]) + noise
end

# Define the gradient (derivative) of the objective function
function g!(G, x)
    G[1] = cos(x[1]) + (10/3) * cos(10/3 * x[1])
end

# Optimise using LBFGS with box constraints
function perform_optimization(x0, lb, ub)
    result = optimize(
        f,               # Objective function
        g!,              # Gradient function
        lb,              # Lower bounds
        ub,              # Upper bounds
        x0,              # Initial guess
        Fminbox(LBFGS()), # Optimizer with box constraints
        Optim.Options(
            store_trace=true,       # Store the trace
            extended_trace=true,   
            show_trace=true,       
            show_every=1,          
            iterations=100,        
            f_tol=1e-3,            
            g_tol=1e-3
        )
    )
    return result
end

# Plot optimization progress
function plot_optimization_progress(trace)
    iterations = 1:length(trace)
    f_values = [t.value for t in trace]
    x_values = [t.metadata["x"][1] for t in trace]
    
    figure(1)
    clf()  # Clear previous plots

    subplot(1, 2, 1)
    plot(iterations, f_values, marker="o", color="black", linewidth=2)
    xlabel("Iteration")
    ylabel("Objective Value")
    title("Objective Function Progress")

    subplot(1, 2, 2)
    plot(iterations, x_values, marker="o", color="black", linewidth=2)
    xlabel("Iteration")
    ylabel("Parameter Value")
    title("Parameter Progress")

    tight_layout()
    show()
end

# Plot the solution space
function plot_solution_space(x_range, f_range, optimal_x, optimal_f, x0)
    figure(2)
    clf()  # Clear figure before plotting
    plot(x_range, f_range, color="black", linestyle="--", linewidth=2, label="f(x)", alpha=0.5)
    scatter([optimal_x[1]], [optimal_f], color="red", s=100, label="Optimal Solution")
    xlabel("x", fontsize=16)  # Set fontsize for xlabel
    ylabel("f(x)", fontsize=16)  # Set fontsize for ylabel
    title("Solution Space, x0 = $(x0[1])", fontsize=18)  # Set fontsize for title and include x0 in title
    legend(loc="upper left", fontsize=14)  # Position legend at the top-right and set fontsize
    tick_params(axis="both", labelsize=14)  # Set fontsize for axis ticks
    show()
end


# Main function 
function main()

    # Initial guess
    x0 = [10.0]          # Starting-point, within the bounds

    # Bounds 
    lb = [0.0]          # Lower bound
    ub = [15.0]          # Upper bound


    # Perform optimization
    result = perform_optimization(x0, lb, ub)

    # Check for convergence
    if Optim.converged(result)
        println("Optimization converged successfully!")
    else
        println("Optimization did not converge.")
    end

    # Extract the optimal values and trace data
    optimal_x = Optim.minimizer(result)
    optimal_f = Optim.minimum(result)
    trace = Optim.trace(result)

    # Output the number of function calls (size of the trace)
    num_function_calls = length(trace)
    println("Number of function calls: ", num_function_calls)

    # Plot the optimization progress
    plot_optimization_progress(trace)

    # Generate solution space data for plotting
    x_range = range(lb[1], ub[1], length=1000)
    f_range = f.(x_range)

    # Plot the solution space
    plot_solution_space(x_range, f_range, optimal_x, optimal_f, x0)
end

# Run the main function
main()
