using Surrogates
using Statistics
using Random
using PyPlot

# Clear current figure
PyPlot.clf()

# Define the objective function for Problem02
function f(x)
    return sin(x[1]) + sin(10/3 * x[1]) 
end

# Function to minimise Kriging error
function minimise_Kriging_error(my_krig, x_refined, y_refined, f, x_samples, y_samples, max_error_Kriging_model, lower_bound, upper_bound, p_value)

    # pre-allocate
    max_mse_std_err = 1
    old_y_infill_point = Inf

    # while Kriging model error >= max error
    while max_mse_std_err >= max_error_Kriging_model

        # Kriging predictions objective function response, refined solution space
        krig_approx = my_krig.(x_refined)

        # Kriging predicted Standard errors, refined solution space
        std_err = std_error_at_point.(my_krig, x_refined)

        # Compute Mean square error
        mse_std_err = std_err.^2
        max_mse_std_err, max_index = findmax(mse_std_err)

        # Get x value for max error
        x_infill_point = x_refined[max_index]

        # Compute real objective function response for x_infill_point 
        y_infill_point = f.(x_infill_point)

        # Break the loop if max_mse_std_err is within tolerance
        if y_infill_point == old_y_infill_point
            println("Stopping; repetitive infill-point")
            break
        end

        # rename infill point
        old_y_infill_point = y_infill_point
        println("Infill_point: ", round(x_infill_point, digits=2))

        # Update the Kriging model with the new infill point
        push!(x_samples, x_infill_point)
        push!(y_samples, y_infill_point)
        my_krig = Kriging(x_samples, y_samples, lower_bound, upper_bound, p=p_value)

        # Predictions on refined solution grid
        krig_approx = my_krig.(x_refined)

        println("")

        # Best value
        best_value = minimum(krig_approx)
        println("best solution = ", best_value)

        # Best x value
        best_index = argmin(krig_approx)
        best_pos = x_refined[best_index]
        println("best pos = ", best_pos)

        # print number of function calls
        println("function calls = ", length(y_samples))


        # Plotting
        plotting(x_infill_point,x_refined,y_refined,x_samples,y_samples,krig_approx,std_err,mse_std_err,max_mse_std_err)

        # Break the loop if max_mse_std_err is within tolerance
        if max_mse_std_err <= max_error_Kriging_model
            println("Stopping; max_error_Kriging_model")
            break
        end
    end
end


function plotting(x_infill_point,x_refined,y_refined,x_samples,y_samples,krig_approx,std_err,mse_std_err,max_mse_std_err)

    # round error for plotting
    max_mse_std_err_plot = round(max_mse_std_err, digits=5)
    
    # Plotting
    font_size = 18
    PyPlot.clf()

    # First subplot: Kriging approximation
    PyPlot.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    PyPlot.plot(x_refined, y_refined, "k--", label="True function", linewidth=1.25)
    PyPlot.scatter(x_samples, y_samples, label="Sample points", color="yellow", s=100, marker="o", edgecolor="black", linewidth=1.25, alpha=1)  # Initial sample points
    PyPlot.plot(x_refined, krig_approx, color="red", linestyle="-", linewidth=2, label="Kriging model") 
    PyPlot.fill_between(x_refined, krig_approx .- std_err, krig_approx .+ std_err, color="green", alpha=0.2, label="± MSE error")
    PyPlot.title("Kriging Model", fontsize=font_size + 6)
    PyPlot.xlabel("x", fontsize=font_size)
    PyPlot.ylabel("f(x)", fontsize=font_size)
    PyPlot.legend(fontsize=font_size-4)
    PyPlot.xticks(fontsize=font_size-4)
    PyPlot.yticks(fontsize=font_size-4)
    PyPlot.grid(true)

    # Second subplot: Standard Error
    PyPlot.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    PyPlot.plot(x_refined, mse_std_err, color="green", linewidth=2)
    PyPlot.fill_between(x_refined, mse_std_err, color="green", alpha=0.9)  # Fill the area under the curve
    PyPlot.scatter(x_infill_point, max_mse_std_err, color="lightblue", s=100, marker="o", edgecolor="black", linewidth=1.25, label="Max error = $max_mse_std_err_plot")
    PyPlot.title("Predicted error", fontsize=font_size + 6)
    PyPlot.xlabel("x", fontsize=font_size)
    PyPlot.ylabel("MSE error(x)", fontsize=font_size)
    PyPlot.legend(fontsize=font_size-4)
    PyPlot.xticks(fontsize=font_size-4)
    PyPlot.yticks(fontsize=font_size-4)
    PyPlot.ylim(0, max_mse_std_err * 1.1)  # Set lower limit to 0, upper limit will auto-adjust
    PyPlot.grid(true)

    # Adjust layout
    PyPlot.tight_layout()
    PyPlot.draw()
    PyPlot.sleep(1)
end


function main()

# Bounds
lower_bound = 0
upper_bound = 10

# Initial samples, DoE
n_DoE = 2                     
max_error_Kriging_model = 0.01
p_value = 1.9  # Polynomial degree


# Create sample plan, DoE
x_samples = sample(n_DoE, lower_bound, upper_bound, LatinHypercubeSample())

# Compute real objective function response at sample points  
y_samples = f.(x_samples)

# Create Kriging model from sample points
my_krig = Kriging(x_samples, y_samples, lower_bound, upper_bound, p=p_value)

# Compute real function values, (for plotting only)
x_refined = lower_bound:0.001:upper_bound  # resolution for solution space
y_refined = f.(x_refined)

# Minimise Kriging error and plot the results
minimise_Kriging_error(my_krig, x_refined, y_refined, f, x_samples, y_samples, max_error_Kriging_model, lower_bound, upper_bound, p_value)


# add infill points, EI?
end

# run main 
main()