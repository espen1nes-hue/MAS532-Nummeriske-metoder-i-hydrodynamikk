using WaterLily
using StaticArrays
using Statistics
using FFTW
using Printf
using Plots

# ── Geometry: exactly as in the WaterLily README ───────────────────────────
function circle(n, m; Re=100, U=1)
    radius, center = m / 8, m / 2 - 1
    sdf(x, t) = √sum(abs2, x .- center) - radius
    Simulation((n, m), (U, 0), 2radius;
        ν=U * 2radius / Re,
        body=AutoBody(sdf))
end

# ── Strouhal number ────────────────────────────────────────────────────────
function compute_strouhal(times, Cl; t_transient)
    idx = findfirst(t -> t ≥ t_transient, times)
    Cl_dev = Cl[idx:end] .- mean(Cl[idx:end])
    N = length(Cl_dev)
    dt = times[2] - times[1]
    Cl_fft = abs.(fft(Cl_dev))
    freqs = (0:N-1) .* (1 / (dt * N))
    half = N ÷ 2
    peak = argmax(Cl_fft[2:half]) + 1
    return freqs[peak]
end

# ── Main ──────────────────────────────────────────────────────────────────
function main()
    Re = 100
    p = 5                       # Mesh: p=5→96×64, p=6→192×128, p=7→384×256
    t_max = 150.0
    dt_out = 0.1
    t_transient = t_max / 3     # Discard first third as transient for statistics

    # GIF settings
    make_gif = true             # true = create vorticity animation
    gif_start = 0.0             # Start capturing from this time
    gif_step = 0.1              # Time between frames (= dt_out, every step)
    gif_fps = 15                # Frames per second

    # Create simulation
    n = 3 * 2^p
    m = 2^(p + 1)
    sim = circle(n, m; Re)
    R = inside(sim.flow.p)

    println("Grid: $(n) × $(m) = $(n*m) cells")
    println("L = $(sim.L), ν = $(round(sim.flow.ν, digits=6))")
    println("Blockage: $(round(sim.L/m*100, digits=1))%")
    println("Threads: $(Threads.nthreads())")
    make_gif && println("GIF: capturing tU/L = $gif_start → $t_max, step $gif_step")

    # ── Run simulation ────────────────────────────────────────────────────
    time_range = collect(1:dt_out:t_max)
    drag = Float64[]
    lift = Float64[]
    framedir = make_gif ? mktempdir() : ""
    frame_num = 0
    t_next_frame = gif_start

    for t in time_range
        sim_step!(sim, t, remeasure=false)
        print("\rtU/L = ", round(t, digits=1), " / ", t_max)

        # Forces on body (negate: total_force returns force on fluid)
        f = -WaterLily.total_force(sim) ./ (0.5 * sim.L * sim.U^2)
        push!(drag, f[1])
        push!(lift, f[2])

        # Capture GIF frame as PNG
        if make_gif && t ≥ t_next_frame
            @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
            flood(sim.flow.σ[R], clims=(-10, 10), border=:none,
                title="ω  (tU/L = $(round(t, digits=1)),  Re = $Re)",
                background_color=:white,
                size=(1000, 400))
            body_plot!(sim)
            frame_num += 1
            savefig(joinpath(framedir, @sprintf("%06d.png", frame_num)))
            t_next_frame += gif_step
        end
    end
    println()

    # ── Statistics ────────────────────────────────────────────────────────
    idx = findfirst(t -> t ≥ t_transient, time_range)
    println("\n--- Results (Re=$Re, tU/L > $t_transient) ---")
    println("Mean Cd: ", round(mean(drag[idx:end]), digits=4))
    println("Mean Cl: ", round(mean(lift[idx:end]), digits=4))
    println("RMS  Cl: ", round(sqrt(mean(lift[idx:end] .^ 2)), digits=4))

    St = compute_strouhal(time_range, lift; t_transient)
    println("Strouhal (St): ", round(St, digits=4))

    # ── Figure 1: Force coefficients ──────────────────────────────────────
    p1 = plot(time_range, [drag lift],
        labels=["drag" "lift"],
        xlabel="tU/L",
        ylabel="Force coefficients",
        linewidth=2, grid=true,
        title="Re = $Re,  St = $(round(St, digits=3))",
        background_color=:white,
        titlefontsize=18, xlabelfontsize=16,
        ylabelfontsize=16, legendfontsize=13,
        tickfontsize=12, size=(1000, 600),
        left_margin=15Plots.px)
    savefig(p1, "Force_coefficients.png")
    println("Saved: Force_coefficients.png")

    # ── Figure 2: Vorticity field ─────────────────────────────────────────
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    p2 = flood(sim.flow.σ[R], clims=(-10, 10), border=:none,
        title="Vorticity  (tU/L = $(Int(t_max)),  Re = $Re)",
        background_color=:white,
        titlefontsize=18, xlabelfontsize=16,
        ylabelfontsize=16, tickfontsize=12,
        size=(1200, 500),
        left_margin=10Plots.px)
    body_plot!(sim)
    savefig(p2, "Vorticity.png")
    println("Saved: Vorticity.png")

    # ── Animation ──────────────────────────────────────────────────────────
    if make_gif
        println("Assembling $frame_num frames into GIF...")
        try
            run(`ffmpeg -v 16 -framerate $gif_fps -i $framedir/%06d.png
                 -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
                 -loop 0 -y Vorticity.gif`)
            println("Saved: Vorticity.gif")
        catch
            println("GIF failed — install system ffmpeg: sudo apt install ffmpeg")
        end
        rm(framedir, recursive=true)
    end

    display(p1)
    display(p2)
end

@time main()
gui()
