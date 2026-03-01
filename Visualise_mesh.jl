using Plots

# ── Parameters (match WaterLily README) ───────────────────────────────────
p = 5                           # Mesh power: p=5→96×64, p=6→192×128, p=7→384×256
n = 3 * 2^p                    # x-cells
m = 2^(p + 1)                  # y-cells
radius = m / 8                 # WaterLily: m/8
center = m / 2 - 1             # WaterLily: m/2 - 1
L = 2 * radius                 # Diameter = characteristic length

println("Grid: $n × $m = $(n*m) cells")
println("Radius: $radius cells")
println("Center: ($center, $center)")
println("Diameter (L): $L cells")
println("Blockage: $(round(L/m*100, digits=1))%")
println("Cells/D: $(Int(L))")

# ── Build SDF field (NaN outside body for transparent background) ─────────
body = [sqrt((i + 0.5 - center)^2 + (j + 0.5 - center)^2) - radius ≤ 0 ? 1.0 : NaN
        for i in 0:n-1, j in 0:m-1]

# ── Plot ──────────────────────────────────────────────────────────────────
plt = heatmap(0.5:n-0.5, 0.5:m-0.5, body',
    c=cgrad([:gray30, :gray30]),
    colorbar=false,
    aspect_ratio=:equal,
    xlims=(0, n), ylims=(0, m),
    xlabel="x (cells)", ylabel="y (cells)",
    title="Mesh  ($n × $m,  p=$p,  L=$(Int(L)) cells/D)",
    background_color=:white,
    titlefontsize=14, xlabelfontsize=16,
    ylabelfontsize=16, tickfontsize=14,
    legendfontsize=11,
    size=(980, 420),
    grid=false,
    bottom_margin=30Plots.px,
    left_margin=20Plots.px,
    right_margin=10Plots.px,
    top_margin=10Plots.px,
    dpi=200)

# Grid lines — NaN-separated segments in two plot! calls
xsegs_x, xsegs_y = Float64[], Float64[]
for x in 0:n
    append!(xsegs_x, [x, x, NaN])
    append!(xsegs_y, [0, m, NaN])
end
plot!(plt, xsegs_x, xsegs_y,
    color=:gray60, linewidth=0.4, label=false)

ysegs_x, ysegs_y = Float64[], Float64[]
for y in 0:m
    append!(ysegs_x, [0, n, NaN])
    append!(ysegs_y, [y, y, NaN])
end
plot!(plt, ysegs_x, ysegs_y,
    color=:gray60, linewidth=0.4, label=false)

# Domain boundary (thick black)
plot!(plt, [0, n, n, 0, 0], [0, 0, m, m, 0],
    linewidth=2, linecolor=:black, label=false)

# Circle outline (exact SDF = 0)
θ = range(0, 2π, length=200)
plot!(plt,
    center .+ radius .* cos.(θ),
    center .+ radius .* sin.(θ),
    linewidth=2.5, linecolor=:red, linestyle=:solid,
    label="SDF = 0  (D = $(Int(L)) cells)")

savefig(plt, "Mesh.png")
println("Saved: Mesh.png")
