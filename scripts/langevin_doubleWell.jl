# using Random, Plots, Statistics, LinearAlgebra, OMEinsum, Plots.PlotMeasures, Roots, QuadGK
# include("../utils/init.jl")
# include("../scripts/exact_partition.jl")
# include("../scripts/PhaseAnalysis.jl")

using Random, Plots, Statistics, Plots.PlotMeasures, Roots

function brownian_motion(n::Int64, x0::Array{Float64}, dt::Float64, ∂U::Function)
    """
        N-dimensional Brownian Motion in the presence of a generic force -∂U.
        n: number of timeSteps
        x0: Initial position
        dt: timeSteps
    """
    x = x0
    t = 0
    samples = size(x0,1)
    traj = zeros(n+1, samples)
    traj[1,:] = x
    timeSteps = zeros(n+1)
    F = ∂U
  
    for i in 1:n
        x = x + F(x) + sqrt(2*dt)*randn(size(x0,1))
        t = t + dt
        # push!(traj,x)
        # push!(timeSteps,t)
        traj[i+1,:]=x
        timeSteps[i+1]=t
    end
    return traj#, timeSteps
end

U_eff(z::Float64,λ::Float64,k::Float64,
    γ::Float64,α::Float64,μ::Float64, 
    s::Float64=1.0) = - s * 0.5 * λ^2 * z^2 + 0.5 * k * (z - μ)^2 + 1/3 * γ * (z - μ)^3 + 0.25 * α * (z - μ)^4
dU_effdz(z::Array{Float64},λ::Array{Float64},k::Array{Float64},
    γ::Array{Float64},α::Array{Float64},μ::Array{Float64}, 
    s::Array{Float64}, dt::Float64=0.01, β::Float64=1.0) = β .* (.- s .* λ .^ 2 .* z .+ k .* (z .- μ) .+ γ .* (z .- μ) .^ 2 .+ α .* (z .- μ) .^ 3) .* dt
dU_effdz(z::Float64,λ::Float64,k::Float64,
    γ::Float64,α::Float64,μ::Float64, 
    s::Float64, dt::Float64=0.01, β::Float64=1.0) = β * (- s * λ ^ 2 * z + k * (z - μ) + γ * (z - μ) ^ 2 + α * (z - μ) ^ 3) * dt
d2U_effdz2(z::Float64,λ::Float64,k::Float64,
    γ::Float64,α::Float64,μ::Float64, 
    s::Float64, dt::Float64=0.01, β::Float64=1.0) = β * (- s * λ ^ 2 + k + 2 * γ * (z - μ) + 3 * α * (z - μ) ^ 2) * dt


momnt(x,n) = mean(x .^ n)
constr(z0::Array{Float64}, λ::Float64,
    k::Float64, γ::Float64,α::Float64, μ::Float64) =  momnt(z0 ,4) * α +  momnt(z0 ,3) * (γ - 3 * μ * α) +  
    momnt(z0 ,2) * (-λ^2+k - 2*μ*γ + 3*μ^2*α) +  momnt(z0 ,1) * (-k*μ + γ*μ^2-μ^3*α)
constr_1(z0::Array{Float64}, μ::Float64) =  momnt(z0 ,2) - μ * momnt(z0 ,1)
constr_2(z0::Array{Float64}, μ::Float64) =  momnt(z0 ,3) - 2 * μ * momnt(z0 ,2) + μ^2 * momnt(z0 ,1)
constr_3(z0::Array{Float64}, μ::Float64) =  momnt(z0 ,4) - 3 * μ * momnt(z0 ,3) + 3 * μ^2 * momnt(z0 ,2) - μ^3 * momnt(z0 ,1)
    
function find_params(z::Float64,Λ::Float64,k::Float64,γ::Float64,α::Float64,μ::Float64,n_steps::Int64, lr1::Float64=0.0001,samples::Int64=800, t_max::Int64=2000, dt::Float64=0.0001)
    # z0 = -2.20
    # k_t,γ_t,α_t = (40.2, 1200.0, 600.7, 100.5)
    z0,k_t,γ_t,α_t = (z,k,γ,α)
    plot()
    for i in 1:n_steps
        i % 10 == 0 ? (@info i, k_t, γ_t, α_t) : nothing
        k_t = k_t - lr1 * dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) * (z0-μ)
        γ_t = γ_t - lr1 * dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) * (z0-μ)^2
        α_t = α_t - lr1 * dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) * (z0-μ)^3
        
        p = plot!([i],[dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0)], marker=:circle, markerstrokewidth=0, markersize=10, label=false, color=:magenta)
        
        display(p)
    end
    func = x->dU_effdz(x,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0)
    rts = fzeros(func, -20.5,30.0)
    @info "Force at z0:", dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0)
    @info "Minimum?", d2U_effdz2(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) > 0
    @info "Roots:", rts
    res = brownian_motion(t_max, z0 .* ones(samples), dt, x->-dU_effdz(x,Λ .* ones(samples), k_t .* ones(samples), γ_t .* ones(samples), 
            α_t .* ones(samples), μ .* ones(samples), ones(samples),dt, 1.0))
    @info "Moment constraint:", constr(res[end,:], Λ,k_t,γ_t,α_t, μ)
    z0,Λ,k_t,γ_t,α_t,μ
end

function find_params_2(z::Float64,Λ::Float64,k::Float64,γ::Float64,α::Float64,μ::Float64,n_steps::Int64, 
    lr1::Float64=0.00001, lr2::Float64=0.01, samples::Int64=800, t_max::Int64=2000, dt::Float64=0.0001)
    L1,L2 = [],[]
    z0,k_t,γ_t,α_t = (z,k,γ,α)
    local res
    for i in 1:n_steps
        @info i, k_t, γ_t, α_t
        res = brownian_motion(t_max, z0 .* ones(samples), dt, x->-dU_effdz(x,Λ .* ones(samples), k_t .* ones(samples), γ_t .* ones(samples), 
            α_t .* ones(samples), μ .* ones(samples), ones(samples),dt, 1.0))
        k_t = k_t - lr1 * (constr(res[end,:], Λ,k_t,γ_t,α_t, μ)-1) * constr_1(res[end,:], μ)
        γ_t = γ_t - lr1 * (constr(res[end,:], Λ,k_t,γ_t,α_t, μ)-1) * constr_2(res[end,:], μ)
        α_t = α_t - lr1 * (constr(res[end,:], Λ,k_t,γ_t,α_t, μ)-1) * constr_3(res[end,:], μ)

        k_t = k_t - lr2 * dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) * (z0-μ)
        γ_t = γ_t - lr2 * dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) * (z0-μ)^2
        α_t = α_t - lr2 * dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) * (z0-μ)^3

        append!(L1, constr(res[end,:], Λ,k_t,γ_t,α_t, μ)-1)
        append!(L2, dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0))

        if abs(constr(res[end,:], Λ,k_t,γ_t,α_t, μ) - 1) < 0.1
            @info constr(res[end,:], Λ,k_t,γ_t,α_t, μ)
            break
        end
    end
    p1 = plot(L1, marker=:circle, markerstrokewidth=0, markersize=10, label=false, color=:magenta)
    p2 = plot(L2, marker=:circle, markerstrokewidth=0, markersize=10, label=false, color=:magenta)
    display(plot(p1,p2, layout=(2,1)))
    func = x->dU_effdz(x,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0)
    rts = fzeros(func, -20.5,30.0)
    @info "Force at z0:", dU_effdz(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0)
    @info "Minimum?", d2U_effdz2(z0,Λ,k_t,γ_t,α_t,μ,1.0,1.0,1.0) > 0
    @info "Roots:", rts
    @info "Moment constraint:", constr(res[end,:], Λ,k_t,γ_t,α_t, μ)
    z0,Λ,k_t,γ_t,α_t,μ
end

# z0,Λ,k_t,γ_t,α_t,μ = find_params(-2.20,39.9,40.2, 1100.0, 40.7,0.99,250)

# z0,Λ,k_t,γ_t,α_t,μ = find_params_2(z0,Λ,k_t,γ_t,α_t,μ,250,1e-6, 1e-4)

# z0,Λ,k_t,γ_t,α_t,μ = find_params(z0,Λ,k_t,γ_t,α_t,μ,250)

# plot(-5:0.02:5, x -> U_eff(x,Λ,k_t,γ_t,α_t,μ))
# plot!([z0],  [U_eff(z0,Λ,k_t,γ_t,α_t,μ)], marker=:circle, lw=0 )


if abspath(PROGRAM_FILE) == @__FILE__
    z0,Λ,k_t,γ_t,α_t,μ = find_params(-2.20,39.9,40.2, 1100.0, 40.7,0.99,250)

    z0,Λ,k_t,γ_t,α_t,μ = find_params_2(z0,Λ,k_t,γ_t,α_t,μ,250,1e-6, 1e-4)

    z0,Λ,k_t,γ_t,α_t,μ = find_params(z0,Λ,k_t,γ_t,α_t,μ,250)

    plot(-5:0.02:5, x -> U_eff(x,Λ,k_t,γ_t,α_t,μ))
    plot!([z0],  [U_eff(z0,Λ,k_t,γ_t,α_t,μ)], marker=:circle, lw=0 )
end

