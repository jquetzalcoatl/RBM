using Random, Plots, Statistics, LinearAlgebra, OMEinsum, Plots.PlotMeasures
include("../utils/init.jl")
include("../scripts/exact_partition.jl")
include("../scripts/PhaseAnalysis.jl")

function brownian_motion(n::Int64, x0::Float64, dt::Float64)
    """
        Simple 1D Brownian Motion.
        n: number of timeSteps
        x0: Initial position
        dt: timeSteps
    """
    x = x0
    t = 0
    traj = [x]
    timeSteps = [0.0]
  
    for i in 1:n
        x = x + sqrt(2*dt)*randn()
        t = t + dt
        push!(traj,x)
        push!(timeSteps,t)
    end
    return traj#, timeSteps
end

function brownian_motion(n::Int64, x0::Float64, dt::Float64, α::Float64, xp::Float64)
    """
        1D Brownian Motion in the presence of a potential.
        n: number of timeSteps
        x0: Initial position
        dt: timeSteps
    """
    x = x0
    t = 0
    traj = [x]
    timeSteps = [0.0]
  
    for i in 1:n
        x = x + dpotential(α,x, xp,dt) + sqrt(2*dt)*randn()
        t = t + dt
        push!(traj,x)
        push!(timeSteps,t)
    end
    return traj#, timeSteps
end

function brownian_motion(n::Int64, x0::Array{Float64}, dt::Float64, 
    α::Array{Float64}, xp::Array{Float64})
    """
        N-dimensional Brownian Motion in the presence of a 1D potential (args to potential are broadcasted).
        n: number of timeSteps
        x0: Initial position
        dt: timeSteps
    """
    x = x0
    t = 0
    traj = [x]
    timeSteps = [0.0]
  
    for i in 1:n
        x = x + dpotential.(α,x, xp,dt) + sqrt(2*dt)*randn(size(x0,1))
        t = t + dt
        push!(traj,x)
        push!(timeSteps,t)
    end
    return traj#, timeSteps
end
function dpotential(α::Float64,x::Float64, xp::Float64,dt::Float64)
    return - α*(x-xp)*dt
end

function brownian_motion(n::Int64, x0::Array{Float64}, dt::Float64, ∂U::Function)
    """
        N-dimensional Brownian Motion in the presence of a generic force -∂U.
        n: number of timeSteps
        x0: Initial position
        dt: timeSteps
    """
    x = x0
    t = 0
    traj = [x]
    timeSteps = [0.0]
    F = ∂U
  
    for i in 1:n
        x = x + F(x) + sqrt(2*dt)*randn(size(x0,1))
        t = t + dt
        push!(traj,x)
        push!(timeSteps,t)
    end
    return traj#, timeSteps
end

function ∂Udt(x::Array{Float64}, Γ::Array{Float64}, Ω::Array{Float64}, dt::Float64)
    return (Γ .* x .- Ω) .* dt
end

function ∂Udt_4(x::Array{Float64}, Λ::Array{Float64}, μ::Array{Float64}, dt::Float64, filter::Bool)
    if filter
        return ((Λ .+ 0.25) .* x .- 0.25 .* μ .- 1/144 .* (x .- μ) .^ 3 + 1/32 .* (x .- μ)  .* ((x .- μ)' * (x .- μ))) .* dt
    else
        return ((Λ .+ 0.25) .* x .- 0.25 .* μ ) .* dt
    end
end

function Dwelldx(x::Array{Float64}, Λ::Array{Float64}, μ::Array{Float64}, k::Array{Float64}, α::Array{Float64}, dt::Float64)
    return (.- Λ .^ 2 .* x .+ k .* (x .- μ ) .+ α .* (x .- μ ) .^ 3 ) .* dt
end

function Dwell(x::Float64, Λ::Float64, μ::Float64, k::Float64, α::Float64)
    return (.- 0.5 * Λ .^ 2 .* x .^ 2 .+ 0.5 .* k .* (x .- μ ) .^ 2 .+ 0.25 .* α .* (x .- μ ) .^ 4 )
end

function generate_corr_functions(J::Weights, hparams::HyperParams)
    @info "Generate our data"
    F = LinearAlgebra.svd(J.w, full=true)
    rbm_vectors = generate_binary_states(hparams.nv + hparams.nh)
    vs = hcat(rbm_vectors...)[1:hparams.nv,:]
    hs = hcat(rbm_vectors...)[hparams.nv+1:end,:]

    ys = F.Vt * hs
    xs = F.U' * vs
    a = F.U' * J.a
    b = F.Vt * J.b
    x0 = -b ./ F.S
    y0 = -a[1:hparams.nh] ./ F.S

    us = 1/sqrt(2) .* (xs[1:hparams.nh,:] .- ys .- x0 .+ y0)
    ws = 1/sqrt(2) .* (xs[1:hparams.nh,:] .+ ys .- x0 .- y0)

    zs = vcat(us,ws,xs[hparams.nh+1:end,:])
    μ = mean(zs,dims=2)
    Δ = (zs .- mean(zs,dims=2))

    @ein cor2body[i,j] := Δ[i,k] * Δ[j,k]
    cor2body = cor2body * 2.0^(-(hparams.nv+hparams.nh))

    #three body
    @ein cor3body[i,j,l] := Δ[i,k] * Δ[j,k] * Δ[l,k]
    cor3body = cor3body * 2.0^(-(hparams.nv+hparams.nh))
    # heatmap(cat([cat([cor3body[:,:,i+4*j] for i in 1:4]...,dims=2) for j in 0:3]...,dims=1))

    #four body
    @ein cor4body[i,j,l,m] := Δ[i,k] * Δ[j,k] * Δ[l,k] * Δ[m,k]
    cor4body = cor4body * 2.0^(-(hparams.nv+hparams.nh))
    cor2body, cor3body, cor4body, μ, F.S
end

function ∂UdtRBM(x::Array{Float64}, μ::Array{Float64}, k2::Array{Float64,2},k3::Array{Float64,3},k4::Array{Float64,4}, λ::Array{Float64},dt::Float64, hparams::HyperParams, D::Float64=1.0)
    Δ = x .- μ
    @ein V2[m] := k2[m,j]*Δ[j]
    @ein V3[m] := k3[m,i,j]*Δ[i]*Δ[j]
    @ein V4[m] := k4[m,i,j,k]*Δ[i]*Δ[j]*Δ[k]
    return (vcat(- λ .* x[1:hparams.nh], λ .* x[hparams.nh+1:2*hparams.nh],zeros(hparams.nv-hparams.nh)) .+ (V2 .+ 0.5*V3 .+ 1/6*V4) .* D)*dt
end

begin
    t_max = 500
    dt=0.01
    samples=800
    α=0.5
    xp=0.2
    # res = cat(brownian_motion.(t_max,zeros(samples), dt)...,dims=2)
    res = cat(brownian_motion.(t_max,zeros(samples), dt, α, xp)...,dims=2)
    plot(res, label=false)
    plot!(mean(res, dims=2), lw=5, ribbon=std(res,dims=2), label=false)
end

begin
    t_max = 1000
    dt=0.01
    samples=800
    α=0.5
    xp=0.2
    res = cat(brownian_motion(t_max, zeros(samples), dt, x->-∂Udt(x,0.5 .* ones(samples),2 .* ones(samples),dt))...,dims=2)'
    plot(res, label=false)
    plot!(mean(res, dims=2), lw=5, ribbon=std(res,dims=2), label=false)
end
begin
    p1 = plot(mean(res,dims=2))
    p1 = plot!(1:1000,t->4*(1-exp(-0.5*t*dt)))

    p2 = plot(std(res,dims=2))
    p2 = plot!(1:1000,t->√(2*(1-exp(-2*0.5*t*dt))))
    plot(p1,p2)
end

begin
    t_max = 1000
    dt=0.01
    samples=800
    x = zeros(samples)
    Λ = 1.2 .* ones(samples)
    μ = 2 .* ones(samples)
    k = 1.0 .* ones(samples)
    α = 0.5 .* ones(samples)
    res = cat(brownian_motion(t_max, x, dt, x->-∂Udt_4(x,Λ,μ,dt,true))...,dims=2)'
    plot(res, label=false)
    plot!(mean(res, dims=2), lw=5, ribbon=std(res,dims=2), label=false)
end


begin
    c1 = 1.2+0.25 # 
    c2 = 2*0.25/c1
    p1 = plot(mean(res,dims=2))
    p1 = plot!(1:1000,t->c2*(1-exp(-c1*t*dt)))

    p2 = plot(std(res,dims=2))
    p2 = plot!(1:1000,t->√((1-exp(-2*c1*t*dt))/c1))
    plot(p1,p2)
end

function BM_loop_opt(t_max::Int64, replicas::Int64, μ::Array{Float64,2}, λ::Array{Float64},dt::Float64, hparams::HyperParams)
    
    x = zeros(hparams.nv+hparams.nh)
    Λ = vcat(- λ , λ ,zeros(hparams.nv-hparams.nh)) #1.2 .* ones(samples)
    res = cat(brownian_motion(t_max, x, dt, x->-∂Udt_4(x,Λ,μ[:,1],dt,true))...,dims=2)'
    res_replicas = res
    for r in 1:replicas
        res = cat(brownian_motion(t_max, x, dt, x->-∂Udt_4(x,Λ,μ[:,1],dt,true))...,dims=2)'
        res_replicas = cat(res_replicas,res,dims=3)
    end
    res_replicas
end

function BM_loop(t_max::Int64, replicas::Int64, μ::Array{Float64,2}, k2::Array{Float64,2},k3::Array{Float64,3},k4::Array{Float64,4}, λ::Array{Float64},dt::Float64, hparams::HyperParams)
    samples=hparams.nv + hparams.nh
    x = zeros(samples)
    res = cat(brownian_motion(t_max, x, dt, x->-∂UdtRBM(x,μ[:,1],k2,k3,k4,λ,dt,hparams))...,dims=2)'
    res_replicas = res
    for r in 1:replicas
        res = cat(brownian_motion(t_max, x, dt, x->-∂UdtRBM(x,μ[:,1],k2,k3,k4,λ,dt,hparams))...,dims=2)'
        res_replicas = cat(res_replicas,res,dims=3)
    end
    # color_gradient = reshape(range(colorant"red", stop=colorant"blue", length=size(samples, 2)),1,size(samples,2));
    # plot(res, label=false, color=color_gradient)
    res_replicas
end

dt=0.01
timeSteps=1000
rbm, J, m, hparams, rbmZ = initModel(nv=10, nh=6, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")
k2,k3,k4,μ, λ = generate_corr_functions(J,hparams)
res = BM_loop(timeSteps,800,μ,k2,k3,k4,Array{Float64}(λ),dt,hparams)

res = BM_loop_opt(timeSteps,800,μ,Array{Float64}(λ),dt,hparams)

begin
    PATH = "/home/javier/Projects/RBM/NewResults/PhaseDiagrams/"
    color_gradient = reshape(range(colorant"red", stop=colorant"blue", length=hparams.nv+hparams.nh),1,hparams.nv+hparams.nh);
    p1 = plot(0:dt:timeSteps*dt, reshape(mean(res,dims=3),:,hparams.nv+hparams.nh), label=false, color=color_gradient, 
        frame=:box, xlabel="Time", ylabel="Mean")
    p2 = plot(0:dt:timeSteps*dt, reshape(std(res,dims=3),:,hparams.nv+hparams.nh), label=false, color=color_gradient,
    frame=:box, xlabel="Time", ylabel="STD")

    p3 = plot(frame=:box)
    for i in 1:hparams.nv+hparams.nh
        p3 = plot!(res[end,i,:], st=:histogram, lw=0, opacity=0.5, 
            normalize=true, label="$i", color=color_gradient[1,i])
    end
    p3 = plot!(xlabel="Position", ylabel="PDF")
    p = plot(p1,p2,p3, size=(900,500), left_margin=5mm)
    # savefig(p, PATH * "fig_langevin_nv8_nh4.png")
    p
end


hline!([μ[15,1]])
plot(0:dt:timeSteps*dt, mean(res,dims=3)[:,15,1], label=false, color=color_gradient, 
        frame=:box, xlabel="Time", ylabel="Mean")


#######################
####################
subWeights = J.w[1:10,1:6]
suba = J.a[1:10]
subb = J.b[1:6]

J.w = subWeights |> cpu
J.a = suba |> cpu
J.b = subb |> cpu
######################

J.w = rand(size(J.w)[1], size(J.w)[2])
J.a = rand(size(J.a)[1])
J.b = rand(size(J.b)[1])

F = LinearAlgebra.svd(J.w, full=true)
J.w = F.U * vcat(Diagonal(F.S .+ 7), zeros(hparams.nv-hparams.nh, hparams.nh)) * F.Vt

include("../scripts/PhaseAnalysis.jl")
config.model_analysis["files"]

modelName = "PCD-FMNIST-500-replica1-L" #config.model_analysis["files"][1]
# modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = config.model_analysis["files"][1]
modelName = "Random-RBM_small"
rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);

######################
F = LinearAlgebra.svd(J.w, full=true)
dev=cpu

v_gibbs, h_gibbs = gibbs_sample(J, hparams, 20000, 500)
ys = F.Vt * h_gibbs
xs = F.U' * v_gibbs

rbm_vectors = generate_binary_states(hparams.nv + hparams.nh)
vs = hcat(rbm_vectors...)[1:hparams.nv,:]
hs = hcat(rbm_vectors...)[hparams.nv+1:end,:]
ys = F.Vt * hs #h_gibbs
xs = F.U' * vs #v_gibbs


a = F.U' * J.a
b = F.Vt * J.b
x0 = -b ./ F.S
y0 = -a[1:hparams.nh] ./ F.S

us = 1/sqrt(2) .* (xs[1:hparams.nh,:] .- ys .- x0 .+ y0)
ws = 1/sqrt(2) .* (xs[1:hparams.nh,:] .+ ys .- x0 .- y0)

zs = vcat(us,ws,xs[hparams.nh+1:end,:])

p3 = plot(frame=:box)
for i in 1:hparams.nv+hparams.nh
    p3 = plot!(zs[i,:], st=:histogram, lw=0, opacity=0.5, 
        normalize=true, label="$i", color=color_gradient[1,i])
end
p3 = plot!(xlabel="Position", ylabel="PDF")