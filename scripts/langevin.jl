using Random, Plots

function brownian_motion(n::Int64, x0::Float64, dt::Float64)
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

function brownian_motion(n::Int64, x0::Float64, dt::Float64, 
    α::Float64, xp::Float64)
    x = x0
    t = 0
    traj = [x]
    timeSteps = [0.0]
  
    for i in 1:n
        x = x + potential(α,x, xp,dt) + sqrt(2*dt)*randn()
        t = t + dt
        push!(traj,x)
        push!(timeSteps,t)
    end
    return traj#, timeSteps
end

function brownian_motion(n::Int64, x0::Array{Float64}, dt::Float64, 
    α::Array{Float64}, xp::Array{Float64})
    x = x0
    t = 0
    traj = [x]
    timeSteps = [0.0]
  
    for i in 1:n
        x = x + potential.(α,x, xp,dt) + sqrt(2*dt)*randn(size(x0,1))
        t = t + dt
        push!(traj,x)
        push!(timeSteps,t)
    end
    return traj#, timeSteps
end

function potential(α::Float64,x::Float64, xp::Float64,dt::Float64)
    return - α*(x-xp)*dt
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
    plot!(mean(res, dims=2), lw=5, ribbon=std(res,dims=2))
end
plot(0:dt:t_max*dt,std(res,dims=2))
plot!(0:dt:t_max*dt, t->√(2*t))
plot!(0:dt:t_max*dt, t->1/√(α))

potential.(ones(10),collect(1:10),zeros(10),0.01)
cat(brownian_motion(10, zeros(10), 0.01, ones(10), zeros(10))...,dims=2)