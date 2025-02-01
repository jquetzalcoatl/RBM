using LinearAlgebra, Plots, OMEinsum
using CUDA
# CUDA.device_reset!()
# CUDA.device!(1)
include("../scripts/exact_partition.jl")
include("../utils/train.jl")

include("../configs/yaml_loader.jl")
PATH = "/home/javier/Projects/RBM/NewResults/PhaseDiagrams/"
isdir(PATH) || mkpath(PATH)
config, _ = load_yaml_iter();
if config.phase_diagrams["gpu_bool"]
    dev = gpu
else
    dev = cpu
end

function magnetization(J::Weights, hparams::HyperParams)
    v = cat(generate_binary_states(hparams.nv)...,dims=2)
    h = cat(generate_binary_states(hparams.nh)...,dims=2)
    W = v' * J.w * h
    A = ones(2^hparams.nv,2^hparams.nh) .* (v' * J.a)
    B = ones(2^hparams.nv,2^hparams.nh) .* (J.b' * h)    
    B_weights = exp.( (W + A + B))
    Z = sum(exp.( (W + A + B)))
    mag = (sum(v', dims=2) .+ sum(h,dims=1)) ./ (hparams.nv + hparams.nh)
    num = sum(mag .* B_weights)
    num/Z
end

function magnetization_Gibbs(J::Weights, hparams::HyperParams, num::Int=5000, steps::Int=1000)
    v,h = gibbs_sample(J, hparams, num, steps)
    mag = mean(cat([v,h]...,dims=1))
    mag
end

function magnetization2_Gibbs(J::Weights, hparams::HyperParams, num::Int=5000, steps::Int=1000)
    v,h = gibbs_sample(J, hparams, num, steps)
    mag = mean(mean(cat([v,h]...,dims=1),dims=2) .^ 2)
    mag
end

function gibbs_sample(J::Weights, hparams::HyperParams, num::Int=5000, steps::Int=1000)
    v = rand([0,1],hparams.nv, num) |> dev
    local h
    for _ in 1:steps
        h = sign.(rand(hparams.nh, num) |> dev .< σ.(J.w' * v .+ J.b))
        v = sign.(rand(hparams.nv, num) |> dev .< σ.(J.w * h .+ J.a))
    end
    v, h
end

function gibbs_sample(v, J::Weights, hparams::HyperParams, num::Int=5000, steps::Int=1000)
    v = v |> dev
    local h
    for _ in 1:steps
        h = sign.(rand(hparams.nh, num) |> dev .< σ.(J.w' * v .+ J.b))
        v = sign.(rand(hparams.nv, num) |> dev .< σ.(J.w * h .+ J.a))
    end
    v, h
end

########################
function create_diagram_exact(J::Weights, hparams::HyperParams)
    phase_array = zeros(size(collect(0:0.1:1.5),1),size(collect(0.1:0.1:1.5),1)) 
    for (i,x0) in enumerate(0:0.1:1.5)
        @info i
        for (k,y0) in enumerate(0.1:0.1:1.5)
            j = 1/((hparams.nv + hparams.nh)*y0/2)
            j0 = x0*j
            J.w = randn(size(J.w)) .* j .+ j0 |> dev
            J.b = zeros(size(J.b)) |> dev
            J.a = zeros(size(J.a)) |> dev

            mag_exact = magnetization(J, hparams)
            phase_array[i,k] = mag_exact
        end
    end
    phase_array, collect(0:0.1:1.5), collect(0.1:0.1:1.5)
end

function create_diagram(J::Weights, hparams::HyperParams, num::Int=5000, steps::Int=1000)
    phase_array = zeros(size(collect(0:0.1:1.5),1),size(collect(0.1:0.1:1.5),1)) 
    for (i,x0) in enumerate(0:0.1:1.5)
        @info i
        for (k,y0) in enumerate(0.1:0.1:1.5)
            j = 1/((hparams.nv + hparams.nh)*y0/2)
            j0 = x0*j
            J.w = randn(size(J.w)) .* j .+ j0 |> dev
            J.b = zeros(size(J.b)) |> dev
            J.a = zeros(size(J.a)) |> dev

            mag_estimated = magnetization_Gibbs(J, hparams, num,steps)
            phase_array[i,k] = mag_estimated
        end
    end
    phase_array, collect(0:0.1:1.5), collect(0.1:0.1:1.5)
end

function create_diagram_EA(J::Weights, hparams::HyperParams, num::Int=5000, steps::Int=1000)
    phase_array = zeros(size(collect(0:0.1:1.5),1),size(collect(0.1:0.1:1.5),1)) 
    for (i,x0) in enumerate(0:0.1:1.5)
        @info i
        for (k,y0) in enumerate(0.1:0.1:1.5)
            j = 1/((hparams.nv + hparams.nh)*y0/2)
            j0 = x0*j
            J.w = randn(size(J.w)) .* j .+ j0 |> dev
            J.b = zeros(size(J.b)) |> dev
            J.a = zeros(size(J.a)) |> dev

            mag_estimated = magnetization2_Gibbs(J, hparams, num, steps)
            phase_array[i,k] = mag_estimated
        end
    end
    phase_array, collect(0:0.1:1.5), collect(0.1:0.1:1.5)
end

# num=500
# dev = gpu
# rbm, J, m, hparams, rbmZ = initModel(nv=6, nh=6, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
# @info dev
# @time gibbs_sample(J, hparams, 1000, 5000)
# J
# v,h = gibbs_sample(J, hparams, 500, 1000)
# cat([v,h]...,dims=1)
# mean(cat([v,h]...,dims=1),dims=2)
# mag = mean(mean(cat([v,h]...,dims=1),dims=2) .^ 2)

function main(dict::Dict)
    @info "Starting..."
    @info dict
    rbm, J, m, hparams, rbmZ = initModel(nv=dict["nv"], nh=dict["nh"], batch_size=500, lr=1.5, t=10, gpu_usage = dict["gpu_bool"], optType="Adam")
    @info "RBM initialized... Generating magnetization phase diagram"
    phase_array_m, x_ph, y_ph = create_diagram(J, hparams, dict["num"], dict["steps"])
    for i in 2:dict["disorder_avg"]
        @info "Replica $i"
        phase_array, x_ph, y_ph = create_diagram(J, hparams, dict["num"], dict["steps"])
        phase_array_m = phase_array_m .+ phase_array
    end
    phase_array_m = phase_array_m ./ dict["disorder_avg"]

    @info "Generating plots"
    fig1 = contourf(x_ph, y_ph, phase_array_m', 
        color=cgrad(:gnuplot,10, rev = false, categorical = false), linewidth=0, frame=:box)
    fig_filename = "m_$(config.phase_diagrams["fig1"])_$(config.phase_diagrams["nv"])_$(config.phase_diagrams["nh"])_$(config.phase_diagrams["disorder_avg"]).png"
    savefig(fig1, PATH * fig_filename)

    fig2 = contourf(x_ph, y_ph, phase_array_m', 
            color=cgrad(:gnuplot,10, rev = false, categorical = false), linewidth=0, frame=:box)
    fig2 = hline!([1],lw=2, label=false)
    fig2 = plot!([1,2], x->x, lw=2, label=false)
    fig2 = vline!([1], lw=2, label=false)
    fig_filename = "m_$(config.phase_diagrams["fig2"])_$(config.phase_diagrams["nv"])_$(config.phase_diagrams["nh"])_$(config.phase_diagrams["disorder_avg"]).png"
    savefig(fig2, PATH * fig_filename)

    @info "Generating EA phase diagram"
    phase_array_EA_m, x_ph, y_ph = create_diagram_EA(J, hparams, dict["num"], dict["steps"])
    for i in 2:dict["disorder_avg"]
        @info "Replica $i"
        phase_array, x_ph, y_ph = create_diagram_EA(J, hparams, dict["num"], dict["steps"])
        phase_array_EA_m = phase_array_EA_m .+ phase_array
    end
    phase_array_EA_m = phase_array_EA_m ./ dict["disorder_avg"]

    @info "Generating plots"
    fig1 = contourf(x_ph, y_ph, phase_array_EA_m', 
        color=cgrad(:gnuplot,10, rev = false, categorical = false), linewidth=0, frame=:box)
    fig_filename = "EA_$(config.phase_diagrams["fig1"])_$(config.phase_diagrams["nv"])_$(config.phase_diagrams["nh"])_$(config.phase_diagrams["disorder_avg"]).png"
    savefig(fig1, PATH * fig_filename)

    fig2 = contourf(x_ph, y_ph, phase_array_EA_m', 
            color=cgrad(:gnuplot,10, rev = false, categorical = false), linewidth=0, frame=:box)
    fig2 = hline!([1],lw=2, label=false)
    fig2 = plot!([1,2], x->x, lw=2, label=false)
    fig2 = vline!([1], lw=2, label=false)
    fig_filename = "EA_$(config.phase_diagrams["fig2"])_$(config.phase_diagrams["nv"])_$(config.phase_diagrams["nh"])_$(config.phase_diagrams["disorder_avg"]).png"
    savefig(fig2, PATH * fig_filename)
    @info "Finished!"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(config.phase_diagrams)
end