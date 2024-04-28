using CUDA
using Plots, Statistics, Dates

include("init.jl")
include("structs.jl")
include("adamOpt.jl")
include("loader.jl")
include("en.jl")
include("tools.jl")

# Training function
function train(dict ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, 
    learnType="Rdm", gpu_usage = false, t_samp=100, num=25, optType="SGD", numbers=[0,1], snapshot=50, 
    savemodel=true, γ=0.001, logging=false, io=nothing, annealing=false, β2=1.0)
    try
        Int(sqrt(num))
    catch
        @warn "num's root needs to be an integer."
        return 0,0,0,0,0
    end
    
    rbm, J, m, hparams, rbmZ = initModel(; nv, nh, batch_size, lr, γ, t, gpu_usage, optType)
    x, y = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    TS = Test(x,y)

    if optType=="Adam"
        opt = initOptW(hparams, J) 
    elseif optType=="SGD"
        opt = 0
    end
    dev = selectDev(hparams)
    
    x = loadData(; hparams, dsName="MNIST01", numbers)
    
    if learnType == "Rdm"
        x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
        lProtocol = learnType
    elseif learnType in ["CD", "PCD"]
        x_Gibbs = x
        lProtocol = learnType
    end

    if annealing
        β0 = β2
        if "tout" in keys(dict)
            ΔT = (1/(β0*dict["tout"]) - 1)*1/(epochs-5)      #(T₀/Tₙ - 1)/epochs
        else
            ΔT = (1/(β0*10) - 1)*1/(epochs-5)      #(T₀/Tₙ - 1)/epochs
        end
        # β = β2
    end

    genSample(rbm, J, hparams, m; num, t=t_samp, plotSample, epoch=0, dict, dev, TS) 
    for epoch in 1:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch, ZEpoch = [], [], [], [], []
        
        for i in eachindex(x)
            # @info i
            
            Δw, Δa, Δb = loss(rbm, J, x[i], x_Gibbs[i]; β2, hparams, dev, lProtocol, bw=dict["bw"])
            if learnType == "PCD"
                x_Gibbs[i] = rbm.v |> cpu
            end
            
            updateJ!(J, Δw, Δa, Δb, opt; hparams)

            append!(enEpoch, avgEn(rbm,J, 1) |> cpu)
            append!(ZEpoch, sum(exp.(- H(rbm, J))) |> cpu)
            append!(ΔwEpoch, mean(Δw) |> cpu)
            append!(ΔaEpoch, mean(Δa) |> cpu)
            append!(ΔbEpoch, mean(Δb) |> cpu)
        end
        if sign(sum(isnan.(J.w))) == 1
            @warn "Weights are NaN. Training will be stopped"
            break
        end

        a,b = EnRBM(J, hparams, 1; dev)
        append!(m.enData, mean(enEpoch))
        append!(m.enDataSD, std(enEpoch))
        append!(m.enRBM, a)
        append!(m.enSP, saddlePointEnergy(J, hparams; dev))
        append!(m.Zdata, mean(ZEpoch))
        append!(m.Zrbm, b)
        append!(m.T, 1)
        
        append!(m.ΔwList, mean(ΔwEpoch))
        append!(m.ΔwSDList, std(ΔwEpoch))
        append!(m.ΔaList, mean(ΔaEpoch))
        append!(m.ΔaSDList, std(ΔaEpoch))
        append!(m.ΔbList, mean(ΔbEpoch))
        append!(m.ΔbSDList, std(ΔbEpoch))
        append!(m.wMean, MatrixMean(J.w))
        append!(m.wVar, MatrixVar(J.w))
        append!(m.wTrMean, MatrixMean(J.w'))
        append!(m.wTrVar, MatrixVar(J.w'))

        @info string(now())[1:end-4], epoch, m.enData[end], m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β2
        logging ? flush(io) : nothing
        if epoch % snapshot == 0 
            savemodel ? saveModel(rbm, J, m, hparams; opt, path = dict["msg"], baseDir = dict["bdir"], epoch) : nothing
            genSample(rbm, J, hparams, m; num, t=t_samp, plotSample, epoch, dict, dev, TS)         
        end
        if annealing && epoch > 5 #0.5*epochs
            β2 = β2 + β0*ΔT
            # β = β2
        end
        
        x = reshuffle(x; hparams)
        if learnType == "Rdm"
            x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
        elseif learnType == "CD"
            x_Gibbs = x #reshuffle(x_Gibbs; hparams)
        elseif learnType == "PCD"
            x_Gibbs = reshuffle(x_Gibbs; hparams)
        elseif learnType == "Hybrid"
            lProtocol = ["Eigen", "CD"][sign(mod(epoch,3))+1]
            x_Gibbs = x
        elseif learnType in ["Eigen", "EigenCD"]
            x_Gibbs = x
        end
        
    end
    rbm, J, m, hparams, opt
end

# function train(dict ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, β2 = 1.0, learnType="Rdm", gpu_usage = false, t_samp=100, num=25, optType="SGD", numbers=[0,1], snapshot=50, savemodel=true, γ=0.001, logging=false, io=nothing)
#     try
#         Int(sqrt(num))
#     catch
#         @warn "num's root needs to be an integer."
#         return 0,0,0,0,0
#     end
    
#     rbm, J, m, hparams, rbmZ = initModel(; nv, nh, batch_size, lr, γ, t, gpu_usage, optType)
#     x, y = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
#     TS = Test(x,y)

#     if optType=="Adam"
#         opt = initOptW(hparams, J) 
#     elseif optType=="SGD"
#         opt = 0
#     end
#     dev = selectDev(hparams)
    
#     x = loadData(; hparams, dsName="MNIST01", numbers)
    
#     if learnType == "Rdm"
#         x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
#         lProtocol = learnType
#     elseif learnType in ["CD", "PCD"]
#         x_Gibbs = x
#         lProtocol = learnType
#     elseif learnType == "Hybrid"
#         x_Gibbs = x
#         lProtocol = "CD"
#     elseif learnType in ["Eigen", "EigenCD", "CQA"]
#         x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
#         lProtocol = learnType
#     end
    
    
#     if annealing
#         β0 = β2
#         if "tout" in keys(dict)
#             ΔT = (1/(β0*dict["tout"]) - 1)*1/(epochs-5)      #(T₀/Tₙ - 1)/epochs
#         else
#             ΔT = (1/(β0*10) - 1)*1/(epochs-5)      #(T₀/Tₙ - 1)/epochs
#         end
#         # β = β2
#     end 

#     genSample(rbm, J, hparams, m; num, β, β2, t=t_samp, plotSample, epoch=0, dict, dev, TS) 
#     for epoch in 1:epochs
#         enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch, ZEpoch = [], [], [], [], []
        
#         for i in eachindex(x)
#             # @info i
            
#             Δw, Δa, Δb = loss(rbm, J, x[i], x_Gibbs[i]; hparams, β, β2, dev, lProtocol, bw=dict["bw"])
#             if learnType == "PCD"
#                 x_Gibbs[i] = rbm.v |> cpu
#             end
            
#             updateJ!(J, Δw, Δa, Δb, opt; hparams)

#             append!(enEpoch, avgEn(rbm,J, β2) |> cpu)
#             append!(ZEpoch, sum(exp.(- β2 .* H(rbm, J))) |> cpu)
#             append!(ΔwEpoch, mean(Δw) |> cpu)
#             append!(ΔaEpoch, mean(Δa) |> cpu)
#             append!(ΔbEpoch, mean(Δb) |> cpu)
#         end
#         if sign(sum(isnan.(J.w))) == 1
#             @warn "Weights are NaN. Training will be stopped"
#             break
#         end

#         a,b = EnRBM(J, hparams, β2; dev)
#         append!(m.enData, mean(enEpoch))
#         append!(m.enDataSD, std(enEpoch))
#         append!(m.enRBM, a)
#         append!(m.enSP, saddlePointEnergy(J, hparams; dev))
#         append!(m.Zdata, mean(ZEpoch))
#         append!(m.Zrbm, b)
#         append!(m.T, 1/β2)
        
#         append!(m.ΔwList, mean(ΔwEpoch))
#         append!(m.ΔwSDList, std(ΔwEpoch))
#         append!(m.ΔaList, mean(ΔaEpoch))
#         append!(m.ΔaSDList, std(ΔaEpoch))
#         append!(m.ΔbList, mean(ΔbEpoch))
#         append!(m.ΔbSDList, std(ΔbEpoch))
#         append!(m.wMean, MatrixMean(J.w))
#         append!(m.wVar, MatrixVar(J.w))
#         append!(m.wTrMean, MatrixMean(J.w'))
#         append!(m.wTrVar, MatrixVar(J.w'))

#         @info string(now())[1:end-4], epoch, m.enData[end], m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β2
#         logging ? flush(io) : nothing
#         if epoch % snapshot == 0 
#             savemodel ? saveModel(rbm, J, m, hparams; opt, path = dict["msg"], baseDir = dict["bdir"], epoch) : nothing
#             genSample(rbm, J, hparams, m; num, β, β2, t=t_samp, plotSample, epoch, dict, dev, TS)         
#         end
#         if annealing && epoch > 5 #0.5*epochs
#             β2 = β2 + β0*ΔT
#             # β = β2
#         end
        
#         x = reshuffle(x; hparams)
#         if learnType == "Rdm"
#             x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
#         elseif learnType == "CD"
#             x_Gibbs = x #reshuffle(x_Gibbs; hparams)
#         elseif learnType == "PCD"
#             x_Gibbs = reshuffle(x_Gibbs; hparams)
#         elseif learnType == "Hybrid"
#             lProtocol = ["Eigen", "CD"][sign(mod(epoch,3))+1]
#             x_Gibbs = x
#         elseif learnType in ["Eigen", "EigenCD"]
#             x_Gibbs = x
#         end
        
#     end
#     rbm, J, m, hparams, opt
# end

function reshuffle(x; hparams)
    cat_state = cat(x..., dims=2)
    idx = randperm(size(cat_state,2))
    new_state = cat_state[:,idx]
    [new_state[:,i] for i in Iterators.partition(1:size(new_state,2), hparams.batch_size)]
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end