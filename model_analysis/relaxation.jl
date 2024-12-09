begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    using BSplineKit
    using QuadGK
    using FFTW, CurveFit
    CUDA.device_reset!()
    CUDA.device!(1)
    Threads.nthreads()
end

include("../utils/train.jl")

include("../scripts/exact_partition.jl")
include("../scripts/gaussian_partition.jl")
include("../scripts/gaussian_orth_partition.jl")
include("../scripts/RAIS.jl")



# Random.seed!(1234);
# rbm, J, m, hparams, rbmZ = initModel(nv=5, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

begin
    # include("../therm.jl")
    include("../configs/yaml_loader.jl")
    PATH = "/home/javier/Projects/RBM/Results/"
    # dev = gpu
    # β = 1.0
    config, _ = load_yaml_iter();
end
config.model_analysis["files"]

modelName = "PCD-FMNIST-500-replica1-L" #config.model_analysis["files"][1]
# modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
# modelName = config.model_analysis["files"][6]
modelName = "Random-RBM"
rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);
PATH="/home/javier/Projects/RBM/NewResults/$modelName/"
isdir(PATH) ? (@info "Directory exists") : mkdir(PATH)

F = LinearAlgebra.svd(J.w, full=true)
plot(F.S)

# rbm, J, m, hparams, rbmZ = initModel(nv=1784, nh=1784, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
rbm, J, m, hparams, rbmZ = initModel(nv=10, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
# J.w = J.w * 100
J.w = F.U * gpu(vcat(diagm(F.S * 10), zeros(hparams.nv-hparams.nh,hparams.nh))) * F.Vt

plot(Array(reshape(J.w,:)), st=:histogram)

###########Corr
function gibbs_sample(v, J, dev)
    h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(J.w' * v .+ J.b))) |> dev
    v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(J.w * h .+ J.a))) |> dev
    return v, h
end

function self_correlation(arr, t, τ)
    μ = mean(arr, dims=1)
    C = mean(arr[:,:,t+τ] .* arr[:,:,t], dims=1) .- μ[:,:,t+τ] .* μ[:,:,t]
    # mean(C, dims=1) / var(arr[:,:,t], dims=1)
    C ./ (var(arr[:,:,t], dims=1) .+ eps(eltype(arr)))
    # C ./ var(arr[:,:,t], dims=1)
end

function c_i_tau(arr,τ, steps=size(arr,3))
    CT = Array{Float32}(undef, size(1:steps-τ,1), size(arr,2))
    Threads.@threads for t in 1:steps-τ
        CT[t,:] = self_correlation(arr, t, τ)
    end
    mean(CT, dims=1)
end

function generate_samples(num_iterations, num, J, hparams, gpu, burnout=100, step=1)
    n_it = Int(floor(num_iterations/step))
    F = LinearAlgebra.svd(J.w, full=true);
    vs = Array{Float32,3}(undef, num, hparams.nv, n_it)
    hs = Array{Float32,3}(undef, num, hparams.nh, n_it)
    xs = Array{Float32,3}(undef, num, hparams.nv, n_it)
    ys = Array{Float32,3}(undef, num, hparams.nh, n_it)
    us = Array{Float32,3}(undef, num, hparams.nh, n_it)
    ws = Array{Float32,3}(undef, num, hparams.nh, n_it)

    v = rand([0,1], hparams.nv, num) |> gpu
    for i in 1:burnout
        v, h = gibbs_sample(v, J, gpu)
    end
    for i in 1:num_iterations
        v, h = gibbs_sample(v, J, gpu)
        if i % step == 0
            x, y = F.U' * v, F.Vt * h
            u, w = generate_uws(J, hparams, x, y, F)

            vs[:,:,Int(i/step)] = Array(transpose(v))
            hs[:,:,Int(i/step)] = Array(transpose(h))
            xs[:,:,Int(i/step)] = Array(transpose(x))
            ys[:,:,Int(i/step)] = Array(transpose(y))
            us[:,:,Int(i/step)] = Array(transpose(u))
            ws[:,:,Int(i/step)] = Array(transpose(w))
        end
    end
 
    return vs, hs, xs, ys, us, ws
end

function generate_uws(J, hparams,xs,ys, F)
    a0 = (F.U' * J.a)[1:hparams.nh]
    b0 = (F.Vt * J.b)
    x_sp = - b0 ./ F.S
    y_sp = - a0 ./ F.S
    u = 1/√2 .* ((xs[1:hparams.nh,:] + ys) .- x_sp .- y_sp)
    w = 1/√2 .* ((xs[1:hparams.nh,:] - ys) .- x_sp .+ y_sp)
    return u, w
end

function build_correlation_functions(vs, hs, tmax, Δt=1)
    c_dict = Dict()
    for (i,ar) in enumerate([vs, hs])
        @info i
        CT = Array{Float32}(undef, size(0:Δt:tmax,1), size(ar,2))
        for (j,τ) in enumerate(0:Δt:tmax-1)
        # time_steps = collect(0:Δt:tmax-1)
        # Threads.@threads for j in eachindex(time_steps)
            # τ = time_steps[j]
            @info τ
            CT[j,:] = c_i_tau(ar, τ)
        end
        c_dict[i] = CT
    end
    return c_dict  
end

num_iterations = 100  # replace with the number of iterations you want
num=100
burnout=5000
step=1
cor_step = 1
@time vs, hs, xs, ys, us, ws = generate_samples(num_iterations, num, J, hparams, gpu, burnout, step)

vs_n, hs_n = hparams.nv, hparams.nh #100,100
c_dict_vh = build_correlation_functions(vs[:,1:vs_n,:], hs[:,1:hs_n,:], Int(num_iterations/step), cor_step)
c_dict = build_correlation_functions(xs[:,1:vs_n,:], ys[:,1:hs_n,:], Int(num_iterations/step), cor_step)
c_dict_uw = build_correlation_functions(Array{Float64}(us[:,1:hs_n,:]), Array{Float64}(ws[:,1:hs_n,:]), Int(num_iterations/step), cor_step)

begin
    # Generate a gradient from blue to red
    color_gradient = range(colorant"red", stop=colorant"blue", length=size(c_dict[1], 2));
    # Plot each x-y curve with a color from the gradient
    p1 = plot(legend=false, lw=2, xlabel="Time (x$(step*cor_step))", ylabel="Correlation", title="Visible layer x", frame=:box, size=(700,500))
    for i in 1:size(c_dict[1], 2)
        p1 = plot!(c_dict[1][1:end-1, i], color=color_gradient[i], alpha=0.5, lw=2)
    end
    p1 = plot!(size=(700,500))

    p2 = plot(legend=false, lw=2, xlabel="Time (x$(step*cor_step))", ylabel="Correlation", title="Hidden layer y")
    for i in 1:size(c_dict[2], 2)
        p2 = plot!(c_dict[2][1:end-1, i], color=color_gradient[i], lw=2, alpha=0.5)
    end
    p2 = plot!(size=(700,500))

    # Plot each u-w curve with a color from the gradient
    p3 = plot(legend=false, lw=2, xlabel="Time (x$(step*cor_step))", ylabel="Correlation", title="Visible layer u", frame=:box, size=(700,500))
    for i in 1:size(c_dict_uw[1], 2)
        p3 = plot!(c_dict_uw[1][1:end-1, i], color=color_gradient[i], alpha=0.5, lw=2)
    end
    p3 = plot!(size=(700,500))

    p4 = plot(legend=false, lw=2, xlabel="Time (x$(step*cor_step))", ylabel="Correlation", title="Hidden layer w")
    for i in 1:size(c_dict_uw[2], 2)
        p4 = plot!(c_dict_uw[2][1:end-1, i], color=color_gradient[i], lw=2, alpha=0.5)
    end
    p4 = plot!(size=(700,500))

    # Plot each v-h curve with a color from the gradient
    p5 = plot(legend=false, lw=2, xlabel="Time (x$(step*cor_step))", ylabel="Correlation", title="Visible layer v", frame=:box, size=(700,500))
    for i in 1:size(c_dict_vh[1], 2)
        p5 = plot!(c_dict_vh[1][1:end-1, i], color=color_gradient[i], alpha=0.5, lw=2)
    end
    p5 = plot!(size=(700,500))

    p6 = plot(legend=false, lw=2, xlabel="Time (x$(step*cor_step))", ylabel="Correlation", title="Hidden layer h")
    for i in 1:size(c_dict_vh[2], 2)
        p6 = plot!(c_dict_vh[2][1:end-1, i], color=color_gradient[i], lw=2, alpha=0.5)
    end
    p6 = plot!(size=(700,500))

    p = plot(p1,p2,p3,p4,p5,p6, size=(1200,900), layout=(3,2))
    savefig(p, PATH * "correlation_visible_$modelName.png")
    p
end

begin
    v = permutedims(vs[:,:,end], [2,1])
    Array(transpose(v))
    lnum=Int(sqrt(100))
    mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
    mat_rot = reverse(transpose(mat), dims=1)
    f1 = heatmap(cpu(mat_rot), size=(900,900))
    savefig(f1, PATH * "Sample_$modelName.png")
    f1
end

####Char time extraction
function find_zero(ar, thrsh=0)
    counter = 0
    for el in ar
        if el > thrsh
            counter = counter + 1
        else
            break
        end
    end
    counter
end

function char_time(ar, n=2)
    t_max = find_zero(ar)
    itp = interpolate(1:size(ar,1), ar, BSplineOrder(n))
    i, e = quadgk(itp, 0, t_max, rtol=1e-3)
    i
end

function char_time_ft(ar, res=1)
    signal = ar
    ft = fft(signal)
    freq = - 0.5 * 1/minimum(imag.(ft)[1:Int(floor(size(ar,1)/2))]) * 1/res
    # @info imag.(ft)[1:Int(floor(size(ar,1)/2))]
    return 1/freq
end

begin
    idx = 10
    τ = char_time(c_dict_uw[1][:,idx])
    τ_ft = char_time_ft(c_dict_uw[1][:,idx])

    plot(c_dict_uw[1][1:end-1,1], label="1")
    plot!(c_dict_uw[1][1:end-1,idx], label="$idx")
    plot!(0:100, t->exp(-t/τ), lw=2, label="fit $τ")
    plot!(0:100, t->exp(-t/τ_ft), lw=2, label="fit ft $τ_ft")
    hline!([0], lw=2, color=:black)
end

begin
    p1 = plot(frame=:box, size=(700,500), title="Int over C(t)")
    τ = [char_time(c_dict_uw[1][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p1 = plot!(log.(τ ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="u")
    τ_2 = [char_time(c_dict[1][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p1 = plot!(log.(τ_2 ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="x", alpha=0.7)
        y = (log.(τ ./ cpu(F.S[1:hs_n])))
        x = findall(x->x!=-Inf, y)
        # b = y[x]\x
        a,b = linear_fit(x, y[x])
    p1 = plot!(1:size(y,1), λ-> a+b*λ , lw=3, c=:black, label=round(b, digits=3))

    p1b = plot(frame=:box, size=(700,500), title="FT")
    τ = [char_time_ft(c_dict_uw[1][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p1b = plot!(log.(τ ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0.1, label="u")
    τ_2 = [char_time_ft(c_dict[1][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p1b = plot!(log.(τ_2 ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0.1, label="x", alpha=0.7)
        y = (log.(τ ./ cpu(F.S[1:hs_n])))
        x = findall(x->x!=-Inf, y)
        a,b = linear_fit(x, y[x])
    p1b = plot!(1:size(y,1), λ-> a+b*λ , lw=3, c=:black, label=round(b, digits=3))
    

    p2 = plot(frame=:box, size=(700,500), title="Int over C(t)")   
    τ = [char_time(c_dict_uw[2][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p2 = plot!(log.(τ ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="w")
    τ_2 = [char_time(c_dict[2][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p2 = plot!(log.(τ_2 ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="y", alpha=0.7)
        y = (log.(τ ./ cpu(F.S[1:hs_n])))
        x = findall(x->x!=-Inf, y)
        a,b = linear_fit(x, y[x])
    p2 = plot!(1:size(y,1), λ-> a+b*λ , lw=3, c=:black, label=round(b, digits=3))

    p2b = plot(frame=:box, size=(700,500), title="FT")   
    τ = [char_time_ft(c_dict_uw[2][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p2b = plot!(log.(τ ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="w")
    τ_2 = [char_time_ft(c_dict[2][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p2b = plot!(log.(τ_2 ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="y", alpha=0.7)
        y = (log.(τ ./ cpu(F.S[1:hs_n])))
        x = findall(x->x!=-Inf, y)
        a,b = linear_fit(x, y[x])
    p2b = plot!(1:size(y,1), λ-> a+b*λ , lw=3, c=:black, label=round(b, digits=3))

    p3 = plot(frame=:box, size=(700,500), title="Int over C(t)")   
    τ = [char_time(c_dict_vh[1][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p3 = plot!(log.(τ ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="v")
    τ_2 = [char_time(c_dict_vh[2][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p3 = plot!(log.(τ_2 ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="h", alpha=0.7)
        y = (log.(τ ./ cpu(F.S[1:hs_n])))
        x = findall(x->x!=-Inf, y)
        a,b = linear_fit(x, y[x])
    p3 = plot!(1:size(y,1), λ-> a+b*λ , lw=3, c=:black, label=round(b, digits=3))

    p3b = plot(frame=:box, size=(700,500), title="FT")   
    τ = [char_time_ft(c_dict_vh[1][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p3b = plot!(log.(τ ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="v")
    τ_2 = [char_time_ft(c_dict_vh[2][1:end-1,idx]) for idx in 1:hs_n] .* (step * cor_step)
    p3b = plot!(log.(τ_2 ./ cpu(F.S[1:hs_n])), 
        marker=:circle, markerstrokewidth=0, label="h", alpha=0.7)
        y = (log.(τ ./ cpu(F.S[1:hs_n])))
        x = findall(x->x!=-Inf, y)
        a,b = linear_fit(x, y[x])
    p3b = plot!(1:size(y,1), λ-> a+b*λ , lw=3, c=:black, label=round(b, digits=3))

    p = plot(p1, p1b, p2, p2b, p3, p3b, layout=(3,2), size=(700,700), xlabel="index", ylabel="log(τ/λ)")
    savefig(p, PATH * "tau_lambda_ratio_$modelName.png")
    p
end

plot( log.([char_time(c_dict[1][:,idx]) for idx in 1:hs_n]), marker=:circle, markerstrokewidth=0)
plot(cpu(F.S), [char_time_ft(c_dict[1][1:end-1,idx]) for idx in 1:hs_n], xscale=:log, marker=:circle, markerstrokewidth=0)

plot(F.S, [char_time(c_dict_vh[1][:,idx]) for idx in 1:hs_n], xscale=:log, marker=:circle, markerstrokewidth=0)
# plot!([char_time_ft(c_dict[2][:,idx]) for idx in 1:hs_n], marker=:circle, markerstrokewidth=0)
# plot([char_time(c_dict_uw[1][:,idx]) for idx in 1:100])
# plot!([char_time_ft(c_dict_uw[1][:,idx]) for idx in 1:100])
# plot([char_time(c_dict_uw[2][:,idx]) for idx in 1:100])
# plot!([char_time_ft(c_dict_uw[2][:,idx]) for idx in 1:100])
# plot([char_time_ft(c_dict_vh[1][1:end-1,idx]) for idx in 1:100])
# plot!([char_time(c_dict_vh[1][1:end-1,idx]) for idx in 1:100])
# plot([char_time_ft(c_dict_vh[2][1:end-1,idx]) for idx in 1:100])
# plot!([char_time(c_dict_vh[2][1:end-1,idx]) for idx in 1:100])


128*3*3*3
using CurveFit
a,b = linear_fit(x, y[x])
y

y = (log.([char_time(c_dict_uw[1][1:end-1,idx]) for idx in 1:hs_n] ./ cpu(F.S[1:hs_n])))
x = findall(x->x!=-Inf, y)
b = x\y[x]

##########Simple example
signal = [ 0.5 * cos(15.3 * t) + 0.5 * cos(1.3 * t) + 1.0*rand() for t in 0:0.001:100]
plot(signal)
ft = fft(signal)

plot(real(ft))
plot(imag(ft))
argmax(imag.(ft)[1:250])
sortperm(imag.(ft)[1:250], rev=true)
2pi*21/100


signal = [ 1/(2.0)*(1.0 *exp(-15.3 * t) + 1.0 * exp(-1.3578 * t) + 0.001*rand()) for t in 0:0.001:100]
plot(signal)
ft = fft(signal)

plot(real(ft))
plot(imag(ft))
imag(ft)
# plot(abs.(ft))
# argmax(imag.(ft)[1:250])
- 0.5 * 1/minimum(imag(ft)[1:50000]) * 1/0.001
sortperm(imag.(ft)[1:250], rev=true)



################
#######################
#RW
function rw(v,h,J; mcs=5000, dev0=cpu, evry=10)
    β=1
    dev = gpu
    nh = size(h,1)
    nv = size(v,1)
    num= size(v,2)
    # v = gpu(v[:,num])
    # h = gpu(h[:,num])
    v = gpu(v)
    h = gpu(h)
    J.w = gpu(J.w)
    J.b = gpu(J.b)
    J.a = gpu(J.a)
    F = LinearAlgebra.svd(J.w, full=true);

    x = zeros(size(v,1),size(v,2),Int(floor(mcs/evry))+1)
    y = zeros(size(h,1),size(h,2),Int(floor(mcs/evry))+1)
    @info size(y)
    
    counter = 1
    for i in 1:mcs
        h = Array{Float32}(sign.(rand(nh, num) |> dev .< σ.(β .* (J.w' * v .+ J.b)))) |> dev
        
        v = Array{Float32}(sign.(rand(nv, num) |> dev .< σ.(β .* (J.w * h .+ J.a)))) |> dev 
        if i % evry == 0
            y[:,:,counter+1] = cpu(F.Vt * h)
            x[:,:,counter+1] = cpu(F.U' * v)
            counter = counter + 1
        end
    end
    return dev0(x),dev0(y), dev0(v),dev0(h)
end

# dims are variable x samples x time
JJ = initWeights(hparams)

begin
    mc_steps=2000
    save_every = 1
    ar_size = Int(mc_steps / save_every) + 1
    sample_size = 400
    x,y, v,h = rw(zeros(hparams.nv,sample_size),zeros(hparams.nh,sample_size),J; mcs=mc_steps, dev0=cpu, evry=save_every);
    # x,y, v,h = rw(rand([0,1],hparams.nv,sample_size),rand([0,1],hparams.nh,sample_size),J; mcs=mc_steps, dev0=cpu, evry=save_every);



    lnum=Int(sqrt(sample_size))
    mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
    mat_rot = reverse(transpose(mat), dims=1)
    f1 = heatmap(cpu(mat_rot), size=(900,900))
end

begin
    f2 = plot(reshape(mean(x,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="⟨x⟩", frame=:box, title=modelName, opacity=0.5);

    f3 = plot(reshape(std(x,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="std(x)", frame=true, opacity=0.5);

    f4 = plot(reshape(mean(y,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none, 
        xlabel="MC step (x $(save_every))", ylabel="⟨y⟩", frame=:box, opacity=0.5);

    f5 = plot(reshape(std(y,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="std(y)", frame=:box, opacity=0.5);

    # f6 = plot([kurtosis(x[i,:,j]) for i in 1:hparams.nv, j in 2:ar_size]', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
    # xlabel="MC step (x $(save_every))", ylabel="kutorsis(x)", frame=:box, opacity=0.5);

    # f7 = plot([kurtosis(y[i,:,j]) for i in 1:hparams.nh, j in 2:ar_size]', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
    # xlabel="MC step (x $(save_every))", ylabel="kutorsis(y)", frame=:box, opacity=0.5);

    plot(f1,f2,f3,f4,f5, size=(1200,900))
end

plot(reshape(mean(x,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="⟨x⟩", frame=:box, title=modelName, opacity=0.5);

plot(reshape(std(x,dims=2),:,ar_size)[1:20,1:100]', legend=false)