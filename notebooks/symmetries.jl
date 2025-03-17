using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(0)

include("../utils/init.jl")
include("../scripts/PhaseAnalysis.jl")

PATH = "/home/javier/Projects/RBM/NewResults/"

modelName = config.model_analysis["files"][6]
modelName = "CD-500-T1-BW-replica1"
modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = "PCD-100-replica1"
modelName = "PCD-MNIST-500-lr-replica2"
rbm, J, m, hparams, opt = loadModel(modelName, dev, idx=100);
rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
dict = loadDict(modelName)
x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);

#  Implementation of the Aguilera-Perez Algorithm.
#  Aguilera, Antonio, and Ricardo Pérez-Aguila. "General n-dimensional rotations." (2004).
# function rotmnd(v::AbstractMatrix, theta::Real)
#     n = size(v, 1)
#     # Create an n×n identity matrix; we convert it to a full matrix because we will update its entries.
#     M = Matrix{Float64}(I, n, n)
    
#     # Loop over columns 1 to n-2.
#     for c in 1:(n-2)
#         # Loop over rows from n down to c+1.
#         for r in n:-1:(c+1)
#             # Compute the rotation angle to zero out the element v[r, c].
#             # Note: Julia’s atan(y, x) returns the angle whose tangent is y/x, similar to MATLAB’s atan2.
#             t = atan(v[r, c], v[r-1, c])
            
#             # Build a rotation matrix R (n×n identity, then modify a 2×2 block)
#             R = Matrix{Float64}(I, n, n)
#             # Here we update the (r-1, r-1), (r-1, r), (r, r-1), and (r, r) entries.
#             # MATLAB uses the index order [r, r-1] for rows and columns; we mimic that order explicitly.
#             indices = [r, r-1]
#             R[indices, indices] = [cos(t) -sin(t); sin(t) cos(t)]
            
#             # Apply the rotation to v.
#             v = R * v
#             # Accumulate the rotation into M.
#             M = R * M
#         end
#     end

#     # Build the final rotation matrix R that rotates the last two dimensions by theta.
#     R = Matrix{Float64}(I, n, n)
#     indices = [n-1, n]
#     R[indices, indices] = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    
#     # The final operation is a similarity transformation: M = inv(M) * R * M.
#     # In MATLAB, M\R*M is equivalent to inv(M)*R*M.
#     M = inv(M) * R * M

#     return M
# end

# Helper function: rotates rows i and j of matrix A in place.
@inline function plane_rotate!(A::AbstractMatrix{Float64}, i::Int, j::Int, ct::Float64, st::Float64)
    n = size(A, 2)
    @inbounds for k in 1:n
        a = A[i, k]
        b = A[j, k]
        A[i, k] = ct * a - st * b
        A[j, k] = st * a + ct * b
    end
end

function rotmnd_optimized(v::AbstractMatrix{Float64}, theta::Real)
    n = size(v, 1)
    # Make sure v is square and theta is a real number.
    # @assert size(v, 1) == size(v, 2)
    
    # Initialize M as the identity matrix (accumulated rotations).
    M = Matrix{Float64}(I, n, n)
    
    # Loop over columns 1 to n-2.
    for c in 1:(n - 2)
        # Loop over rows from n down to c+1.
        for r in n:-1:(c + 1)
            # Compute the rotation angle that will zero out v[r, c].
            # Note: atan(y, x) returns the angle whose tangent is y/x.
            t = atan(v[r, c], v[r - 1, c])
            ct = cos(t)
            st = sin(t)
            
            # Update the affected rows in v (left multiplication by a Givens rotation).
            plane_rotate!(v, r - 1, r, ct, st)
            # Update the accumulated rotations M in the same way.
            plane_rotate!(M, r - 1, r, ct, st)
        end
    end
    
    # Construct the final rotation matrix R_final that rotates the last two dimensions by theta.
    R_final = Matrix{Float64}(I, n, n)
    R_final[n - 1, n - 1] = cos(theta)
    R_final[n - 1, n]     = -sin(theta)
    R_final[n, n - 1]     = sin(theta)
    R_final[n, n]         = cos(theta)
    
    # Perform the final similarity transformation.
    # Since M is a product of rotation matrices (hence orthogonal), we have inv(M) = Mᵀ.
    M_final = transpose(M) * R_final * M
    return M_final
end

@time m2 = rotmnd_optimized(Diagonal(ones(3000))[:,3:3000],π)

####################################
############## Hierarchical probing
modelName = config.model_analysis["files"]
rbm, J, m, hparams, opt = loadModel("PCD-500-replica1", dev, idx=100);
rbm, J, m, hparams, opt = loadModel("PCD-500-replica1", dev, idx=100);
rbm, J, m, hparams, opt = loadModel("CD-500-T1000-5-BW-replica1-L", dev, idx=1000);

v,h = gibbs_sample(J, hparams, 200,2000)

lnum=10
mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
fig = heatmap(cpu(mat_rot), legend=:none, ticks=:none, frame=:box, size=(550,500),
    c=cgrad(:buda, 2, rev=true, scale = :linear, categorical=true))
savefig(fig, PATH * "MS/sample_gibbs_rot0.pdf")

F = LinearAlgebra.svd(J.w, full=true);
x = cpu(F.U' * v)
y = cpu(F.Vt * h);

Mrot = Matrix{Float64}(I, hparams.nv, hparams.nv)
co = 5
for j in 1:10
    @info "\t" j
    indxs = vcat(collect(1:co),shuffle(collect(co+1:hparams.nv))[1:hparams.nv-(2+co)])
    Mrot = Mrot*rotmnd_optimized(Diagonal(ones(hparams.nv))[:,indxs],π)
end
for j in 10:2:20
    @info "\t" j
    Mrot = Mrot * rotmnd_optimized(Diagonal(ones(hparams.nv))[:,vcat(collect(1:j), collect(j+3:784))],π/2)
end
Mrot2 = rotmnd_optimized(Diagonal(ones(hparams.nv))[:,vcat([],collect(3:7),collect(8:hparams.nv))],π/2)

Σ = cat(cpu(Diagonal(F.S)), (hparams.nv - hparams.nh > 0 ? zeros(abs(hparams.nv - hparams.nh), hparams.nh) : zeros(hparams.nv, abs(hparams.nv - hparams.nh))), 
    dims=(hparams.nv - hparams.nh > 0 ? 1 : 2) )
v_loop = σ.(Mrot * cpu(F.U) * Mrot' * Σ * y .+ cpu(J.a));
v_loop = σ.(Mrot2 * cpu(F.U) * Mrot2' * Σ * y .+ cpu(J.a));
v_loop = Array{Float32}(sign.(rand(hparams.nv, size(v,2)) .< v_loop));


mat = cat([cat([reshape(v_loop[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
fig = heatmap(cpu(mat_rot), legend=:none, ticks=:none, frame=:box, size=(550,500),
    c=cgrad(:buda, 2, rev=true, scale = :linear, categorical=true))
savefig(fig, PATH * "MS/sample_gibbs_rot1and2.pdf")

################################
####################################
#############Marcenko stuff
λp(N,M,σ) = √(N*σ^2) + √(M*σ^2)
λm(N,M,σ) = √(N*σ^2) - √(M*σ^2)
ρ(λ,λp,λm,q) = 1/(π*q*λ) * √((λp^2 - λ^2)*(λ^2 - λm^2))



Λ = zeros(100,num,50)
for j in 0:49
    num=200
    rbm, J, m, hparams, rbmZ = initModel(nv=100, nh=100-j, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    μ = zeros(hparams.nv+hparams.nh, num)

    for i in 1:num
        rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
        F = LinearAlgebra.svd(J.w, full=true);
        μ[:,i] = vcat(sum(cpu(F.U)', dims=2)[:,1], sum(cpu(F.Vt), dims=2)[:,1])
        Λ[:,i,j+1] = vcat(cpu(F.S),zeros(max(hparams.nv,hparams.nh)-min(hparams.nv,hparams.nh)))
    end
end

num=20
rbm, J, m, hparams, rbmZ = initModel(nv=3000, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
Λ = zeros(min(hparams.nv,hparams.nh),num)
for i in 1:num
    rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    # J.w = J.w*100
    F = LinearAlgebra.svd(J.w, full=true);
    Λ[:,i] = cpu(F.S)
end

fig = plot(reshape(Λ,:), st=:histogram, normalize=true, bins=100, 
    linewidth=0, frame=:box, label=:none, xlabel="λ", ylabel="ρ(λ)",
    tickfontsize=15, labelfontsize=15, legendfontsize=15, size=(700,500),
    left_margin=3mm, bottom_margin=2mm, color=:magenta)
# vline!([sqrt(3000-784)*10^(-2)])
# q = hparams.nh/10^4
# λp = (√(hparams.nv/10000) + √q)
# λm = (√(hparams.nv/10000) - √q)

Jstd = 10^(-2)

# ρ(0.4,λp(hparams.nv,hparams.nh,Jstd),λm(hparams.nv,hparams.nh,Jstd),(hparams.nh*Jstd^(2)))
plot!(λm(hparams.nv,hparams.nh,Jstd):0.0001:λp(hparams.nv,hparams.nh,Jstd), 
    x-> ρ(x,λp(hparams.nv,hparams.nh,Jstd),λm(hparams.nv,hparams.nh,Jstd),(hparams.nh*Jstd^(2))), 
    color=:black, label=:none, lw=7)
# √((λp^2-x^2)*(x^2-λm^2))/(π*q*x)

savefig(fig, PATH * "MS/rho_lambda_random_M$(hparams.nv).pdf")

modelName = config.model_analysis["files"]
modelSize=[500,784,784,784,500]
lambda_models = Dict()
for (j,lab) in enumerate(["PCD-500-replica","PCD-500-784-replica", "PCD-500-1200-replica","PCD-500-3000-replica", "CD-500-T1000-5-BW-replica"])
    Λ = zeros(modelSize[j],5)
    if contains(lab, "PCD")
        for i in 1:5
            rbm, J, m, hparams, opt = loadModel(lab*"$i", dev, idx=100);
            F = LinearAlgebra.svd(J.w, full=true);
            Λ[:,i] = cpu(F.S)
        end
    else
        for i in 1:5
            rbm, J, m, hparams, opt = loadModel(lab*"$i-L", dev, idx=1000);
            F = LinearAlgebra.svd(J.w, full=true);
            Λ[:,i] = cpu(F.S)
        end
    end
    lambda_models[lab] = Λ
end

lab = ["500 PCD","784 PCD", "1200 PCD","3000 PCD", "500 CD + annealing"]
fig = plot()
for (i,key) in enumerate(sort(collect(keys(lambda_models)), rev=true))
    fig = plot!(reshape(lambda_models[key],:), st=:histogram, normalize=true, bins=range(0,41,40), 
    linewidth=0.1, yscale=:log10, frame=:box, alpha=1., label=lab[i], xlabel="λ", 
    ylabel="ρ(λ)", tickfontsize=15, labelfontsize=15, legendfontsize=15, size=(700,500),
    left_margin=3mm, bottom_margin=2mm, palette=:seaborn_bright)
end
fig = plot!(legend = :topright)

savefig(fig, PATH * "MS/rho_lambda_trained_models.pdf")

keys(lambda_models)
plot()
for (i,key) in enumerate(sort(collect(keys(lambda_models)), rev=true))
    plot!((- sum(lambda_models[key] .* 
        log.(lambda_models[key]), dims=1))[:], linewidth=4.1, frame=:box, label=key, xlabel="replica", 
        ylabel="Weight matrix entropy", tickfontsize=15, labelfontsize=15, legendfontsize=15, size=(700,500),
        left_margin=3mm, bottom_margin=2mm)
end
plot!()

s_mat = []
for s in [500,784,1200,3000,5000]
    rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=s, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    F = LinearAlgebra.svd(J.w, full=true);
    append!(s_mat, -sum(cpu(F.S) .* log.(cpu(F.S))))
end

sort(collect(keys(lambda_models)), rev=true)
plot([500,784,1200,3000], [mean(- mean(lambda_models[key] .* log.(lambda_models[key]), dims=1))
     for key in sort(collect(keys(lambda_models)), rev=true)][[1,2,4,3]])
plot([500,784,1200,3000,5000], s_mat)


#####################
################################
############rotational symmetry

rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
rbm, J, m, hparams, opt = loadModel("PCD-500-3000-replica1", dev, idx=100);
F = LinearAlgebra.svd(J.w, full=true);
begin
    MrotU = Matrix{Float64}(I, hparams.nv, hparams.nv)
    co = 0
    for j in 1:Int(floor(hparams.nv*0.1))
        @info "\t" j
        indxs = vcat(collect(1:co),shuffle(collect(co+1:hparams.nv))[1:hparams.nv-(2+co)])
        MrotU = MrotU*rotmnd_optimized(Diagonal(ones(hparams.nv))[:,indxs],2π*rand())
    end
    MrotV = Matrix{Float64}(I, hparams.nh, hparams.nh)
    co = 0
    for j in 1:Int(floor(hparams.nh*0.1))
        @info "\t" j
        indxs = vcat(collect(1:co),shuffle(collect(co+1:hparams.nh))[1:hparams.nh-(2+co)])
        MrotV = MrotV*rotmnd_optimized(Diagonal(ones(hparams.nh))[:,indxs],2π*rand())
    end
end
# MrotU = rotmnd_optimized(Diagonal(ones(hparams.nv))[:,3:end],π)
# MrotV = rotmnd_optimized(Diagonal(ones(hparams.nh))[:,3:end],π)
fig = plot(reshape(cpu(J.w),:), st=:histogram, lw=0, normalize=true, 
    yscale=:log10, color=:magenta, label="Before rotation")
Σ = cat(cpu(Diagonal(F.S)), (hparams.nv - hparams.nh > 0 ? zeros(abs(hparams.nv - hparams.nh), hparams.nh) : zeros(hparams.nv, abs(hparams.nv - hparams.nh))), 
    dims=(hparams.nv - hparams.nh > 0 ? 1 : 2) )
fig = plot!(cpu(reshape(MrotU*cpu(F.U)*MrotU'*Σ*MrotV*cpu(F.Vt)*MrotV,:)), 
    st=:histogram, color=:blue, lw=0, normalize=true, yscale=:log10, alpha=0.5, label="After rotation")
fig = plot!(xlabel="Weights", ylabel="PDF", 
    tickfontsize=15, labelfontsize=15, legendfontsize=15, 
    frame=:box, size=(700,500), left_margin=3mm, bottom_margin=2mm, legend = :topleft)
savefig(fig, PATH * "MS/trained_weights_rot_1200.png")
savefig(fig, PATH * "MS/random_weights_rot.png")


############Some featured plots
num = 10000
rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
F = LinearAlgebra.svd(J.w, full=true);
samp_x_untr = zeros(784,num)
samp_y_untr = zeros(500,num)
for i in 1:num
    samp_x_untr[:,i] = cpu(F.U)' * rand([0,1],784) 
    samp_y_untr[:,i] = cpu(F.Vt) * rand([0,1],500) 
end
μ_x_untr = reshape(sum(cpu(F.U'), dims=2)/2,:)
μ_y_untr = reshape(sum(cpu(F.Vt), dims=2)/2,:)

rbm, J, m, hparams, opt = loadModel("PCD-500-replica1", dev, idx=100);
F = LinearAlgebra.svd(J.w, full=true);
samp_x_tr = zeros(784,num)
samp_y_tr = zeros(500,num)
for i in 1:num
    samp_x_tr[:,i] = cpu(F.U)' * rand([0,1],784) 
    samp_y_tr[:,i] = cpu(F.Vt) * rand([0,1],500) 
end
μ_x_tr = reshape(sum(cpu(F.U'), dims=2)/2,:)
μ_y_tr = reshape(sum(cpu(F.Vt), dims=2)/2,:)

s = ones(784)/2
plot_list = []
for i in [1,2,3,4,450,784]
    # p = plot(samp_x_tr[i,:], st=:hist, frame=:box, linewidth=0, bins=60, normalize=true, color=:magenta, label="Trained", xlabel="xᵢ")
    p = plot(samp_x_untr[i,:], st=:hist, frame=:box, linewidth=0, bins=60, normalize=true, color=:blue, opacity=0.5, label="Untrained")
    # p = plot!(-4*s[i]+μ_x_tr[i]:0.01:4*s[i]+μ_x_tr[i], x-> 1/√(2π*s[i]^2)*exp(-(x-μ_x_tr[i])^2/(2*s[i]^2)), lw=3, color=:black, label="N(μᵢ,σᵢ)")
    p = plot!(-4*s[i]+μ_x_untr[i]:0.01:4*s[i]+μ_x_untr[i], x-> 1/√(2π*s[i]^2)*exp(-(x-μ_x_untr[i])^2/(2*s[i]^2)), lw=3, color=:red, label="N(μᵢ,σᵢ)")
    push!(plot_list, p)
end
for i in [1,2,3,4,450,500]
    p = plot(samp_y_tr[i,:], st=:hist, frame=:box, linewidth=0, bins=60, normalize=true, color=:magenta, label="Trained", xlabel="yᵢ")
    p = plot!(samp_y_untr[i,:], st=:hist, frame=:box, linewidth=0, bins=60, normalize=true, color=:blue, opacity=0.5, label="Untrained")
    p = plot!(-4*s[i]+μ_y_tr[i]:0.01:4*s[i]+μ_y_tr[i], x-> 1/√(2π*s[i]^2)*exp(-(x-μ_y_tr[i])^2/(2*s[i]^2)), lw=3, color=:black, label="N(μᵢ,σᵢ)")
    p = plot!(-4*s[i]+μ_y_untr[i]:0.01:4*s[i]+μ_y_untr[i], x-> 1/√(2π*s[i]^2)*exp(-(x-μ_y_untr[i])^2/(2*s[i]^2)), lw=3, color=:red, label="N(μᵢ,σᵢ)")
    push!(plot_list, p)
end
fig = plot(plot_list..., layout=(2,3), size=(900,900), ylabel="PDF", left_margin=3mm, bottom_margin=1mm)
savefig(fig, PATH * "MS/PDF_rand_trained.pdf")

v,h = gibbs_sample(J, hparams, 1000,1000)
samp_x_tr_g = cpu(F.U)' * cpu(v)
samp_y_tr_g = cpu(F.Vt) * cpu(h)

plot_list = []
for i in [1,2,3,4,450,500]
    p = plot(samp_x_tr[i,:], samp_y_tr[i,:], marker=:circle, markersize=7, lw=0.0, markerstrokewidth=0.1, 
        color=:magenta, label="Random Samples", xlabel="xᵢ", ylabel="yᵢ")
    p = plot!(samp_x_tr_g[i,:], samp_y_tr_g[i,:], marker=:circle, markersize=7, lw=0.0, markerstrokewidth=0.1,
        color=:cyan, label="Gibbs Samples", frame=:box)
    push!(plot_list, p)
end
fig = plot(plot_list..., layout=(2,3), size=(900,500), left_margin=3mm, bottom_margin=2mm)
savefig(fig, PATH * "MS/Scatter_plot_rand_trained.pdf")

###########kurtosis
function get_krts_random_rbm_v2(num::Int, v_size::Array{Int})
    # num = 10000
    k_gauss_dict = Dict()
    k_x_dict = Dict()
    k_y_dict = Dict()

    for s in v_size
        @info s
        rbm, J, m, hparams, rbmZ = initModel(nv=s, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")
        J.w = cpu(J.w)
        F = LinearAlgebra.svd(J.w, full=true);
        samp_x_untr = zeros(hparams.nv,num)
        samp_y_untr = zeros(hparams.nh,num)
        for i in 1:num
            samp_x_untr[:,i] = F.U' * rand([0,1],hparams.nv) 
            samp_y_untr[:,i] = F.Vt * rand([0,1],hparams.nh) 
        end
        # z_gauss = randn(hparams.nv,num)

        # krts_gauss = reshape( mean( (z_gauss .- mean(z_gauss, dims=2)) .^4, dims=2 ) ./ (mean( (z_gauss .- mean(z_gauss, dims=2)) .^2, dims=2 )) .^ 2, :)
        krts_x = reshape( mean( (samp_x_untr .- mean(samp_x_untr, dims=2)) .^4, dims=2 ) ./ (mean( (samp_x_untr .- mean(samp_x_untr, dims=2)) .^2, dims=2 )) .^ 2, :)
        krts_y = reshape( mean( (samp_y_untr .- mean(samp_y_untr, dims=2)) .^4, dims=2 ) ./ (mean( (samp_y_untr .- mean(samp_y_untr, dims=2)) .^2, dims=2 )) .^ 2, :)

        # k_gauss_dict[string(s)] = Dict("mean" => mean(krts_gauss), "std" => std(krts_gauss))
        k_x_dict[string(s)] = Dict("mean" => mean(krts_x[1:hparams.nh]), "std" => std(krts_x[1:hparams.nh]))
        k_y_dict[string(s)] = Dict("mean" => mean(krts_y), "std" => std(krts_y))
    end

    return k_x_dict, k_y_dict #k_gauss_dict, 
end

function get_krts_random_rbm(num::Int, v_size::Array{Int})
    # num = 10000
    k_gauss_dict = Dict()
    k_x_dict = Dict()
    k_y_dict = Dict()

    for s in v_size
        @info s
        rbm, J, m, hparams, rbmZ = initModel(nv=s, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")
        J.w = cpu(J.w)
        F = LinearAlgebra.svd(J.w, full=true);
        samp_x_untr = F.U' * rand([0,1],hparams.nv,num)
        samp_y_untr = F.Vt * rand([0,1],hparams.nh,num)
        # for i in 1:num
        #     samp_x_untr[:,i] = F.U' * rand([0,1],hparams.nv) 
        #     samp_y_untr[:,i] = F.Vt * rand([0,1],hparams.nh) 
        # end
        # z_gauss = randn(hparams.nv,num)

        # krts_gauss = reshape( mean( (z_gauss .- mean(z_gauss, dims=2)) .^4, dims=2 ) ./ (mean( (z_gauss .- mean(z_gauss, dims=2)) .^2, dims=2 )) .^ 2, :)
        krts_x = reshape( mean( (samp_x_untr .- mean(samp_x_untr, dims=2)) .^4, dims=2 ) ./ (mean( (samp_x_untr .- mean(samp_x_untr, dims=2)) .^2, dims=2 )) .^ 2, :)
        krts_y = reshape( mean( (samp_y_untr .- mean(samp_y_untr, dims=2)) .^4, dims=2 ) ./ (mean( (samp_y_untr .- mean(samp_y_untr, dims=2)) .^2, dims=2 )) .^ 2, :)

        # k_gauss_dict[string(s)] = Dict("mean" => mean(krts_gauss), "std" => std(krts_gauss))
        k_x_dict[string(s)] = Dict("mean" => mean(krts_x[1:hparams.nh]), "std" => std(krts_x[1:hparams.nh]))
        k_y_dict[string(s)] = Dict("mean" => mean(krts_y), "std" => std(krts_y))
    end

    return k_x_dict, k_y_dict #k_gauss_dict, 
end

l = [500,700,784,1200,2000]
# @time k_x_dict, k_y_dict = get_krts_random_rbm(10000,l)
@time k_x_dict, k_y_dict = get_krts_random_rbm_v2(7000000,l)
l = [2000]
@time k_x_dict_2, k_y_dict_2 = get_krts_random_rbm_v2(7000000,l)
k_x_dict, k_y_dict = k_x_dict_2, k_y_dict_2

k_x_dict["2000"] = k_x_dict_2["2000"]

fig = plot([1/key for key in l], [k_x_dict[string(key)]["mean"] for key in l], 
    ribbon=[k_x_dict[string(key)]["std"] for key in l], lw=0, marker=:circle, markersize=15,
    markerstrokewidth=0, label="RBM", color=:magenta)
# plot!([1/key for key in l], [k_gauss_dict[string(key)]["mean"] for key in l], 
    # ribbon=[k_gauss_dict[string(key)]["std"] for key in l])
# plot!([1/key for key in l], [k_y_dict[string(key)]["mean"] for key in l], 
    # ribbon=[k_y_dict[string(key)]["std"] for key in l])
fig = hline!([3],lw=3,ls=:dash, color=:black, label="Gaussian")
fig = plot!(0.:0.0001:0.002, x->fit.b + fit.a*x, label="Linear Fit", lw=3, ribbon=1e-4, color=:blue)
fig = plot!(xlabel="1/N", ylabel="Kurtosis", tickfontsize=15, labelfontsize=15, legendfontsize=15, 
    frame=:box, size=(700,650), left_margin=3mm, right_margin=5mm, bottom_margin=2mm, legend = :topright)

savefig(fig, PATH * "MS/Kurtosis_vs_RBMSize.pdf")



using EasyFit, LaTeXStrings

fit = fitlinear([1/key for key in l],[k_x_dict[string(key)]["mean"] for key in l])
fit.ypred
fit.residues