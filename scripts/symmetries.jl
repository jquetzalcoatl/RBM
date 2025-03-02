using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(0)

include("../utils/init.jl")
include("../scripts/PhaseAnalysis.jl")

PATH = "/home/javier/Projects/RBM/NewResults/"

modelName = config.model_analysis["files"][1]
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

@time m2 = rotmnd_optimized(Diagonal(ones(700))[:,3:700],π)

##############
rbm, J, m, hparams, opt = loadModel("PCD-500-replica1", dev, idx=100);

v,h = gibbs_sample(J, hparams, 200,2000)

lnum=10
mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
plot(heatmap(cpu(mat_rot)))

F = LinearAlgebra.svd(J.w, full=true);
x = cpu(F.U' * v)
y = cpu(F.Vt * h);

Mrot = Matrix{Float64}(I, hparams.nv, hparams.nv)
co = 150
for j in 1:100
    @info "\t" j
    indxs = vcat(collect(1:co),shuffle(collect(co+1:hparams.nv))[1:hparams.nv-(2+co)])
    Mrot = Mrot*rotmnd_optimized(Diagonal(ones(hparams.nv))[:,indxs],π)
end
for j in 10:2:100
    @info "\t" j
    Mrot = Mrot * rotmnd_optimized(Diagonal(ones(hparams.nv))[:,vcat(collect(1:j), collect(j+3:784))],π/2)
end
Mrot = rotmnd_optimized(Diagonal(ones(hparams.nv))[:,3:end],π)
Σ = cat(cpu(Diagonal(F.S)), (hparams.nv - hparams.nh > 0 ? zeros(abs(hparams.nv - hparams.nh), hparams.nh) : zeros(hparams.nv, abs(hparams.nv - hparams.nh))), 
    dims=(hparams.nv - hparams.nh > 0 ? 1 : 2) )
v_loop = σ.(Mrot * cpu(F.U) * Mrot' * Σ * y .+ cpu(J.a));
v_loop = Array{Float32}(sign.(rand(hparams.nv, size(v,2)) .< v_loop));
# v_loop = sign.(Mrot * cpu(F.U) * Mrot' * x)

mat = cat([cat([reshape(v_loop[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
plot(heatmap(cpu(mat_rot)))


###Test set
#Challenge 0
rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
v_test = x_i[:,1:400] |> dev
_, h_test = gibbs_sample(v_test, J, hparams, size(v_test,2), 1)

mat = cat([cat([reshape(v_test[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
plot(heatmap(cpu(mat_rot)))

#Challenge 1
F = LinearAlgebra.svd(J.w, full=true);
x = cpu(F.U' * v_test)
y = cpu(F.Vt * h_test);

Σ = cat(cpu(Diagonal(F.S)), (hparams.nv - hparams.nh > 0 ? zeros(abs(hparams.nv - hparams.nh), hparams.nh) : zeros(hparams.nv, abs(hparams.nv - hparams.nh))), 
    dims=(hparams.nv - hparams.nh > 0 ? 1 : 2) )
v_back = σ.(cpu(F.U) * Σ * y .+ cpu(J.a));
v_back = Array{Float32}(sign.(rand(hparams.nv, size(v_back,2)) .< v_back));
mat = cat([cat([reshape(v_back[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
plot(heatmap(cpu(mat_rot)))

Mrot = Matrix{Float64}(I, hparams.nv, hparams.nv)
co = 10
for j in 1:100
    @info "\t" j
    indxs = vcat(collect(1:co),shuffle(collect(co+1:hparams.nv))[1:hparams.nv-(2+co)])
    Mrot = Mrot*rotmnd_optimized(Diagonal(ones(hparams.nv))[:,indxs],π)
end
Mrot = rotmnd_optimized(Diagonal(ones(hparams.nv))[:,3:end],π)
Σ = cat(cpu(Diagonal(F.S)), (hparams.nv - hparams.nh > 0 ? zeros(abs(hparams.nv - hparams.nh), hparams.nh) : zeros(hparams.nv, abs(hparams.nv - hparams.nh))), 
    dims=(hparams.nv - hparams.nh > 0 ? 1 : 2) )
v_back = sign.(Mrot * cpu(F.U) * Mrot' * x)
mat = cat([cat([reshape(v_back[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
plot(heatmap(cpu(mat_rot)))


#############
μ = zeros(784, 100)
for i in 1:100
    rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=800, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    F = LinearAlgebra.svd(J.w, full=true);

    μ[:,i] = sum(cpu(F.U)', dims=2)[:,1]
end
plot(mean(μ,dims=2)[:,1], ribbon=std(μ,dims=2)[:,1])

rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=800, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
F = LinearAlgebra.svd(J.w, full=true);
samp = zeros(784,5000)
for i in 1:5000
   samp[:,i] = cpu(F.U)' * rand([0,1],784) 
end
plot(samp[1,:], st=:hist)
mean(samp, dims=2)