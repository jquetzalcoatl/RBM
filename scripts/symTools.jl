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

function div_(p::Vector{Float64},q::Vector{Float64}, ϵ::Float64=1e-9)
    #KL
    kl = sum(p .* log.((p .+ ϵ) ./ (q .+ ϵ)))
    # JS
    js = 0.5 * (sum(p .* log.(2 .* (p .+ ϵ) ./ (q .+ p .+ ϵ))) + sum(q .* log.(2 .* (q .+ ϵ) ./ (p .+ q .+ ϵ))))
    return kl, js
end