using LinearAlgebra
using Distributions
using Random
using BenchmarkTools
using Plots
using DataFrames


n=2000
matrix_1 = fill(0.9,(n,n))
for i in 1:n
    matrix_1[i,i]=1.0
end
matrix_1[1,2] = 0.7357
matrix_1[2,1] = 0.7357


@time begin


#Near PSD Matrix
function near_psd(a; epsilon=0.0)
    n = size(a,1)

    invSD = nothing
    out = copy(a)

    #calculate the correlation matrix if we got a covariance
    if count(x->x ≈ 1.0,diag(out)) != n
        invSD = diagm(1 ./ sqrt.(diag(out)))
        out = invSD * out * invSD
    end

    #SVD, update the eigen value and scale
    vals, vecs = eigen(out)
    vals = max.(vals,epsilon)
    T = 1 ./ (vecs .* vecs * vals)
    T = diagm(sqrt.(T))
    l = diagm(sqrt.(vals))
    B = T*vecs*l
    out = B*B'

    #Add back the variance
    if invSD !== nothing 
        invSD = diagm(1 ./ diag(invSD))
        out = invSD * out * invSD
    end
    return out
end


psd_matrix = near_psd(matrix_1)
eigenvals = eigen(Matrix(psd_matrix)).values

end

function print_smallest_eigenvalue(eigenvals)
    println("The smallest eigenvalue is: ", minimum(eigenvals))
end
print_smallest_eigenvalue(eigenvals)


n=2000
matrix_2 = fill(0.9,(n,n))
for i in 1:n
    matrix_2[i,i]=1.0
end
matrix_2[1,2] = 0.7357
matrix_2[2,1] = 0.7357



function Higham(A, tolerance=1e-8)
    delta_s = fill(0, size(A))
    Y = copy(A)
    gamma_last = typemax(Float64)
    gamma_now = 0
    i = 1

    while true
        R = Y - delta_s
        
        Rval, Rvec = eigen(R)
        Rval_real = real(Rval)
        Rval_real[Rval_real .< 0] .= 0
        Rval = ComplexF64.(Rval_real) # change this line
        Rvec_transpose = transpose(Rvec)
        
        X = Rvec * Diagonal(Rval) * Rvec_transpose
        
        delta_s = X - R
        
        size_X = size(X)
        for i in 1:size_X[1]
            for j in 1:size_X[2]
                Y[i,j] = X[i,j]
            end
        end
        
        difference_mat = Y - A
        gamma_now = norm(difference_mat, 2)
        
        Yval, Yvec = eigen(Y)
        
        Yval_real = real(Yval) # change this line
        if minimum(Yval_real) > -1 * tolerance
            break
        else
            gamma_last = gamma_now
        end
    end
    
    return Y
end

function frobenius_norm(A)
    return norm(vec(A))
end

@time begin
psd_2 = Higham(matrix_2)
eigenvals_2 = eigen(Matrix(psd_matrix)).values
print_smallest_eigenvalue(psd_2)
end

println("F-norm of the near_psd: ",frobenius_norm(psd_matrix-matrix_1))
println("F-norm of the h is : ",frobenius_norm(psd_2-matrix_1))


