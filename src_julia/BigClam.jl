module BigClam
using JuMP
using Ipopt
# using AmplNLWriter, SCIP, Bonmin_jll, HiGHS, Couenne_jll
using Random: bitrand
using Graphs
using LinearAlgebra
using Base.Threads
using Plots
using PyPlot
include("utils.jl")
using .utils


"""
Direct implementation of BigClam
http://i.stanford.edu/~crucis/pubs/paper-nmfagm.pdf
"""

    const MinVal = 0
    const MaxVal = 1000


    """
        Run the original implementation writen in cpp by JUre Leskovec
    """
    function run_bigclam_cpp(filename="output_graph.txt",num_communities=3)
        path = "./bigclam"
        run(`$path -i:$(filename) -c:$(num_communities)`)
        cmtyvv = utils.read_bipartite_file("cmtyvv.txt","\t")
        return cmtyvv
    end

    function non_neighbors(G, u)
        all_nodes = collect(vertices(G))
        neighbor_set = Set(neighbors(G, u))
        return filter(v -> v ∉ neighbor_set, all_nodes)
    end

    function sigm(x)
        n=length(x)
        term = zeros(n)
        try
            term = exp.(-x) ./ (1 .- exp.(-x))
        catch
            term = zeros(n)
        end
        return term
    end

    function log_likelihood_row(F, G, u, sum_Fv)
        total_edges = 0
        for v in neighbors(G, u)
            total_edges += log(1 - exp(-dot(F[u,:], F[v,:])))
        end

        total_non_edges = 0
        sum_Fv = copy(sum_Fv)
        for v in neighbors(G, u)
            sum_Fv .-= F[v,:]
        end
        sum_Fv .-= F[u,:]
        total_non_edges += dot(F[u,:], sum_Fv)
        return total_edges - total_non_edges
    end

    function log_likelihood(F, G, sum_Fv)
        total = sum([log_likelihood_row(F, G, u, sum_Fv) for u in vertices(G)])
        return total
    end

    function line_search(F, G, u, DeltaV, GradV, Alpha, Beta, MaxIter, sum_Fv)
        StepSize = 1.0
        InitLikelihood = log_likelihood_row(F, G, u, sum_Fv)
        NewVarV = similar(DeltaV)
        k = size(DeltaV)
        for i in 1:MaxIter
            for j in 1:k[1]
                NewVal = F[u, j] + StepSize * DeltaV[j]
                if NewVal < MinVal
                    NewVal = MinVal
                end
                if NewVal > MaxVal
                    NewVal = MaxVal
                end
                NewVarV[j] = NewVal
            end
            F_new = copy(F)
            F_new[u, :] .= NewVarV
            if log_likelihood_row(F_new, G, u, sum_Fv) < InitLikelihood + Alpha * StepSize * dot(GradV, DeltaV)
                StepSize *= Beta
            else
                break
            end
            if i == MaxIter
                StepSize = 0
                break
            end
        end
        return StepSize
    end

    function gradient_efficient(F, G, u, sum_Fv)
        N, C = size(F)
        sum_neigh = zeros(C)
        for v in neighbors(G, u)
            dotproduct = dot(F[v,:], F[u,:])
            sum_neigh .+= F[v,:] .* sigm(dotproduct)
        end

        sum_nneigh = copy(sum_Fv)
        for v in neighbors(G, u)
            sum_nneigh .-= F[v,:]
        end
        sum_nneigh .-= F[u,:]
        grad = sum_neigh - sum_nneigh
        grad = clamp.(grad,-10,10)
        #grad = max.(-10, min.(10, grad))
        return grad
    end

    function fit(;
        G=nothing,
        C=nothing,
        iterations=100, 
        lr=0.005, 
        display_loss=false, 
        F_init=nothing, 
        use_line_search=true,
        Thres=0.001,
        run_original=false)
        if run_original
            cmtyvv = run_bigclam_cpp("output_graph.txt",C)
            F = utils.cmtyvv_to_membership_matrix(cmtyvv, nv(G))
            return F
        else
            return  fit( G, C, iterations, lr, display_loss, F_init,  use_line_search, Thres)
        end
    end
    
    function fit(
                    G=nothing,
                    C=nothing,
                    iterations=100, 
                    lr=0.005, 
                    display_loss=false, 
                    F_init=nothing, 
                    use_line_search=true,
                    Thres=0.001)
        N = length(vertices(G))
        if F_init !== nothing
            F = F_init
        else
            F = rand(N, C)
        end
        losses = []
        sum_Fv = zeros(C)
        for u in 1:N
            sum_Fv .+= F[u, :]
        end

        Cur_ll = log_likelihood(F, G, copy(sum_Fv))
        Prev_ll = -Inf64
        
        n = 0
        while n < iterations && abs(Cur_ll - Prev_ll) > Thres * abs(Cur_ll)
            Prev_ll = Cur_ll
            Threads.@threads for u in 1:N
                grad = gradient_efficient(F, G, u, copy(sum_Fv))
                prev_Fu = copy(F[u, :])
                if use_line_search
                    alpha = line_search(F, G, u, grad, grad, 0.05, 0.3, 5, copy(sum_Fv))
                    F[u, :] .+= alpha .* grad
                else
                    F[u, :] .+= lr .* grad
                end
                F[u, :] .= max.(0.001, F[u, :]) # F should be nonnegative
                new_Fu = F[u, :]
                delta_Fu = new_Fu .- prev_Fu
                sum_Fv .+= delta_Fu
            end
            if display_loss
                Cur_ll = log_likelihood(F, G, copy(sum_Fv))
                push!(losses, Cur_ll)
            end
            n += 1
        end
        if display_loss
            xs = 1:length(losses)
            PyPlot.plot(xs, losses)
            PyPlot.xlabel("iterations")
            PyPlot.ylabel("log_likelihood")
            PyPlot.display()
            #close()
        end
        return F
    end

end