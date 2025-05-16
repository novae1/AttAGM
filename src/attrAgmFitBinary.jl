__precompile__()

module attrAgmFitBinary

    using JuMP
    using Gurobi
    using Ipopt
    using AmplNLWriter, SCIP, Bonmin_jll, HiGHS, Couenne_jll
    using Random: bitrand
    using Graphs
    using PyPlot
    using LinearAlgebra
    using Base.Threads
    """
    Implementation of the attrAgmFit model in Julia.
    TO SET THE NUMBER OF THREADS DO:
        export JULIA_NUM_THREADS=1
    """
    Base.@kwdef mutable struct AttrAgmFitBinary
        MinVal::Float64 = 0.0
        MaxVal::Float64 = 1000.0
        n_clusters::Int = 3
        N::Union{Nothing, Int} = nothing
        D_attr::Union{Nothing, Int} = nothing
        D_weight::Union{Nothing, Int} = nothing
        update_F_iterations::Int = 1
        update_F_lr::Float64 = 0.005
        update_F_use_line_search::Bool = false
        update_F_use_riemmanian_grad::Bool = false
        display_ll::Bool = true
        line_search_Alpha::Float64 = 0.05
        line_search_Beta::Float64 = 0.3
        line_search_MaxIter::Int = 5
        sum_Fv::Union{Nothing, Vector{Float64}} = nothing
    end

    function iterate_non_neighbors(g::Graph, u::Int)
        all_vertices = Set(vertices(g))
        neighbors_u = Set(neighbors(g, u))
        non_neighbors = setdiff(all_vertices, neighbors_u, [u])  # Exclude u itself
        return non_neighbors
    end

    function sigmoid(x)
        n=length(x)
        term = zeros(n)
        try
            term = exp.(-x) ./ (1 .- exp.(-x))
        catch
            term = zeros(n)
        end
        return term
    end

    function get_p_uv(F,u,v,probs,n_clusters)
        res = prod([F[v,i] && F[u,i] ? (1 - probs[i]) : 1 for i in 1:n_clusters])
        return 1 - res
    end

    function log_likelihood_row(F::Matrix{Bool},
                                probs::Vector{Float64},
                                G::SimpleGraph, 
                                Y::Matrix{Float64},
                                A::Matrix{Float64}, 
                                B::Matrix{Float64}, 
                                edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                                u::Int, 
                                model::AttrAgmFit)::Float64
        total_edges = 0
        for v in neighbors(G, u)
            p_uv = get_p_uv(F,u,v,probs,model.n_clusters)
            total_edges += p_uv
        end

        total_non_edges = 0
        for v in iterate_non_neighbors(G,u)
            total_non_edges += log( 1 - get_p_uv(F,u,v,probs,model.n_clusters) )
        end  

        graph_ll = total_edges - total_non_edges
        attr_ll = -0.5 * norm(Y[u, :] - A' * F[u, :])^2

        weights_ll = 0.0
        for edge in edges(G)
            u, v = src(edge), dst(edge)
            if u > v
                u, v = v, u
            end
            arr = vcat(F[u, :], F[v, :])
            weights_ll -= 0.5 * norm(edges_data[(u,v)] - B' * arr)^2
        end

        return graph_ll + attr_ll + weights_ll
    end

    function log_likelihood(F::Matrix{Bool}, 
                            G::SimpleGraph,
                            Y::Matrix{Float64},
                            A::Matrix{Float64}, 
                            B::Matrix{Float64},
                            edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}}, 
                            model::AttrAgmFit)::Float64
        return sum(log_likelihood_row(F, G, Y, A, B, edges_data, u, model) for u in 1:nv(G))
    end

    function line_search(F::Matrix{Float64},
                        G::SimpleGraph, 
                        Y::Matrix{Float64}, 
                        A::Matrix{Float64},
                        B::Matrix{Float64}, 
                        edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                        u::Int, 
                        DeltaV::Vector{Float64}, 
                        GradV::Vector{Float64},
                        model::AttrAgmFit)::Float64
        StepSize = 1.0
        InitLikelihood = log_likelihood_row(F, G, Y, A, B, edges_data, u, model)
        NewVarV = similar(DeltaV)
        k = size(DeltaV)[1]
        for _ in 1:model.line_search_MaxIter
            for j in 1:k
                NewVal = F[u, j] + StepSize * DeltaV[j]
                NewVarV[j] = clamp(NewVal, model.MinVal, model.MaxVal)
            end

            F_new = copy(F)
            F_new[u, :] .= NewVarV

            if log_likelihood_row(F_new, G, Y, A, B, edges_data, u, model) < InitLikelihood +
            model.line_search_Alpha * StepSize * dot(GradV, DeltaV)
                StepSize *= model.line_search_Beta
            else
                break
            end

            if StepSize == 0.0
                break
            end
        end

        return StepSize
    end

    ## attributes centers
    function update_A(F::Matrix{Float64}, Y::Matrix{Float64})::Matrix{Float64}
        return inv(F' * F) * F' * Y
    end

    function update_A_exact(F,Y)::Matrix{Float64}
        N, C = size(F)
        D = size(Y)[2]
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        register(model, :sum, 1, sum; autodiff = true)
        @variable(model, A_var[i=1:C, j=1:D] , base_name="A_var")
        @NLobjective(model, Min, sum( sum((Y[i,j] - (F[i,:]' * A_var)[j])^2 for j in 1:D) for i in 1:N) )
        optimize!(model)
        return value.(A_var)
    end

    function update_B(F::Matrix{Float64}, 
                    G::SimpleGraph,
                    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                    model::AttrAgmFit)::Matrix{Float64}
        
        num_edges = ne(G)
        stacked_arr = zeros(num_edges, 2 * model.n_clusters)
        stacked_edges = zeros(num_edges, model.D_weight)
        i = 1
        for edge in edges(G)
            u, v = src(edge), dst(edge)
            if u > v
                u, v = v, u
            end
            stacked_arr[i, :] = vcat(F[u, :], F[v, :])
            stacked_edges[i, :] = edges_data[(u,v)]
            i += 1
        end
        return inv(stacked_arr' * stacked_arr) * stacked_arr' * stacked_edges
    end

    function update_B_exact(F::Matrix{Float64}, 
                            G::SimpleGraph,
                            edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                            mymodel::AttrAgmFit)::Matrix{Float64}
        
        num_edges = ne(G)
        stacked_arr = zeros(num_edges, 2 * mymodel.n_clusters)
        stacked_edges = zeros(num_edges, mymodel.D_weight)
        i = 1
        for edge in edges(G)
            u, v = src(edge), dst(edge)
            if u > v
                u, v = v, u
            end
            stacked_arr[i, :] = vcat(F[u, :], F[v, :])
            stacked_edges[i, :] = edges_data[(u,v)]
            i += 1
        end
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        register(model, :sum, 1, sum; autodiff = true)
        @variable(model, B_var[i=1:2*mymodel.n_clusters, j=1:mymodel.D_weight] , base_name="B_var")
        @NLobjective(model, Min, sum( sum((stacked_edges[i,j] - (stacked_arr[i,:]' * B_var)[j])^2 for j in 1:mymodel.D_weight) for i in 1:num_edges) )
        optimize!(model)
        return value.(B_var)
    end

    function gradient_efficient(F::Matrix{Float64}, 
                                G::SimpleGraph,
                                Y::Matrix{Float64},
                                A::Matrix{Float64},
                                B::Matrix{Float64}, 
                                edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                                u::Int, 
                                model::AttrAgmFit)::Vector{Float64}
        sum_neigh = zeros(model.n_clusters)
        
        for v in neighbors(G, u)
            dotproduct = dot(F[v, :], F[u, :])
            sum_neigh .+= F[v, :] .* sigmoid(dotproduct)
        end

        sum_nneigh = copy(model.sum_Fv)
        for v in neighbors(G, u)
            sum_nneigh .-= F[v, :]
        end
        sum_nneigh .-= F[u, :]

        # n_vu = length(neighbors(G,u))
        # nn_vu = nv(G) - n_vu
        grad_net = sum_neigh - sum_nneigh
        grad_att = A * (Y[u, :] - A' * F[u, :] ) 
        grad_weights = zeros(model.n_clusters)
        for v in neighbors(G, u)
            total, constant_vec = zeros(model.D_weight), zeros(model.D_weight)
            if v > u
                total = B[1:model.n_clusters, :]' * F[u, :]
                constant_vec = B[model.n_clusters + 1:end, :]' * F[v, :]
                grad_weights .+= B[1:model.n_clusters, :] * (edges_data[(u,v)] - total - constant_vec)
            else
                total = B[model.n_clusters + 1:end, :]' * F[u, :]
                constant_vec = B[1:model.n_clusters, :]' * F[v, :]
                grad_weights .+= B[1:model.n_clusters, :] * (edges_data[(v,u)] - total - constant_vec)
            end
        end

        grad = grad_net + grad_att + grad_weights
        n_grad = norm(grad)
        if n_grad>10
            return 10*grad./n_grad
        end
        return grad
        #return clamp.(grad, -10.0, 10.0)
    end

    function update_F(G::SimpleGraph, 
                    Y::Matrix{Float64}, 
                    A::Matrix{Float64}, 
                    B::Matrix{Float64},
                    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                    model::AttrAgmFit, 
                    F_init=nothing)::Matrix{Float64}
        F = F_init !== nothing ? F_init : rand(nv(G), model.n_clusters)

        Threads.@threads for u in 1:nv(G)
                prev_Fu = copy(F[u, :])

                F[u, :] .= max.(10^-5, F[u, :])
                new_Fu = F[u, :]
                delta_Fu = new_Fu .- prev_Fu
                model.sum_Fv .+= delta_Fu
        end
        

        return F
    end

    function fit_MLE(G::SimpleGraph, 
                    Y::Matrix{Float64}, 
                    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                    C::Int, 
                    iterations::Int, 
                    model::AttrAgmFit,
                    F_init=nothing,
                    Thres=0.001)
        println("USING ",Threads.nthreads()," thread(s)")
        model.n_clusters = C
        model.N, model.D_attr = size(Y)
        for i in values(edges_data)
            model.D_weight = size(i)[1]
            break
        end

        F = rand(model.N, C)
        if F_init !== nothing
            F = F_init
        end

        model.sum_Fv = zeros(C)
        for u in 1:model.N
            model.sum_Fv .+= F[u, :]
        end

        # A = update_A(F, Y)
        A = update_A_exact(F,Y)
        B = update_B_exact(F, G, edges_data, model)
        losses = Float64[]
        Cur_ll = log_likelihood(F, G, Y, A, B, edges_data, model)
        Prev_ll = -Inf64

        i = 0
        get_F = update_F
        if model.update_F_use_riemmanian_grad
            get_F = update_F_riemmanian
        end

        while i < iterations && abs(Cur_ll - Prev_ll) > Thres * abs(Cur_ll)
            Prev_ll = Cur_ll
            F = get_F(G, Y, A, B, edges_data, model, F)
            A = update_A_exact(F, Y)
            B = update_B_exact(F, G, edges_data, model)
            Cur_ll = log_likelihood(F, G, Y, A, B, edges_data, model)

            if model.display_ll
                push!(losses, Cur_ll)
            end

            i += 1
        end

        if model.display_ll
            xs = 1:length(losses)
            plot(xs, losses)
            xlabel("iterations")
            ylabel("log_likelihood")
            display()
        end

        println("Finished with ", i ," iterations")
        return F, A, B
    end

    function BIC(ll,N,K,num_edges)
        return -2*ll + N*K*log(num_edges)
    end 

    function get_best_init(F1, F2, G, Y, edges_data)
        model = AttrAgmFit()

        model.N, model.n_clusters = size(F1)
        model.sum_Fv = zeros(model.n_clusters)
        for u in 1:model.N
            model.sum_Fv .+= F1[u, :]
        end

        for v in values(edges_data)
            model.D_weight = size(v)[1]
            break
        end
        A = update_A_exact(F1,Y)
        B = update_B_exact(F1, G, edges_data, model)
        l1 = log_likelihood(F1, G, Y, A, B, edges_data, model)
        

        model.n_clusters = size(F2)[2]
        model.sum_Fv = zeros(model.n_clusters)
        for u in 1:model.N
            model.sum_Fv .+= F2[u, :]
        end        
        A = update_A_exact(F2,Y)
        B = update_B_exact(F2, G, edges_data, model)
        l2 = log_likelihood(F2, G, Y, A, B, edges_data, model)

        N = nv(G)
        num_edges = ne(G) 
        if BIC(l1,N,size(F1)[2],num_edges) >  BIC(l2,N,size(F2)[2],num_edges)   
            return F2
        else
            return F1
        end
    end
end