__precompile__()

module attrAgmFit

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
    Base.@kwdef mutable struct AttrAgmFit
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

    function expm1(x)
        return 1 .- exp.(-x)
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

    function log_likelihood_row(F::Matrix{Float64},
                                G::SimpleGraph, 
                                Y::Matrix{Float64},
                                A::Matrix{Float64}, 
                                B::Matrix{Float64}, 
                                edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                                u::Int, 
                                model::AttrAgmFit)::Float64
        total_edges = 0
        for v in neighbors(G, u)
            total_edges += log(1 - exp(-dot(F[u, :], F[v, :])))
        end

        sum_Fv = copy(model.sum_Fv)
        for v in neighbors(G, u)
            sum_Fv .-= F[v, :]
        end
        sum_Fv .-= F[u, :]

        total_non_edges = dot(F[u, :], sum_Fv)
        graph_ll = total_edges - total_non_edges

        exp_F = expm1(F)
        attr_ll = -0.5 * norm(Y[u, :] - A' * exp_F[u, :])^2

        weights_ll = 0.0
        for edge in edges(G)
            u, v = src(edge), dst(edge)
            if u > v
                u, v = v, u
            end
            arr = vcat(exp_F[u, :], exp_F[v, :])
            weights_ll -= 0.5 * norm(edges_data[(u,v)] - B' * arr)^2
        end

        return graph_ll + attr_ll + weights_ll
    end

    function log_likelihood(F::Matrix{Float64}, 
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
        exp_F = expm1(F)
        return inv(exp_F' * exp_F) * exp_F' * Y
    end

    function update_A_exact(F,Y)::Matrix{Float64}
        exp_F = expm1(F)
        N, C = size(F)
        D = size(Y)[2]
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        register(model, :sum, 1, sum; autodiff = true)
        @variable(model, A_var[i=1:C, j=1:D] , base_name="A_var")
        @NLobjective(model, Min, sum( sum((Y[i,j] - (exp_F[i,:]' * A_var)[j])^2 for j in 1:D) for i in 1:N) )
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
        exp_F = expm1(F)
        for edge in edges(G)
            u, v = src(edge), dst(edge)
            if u > v
                u, v = v, u
            end
            stacked_arr[i, :] = vcat(exp_F[u, :], exp_F[v, :])
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
        exp_F = expm1(F)
        for edge in edges(G)
            u, v = src(edge), dst(edge)
            if u > v
                u, v = v, u
            end
            stacked_arr[i, :] = vcat(exp_F[u, :], exp_F[v, :])
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

    function F_vanilla_gradient(F::Matrix{Float64}, 
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
        exp_F = expm1(F)
        grad_att = A * (Y[u, :] - A' * exp_F[u, :] ) 
        grad_weights = zeros(model.n_clusters)
        for v in neighbors(G, u)
            total, constant_vec = zeros(model.D_weight), zeros(model.D_weight)
            if v > u
                total = B[1:model.n_clusters, :]' * exp_F[u, :]
                constant_vec = B[model.n_clusters + 1:end, :]' * exp_F[v, :]
                grad_weights .+= B[1:model.n_clusters, :] * (edges_data[(u,v)] - total - constant_vec)
            else
                total = B[model.n_clusters + 1:end, :]' * exp_F[u, :]
                constant_vec = B[1:model.n_clusters, :]' * exp_F[v, :]
                grad_weights .+= B[1:model.n_clusters, :] * (edges_data[(v,u)] - total - constant_vec)
            end
        end
        grad_att = grad_att.*exp.(-F[u,:])
        grad_weights = grad_weights.*exp.(-F[u,:])
        #grad = grad_net./norm(grad_net) + grad_att./norm(grad_att) + grad_weights./norm(grad_weights)
        grad = grad_net + grad_att + grad_weights
        n_grad = norm(grad)
        if n_grad>10
            return 10*grad./n_grad
        end
        return grad
    end

    function update_F(G::SimpleGraph, 
                    Y::Matrix{Float64}, 
                    A::Matrix{Float64}, 
                    B::Matrix{Float64},
                    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                    model::AttrAgmFit, 
                    F_init=nothing)::Matrix{Float64}
        F = F_init !== nothing ? F_init : rand(nv(G), model.n_clusters)

        # model.sum_Fv = sum(F, dims=1)[:]

        for i in 1:model.update_F_iterations
            Threads.@threads for u in 1:nv(G)
                prev_Fu = copy(F[u, :])
                grad = F_vanilla_gradient(F, G, Y, A, B, edges_data, u, model)
                if model.update_F_use_line_search
                    alpha = line_search(F, G, Y, A, B, edges_data, u, grad, grad, model)
                    F[u, :] .+= alpha * grad
                else
                    F[u, :] .+= model.update_F_lr * grad
                end
                F[u, :] .= max.(10^-5, F[u, :])
                new_Fu = F[u, :]
                delta_Fu = new_Fu .- prev_Fu
                model.sum_Fv .+= delta_Fu
            end
        end

        return F
    end

    function gradient_riemmanian(D::Matrix{Float64}, 
                                G::SimpleGraph,
                                Y::Matrix{Float64},
                                A::Matrix{Float64},
                                B::Matrix{Float64}, 
                                edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                                u::Int, 
                                model::AttrAgmFit)::Vector{Float64}
        grad = F_vanilla_gradient(D.^2, G, Y, A, B, edges_data, u, model)
        grad .*= 2*D[u,:]
        n_grad = norm(grad)
        if n_grad>10
            return 10*grad./n_grad
        end
        return grad
        #return clamp.(grad, -10.0, 10.0)
    end

    function update_F_riemmanian(G::SimpleGraph, 
                    Y::Matrix{Float64}, 
                    A::Matrix{Float64}, 
                    B::Matrix{Float64},
                    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                    model::AttrAgmFit, 
                    F_init=nothing)::Matrix{Float64}
        D = F_init !== nothing ? F_init.^0.5 : rand(nv(G), model.n_clusters)
        # model.sum_Fv = sum(F, dims=1)[:]

        for i in 1:model.update_F_iterations
            Threads.@threads for u in 1:nv(G)
                prev_Du = copy(D[u, :])
                grad = gradient_riemmanian(D, G, Y, A, B, edges_data, u, model)
                if model.update_F_use_line_search
                    F_grad = grad.^2 .+ 2*D[u,:].*grad
                    alpha = line_search(D.^2, G, Y, A, B, edges_data, u, F_grad, F_grad, model)
                    D[u, :] .+= alpha * grad
                else
                    D[u, :] .+= model.update_F_lr * grad
                end
                new_Du = D[u, :]
                delta_Du = new_Du .- prev_Du
                model.sum_Fv .+= (delta_Du.^2 + 2*delta_Du.*prev_Du)
            end
        end

        return D.^2
    end

    function update_F_riemman_nesterov(G::SimpleGraph, 
                    Y::Matrix{Float64}, 
                    A::Matrix{Float64}, 
                    B::Matrix{Float64},
                    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                    model::AttrAgmFit, 
                    F_init=nothing)::Matrix{Float64}
        D = F_init !== nothing ? F_init.^0.5 : rand(nv(G), model.n_clusters)
        
        # Initialize momentum
        momentum = zeros(size(D))
        beta = 0.9  # Momentum parameter
        
        for i in 1:model.update_F_iterations
            Threads.@threads for u in 1:nv(G)
                # Nesterov momentum: look ahead
                D_lookahead = D[u, :] + beta * momentum[u, :]
                
                # Create a temporary D matrix with the lookahead value
                D_temp = copy(D)
                D_temp[u, :] = D_lookahead
                
                # Compute gradient at lookahead point
                grad = gradient_riemmanian(D_temp, G, Y, A, B, edges_data, u, model)
                
                # Update momentum
                momentum[u, :] = beta * momentum[u, :] + model.update_F_lr * grad
                
                # Update D with momentum
                D[u, :] .+= momentum[u, :]
                
                # Ensure numerical stability
                D[u, :] .= max.(1e-8, D[u, :])
                
                # Update sum_Fv
                new_Du = D[u, :]
                prev_Du = D[u, :] - momentum[u, :]
                delta_Du = new_Du .- prev_Du
                model.sum_Fv .+= (delta_Du.^2 + 2*delta_Du.*prev_Du)
            end
        end

        return D.^2
    end

    function update_F_nesterov(G::SimpleGraph, 
                    Y::Matrix{Float64}, 
                    A::Matrix{Float64}, 
                    B::Matrix{Float64},
                    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                    model::AttrAgmFit, 
                    F_init=nothing)::Matrix{Float64}
        F = F_init !== nothing ? F_init : rand(nv(G), model.n_clusters)
        
        # Initialize momentum
        momentum = zeros(size(F))
        beta = 0.9  # Momentum parameter
        
        for i in 1:model.update_F_iterations
            Threads.@threads for u in 1:nv(G)
                # Nesterov momentum: look ahead
                F_lookahead = F[u, :] + beta * momentum[u, :]
                
                # Create a temporary F matrix with the lookahead value
                F_temp = copy(F)
                F_temp[u, :] = F_lookahead
                
                # Compute gradient at lookahead point
                grad = F_vanilla_gradient(F_temp, G, Y, A, B, edges_data, u, model)
                
                # Update momentum
                momentum[u, :] = beta * momentum[u, :] + model.update_F_lr * grad
                
                # Update F with momentum
                F[u, :] .+= momentum[u, :]
                
                # Ensure non-negativity
                F[u, :] .= max.(model.MinVal, F[u, :])
                
                # Update sum_Fv
                new_Fu = F[u, :]
                prev_Fu = F[u, :] - momentum[u, :]
                delta_Fu = new_Fu .- prev_Fu
                model.sum_Fv .+= delta_Fu
            end
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
        get_F = update_F_nesterov
        if model.update_F_use_riemmanian_grad
            get_F = update_F_riemman_nesterov
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