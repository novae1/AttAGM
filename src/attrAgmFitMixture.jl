__precompile__()

module attrAgmFitMixture 

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

function one_minus_exp(F,u,v)
    return 1 .- exp.(-F[u,:].*F[v,:])
end 

function exp_likelihood(lambdas,w)
    return lambdas .* exp.(-lambdas*w)
end

function compute_pi_uv(F,u,v,model)
    pi = zeros(model.n_clusters)
    for i in 1:model.n_clusters
        pi[i] = 1 - exp(-F[u,i]*F[v,i])
    end
    #Compute the normalization factor
    #Z = 1 - exp(-dot(F[u,:],F[v,:]))
    Z = sum(pi)
    pi ./= Z
    return pi
end

function log_likelihood_row(F::Matrix{Float64},
                            G::SimpleGraph, 
                            Y::Matrix{Float64},
                            A::Matrix{Float64}, 
                            lambdas::Vector{Float64}, 
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
    attr_ll = -0.5 * norm(Y[u, :] - A' * F[u, :])^2

    weights_ll = 0.0
    for v in neighbors(G, u)
        pi = compute_pi_uv(F,u,v,model)
        if u > v
            u_p, v_p = v, u
        else
            u_p, v_p = u, v
        end
        w = edges_data[(u_p,v_p)][1]
        weights_ll += log(
                            sum( pi[i]*lambdas[i]*exp(-lambdas[i]*w) for i in 1:model.n_clusters )
                        )
    end

    return graph_ll + attr_ll + weights_ll
end

function log_likelihood(F::Matrix{Float64}, 
                        G::SimpleGraph,
                        Y::Matrix{Float64},
                        A::Matrix{Float64}, 
                        lambdas::Vector{Float64},
                        edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}}, 
                        model::AttrAgmFit)::Float64
    return sum(log_likelihood_row(F, G, Y, A, lambdas, edges_data, u, model) for u in 1:nv(G))
end

function line_search(F::Matrix{Float64},
                     G::SimpleGraph, 
                     Y::Matrix{Float64}, 
                     A::Matrix{Float64},
                     lambdas::Vector{Float64}, 
                     edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                     u::Int, 
                     DeltaV::Vector{Float64}, 
                     GradV::Vector{Float64},
                     model::AttrAgmFit)::Float64
    StepSize = 1.0
    InitLikelihood = log_likelihood_row(F, G, Y, A, lambdas, edges_data, u, model)
    NewVarV = similar(DeltaV)
    k = size(DeltaV)[1]
    for _ in 1:model.line_search_MaxIter
        for j in 1:k
            NewVal = F[u, j] + StepSize * DeltaV[j]
            NewVarV[j] = clamp(NewVal, model.MinVal, model.MaxVal)
        end

        F_new = copy(F)
        F_new[u, :] .= NewVarV

        if log_likelihood_row(F_new, G, Y, A, lambdas, edges_data, u, model) < InitLikelihood +
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

function update_A(F::Matrix{Float64}, Y::Matrix{Float64})::Matrix{Float64}
    return inv(F' * F) * F' * Y
end

function update_lambdas(F::Matrix{Float64}, 
                  G::SimpleGraph,
                  edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                  model::AttrAgmFit)::Vector{Float64}
    opt_model = Model(Ipopt.Optimizer)
    set_silent(opt_model)
    @variable(opt_model, lambda_var[i=1:model.n_clusters] ,lower_bound=0, base_name="lambda_var")    
    @NLobjective(opt_model, Max, 
        sum(
            log( 
                sum( 
                    compute_pi_uv(F,e.src,e.dst,model)[i]*lambda_var[i]*exp( -lambda_var[i]*edges_data[(e.src,e.dst)][1] ) 
                    for i in 1:model.n_clusters
                )
            )
            for e in edges(G)
        )
    )
    optimize!(opt_model)
    return value.(lambda_var)
end

function update_lambdas_inexact(F::Matrix{Float64}, 
    G::SimpleGraph,
    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
    model::AttrAgmFit)::Vector{Float64}

    K = model.n_clusters
    lambdas = zeros(K)
    normalization_factor = zeros(K)
    for e in edges(G)
        pi = compute_pi_uv(F,e.src,e.dst,model)
        u,v = e.src, e.dst
        lambdas .+= pi .* edges_data[(u,v)] 
        normalization_factor .+= pi
    end
    return lambdas./normalization_factor
end

function gradient_efficient(F::Matrix{Float64}, 
                            G::SimpleGraph,
                            Y::Matrix{Float64},
                            A::Matrix{Float64},
                            lambdas::Vector{Float64}, 
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

    grad_net = sum_neigh - sum_nneigh
    grad_att = A * (Y[u, :] - A' * F[u, :] ) 
    grad_weights = zeros(model.n_clusters)
    for v in neighbors(G, u)
        if u > v
            u_p, v_p = v, u
        else
            u_p, v_p = u, v
        end
        a = one_minus_exp(F,u_p,v_p)
        c = exp_likelihood(lambdas,edges_data[(u_p,v_p)][1])
        x = sum(a)
        y = sum(a.*c)
        for i in 1:model.n_clusters
            grad_weights[i] += F[v,i]*exp(-F[u,i]*F[v,i])*( c[i]/y - 1/x) 
        end
    end

    grad = grad_net + grad_att + grad_weights
    return grad
    #return clamp.(grad, -10.0, 10.0)
end

function update_F(G::SimpleGraph, 
                  Y::Matrix{Float64}, 
                  A::Matrix{Float64}, 
                  lambdas::Vector{Float64},
                  edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                  model::AttrAgmFit, 
                  F_init=nothing)::Matrix{Float64}
    F = F_init !== nothing ? F_init : rand(nv(G), model.n_clusters)

    # model.sum_Fv = sum(F, dims=1)[:]

    for i in 1:model.update_F_iterations
        Threads.@threads for u in 1:nv(G)
            prev_Fu = copy(F[u, :])
            grad = gradient_efficient(F, G, Y, A, lambdas, edges_data, u, model)
            n_grad = norm(grad)
            if n_grad>10
                return 10*grad./n_grad
            end
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
                            lambdas::Vector{Float64}, 
                            edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                            u::Int, 
                            model::AttrAgmFit)::Vector{Float64}

    grad = gradient_efficient(D.^2, G, Y, A, lambdas, edges_data, u, model)
    grad .*= 2*D[u,:]
    return grad
    #return clamp.(grad, -10.0, 10.0)
end

function update_F_riemmanian(G::SimpleGraph, 
                  Y::Matrix{Float64}, 
                  A::Matrix{Float64}, 
                  lambdas::Vector{Float64},
                  edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                  model::AttrAgmFit, 
                  F_init=nothing)::Matrix{Float64}
    D = F_init !== nothing ? F_init.^0.5 : rand(nv(G), model.n_clusters)
    
    for i in 1:model.update_F_iterations
        Threads.@threads for u in 1:nv(G)
            prev_Du = copy(D[u, :])
            grad = gradient_riemmanian(D, G, Y, A, lambdas, edges_data, u, model)
            n_grad = norm(grad)
            if n_grad>10
                return 10*grad./n_grad
            end
            if model.update_F_use_line_search
                alpha = line_search(D.^2, G, Y, A, lambdas, edges_data, u, grad, grad, model)
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

    A = update_A(F, Y)
    lambdas = update_lambdas(F, G, edges_data, model)
    losses = Float64[]
    Cur_ll = log_likelihood(F, G, Y, A, lambdas, edges_data, model)
    Prev_ll = -Inf64
    i = 0
    get_F = update_F
    if model.update_F_use_riemmanian_grad
        get_F = update_F_riemmanian
    end

    while i < iterations && abs(Cur_ll - Prev_ll) > Thres*abs(Cur_ll)
        Prev_ll = Cur_ll
        F = get_F(G, Y, A, lambdas, edges_data, model, F)
        A = update_A(F, Y)
        lambdas = update_lambdas(F, G, edges_data, model)
        Cur_ll = log_likelihood(F, G, Y, A, lambdas, edges_data, model)

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
    return F, A, lambdas
end

end 