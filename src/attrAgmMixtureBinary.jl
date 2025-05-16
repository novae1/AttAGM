__precomHle__()

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

function compute_H_uv(F,u,v,model)
    H = zeros(model.n_clusters)
    for i in 1:model.n_clusters
        H[i] = 1 - exp(-F[u,i]*F[v,i])
    end
    #Compute the normalization factor
    #Z = 1 - exp(-dot(F[u,:],F[v,:]))
    Z = sum(H)
    H ./= Z
    return H
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
        H = compute_H_uv(F,u,v,model)
        if u > v
            u_p, v_p = v, u
        else
            u_p, v_p = u, v
        end
        w = edges_data[(u_p,v_p)][1]
        weights_ll += log(
                            sum( H[i]*lambdas[i]*exp(-lambdas[i]*w) for i in 1:model.n_clusters )
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

function update_A(F::Matrix{Float64}, Y::Matrix{Float64})::Matrix{Float64}
    return inv(F' * F) * F' * Y
end

function update_lambdas(F::Matrix{Float64}, 
    G::SimpleGraph,
    edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
    model::AttrAgmFit)::Vector{Float64}

    K = model.n_clusters
    lambdas = zeros(K)
    normalization_factor = zeros(K)
    for e in edges(G)
        H = compute_H_uv(F,e.src,e.dst,model)
        u,v = e.src, e.dst
        lambdas .+= H .* edges_data[(u,v)] 
        normalization_factor .+= H
    end
    return lambdas./normalization_factor
end

function dynamic_M(F,G,Y,A, lambdas, edges_data, u, model)
    k = model.n_clusters
    thread_active = zeros(Int, k)
    thread_losses = zeros(Float64, k)
    thread_guess = zeros(Float64, k, k)
    
    for h in 1:k
        thread_active[h] = 1
        m0 = zeros(Float64, k)
        m0[h] = 1
        loss_h = -log_likelihood_row(F,G,Y,A, lambdas, edges_data, u, model)
        
        for r in 2:k
            if thread_active[h] == 1
                new_candidates = fill(Inf, k)
                for p in 1:k
                    if m0[p] == 0
                        m0[p] = 1
                        new_candidates[p] = -log_likelihood_row(F,G,Y,A, lambdas, edges_data, u, model)
                        m0[p] = 0
                    end
                end
                p = argmin(new_candidates)
                if loss_h <= new_candidates[p]
                    thread_active[h] = 0
                else
                    m0[p] = 1
                    loss_h = new_candidates[p]
                end
            end
        end
        thread_losses[h] = -log_likelihood_row(F,G,Y,A, lambdas, edges_data, u, model)
        thread_guess[h, :] = m0
    end
    best_guess = argmin(thread_losses)
    return thread_guess[best_guess, :]
end

function update_F(G::SimpleGraph, 
                  Y::Matrix{Float64}, 
                  A::Matrix{Float64}, 
                  lambdas::Vector{Float64},
                  edges_data::Dict{Tuple{Int64,Int64},Vector{Float64}},
                  model::AttrAgmFit, 
                  F_init=nothing)::Matrix{Float64}
    F = F_init !== nothing ? F_init : bitrand(nv(G), model.n_clusters)

    for i in 1:model.update_F_iterations
        Threads.@threads for u in 1:nv(G)
            new_m = dynamic_M(F,G,Y,A, lambdas, edges_data, u, model)
            M = copy(F)
            M[u,:] = new_m
            if  log_likelihood_row(M,G,Y,A, lambdas, edges_data, u, model) >
                    log_likelihood_row(F,G,Y,A, lambdas, edges_data, u, model)
                F[u, :] = new_m
            end        
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

    A = update_A(F, Y)
    lambdas = update_lambdas(F, G, edges_data, model)
    losses = Float64[]
    Cur_ll = log_likelihood(F, G, Y, A, lambdas, edges_data, model)
    Prev_ll = -Inf64
    i = 0
    get_F = update_F

    while i < iterations && abs(Cur_ll - Prev_ll) > Thres*abs(Cur_ll)
        Prev_ll = Cur_ll
        F = update_F(G, Y, A, lambdas, edges_data, model, F)
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