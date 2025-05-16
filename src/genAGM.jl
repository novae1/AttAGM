__precompile__()

module genAGM
using Distributions
using Graphs
include("utils.jl")
using .utils
    function RndConnectInsideCommunity(G::SimpleGraph{Int64}, 
                                        CmtyV::Union{Vector{Int64}, UnitRange{Int64}}, 
                                        Prob::Float64
                                    )
        CNodes = size(CmtyV)[1]
        if (CNodes < 20)
            b = Binomial(CNodes*(CNodes - 1)/2,Prob)
            CEdges = rand(b,1)[1]
        else
            CEdges = (Prob * CNodes * (CNodes - 1) / 2)
        end

        total_edges = 0
        edges_set = Set()
        while total_edges < CEdges
            SrcNId = sample(CmtyV)
            DstNId = sample(CmtyV)
            SrcNId, DstNId = minimum([SrcNId,DstNId]),maximum([SrcNId,DstNId])
            if SrcNId != DstNId &&  ! ( (SrcNId,DstNId) in edges_set)
                push!(edges_set,(SrcNId,DstNId))
                add_edge!(G,SrcNId,DstNId)
                total_edges += 1
            end 
        end
        return G
    end

    function connect_isolated_nodes(G::SimpleGraph{Int64},v_list::Union{Vector{Int64}, UnitRange{Int64}})
        isolated_nodes = Vector{Int64}()
        non_isolated_nodes = Vector{Int64}()

        for u in v_list
            if degree(G,u) > 0
                push!(non_isolated_nodes,u)
            else
                push!(isolated_nodes,u)
            end
        end

        for u in isolated_nodes
            v = sample(non_isolated_nodes)
            add_edge!(G,u,v)
            push!(non_isolated_nodes,u)
        end
        return G
    end

    function genAgm(CmtyVV::Vector{Vector{Int64}},CProbV::Vector{Float64}, PNoCom::Float64,
                                                                use_MCMC=true,n_iter=10000)
        N = 1
        n_cmty = size(CmtyVV)[1]
        for i in 1:n_cmty
            n = maximum(CmtyVV[i])
            N = maximum([n,N])
        end
        G = SimpleGraph{Int64}(N)
        # v_list = collect(range(1,N))
        # if PNoCom > 0.0
        #     push!(CProbV,PNoCom)
        #     push!(CmtyVV,v_list)
        # end
        # n_cmty = size(CmtyVV)[1]
        # println(n_cmty)
        for i in 1:n_cmty
            G = RndConnectInsideCommunity(G,CmtyVV[i],CProbV[i])
        end
        if PNoCom > 0
            G = RndConnectInsideCommunity(G,1:N,PNoCom)
        end
        G = connect_isolated_nodes(G,1:N)
        if use_MCMC
            G = sample_graph_MCMC(G,CmtyVV,CProbV,n_iter,1:N,PNoCom)
        end
        return G
    end

    function genAgmDS(CmtyVV::Vector{Vector{Int64}}, DensityCoef::Float64=0.6, ScaleCoef::Float64=1.3)
        CProbV = Vector{Float64}()
        n_cmty = size(CmtyVV)[1]
        for i in 1:n_cmty
            size_cmnty = size(CmtyVV[i])[1]
            prob = ScaleCoef* (size_cmnty^-DensityCoef)
            if prob > 1
                prob = 1
            end
            push!(CProbV,prob)        
        end
        return genAgm(CmtyVV,CProbV,-1.) 
    end 

    function genAgmD(CmtyVV::Vector{Vector{Int64}}, DensityCoef::Float64=0.6, TargetEdges::Int64=100)
        TryG = genAgmDS(CmtyVV, DensityCoef, 1.0)
        ScaleCoef = TargetEdges / ne(TryG)
        return genAgmDS(CmtyVV,DensityCoef,ScaleCoef)
    end 

    function compute_proba_edge(CProbV::Vector{Float64},
                                nodes_cmty::Vector{Set{Int64}},
                                u::Int64,
                                v::Int64,
                                PNoCom::Float64)
        u_cmtys = nodes_cmty[u]
        v_cmtys = nodes_cmty[v]
        intersection_uv = intersect(u_cmtys,v_cmtys)
        prob_non_success = 1
        if length(intersection_uv) > 0
            for k in intersection_uv
                prob_non_success *= (1-CProbV[k])
            end
        elseif PNoCom > 0
            prob_non_success = 1-PNoCom
        else
            prob_non_success = 1 
        end
        return 1 - prob_non_success
    end

    ### see paper https://arxiv.org/pdf/1806.11276
    function sample_graph_MCMC( G,
                                CmtyVV::Vector{Vector{Int64}},
                                CProbV::Vector{Float64},
                                n_iter::Int64,
                                v_list::Union{Vector{Int64}, UnitRange{Int64}},
                                PNoCom::Float64
                            )
        N = length(v_list)
        nodes_cmty = utils.get_nodes_cmty(CmtyVV,N)
        for k in 1:n_iter
            SrcNId = sample(v_list)
            DstNId = sample(v_list)
            SrcNId, DstNId = minimum([SrcNId,DstNId]),maximum([SrcNId,DstNId])
            if SrcNId != DstNId 
                p = compute_proba_edge(CProbV, 
                                        nodes_cmty,
                                        SrcNId,
                                        DstNId,
                                        PNoCom)
                if has_edge(G,SrcNId,DstNId)
                    rem_edge!(G,SrcNId,DstNId)
                    if is_connected(G)
                        accept_proba = minimum([(1-p)/p , 1])
                        dist = Bernoulli(accept_proba)
                        accept = rand(dist,1)[1]
                        if ! accept
                            add_edge!(G,SrcNId,DstNId)
                        end  
                    else
                        add_edge!(G,SrcNId,DstNId)    
                    end
                else # The edge is not in the graph. We will try to add
                    add_edge!(G,SrcNId,DstNId)
                    accept_proba = minimum([p/(1-p) , 1])
                    dist = Bernoulli(accept_proba)
                    accept = rand(dist,1)[1]
                    if ! accept
                        rem_edge!(G,SrcNId,DstNId)
                    end
                end
            end
        end
        println("\t\t ### FINISHED MCMC PROCESS ###")
        return G
    end
end
