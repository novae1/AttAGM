include("utils.jl")
include("genAGM.jl")
include("genAttrAGM.jl")
include("attrAgmFit.jl")
include("scores.jl")
include("BigClam.jl")
include("MakeBenchmark.jl")
include("banerjee_overlapping.jl")
using .utils
using .genAGM
using .genAttrAGM
using .attrAgmFit
using .attrAgmFitMixture
using .BigClam
using .MakeBenchmark
using NPZ
using Graphs
using BenchmarkTools
using Printf
using DataFrames
using CSV
using Random: bitrand

function euclidean_distance(p1, p2)
    if length(p1) != length(p2)
        throw(DimensionMismatch("Vectors must have the same length"))
    end
    return sum((p1 .- p2).^2)
end

function evaluate_algorithm(algo_name,algo_func,params_dict,true_,metrics_dict,Thres)
    start = time()
    F = algo_func(;params_dict...)
    elapsed = time() - start
    pred_ = nothing
    pred_ = convert.(Float64,F) .> Thres
    res = compute_all_metrics(true_,pred_)
    push!(metrics_dict[algo_name]["F1"],res[1])
    push!(metrics_dict[algo_name]["NMI"],res[2])
    push!(metrics_dict[algo_name]["ARI"],res[3])
    push!(metrics_dict[algo_name]["exec_time"],elapsed)
    return F
end

function evaluate_proposed_GLM( algos_metrics,
                                G,
                                attr_data,
                                edges_data,
                                M,
                                n_clusters,
                                Thres,
                                F_init=nothing)

    println("Using GLM")
    model = attrAgmFit.AttrAgmFit()
    model.MaxVal = 100
    model.update_F_use_line_search = true
    model.update_F_use_riemmanian_grad = true
    model.display_ll = false

    start = time()
    F,A,B = attrAgmFit.fit_MLE(G,attr_data,edges_data,n_clusters,100,model,F_init,0.001)
    elapsed = time() - start

    F_bin = F.>Thres
    res = compute_all_metrics(M,F_bin)
    push!(algos_metrics["attrAgmRiemann"]["F1"],res[1])
    push!(algos_metrics["attrAgmRiemann"]["NMI"],res[2])
    push!(algos_metrics["attrAgmRiemann"]["ARI"],res[3])
    push!(algos_metrics["attrAgmRiemann"]["exec_time"],elapsed)

    #############################
    model.update_F_use_line_search = false
    model.update_F_use_riemmanian_grad = false
    start = time()
    F,A,B = attrAgmFit.fit_MLE(G,attr_data,edges_data,n_clusters,100,model,copy(F_init),0.001)
    elapsed = time() - start

    F_bin = F.>Thres
    res = compute_all_metrics(M,F_bin)
    push!(algos_metrics["attrAgm"]["F1"],res[1])
    push!(algos_metrics["attrAgm"]["NMI"],res[2])
    push!(algos_metrics["attrAgm"]["ARI"],res[3])
    push!(algos_metrics["attrAgm"]["exec_time"],elapsed)
end

function run_experiments(num_experiments,algos,CmtyVV,params)
    algos_metrics = Dict{String,Dict{String,Vector{Float64}}}()
    for algo in algos
        d = Dict(
                "F1" => Vector{Float64}(),
                "ARI" => Vector{Float64}(),
                "NMI" => Vector{Float64}(),
                "exec_time" => Vector{Float64}(),
            )
        algos_metrics[algo] = d 
    end

    G,edges_data,attr_data,M = MakeBenchmark.generate(CmtyVV,params)
    utils.save_graph(G,"output_graph.txt")
    utils.save_graph_label(G,"output_labels.txt")
    n_edges = ne(G)
    n_vertices = nv(G)
    Thres = 2*n_edges/( n_vertices * (n_vertices - 1) )
    for _ in 1:num_experiments
        # G,edges_data,attr_data,M = MakeBenchmark.generate(CmtyVV,params)
        # n_edges = ne(G)
        # n_vertices = nv(G)
        # Thres = 2*n_edges/( n_vertices * (n_vertices - 1) )
        # F_init = rand(Bool,nv(G),params.n_clusters)
        # F_init = zeros(Bool,nv(G),params.n_clusters)
        # F_init = utils.random_assign_unlabeled_vertices(F_init)

        # F_init = utils.sample_invertible_binary_matrix(nv(G),params.n_clusters)
        # CmtyVV = utils.get_cmtyvv(F_init)
        # utils.save_cmtyvv(CmtyVV,"INIT_cmtyvv.txt")

        params_dict = Dict(
            :G => G,
            :C => params.n_clusters,
            :iterations => 1000, 
            :lr => 0.005, 
            :display_loss => false, 
            :F_init => nothing, 
            :use_line_search => false,
            :Thres => 0.001,
            :run_original => true)

        println(">>>>>>>>>>>>>>>> Running BigClam")
        F_bc = evaluate_algorithm("BigClam",BigClam.fit,params_dict,M,algos_metrics,Thres)
        # cvv = utils.read_bipartite_file("INIT_cmtyvv.txt","\t")
        # F_init = utils.cmtyvv_to_membership_matrix(cvv,nv(G))
        # F_init = utils.random_assign_unlabeled_vertices(F_init)
        ############### BANERJEE ################

        println(">>>>>>>>>>>>>>>> Running Banerjee")
        _,D = size(attr_data)
        b = BanerjeeOverlapping(params.n_clusters, euclidean_distance, n_vertices, D)

        banerjee_dict = Dict(
            :b=>b, 
            :X=>attr_data,
            :n_clus=>params.n_clusters,
            :F_init=>utils.sample_invertible_binary_matrix(nv(G),params.n_clusters),
            :iterations=>1000,
            :original=>false
        )

        F_banerjee = evaluate_algorithm("Banerjee",fit,banerjee_dict,M,algos_metrics,Thres)

        ############### OURS ################

        println(">>>>>>>>>>>>>>>> Running Ours")
        F_init = attrAgmFit.get_best_init(Float64.(F_banerjee),Float64.(F_bc), 
                                        G, attr_data, edges_data)
        F_init_our = Float64.(F_init) .+ 0.1
        evaluate_proposed_GLM(  algos_metrics,
                                G,
                                attr_data,
                                edges_data,
                                M,
                                params.n_clusters,
                                Thres,
                                F_init_our
        )
    end
    return algos_metrics
end



function benchmark_varying_net( N,
                                K,
                                D_attr,
                                D_weight,
                                p_bipartite,
                                P_in_range,
                                PNoCom,
                                lambda_max,
                                attr_radius,
                                weight_radius,
                                num_experiments)
    

    df = DataFrame(
                    mean_F1 = Vector{Float64}(),
                    mean_ARI = Vector{Float64}(),
                    mean_NMI = Vector{Float64}(),
                    mean_exec_time = Vector{Float64}(),
                    std_F1 = Vector{Float64}(),
                    std_ARI = Vector{Float64}(),
                    std_NMI = Vector{Float64}(),
                    std_exec_time = Vector{Float64}(),
                    algo_name = Vector{String}(),
                    p_in = Vector{Float64}()
                )
    
    algos_stats = Dict{String,Dict{String,Vector{Float64}}}()
    algos = ["attrAgm","attrAgmRiemann","BigClam","Banerjee"]  
    for algo in algos
        d = Dict(
                "mean_F1" => Vector{Float64}(),
                "mean_ARI" => Vector{Float64}(),
                "mean_NMI" => Vector{Float64}(),
                "mean_exec_time" => Vector{Float64}(),
                "std_F1" => Vector{Float64}(),
                "std_ARI" => Vector{Float64}(),
                "std_NMI" => Vector{Float64}(),
                "std_exec_time" => Vector{Float64}(),
                "p_in" => Vector{Float64}()
            )
        algos_stats[algo] = d 
    end

    for P_in in P_in_range
        CmtyVV,params = MakeBenchmark.set_params(  N, 
                                                    K, 
                                                    D_attr, 
                                                    D_weight, 
                                                    p_bipartite, 
                                                    P_in, 
                                                    PNoCom, 
                                                    lambda_max, 
                                                    attr_radius, 
                                                    weight_radius)
        d = run_experiments(num_experiments,algos,CmtyVV,params)
        for (algo,metrics) in d
            for (metric_name,metric_vec) in metrics
                push!(algos_stats[algo]["mean_"*metric_name],mean(metric_vec))
                push!(algos_stats[algo]["std_"*metric_name],std(metric_vec))

                push!(df[!,"mean_"*metric_name], mean(metric_vec))
                push!(df[!,"std_"*metric_name], std(metric_vec))
            end
            push!(algos_stats[algo]["p_in"],P_in)
            
            push!(df[!,"p_in"], P_in)
            push!(df[!,"algo_name"], algo)
        end
    end
    # for writing
    p_min = minimum(P_in_range)
    p_max = maximum(P_in_range)
    output_filename = ""
    if D_weight == 1
        output_filename = "varying_net_l_max_$(lambda_max)_attr_R_$(attr_radius)_p_min_$(p_min)_p_max_$(p_max)_N_$(N)_Dattr_$(D_attr)_Dweight_$(D_weight).txt"
    else
        output_filename = "varying_net_weight_R_$(weight_radius)_attr_R_$(attr_radius)_p_min_$(p_min)_p_max_$(p_max)_N_$(N)_Dattr_$(D_attr)_Dweight_$(D_weight).txt"
    end
    CSV.write(output_filename,df)
    return algos_stats
end

function benchmark_varying_weight_R( N,
                                K,
                                D_attr,
                                D_weight,
                                p_bipartite,
                                P_in,
                                PNoCom,
                                lambda_max,
                                attr_radius,
                                weight_radius_range,
                                num_experiments)
    

    df = DataFrame(
                    mean_F1 = Vector{Float64}(),
                    mean_ARI = Vector{Float64}(),
                    mean_NMI = Vector{Float64}(),
                    mean_exec_time = Vector{Float64}(),
                    std_F1 = Vector{Float64}(),
                    std_ARI = Vector{Float64}(),
                    std_NMI = Vector{Float64}(),
                    std_exec_time = Vector{Float64}(),
                    algo_name = Vector{String}(),
                    weight_R = Vector{Float64}()
                )
    
    algos_stats = Dict{String,Dict{String,Vector{Float64}}}()
    algos = ["attrAgm","attrAgmRiemann","BigClam","Banerjee"]  
    for algo in algos
        d = Dict(
                "mean_F1" => Vector{Float64}(),
                "mean_ARI" => Vector{Float64}(),
                "mean_NMI" => Vector{Float64}(),
                "mean_exec_time" => Vector{Float64}(),
                "std_F1" => Vector{Float64}(),
                "std_ARI" => Vector{Float64}(),
                "std_NMI" => Vector{Float64}(),
                "std_exec_time" => Vector{Float64}(),
                "weight_R" => Vector{Float64}()
            )
        algos_stats[algo] = d 
    end

    for weight_radius in weight_radius_range
        CmtyVV,params = MakeBenchmark.set_params(   N, 
                                                    K, 
                                                    D_attr, 
                                                    D_weight, 
                                                    p_bipartite, 
                                                    P_in, 
                                                    PNoCom, 
                                                    lambda_max, 
                                                    attr_radius, 
                                                    weight_radius)        
        d = run_experiments(num_experiments,algos,CmtyVV,params)
        for (algo,metrics) in d
            for (metric_name,metric_vec) in metrics
                push!(algos_stats[algo]["mean_"*metric_name],mean(metric_vec))
                push!(algos_stats[algo]["std_"*metric_name],std(metric_vec))

                push!(df[!,"mean_"*metric_name], mean(metric_vec))
                push!(df[!,"std_"*metric_name], std(metric_vec))
            end
            push!(algos_stats[algo]["weight_R"],weight_radius)
            
            push!(df[!,"weight_R"], weight_radius)
            push!(df[!,"algo_name"], algo)
        end
    end
    # for writing
    r_min = minimum(weight_radius_range)
    r_max = maximum(weight_radius_range)
    output_filename = ""
    if D_weight == 1
        output_filename = "varying_wR_attr_R_$(attr_radius)_r_min_$(r_min)_r_max_$(r_max)_N_$(N)_Dattr_$(D_attr)_Dweight_$(D_weight).txt"
    else
        output_filename = "varying_wR_attr_R_$(attr_radius)_r_min_$(r_min)_r_max_$(r_max)_N_$(N)_Dattr_$(D_attr)_Dweight_$(D_weight).txt"
    end
    CSV.write(output_filename,df)
    return algos_stats
end

function benchmark_varying_attr( 
                                N,
                                K,
                                D_attr,
                                D_weight,
                                p_bipartite,
                                P_in,
                                PNoCom,
                                lambda_max,
                                attr_radius_range,
                                weight_radius,
                                num_experiments)
    

    df = DataFrame(
                    mean_F1 = Vector{Float64}(),
                    mean_ARI = Vector{Float64}(),
                    mean_NMI = Vector{Float64}(),
                    mean_exec_time = Vector{Float64}(),
                    std_F1 = Vector{Float64}(),
                    std_ARI = Vector{Float64}(),
                    std_NMI = Vector{Float64}(),
                    std_exec_time = Vector{Float64}(),
                    algo_name = Vector{String}(),
                    attr_R = Vector{Float64}()
                )
    
    algos_stats = Dict{String,Dict{String,Vector{Float64}}}()
    algos = ["attrAgm","attrAgmRiemann","BigClam","Banerjee"]  
    for algo in algos
        d = Dict(
                "mean_F1" => Vector{Float64}(),
                "mean_ARI" => Vector{Float64}(),
                "mean_NMI" => Vector{Float64}(),
                "mean_exec_time" => Vector{Float64}(),
                "std_F1" => Vector{Float64}(),
                "std_ARI" => Vector{Float64}(),
                "std_NMI" => Vector{Float64}(),
                "std_exec_time" => Vector{Float64}(),
                "attr_R" => Vector{Float64}()
            )
        algos_stats[algo] = d 
    end

    for attr_radius in attr_radius_range
        CmtyVV,params = MakeBenchmark.set_params(  N, 
                                                    K, 
                                                    D_attr, 
                                                    D_weight, 
                                                    p_bipartite, 
                                                    P_in, 
                                                    PNoCom, 
                                                    lambda_max, 
                                                    attr_radius, 
                                                    weight_radius)        
        d = run_experiments(num_experiments,algos,CmtyVV,params)
        for (algo,metrics) in d
            for (metric_name,metric_vec) in metrics
                push!(algos_stats[algo]["mean_"*metric_name],mean(metric_vec))
                push!(algos_stats[algo]["std_"*metric_name],std(metric_vec))

                push!(df[!,"mean_"*metric_name], mean(metric_vec))
                push!(df[!,"std_"*metric_name], std(metric_vec))
            end
            push!(algos_stats[algo]["attr_R"],attr_radius)
            
            push!(df[!,"attr_R"], attr_radius)
            push!(df[!,"algo_name"], algo)
        end
    end
    # for writing
    r_min = minimum(attr_radius_range)
    r_max = maximum(attr_radius_range)
    output_filename = ""
    if D_weight == 1
        output_filename = "varying_attr_r_min_$(r_min)_r_max_$(r_max)_N_$(N)_Dattr_$(D_attr)_Dweight_$(D_weight).txt"
    else
        output_filename = "varying_attr_r_min_$(r_min)_r_max_$(r_max)_N_$(N)_Dattr_$(D_attr)_Dweight_$(D_weight).txt"
    end
    CSV.write(output_filename,df)
    return algos_stats
end

D_weight = 2
N = 100
benchmark_varying_attr( N, ## NUM NODES 
                        3, ## NUM COMMUNITIES 
                        2, ## DIMENSION OF ATTRIBUTES
                        D_weight, ## DIMENSION OF WEIGHTS 
                        0.3, ## Probability of assigning nodes to clusters 
                        0.1, ## Probability of existing an edge inside each community 
                        0.1, ## Probability of connecting random nodes
                        10, ## lambda_max
                        range(0,8,8), ## range of attributes radius. The higher, the easier it is to split
                        0.1, ## weight radius/hipercube size
                        5 ## number of experiments
                        )
                        
benchmark_varying_net(
                      N, ## NUM NODES 
                      3, ## NUM COMMUNITIES 
                      2, ## DIMENSION OF ATTRIBUTES
                      D_weight, ## DIMENSION OF WEIGHTS 
                      0.3, ## Probability of assigning nodes to clusters 
                      range(0,0.5,8), ## Probability of existing an edge inside each community 
                      0.1, ## Probability of connecting random nodes
                      10,  ## lambda_max
                      0.1,  ## Attributes radius. The higher, the easier it is to split
                      0.1,  ## weight radius/hipercube size
                      5   ## number of experiments
                    )

# benchmark_varying_weight_R(
#                            100, ## NUM NODES 
#                            3, ## NUM COMMUNITIES 
#                            2, ## DIMENSION OF ATTRIBUTES
#                            D_weight, ## DIMENSION OF WEIGHTS 
#                            0.3, ## Probability of assigning nodes to clusters 
#                            0.1, ## Probability of existing an edge inside each community 
#                            0.1, ## Probability of connecting random nodes
#                            100, ## lambda_max
#                            1, ## Attributes radius. The higher, the easier it is to split
#                            range(0,100,8), ## weight radius/hipercube size
#                            5 ## number of experiments
#                         )
                        