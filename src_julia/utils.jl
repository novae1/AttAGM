__precompile__()

module  utils
using LinearAlgebra
using Distributions
using Graphs

    #### SOME AUXILIARY FUNCTIONS ####

    """
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/bipartite/generators.html#random_graph
    
    [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.

    n -> num nodes
    m -> num clusters (vertices in the other partition)
    p -> probability of edge
    """
    function generate_bipartite_graph_random(n::Int64,m::Int64,p::Float64)
        CmtyVV = [Vector{Int64}() for i in 1:m ]
        nodes_cmty = [Set{Int64}() for i in 1:n]
        lp = log(1.0 - p)
        v = 0
        w = -1
        while v < n
            lr = log(1.0 - rand())
            w = w + 1 + floor(Int,lr / lp)
            while w >= m && v < n
                w = w - m
                v = v + 1
            end
            if v < n
                push!(CmtyVV[w+1],v+1)## +1 because julia array starts at 1, as opposed to python
                push!(nodes_cmty[v+1],w+1)
            end
        end
        return CmtyVV, nodes_cmty
    end

    """
    n -> num nodes
    m -> num clusters (vertices in the other partition)
    k -> num edges
    """
    function generate_bipartite_graph_gnmk(n,m,k)
        CmtyVV = [Vector{Int64}() for i in 1:m ]
        nodes_cmty = [Set{Int64}() for i in 1:n]
        vertices = 1:n
        comunities = 1:m
        edge_count = 0
        while edge_count < k
            # generate random edge,u,v
            u = sample(vertices)
            c = sample(comunities)
            if u in CmtyVV[c]
                continue
            else
                push!(CmtyVV[c],u)
                push!(nodes_cmty[u],c)
                edge_count += 1
            end
        end
        return CmtyVV, nodes_cmty
    end

    function generate_bipartite_graph_random_non_homogeneous(n::Int64,m::Int64,p::Vector{Float64})
        CmtyVV = Vector{Vector{Int64}}()
        nodes_cmty = [Set{Int64}() for i in 1:n]
        for i in 1:m
            CmtyV, v_cmty = generate_bipartite_graph_random(n,1,p[i])
            println(size(CmtyVV),size(nodes_cmty))
            nodes_cmty = union.(nodes_cmty,v_cmty)
            push!(CmtyVV,CmtyV[1])
        end
        return CmtyVV, nodes_cmty
    end

    function random_assign_unlabeled_vertices(CmtyVV,nodes_cmty)
        N = length(nodes_cmty)
        K = size(CmtyVV)[1]
        for v in 1:N
            if isempty(nodes_cmty[v])
                c = sample(1:K)
                push!(CmtyVV[c],v)
                push!(nodes_cmty[v],c)
            end
        end
        return CmtyVV, nodes_cmty 
    end

    function random_assign_unlabeled_vertices(F)
        N, K = size(F)
        needed_to_fill = 0
        for v in 1:N
            if sum(F[v,:]) == 0
                needed_to_fill += 1
                c = sample(1:K)
                F[v,c] = 1
            end
        end
        println("ASSIGNED ", needed_to_fill, "NODES")
        return F 
    end

    function generate_bipartite_graph(num_nodes=300,
                                    n_clusters=3,
                                    num_edges=nothing,
                                    bipartite_prob=nothing,
                                    CProbV=nothing)
        if bipartite_prob != nothing
            CmtyVV, nodes_cmty = generate_bipartite_graph_random(num_nodes,n_clusters,bipartite_prob)
        elseif num_edges != nothing
            CmtyVV, nodes_cmty = generate_bipartite_graph_gnmk(num_nodes,n_clusters,num_edges)
        elseif CProbV != nothing
            CmtyVV, nodes_cmty = generate_bipartite_graph_random_non_homogeneous(num_nodes,n_clusters,CProbV)
            println(size(CmtyVV),size(nodes_cmty))
        end
        CmtyVV, nodes_cmty  = random_assign_unlabeled_vertices(CmtyVV,nodes_cmty)
        return CmtyVV
    end

    function get_binary_membership_matrix(CmtyVV,N)
        M = cmtyvv_to_membership_matrix(CmtyVV::Vector{Vector{Int64}},N::Int64)
        return Bool.(M)
    end

    # Specify equidistant centers of covariates around a circle of radius R
    function get_uniform_circle_coordinates(n_clusters,radius)
        centers = zeros(Float64,(n_clusters,2))
        for i in 0:n_clusters-1
            centers[i+1,:] = [radius*cos(2*pi*i/n_clusters),
                            radius*sin(2*pi*i/n_clusters)]
        end
        return centers
    end

    function get_nodes_cmty(CmtyVV::Vector{Vector{Int64}},N::Int64)
        nodes_cmty = [Set{Int64}() for i in 1:N]
        n_cmty = size(CmtyVV)[1]
        for c in 1:n_cmty
            for u in CmtyVV[c]
                push!(nodes_cmty[u],c)
            end
        end
        return nodes_cmty
    end

    function cmtyvv_to_membership_matrix(CmtyVV::Vector{Vector{Int64}},N::Int64)
        n_cmty = size(CmtyVV)[1]
        MM = zeros(Bool, N, n_cmty)
        for c in 1:n_cmty
            for u in CmtyVV[c]
                MM[u,c] = 1
            end
        end
        return MM
    end 
   
    function generate_hypercube(samples, dimensions)
        """Returns distinct binary samples of length dimensions."""
        out = sample(0:2^dimensions-1, samples; replace=false)
        coordinates = Matrix{Float64}(undef,samples,dimensions)
        for i in 1:samples 
            coordinates[i,:] = [parse(Float64,j*".0") for j in bitstring(convert(UInt8,out[i]))[8-dimensions+1:8]]
        end
        return coordinates
    end

    function get_cmtyvv(F::Matrix{Bool})
        N,K = size(F)
        CmtyVV = Vector{Vector{Int64}}()
        for c in 1:K    
            CmtyV = Vector{Int64}()
            for u in 1:N
                if F[u,c] > 0
                    push!(CmtyV,u)
                end
            end
            push!(CmtyVV,CmtyV)
        end
        return CmtyVV
    end
    function save_cmtyvv(CmtyVV::Vector{Vector{Int64}},fpath)
        n_cmty = size(CmtyVV)[1]
        open(fpath, "w") do file
            for c in 1:n_cmty
                for u in CmtyVV[c]
                    write(file,"$(u)\t")    
                end
                write(file,"\n")
            end
        end     
    end

    function save_graph(G,fpath)
        open(fpath, "w") do file
            write(file,"# Directed Node Graph\n")
            write(file,"# Artificial attrAGM graph (graph is undirected, each edge is saved twice)\n")
            write(file,"# Nodes: $(nv(G))    Edges: $(ne(G))\n")
            write(file,"# SrcNId	DstNId\n")

            for edge in edges(G)
                u, v = src(edge), dst(edge)
                edge_str = "$(u)\t$(v)\n"
                write(file,edge_str)
                edge_str = "$(v)\t$(u)\n"
                write(file,edge_str)
            end
        end
    end

    function save_graph_label(G,fpath)
        open(fpath, "w") do file
            for vertice in vertices(G)
                label_str = "$(vertice)\t$(vertice)\n"
                write(file,label_str)
            end
        end
    end

    """
    This function reads a txt file and return a vector of vectors.  
    The first dimension is the comunity and the second is the vertices list
    """
    function read_bipartite_file(filename,delim=" ")
        CmntyVV = Vector{Vector{Int64}}()
        open(filename) do io
            while !eof(io)
                line = readline(io)
                vertices_list_str = split(line, delim)
                vertices_list_str = filter(e->!(e in [""]),vertices_list_str)
                vertices_list_int = parse.(Int64,vertices_list_str)
                push!(CmntyVV,vertices_list_int)
            end
        end
        return CmntyVV
    end

    function sample_invertible_binary_matrix(N,C)
        F = zeros(Bool,N,C)
        F = random_assign_unlabeled_vertices(F)
        A = F' * F
        while isapprox(det(A), 0)
            u = sample(1:N)
            c = sample(1:C)
            if sum(F[u]) > 1
                F[u,c] != F[u,c]
            end 
            A = F' * F
        end
        return F
    end
end
