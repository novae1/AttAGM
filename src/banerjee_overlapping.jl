using JuMP
using Gurobi
using Ipopt
using AmplNLWriter, SCIP, Bonmin_jll, HiGHS, Couenne_jll
using Random
using Distances
using Distributions

mutable struct BanerjeeOverlapping
    n_clus::Int
    dist_func
    N::Int
    D::Int
end

function set_BanerjeeOverlapping_func(dist_func)
    return BanerjeeOverlapping(0, dist_func, 0, 0)
end

function update_alphas(M)
    pi = mean(M, dims=1)
    alphas = (pi.^M) .* ((1 .- pi).^((1 .- M)))
    return alphas
end

function dynamic_M(b::BanerjeeOverlapping, x, A)
    k = b.n_clus
    thread_active = zeros(Int, k)
    thread_losses = zeros(Float64, k)
    thread_guess = zeros(Float64, k, k)
    
    for h in 1:k
        thread_active[h] = 1
        m0 = zeros(Float64, k)
        m0[h] = 1
        loss_h = b.dist_func(x, m0' * A)
        
        for r in 2:k
            if thread_active[h] == 1
                new_candidates = fill(Inf, k)
                for p in 1:k
                    if m0[p] == 0
                        m0[p] = 1
                        new_candidates[p] = b.dist_func(x, m0' * A)
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
        thread_losses[h] = b.dist_func(x, m0' * A)
        thread_guess[h, :] = m0
    end
    best_guess = argmin(thread_losses)
    return thread_guess[best_guess, :]
end

function dynamic_M_MCMC(b::BanerjeeOverlapping, x, A, m0)
    k = b.n_clus
    """
    transitions = zeros(Bool,k,k)
    losses = zeros(Float64,k)
    for h in 1:k
        temp = m0
        temp[h] = !temp[h] 
        transitions[h,:] = temp
        losses[h] = b.dist_func(x, temp' * A)
    end
    # Compute probabilities proportional to exp(-loss)
    probabilities = exp.(-losses)
    probabilities /= sum(probabilities)  # Normalize to make it a valid probability distribution
    a = Categorical(probabilities)
    choice = rand(a)
    return transitions[choice,:]
    """
    choice = rand(1:k)
    temp = m0
    temp[choice] = !temp[choice]
    return temp
         
end

function update_M(b::BanerjeeOverlapping, X, M, A)
    for i in 1:b.N
        new_m = dynamic_M(b, X[i, :], A)
        if b.dist_func(X[i, :], new_m' * A) < b.dist_func(X[i, :], M[i, :]' * A)
            M[i, :] = new_m
        end
    end
    return M
end

function update_A(b::BanerjeeOverlapping, X, M, A)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    register(model, :sum, 1, sum; autodiff = true)
    @variable(model, A_var[i=1:b.n_clus, j=1:b.D] , base_name="A_var", start=A[i,j])
    # obj_fun(A_) = sum(b.dist_func(X[i, :], M[i, :]'*A_var) for i in 1:b.N)
    # register(model, :obj_fun, 2, obj_fun; autodiff=true)
    #println("==============>",sum( sqrt(sum((X[i, :]' - M[i, :]'*A).^2)) for i in 1:b.N))
    @NLobjective(model, Min, sum( sum((X[i,j] - (M[i,:]' * A_var)[j])^2 for j in 1:b.D) for i in 1:b.N) )
    optimize!(model)
    return value.(A_var)
end

function update_M_relaxed(b::BanerjeeOverlapping, X, M, A)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    register(model, :sum, 1, sum; autodiff = true)
    @variable(model, M_var[i=1:b.N, j=1:b.n_clus] , base_name="M_var", start=M[i,j],lower_bound=0,upper_bound=1)
    @NLobjective(model, Min, sum( sum((X[i,j] - (M_var[i,:]' * A)[j])^2 for j in 1:b.D) for i in 1:b.N) )
    optimize!(model)
    return value.(M_var)
end

"""
Solves exactly for the whole matrix
    TOO COMPLEX! No solver was able to actualy find the minimum solution
"""
function update_M_exact(b::BanerjeeOverlapping, X, M, A)
    #model = Model(HiGHS.Optimizer)
    #model = Model(Gurobi.Optimizer)
    #model = Model(() -> AmplNLWriter.Optimizer(Bonmin_jll.amplexe))
    #model = Model(SCIP.Optimizer)
    #model = Model(Ipopt.Optimizer)
    model = Model(() -> AmplNLWriter.Optimizer(Couenne_jll.amplexe))
    set_silent(model)
    register(model, :sum, 1, sum; autodiff = true)
    @variable(model, M_var[i=1:b.N, j=1:b.n_clus], Bin, base_name="M_var", start=M[i,j])
    @NLobjective(model, Min, sum( sqrt(sum((X[i,j] - (M_var[i,:]' * A)[j])^2 for j in 1:b.D)) for i in 1:b.N) )
    optimize!(model)
    return value.(M_var)
end

function update_M_MCMC(b::BanerjeeOverlapping, X, M, A)
    for i in 1:b.N
        new_m = dynamic_M_MCMC(b, X[i, :], A, M[i,:])
        new_loss = b.dist_func(X[i, :], new_m' * A)
        prev_loss = b.dist_func(X[i, :], M[i, :]' * A)
        if new_loss < prev_loss || rand() < exp(prev_loss - new_loss)
            M[i, :] = new_m
        end
    end
    return M
end

function log_likelihood(b::BanerjeeOverlapping, X, M, A, alphas)
    total = 0.0
    for i in 1:b.N
        total += b.dist_func(X[i, :], M[i, :]' * A)
    end
    total -= sum(log.(alphas))
    return -total
end


"""
Just like in the original paper from Banerjee:
Model-based Overlapping Clustering
"""
function fit(b::BanerjeeOverlapping, X::Matrix{Float64}, n_clus::Int64, iterations::Int64, F_init::Matrix{Bool}, original=true,make_plot=false)
    b.N = size(X, 1)
    b.D = size(X, 2)
    b.n_clus = n_clus
    M = rand(Bool, b.N, b.n_clus)
    if F_init !== nothing
        M = F_init
    end

    # A = rand(b.n_clus, b.D)
    A = inv(M'*M)*M'*X
    # alphas = rand(b.N, b.n_clus)
    alphas = update_alphas(M)
    project_M = update_M
    if original
        project_M = update_M
    else
        #project_M = update_M_exact REMOVED BECAUSE NO SOLVER WAS ABLE TO SOLVE
        project_M = update_M_MCMC
    end    
    losses = Float64[]
    for i in 1:iterations
        M = project_M(b, X, M, A)
        # A = inv(M'*M)*M'*X
        A = update_A(b, X, M, A)
        alphas = update_alphas(M)
        Cur_ll = log_likelihood(b,X,M,A,alphas)
        push!(losses, Cur_ll)
    end
    if make_plot
        xs = 1:length(losses)
        plot(xs, losses)
        xlabel("iterations")
        ylabel("log_likelihood")
        display()
    end
    return M, A, alphas
end

function fit(;b=nothing, X=nothing, n_clus=nothing, iterations=nothing, F_init=nothing, original=true)
    return fit(b,X,n_clus,iterations,F_init,original)[1]
end


"""
Like in the original paper from Banerjee:
Model-based Overlapping Clustering

=> BUT the M matrix is between 0,1
0 <= M <= 1
"""
function fit_relaxed(b::BanerjeeOverlapping, X, n_clus=2, iterations=100)
    b.N = size(X, 1)
    b.D = size(X, 2)
    b.n_clus = n_clus
    
    M = rand(Bool, b.N, b.n_clus)
    A = rand(b.n_clus, b.D)
    alphas = rand(b.N, b.n_clus)
    for i in 1:iterations
        M = update_M_relaxed(b, X, M, A)
        A = update_A(b, X, M, A)
        alphas = update_alphas(M)
    end
    
    return M, A, alphas
end

function TEST_fit_relaxed(b::BanerjeeOverlapping, X, n_clus=2, iterations=100)
    b.N = size(X, 1)
    b.D = size(X, 2)
    b.n_clus = n_clus
    
    M = rand(Bool, b.N, b.n_clus)
    A = rand(b.n_clus, b.D)
    alphas = rand(b.N, b.n_clus)
    for i in 1:iterations
        M = TEST_update_M_relaxed(b, X, M, A)
        A = update_A(b, X, M, A)
        alphas = update_alphas(M)
    end
    
    return M, A, alphas
end

"""
So far none of the solvers tested can make it
Formulates a single opt problem with M binary
"""
function fit_joint(b::BanerjeeOverlapping, X, n_clus=2)
    b.N = size(X, 1)
    b.D = size(X, 2)
    b.n_clus = n_clus
    
    M = rand(b.N, b.n_clus)
    A = rand(b.n_clus, b.D)
    alphas = rand(b.N, b.n_clus)
    init_pi = mean(M,dims=1)
    #model = Model(Ipopt.Optimizer)
    #model = Model(SCIP.Optimizer)
    model = Model(() -> AmplNLWriter.Optimizer(Bonmin_jll.amplexe))
    #model = Model(HiGHS.Optimizer)
    #model = Model(Gurobi.Optimizer)
    #set_silent(model)

    register(model, :sum, 1, sum; autodiff = true)
    @variable(model, M_var[i=1:b.N, j=1:b.n_clus], Bin, base_name="M_var", start=M[i,j])
    @variable(model, A_var[i=1:b.n_clus, j=1:b.D] , base_name="A_var", start=A[i,j])
    @variable(model, alpha_var[i=1:b.N, j=1:b.n_clus],
                    base_name="alpha_var",
                    lower_bound=0,
                    upper_bound=1)
    @variable(model, pi_var[h=1:b.n_clus], 
                base_name="pi_var",
                start=init_pi[h],
                lower_bound=1/b.N,
                upper_bound=1)

    a_mean = 1/b.N .* ones(b.N)
    println("========================================================")
    println(size(a_mean))
    @constraint(model, [h=1:b.n_clus],
            pi_var[h] == a_mean' * M_var[:,h]
        )    
    @NLconstraint(model,[i=1:b.N, h=1:b.n_clus], 
            alpha_var[i,h] == (pi_var[h] ^ M_var[i,h])*((1 - pi_var[h]) ^ (1 - M_var[i,h]) )
        )

    @NLobjective(model, Min, 
        sum( sqrt(sum((X[i,j] - (M_var[i,:]' * A_var)[j])^2 for j in 1:b.D)) for i in 1:b.N) 
        - sum( alpha_var[i,h] for i in 1:b.N for h in 1:b.n_clus) 
        )
    optimize!(model)
    M = value.(M_var)
    A = value.(A_var)
    alphas = update_alphas(M)
    return M, A, alphas
end

"""
Formulates a single opt problem with M continuous
"""
function fit_joint_relaxed(b::BanerjeeOverlapping, X, n_clus=2)
    b.N = size(X, 1)
    b.D = size(X, 2)
    b.n_clus = n_clus
    
    M = rand(Bool, b.N, b.n_clus)
    A = rand(b.n_clus, b.D)
    alphas = rand(b.N, b.n_clus)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    register(model, :sum, 1, sum; autodiff = true)
    @variable(model, M_var[i=1:b.N, j=1:b.n_clus] , base_name="M_var", start=M[i,j],lower_bound=0,upper_bound=1)
    @variable(model, A_var[i=1:b.n_clus, j=1:b.D] , base_name="A_var", start=A[i,j])
    @variable(model, alpha_var[i=1:b.N, j=1:b.n_clus] , base_name="alpha_var",lower_bound=0,upper_bound=1)
    @variable(model, pi_var[h=1:b.n_clus] , base_name="pi_var",lower_bound=1/b.N,upper_bound=1)

    a_mean = 1/b.N .* ones(b.N)
    println("========================================================")
    println(size(a_mean))
    @constraint(model, [h=1:b.n_clus],
            pi_var[h] == a_mean' * M_var[:,h]
        )    
    @NLconstraint(model,[i=1:b.N, h=1:b.n_clus], 
            alpha_var[i,h] == (pi_var[h] ^ M_var[i,h])*((1 - pi_var[h]) ^ (1 - M_var[i,h]) )
        )
    
    @NLobjective(model, 
                Min, 
                sum( 
                    sqrt(sum((X[i,j] - (M_var[i,:]' * A_var)[j])^2 for j in 1:b.D)) 
                        for i in 1:b.N
                    )
                - sum( alpha_var[i,h] for i in 1:b.N for h in 1:b.n_clus) 
                )
    optimize!(model)
    M = value.(M_var)
    A = value.(A_var)
    alphas = value.(alpha_var)
    return M, A, alphas
end