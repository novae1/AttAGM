"""
Taken from 
https://github.com/ryputtam/Community-Detection-with-BigCLAM/blob/main/Community_Detection_BigCLAM.ipynb

and

https://github.com/RobRomijnders/bigclam/
"""

import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
MinVal = 0
MaxVal = 1000
def sigm(x):
    term = None
    try:
        term = np.divide(np.exp(-1.*x),1.-np.exp(-1.*x))
    except:
        term = np.zeros(x.shape)
    return term

def log_likelihood_row(F, G, u, sum_Fv):
    total_edges = 0
    for v in G.neighbors(u):
        total_edges += np.log(1 - np.exp( -F[u].dot(F[v]) ) )
    
    total_non_edges = 0
    # for v in nx.non_neighbors(G,u):
    #     total_non_edges += F[u].dot(F[v])
    sum_Fv_ = sum_Fv.copy()
    for v in G.neighbors(u):
        sum_Fv_ -= F[v]
    sum_Fv_ -= F[u]
    total_non_edges = F[u].dot(sum_Fv_)
    return total_edges - total_non_edges


def log_likelihood(F, G, sum_Fv):
    """implements equation 2 of 
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
    total = np.sum([log_likelihood_row(F, G, u, sum_Fv) for u in G.nodes()])
    return total

def gradient(F, G, u):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
      * i indicates the row under consideration
    
    The many forloops in this function can be optimized, but for
    educational purposes we write them out clearly
    """
    N, C = F.shape

    sum_neigh = np.zeros((C,))
    for v in G.neighbors(u):
        dotproduct = F[v].dot(F[u])
        sum_neigh += F[v]*sigm(dotproduct)

    sum_nneigh = np.zeros((C,))
    #Speed up this computation using eq.4
    for v in nx.non_neighbors(G,u):
        sum_nneigh += F[v]

    grad = sum_neigh - sum_nneigh
    grad = np.clip(grad,-10,10)
    return grad

def line_search(F,G,u,DeltaV,GradV,Alpha,Beta,MaxIter, sum_Fv):
    StepSize = 1.0
    InitLikelihood = log_likelihood_row(F,G,u,sum_Fv)
    NewVarV = np.zeros_like(DeltaV)
    for i in range(MaxIter):
        for j in range(NewVarV.shape[0]):
            NewVal = F[u,j] + StepSize * DeltaV[j]
            if (NewVal < MinVal):
                NewVal = MinVal
            if (NewVal > MaxVal):
                NewVal = MaxVal
            NewVarV[j]= NewVal
        F_new=F.copy()   
        F_new[u,:] = NewVarV 
        if log_likelihood_row(F_new,G,u,sum_Fv) < InitLikelihood + Alpha * StepSize * GradV.dot(DeltaV):
            StepSize *= Beta
        else:
            break
        if (i == MaxIter - 1):
            StepSize = 0
            break
    return StepSize


def gradient_efficient(F, G, u, sum_Fv):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
      * i indicates the row under consideration
    
    The many forloops in this function can be optimized, but for
    educational purposes we write them out clearly
    """
    N, C = F.shape

    sum_neigh = np.zeros((C,))
    for v in G.neighbors(u):
        dotproduct = F[v].dot(F[u])
        sum_neigh += F[v]*sigm(dotproduct)

    sum_nneigh = sum_Fv.copy()
    #Speed up this computation using eq.4
    for v in nx.neighbors(G,u):
        sum_nneigh -= F[v]
    sum_nneigh -= F[u]
    grad = sum_neigh - sum_nneigh
    grad = np.clip(grad,-10,10)
    return grad

def train_efficient(G, C, iterations = 100,lr=0.005, display_loss=False, F_init=None, use_line_search=True):
    F = None
    if F_init is not None:
        F = F_init
    else:
        # initialize an F
        F = np.random.rand(G.number_of_nodes(),C)
    losses = []
    sum_Fv = np.zeros((C,))
    for u in G.nodes():
        sum_Fv += F[u,:]
    for n in range(iterations):
        for u in G.nodes():
            grad = gradient_efficient(F, G, u, sum_Fv.copy())
            prev_Fu = F[u,:]
            if use_line_search:
                alpha = line_search(F,G,u,grad,grad,0.05,0.3,5,sum_Fv)
                F[u] += alpha*grad
            else:
                F[u] += lr*grad
            F[u,:] = np.maximum(0.001, F[u,:]) # F should be nonnegative
            new_Fu = F[u,:]
            delta_Fu = (new_Fu - prev_Fu)
            sum_Fv += delta_Fu
        if display_loss:
            ll = log_likelihood(F, G,sum_Fv)
            losses.append(ll)
            #print('At step %5i/%5i ll is %5.3f'%(n, iterations, ll))
    if display_loss:
        xs = [x for x in range(len(losses))]
        plt.plot(xs, losses)
        plt.show()
        # Make sure to close the plt object once done
        plt.close()
    return F

"""

def train(G, C, iterations = 100,lr=0.005, display_loss=False, F_init=None, use_line_search=False):
    F = None
    if F_init is not None:
        F = F_init
    else:
        # initialize an F
        F = np.random.rand(G.number_of_nodes(),C)
    losses = []
    for n in range(iterations):
        for u in G.nodes():
            grad = gradient(F, G, u)
            if use_line_search:
                alpha = line_search(F,G,u,grad,grad,0.05,0.3,5)
                F[u] += alpha*grad
            else:
                F[u] += lr*grad
            F[u] = np.maximum(0.001, F[u]) # F should be nonnegative
        if display_loss:
            ll = log_likelihood(F, G)
            losses.append(ll)
            #print('At step %5i/%5i ll is %5.3f'%(n, iterations, ll))
    if display_loss:
        xs = [x for x in range(len(losses))]
        plt.plot(xs, losses)
        plt.show()
        # Make sure to close the plt object once done
        plt.close()
    return F



# A method to update each node in factor matrix as per bigclam v2.0 formula
# Input : Completed Initliazed FACTOR mAtrix, number of iterations, learning rate
def matrix_factorization(G,C,iterations = 100,lr = 0.005):
  n = G.number_of_nodes() 
  fact_mat = np.random.rand(n,C)
  I = iterations
  for iter in range(0,500):   # Taking max i as 500
    if iter < I or all([q>0.001 for q in all_fu_change]): # Checking if i<I or percentage change of fu for all fu <0.001%
      all_fu_change = []
      #print(iter)
      fw=[0]*C
      for w in fact_mat:
        fw = np.add(fw,w) # Finidnf sigma fw term
      for i in range(0,n): # For each node
        fu = fact_mat[i,] # fu value
        fv_neig=[0]*C
        first_term = [0]*C
        for j in nx.neighbors(G,i):
          fv_neig=np.add(fv_neig,fact_mat[j]) # Adding all neighbor nodes
          try:
            term = fact_mat[j]*(math.exp(-(np.dot(fu,fact_mat[j])))/(1-math.exp(-(np.dot(fu,fact_mat[j]))))) # first term in gradient equation
          except: #ZeroDivisionError:
            term = 0 #fact_mat[int(j)]*(math.exp(-(np.dot(fu,fact_mat[int(j)]))))
          #print(term)
          first_term = np.add(first_term,term) # Adding all the first terms
        fv_not_neig = np.subtract(np.subtract(fw,fu),fv_neig) # fc not neighbors value calulatio
        grad_fu = np.subtract(first_term,fv_not_neig) # Finiding the fradient
        fu = fu + (grad_fu*lr) # updating fu
        before = np.sum(fact_mat[i,]) # getting the prev values of fu
        for y in range(0,C):
          if fu[y]< 0:
            fact_mat[i,y] = 0 # projecting fu values to if the update value is neagtive
          else:
            fact_mat[i,y] = fu[y]      # if positving, assiging the new values
        after = np.sum(fact_mat[i,]) # getting the updated value of fu
        fu_change = ((after - before)/before)*100  # calculating the percentage change
        all_fu_change.append(fu_change)
  return fact_mat # returning factor matrix after all iterations or if the change in percent <0.001
"""