"""
According to the paper stated here
https://www.cs.utexas.edu/~ml/papers/moc-submitted-05.pdf
"""

import networkx as nx
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

class banerjee_overlapping():
    def __init__(self,dist_func):
        self.n_clus = None
        self.dist_func = dist_func
        ## num of samples
        self.N = None 
        ## dimension of samples
        self.D = None
        
    def update_alphas(self,M):
        pi = np.mean(M,axis=0)
        alphas = (pi**M)*((1-pi)**(1-M))
        """
        same as
        alphas = np.zeros(F.shape)
        for i in range(alphas.shape[0]):
            for h in range(alphas.shape[1]):
                alphas[i][h] = (pi[h]**F[i][h])*((1-pi[h])**(1-F[i,h]))
                
        but more efficient
        """
        return alphas

    def dynamic_M(self,x,A):
        k = self.n_clus
        thread_active = np.zeros(k)
        thread_losses = np.zeros(k)
        thread_guess = np.zeros((k,k))
        for h in range(k):
            thread_active[h] = 1
            m0 = np.zeros(k)
            m0[h] = 1
            loss_h = self.dist_func(x,m0@A)
            for r in range(1,k):
                if thread_active[h] == 1:
                    new_candidates = np.inf * np.ones(k)
                    for p in range(k):
                        if m0[p] == 0:
                            m0[p] = 1
                            new_candidates[p] = self.dist_func(x,m0@A)
                            m0[p] = 0
                    p = np.argmin(new_candidates)
                    if loss_h <= new_candidates[p]:
                        thread_active[h] = 0
                    else:
                        m0[p] = 1
                        loss_h = new_candidates[p]
            thread_losses[h] = self.dist_func(x,m0@A)
            thread_guess[h,:] = m0
        best_guess = np.argmin(thread_losses)
        return thread_guess[best_guess,:]

    def update_M(self,X,M,A):
        for i in range(self.N):
            new_m = self.dynamic_M(X[i,:],A)
            if self.dist_func(X[i,:],new_m@A) < self.dist_func(X[i,:],M[i,:]@A):
                M[i,:] = new_m
        return M
    
    def update_A(self,X,M,A):
        def obj_func(a):
            A = a.reshape(self.n_clus,self.D)
            MA = M@A
            total = 0
            for i in range(self.N):
                total += self.dist_func(X[i,:],MA[i,:])
            return total
        res = minimize(obj_func,A.flatten())
        return res.x.reshape(self.n_clus,self.D)
         
    def update_M_relaxed(self,X,M,A):
        def obj_func_M(m):
            M = m.reshape(self.N,self.n_clus)
            MA = M@A
            total = 0
            for i in range(self.N):
                total += self.dist_func(X[i,:],MA[i,:])
            return total
        res = minimize(obj_func_M,M.flatten(),bounds=self.M_bounds)
        return res.x.reshape(self.N,self.n_clus)
    
    def log_likelihood(self,X,M,A,alphas):
        total = 0 
        for i in range(self.N):
            total += self.dist_func(X[i,:],M[i,:]@A)
        total -= np.log(alphas).sum()
        return -total
        
    def fit(self,X, n_clus=2, iterations = 100):
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.n_clus = n_clus
        M = np.random.randint(0,1,self.N*self.n_clus)\
            .reshape(self.N,self.n_clus)
        A = np.random.rand(self.n_clus,self.D)\
            .reshape(self.n_clus,self.D)
        losses = []
        for i in range(iterations):
            M = self.update_M(X,M,A)
            A = self.update_A(X,M,A)
            alphas = self.update_alphas(M)
            losses.append(self.log_likelihood(X,M,A,alphas))
        plt.plot(np.arange(len(losses)),losses)
        return M,A,alphas
    
    def fit_relaxed(self, X, n_clus=2, iterations = 100):
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.n_clus = n_clus
        self.M_bounds = [ (0,1) for _ in range(self.N*self.n_clus)]
        M = np.random.randint(0,1,self.N*self.n_clus)\
            .reshape(self.N,self.n_clus)
        A = np.random.rand(self.n_clus,self.D)\
            .reshape(self.n_clus,self.D)
        for i in range(iterations):
            M = self.update_M_relaxed(X,M,A)
            A = self.update_A(X,M,A)
            alphas = self.update_alphas(M)
        return M,A,alphas