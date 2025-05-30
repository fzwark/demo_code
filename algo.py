import numpy as np
from scipy.optimize import minimize

class Router():
    def __init__(self, ann, eps, M, length, base_data, B, alpha, models):
        self.ann = ann 
        self.index = 0
        self.M = M
        self.alpha = alpha
        self.eps = eps
        self.length = length
        self.base_data = base_data
        self.optimal = 0
        self.models = models
        self.B = B
        self.hatd = np.zeros((self.M, self.length))
        self.hatg = np.zeros((self.M, self.length))
        self.gamma = np.zeros(self.M)

    def learning(self, query):
        i = np.random.randint(0, self.M + 1)
        self.ps_esimate(query)

        return i
    
    def calculate(self, indices, distances):
        indices = np.array(indices, dtype=int).tolist()
        data = self.base_data.select(indices)
        d_vals = np.array([np.array(data[model]) for model in self.models])
        g_vals = np.array([np.array(data[f"{model}|total_cost"]) for model in self.models])
        d = np.mean(d_vals, axis=1).tolist()
        g = np.mean(g_vals, axis=1).tolist()

        return d, g

    def ps_esimate(self, query):
        indices, distances = self.ann.search(query)
        d, g = self.calculate(indices, distances)
        self.hatd[:, self.index] = d
        self.hatg[:, self.index] = g

    def optimize(self):

        def F_gamma(gamma):
            term1 = self.eps * np.dot(gamma, self.B)
            scores = (self.hatd[:, :self.index].T * self.alpha) - (self.hatg[:, :self.index].T * gamma)
            term2 = np.sum(np.max(scores, axis=1))
            return term1 + term2

        # Initial guess
        gamma_init = np.ones(self.M)/self.M
        bounds=[(0, 1)] * self.M
        
        result = minimize(F_gamma, gamma_init, method='L-BFGS-B', bounds=bounds)
        return result.x


    def routing(self, query):
        if self.index <= self.eps * self.length:
            i = self.learning(query)
        else:
            if self.optimal == 0:
                self.gamma = self.optimize()
                self.optimal = 1
                print(self.gamma)
            self.ps_esimate(query)
            scores = self.hatd[:, self.index] * self.alpha - self.gamma * self.hatg[:, self.index]
            scores = scores.flatten() 
            i  = np.argmax(scores)
        
        self.index += 1
        return i