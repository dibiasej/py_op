from abc import ABC, abstractmethod
import numpy as np

# We need to add a bachelier simulation if we can

class Simulation(ABC):

    @abstractmethod
    def paths(self):
        pass

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def put(self):
        pass

class BlackScholesSimulation(Simulation):
        
    def paths(self, S: float, T: float, sigma: float, r: float, M: int, I: int) -> list:
        dt = T / M
        S_paths = np.zeros((M + 1, I))
        S_paths[0] = S
        rn = np.random.standard_normal(S_paths.shape)
        # euler discretization 
        for t in range(1, M + 1):   # 1
            S_paths[t] = S_paths[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t])  # 2
        return S_paths
    
    def call(self, S: float, K: int, T: float, sigma: float, r: float = .06, q: float = 0, M = 1000, I = 10000) -> float:

        paths: list = self.paths(S, T, sigma, r, M, I)
    
        return np.exp(-r * T) * np.maximum(paths[-1] - K, 0).mean()

    def put(self, S: float, K: int, T: float, sigma: float, r: float = .05, q: float = 0, M = 1000, I = 10000) -> float:

        paths: list = self.paths(S, T, sigma, r, M, I)

        return np.exp(-r * T) * np.maximum(K - paths[-1], 0).mean()
    
class HestonSimulation(Simulation):

    def paths(self, S: float, T: float, sigma: float, r: float, v0: float, kappa: float, theta: float, rho: float, M: int, I: int):

        corr_mat = np.zeros((2, 2))
        corr_mat[0, :] = [1.0, rho]
        corr_mat[1, :] = [rho, 1.0]
        cho_mat = np.linalg.cholesky(corr_mat)

        dt = T / M

        ran_num = np.random.standard_normal((2, M + 1, I))

        S_paths = np.zeros_like(ran_num[0])
        v = np.zeros_like(ran_num[0])

        S_paths[0] = S
        v[0] = v0

        for t in range(1, M + 1):
            ran = np.dot(cho_mat, ran_num[:, t, :])

            v[t] = (v[t - 1] +
                    kappa * (theta - np.maximum(v[t - 1], 0)) * dt +
                    sigma * np.sqrt(np.maximum(v[t - 1], 0)) *
                    np.sqrt(dt) * ran[1])
            
            S_paths[t] = S_paths[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
                                    np.sqrt(v[t]) * ran[0] * np.sqrt(dt))
            
        return S_paths
    
    def call(self, S: float, K: int, T: float, sigma: float, r: float = .05, q: float = 0, v0: float = .04, kappa: float = 2, theta: float = .04, rho: float = -.7, M: int = 1000, I: int = 10000) -> float:
        """
        Note: In this method we gave default arguments for the parameters v0, kappa, theta and rho but in reality we should not do this and pass in specific args we optimized to find
        """
        paths: list = self.paths(S, T, sigma, r, v0, kappa, theta, rho, M, I)

        #print(f"In Heston price: {np.exp(-r * T) * np.maximum(paths[-1] - K, 0).mean()}")
    
        return np.exp(-r * T) * np.maximum(paths[-1] - K, 0).mean()
    
    def put(self, S: float, K: int, T: float, sigma: float, r: float = .05, q: float = 0, v0: float = .04, kappa: float = 2, theta: float = .04, rho: float = -.7, M: int = 1000, I: int = 10000) -> float:
        """
        Note: In this method we gave default arguments for the parameters v0, kappa, theta and rho but in reality we should not do this and pass in specific args we optimized to find
        """
        paths: list = self.paths(S, T, sigma, r, v0, kappa, theta, rho, M, I)

        return np.exp(-r * T) * np.maximum(K - paths[-1], 0).mean()
    
class VarianceGammaSimulation(Simulation):

    def paths(self, sigma, v, theta, S0, T, r, q, I, M):

        dt = T / M
        w = np.log(1-theta*v-0.5*v*sigma**2)/v
        S = np.zeros((M + 1, I))

        lns0 = np.log(S0)
        S[0] = np.log(S0)
        xt = 0

        rn = np.random.standard_normal(S.shape)
        gamma = np.random.gamma(dt/v, v, S.shape)

        j = np.array([i for i in range(I)])

        Tj = dt*(j+1)
        

        for t in range(1, M + 1):

            xt += theta*gamma[t] + sigma*np.sqrt(gamma[t])*rn[t]

            S[t] = S[0] + (r-q+w)*Tj + xt

        price = np.exp(S)
        return price
    
class SABRSimulation(Simulation):

    def paths(self, S0, alpha, rho, sigma, vol_vol, v, beta, theta, T, r, q, I, M):
    
        dt = T / M
        square_root_dt = np.sqrt(dt)

        va = np.zeros((M + 1, I))
        va_ = np.zeros((M + 1, I))
        va[0] = alpha
        va_[0] = alpha
        rn1 = np.random.standard_normal(va.shape)
        rn2 = np.random.standard_normal(va.shape)

        S = np.zeros((M + 1, I))
        S[0] = S0

        corr_mat = np.zeros((2, 2))
        corr_mat[0, :] = [1.0, rho]
        corr_mat[1, :] = [rho, 1.0]
        cho_mat = np.linalg.cholesky(corr_mat)
        ran_num = np.random.standard_normal((2, M + 1, I))

        for t in range(1, M + 1):
            
            rat = np.dot(cho_mat, ran_num[:, t, :])

            va_[t] = va_[t - 1] * (1 + vol_vol * square_root_dt * rat[1])
            va[t] = np.maximum(0, va_[t])

            F_b = np.abs(S[t - 1]) ** beta
            p = S[t - 1] + va_[t] * F_b * square_root_dt * rat[0]

            if (beta > 0 and beta < 1):
                S[t] = np.maximum(0, p)
            else:
                S[t] = p

        return S
            
# for the two functions below all we need to pass in is the paths from our simulation model, strike (K) and time (T)
    
def mc_call_price(K: int, T: float, r: float = .02, paths = None) -> float:

    if paths != None:
    
        return np.exp(-r * T) * np.maximum(paths[-1] - K, 0).mean()

def mc_put_price(paths, K: int, T: float, r: float = .02) -> float:

    return np.exp(-r * T) * np.maximum(K - paths[-1], 0).mean()