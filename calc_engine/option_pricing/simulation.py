from abc import ABC, abstractmethod
import numpy as np

"""
We need to add q for the simulations.
"""

RISK_FREE_RATE = 0.05

class Simulation(ABC):

    @abstractmethod
    def paths(self):
        pass

    def call(self, S0, K, T, r=RISK_FREE_RATE, M = 100, I = 1000, **kwargs):
        if isinstance(K, (int, float)):
            K = [K]

        paths = self.paths(S0, T, r, M, I, **kwargs)  # model-specific args
        ST = paths[-1][:, np.newaxis]
        K = np.array(K)[np.newaxis, :]

        payoffs = np.maximum(ST - K, 0)
        prices = np.exp(-r * T) * np.mean(payoffs, axis=0)
        return prices

    def put(self, S0, K, T, r=RISK_FREE_RATE, M = 100, I = 1000, **kwargs):
        if isinstance(K, (int, float)):
            K = [K]

        paths = self.paths(S0, T, r, M, I, **kwargs)
        ST = paths[-1][:, np.newaxis]
        K = np.array(K)[np.newaxis, :]

        payoffs = np.maximum(K - ST, 0)
        prices = np.exp(-r * T) * np.mean(payoffs, axis=0)
        return prices

class BlackScholesSimulation(Simulation):
        
    def paths(self, S: float, T: float, r: float, M: int, I: int, sigma: float) -> list:
        dt = T / M
        S_paths = np.zeros((M + 1, I))
        S_paths[0] = S
        rn = np.random.standard_normal(S_paths.shape)
        # euler discretization 
        for t in range(1, M + 1):   # 1
            S_paths[t] = S_paths[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t])  # 2

        return S_paths
    
class BachelierSimulation(Simulation):
    
    def paths(self, S: float, T: float, r: float, M: int, I: int, sigma: float) -> np.ndarray:
        dt = T / M
        S_paths = np.zeros((M + 1, I))
        S_paths[0] = S
        rn = np.random.standard_normal(S_paths.shape)
        # Euler discretization for the Bachelier model
        for t in range(1, M + 1):
            S_paths[t] = S_paths[t-1] + r * dt + sigma * np.sqrt(dt) * rn[t] # might need to use rn[t - 1] to match indexing

        return S_paths
    
class HestonSimulation(Simulation):

    def paths(self, S: float, T: float, r: float, M: int, I: int, sigma: float, v0: float, kappa: float, theta: float, rho: float):

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
    
class VarianceGammaSimulation(Simulation):

    def paths(self, S0, T, r, M, I, sigma, v, theta):

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

            #S[t] = S[0] + (r-q+w)*Tj + xt add q later
            S[t] = S[0] + (r+w)*Tj + xt

        price = np.exp(S)

        return price
    
    def call(self, S0, K, T, sigma=.2, v=.5, theta=.2, r=0, I=1000, M=5000):

        path = self.paths(S0, T, r, M, I, sigma, v, theta)

        return np.exp(-r * T) * np.maximum(path[-1] - K, 0).mean()

    def put(self, S0, K, T, sigma=.2, v=.5, theta=.2, r=0, I=1000, M=5000):

        path = self.paths(S0, T, r, M, I, sigma, v, theta)

        return np.exp(-r * T) * np.maximum(K - path[-1], 0).mean()
    
class SABRSimulation(Simulation):

    def paths(self, S0, T, r, M , I, sigma=0.2, alpha = .8, beta=0.9, rho=-0.5):
    
        dt = T / M
        square_root_dt = np.sqrt(dt)

        va = np.zeros((M + 1, I))
        va_ = np.zeros((M + 1, I))
        va[0] = sigma
        va_[0] = sigma

        S = np.zeros((M + 1, I))
        S[0] = S0

        corr_mat = np.zeros((2, 2))
        corr_mat[0, :] = [1.0, rho]
        corr_mat[1, :] = [rho, 1.0]
        cho_mat = np.linalg.cholesky(corr_mat)
        ran_num = np.random.standard_normal((2, M + 1, I))

        for t in range(1, M + 1):
            
            rat = np.dot(cho_mat, ran_num[:, t, :])

            va_[t] = va_[t - 1] * (1 + alpha * square_root_dt * rat[1])
            va[t] = np.maximum(0, va_[t])

            F_b = np.abs(S[t - 1]) ** beta
            p = S[t - 1] + va_[t] * F_b * square_root_dt * rat[0]

            if (beta > 0 and beta < 1):
                S[t] = np.maximum(0, p)
            else:
                S[t] = p

        return S
    
class rBergomiSimulation(Simulation):

    def paths(self, S: float, T: float, r: float, M: int, I: int, a: float, rho: float, xi: float, eta: float) -> None:
        """
        a: a = H - .5, this out parameter for the Hurst exponent
        xi: forward variance or initial variance
        eta: volatility of volatility
        rho: correlation between dW1 and dB
        n: steps per year
        I: number of paths
        T: maturity 1 = 1 year
        """
    
        e = np.array([0, 0])

        # create covariance matrix
        cov = np.array([[0.,0.],[0.,0.]])
        cov[0,0] = 1./M
        cov[0,1] = 1./((1.*a+1) * M**(1.*a+1))
        cov[1,1] = 1./((2.*a+1) * M**(2.*a+1))
        cov[1,0] = cov[0,1]
        
        s = int(M * T)

        dW1 = np.random.multivariate_normal(e, cov, (I, s))

        Y1 = np.zeros((I, 1 + s))
        Y2 = np.zeros((I, 1 + s)) 

        for i in np.arange(1, 1 + s, 1):
            Y1[:,i] = dW1[:,i-1,1]

        G = np.zeros(1 + s)

        # g = x**a, x = b(k, a/n)
        for k in np.arange(2, 1 + s, 1):
            b = ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)
            #G[k] = g(b(k, self.a)/self.n, self.a)
            G[k] = b ** a

        X = dW1[:,:,0]
        GX = np.zeros((I, len(X[0,:]) + len(G) - 1))

        for i in range(I):
            GX[i,:] = np.convolve(G, X[i,:]) # Convolution!

        Y2 = GX[:,:1 + s]

        Y = np.sqrt(2 * a + 1) * (Y1 + Y2) # volterra process

        dt = 1.0/M
        dW2 = np.random.randn(I, s) * np.sqrt(dt)
        dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2

        # create variance process
        t = np.linspace(0, T, 1 + s)[np.newaxis,:]
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))  # variance process

        increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * dt
        integral = np.cumsum(increments, axis = 1)

        S_path = np.zeros_like(V)
        S_path[:,0] = S
        S_path[:,1:] = S * np.exp(integral)

        return S_path.T
    
    def otm_put_call(self, S: float, strikes: np.ndarray, T: float, a: float, rho: float, xi: float, eta: float, K: float = None, n: int = 100, N: int = 1000):
        """ lets move this into our data processor, this way we only need to implement it once and use model.call() and model.put()
        """
        put_indices = np.where(strikes <= S)[0] # might edit this so it is only <
        call_indices = np.where(strikes > S)[0]

        call_strikes = strikes[call_indices]
        put_strikes = strikes[put_indices]

        model_call = self.call(S, T, call_strikes, a, rho, xi, eta, K, n, N)
        model_put = self.put(S, T, put_strikes, a, rho, xi, eta, K, n, N)

        put_results = list(zip(put_strikes, model_put))
        call_results = list(zip(call_strikes, model_call))

        return [(float(strike), float(price)) for strike, price in (put_results + call_results)]
    
    def atm_put_call(self, S: float, strikes: np.ndarray, T: float, a: float, rho: float, xi: float, eta: float, n: int = 100, N: int = 5000):

        put_indices = np.where(strikes <= S)[0]
        call_indices = np.where(strikes > S)[0]

        call_strikes = strikes[call_indices]
        put_strikes = strikes[put_indices]

        atm_model_call = self.call(S, T, call_strikes, a, rho, xi, eta, S, n, N)
        atm_model_put = self.put(S, T, put_strikes, a, rho, xi, eta, S, n, N)

        return (atm_model_call + atm_model_put) / 2

class BCC97(Simulation):
    pass

class StockSimulation:
    """
    Include a method for each of our simulation classes above, ex add a method def variance_gamma(self)
    """
    pass
            
# for the two functions below all we need to pass in is the paths from our simulation model, strike (K) and time (T)
    
def mc_call_price(K: int, T: float, r: float = .02, paths = None) -> float:

    if paths != None:
    
        return np.exp(-r * T) * np.maximum(paths[-1] - K, 0).mean()

def mc_put_price(paths, K: int, T: float, r: float = .02) -> float:

    return np.exp(-r * T) * np.maximum(K - paths[-1], 0).mean()