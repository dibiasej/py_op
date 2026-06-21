from abc import ABC, abstractmethod
import numpy as np

"""
We need to add q for the simulations.
We add a bunch of ways to simulate below, this is just a collection of code for simulation we may use later 
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
        """
        This is not a Euler discretization of the SDE, this is the exact solution corresponding to GBM
        """

        dt = T / M
        S_paths = np.zeros((M + 1, I))
        S_paths[0] = S
        rn = np.random.standard_normal(S_paths.shape)
        
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
    
class TwoFactorBergomiSmileDynamics2Simulation(Simulation):

    def paths(self, S: float, T: float, r: float, M: int, I: int, xi0_curve: list[float], dtes: list[float], rho: float, rho_sx: float, rho_sy: float, k1: float, k2: float, omega: float, theta: float):
        """
        Monte Carlo simulation engine for the two-factor Bergomi model inspired by Bergomi's Smile Dynamics II framework.

        This class simulates spot paths under a two-factor lognormal instantaneous variance process. The variance dynamics are driven by two mean-reverting
        Gaussian factors, X_t and Y_t, and the spot process is correlated with both variance factors.

        Important
        ---------
        This model requires a full initial variance term structure, not a single variance number.

        The user must pass:
            xi0_curve = np.array(var_swap_curve) ** 2
            dtes = np.array(dtes) / 365

        where `var_swap_curve` is the volatility/variance curve quoted across maturities and `dtes` are the corresponding days-to-expiration.
        The model interpolates the curve at each simulation time t using: xi0_t = np.interp(t, dtes, xi0_curve)
        rho is the correlation between the X and Y OU factors and generally should be positive.
        """
        rng = np.random.default_rng(42)
        t_grid = np.linspace(0, T, M + 1)
        dt = T/M

        X_t = np.zeros((M+1, I))
        Y_t = np.zeros((M+1, I))

        EX2_t = np.zeros(M+1)
        EY2_t = np.zeros(M+1)
        EXY_t = np.zeros(M+1)

        inst_vol = np.zeros((M+1, I))
        inst_var = np.zeros((M + 1, I))
        S_t = np.zeros((M + 1, I))

        var_x_step = (1.0 - np.exp(-2.0 * k1 * dt)) / (2.0 * k1)
        var_y_step = (1.0 - np.exp(-2.0 * k2 * dt)) / (2.0 * k2)
        cov_xy_step = rho * (1.0 - np.exp(-(k1 + k2) * dt)) / (k1 + k2)

        cov_xz_step = rho_sx * (1.0 - np.exp(-k1 * dt)) / k1
        cov_yz_step = rho_sy * (1.0 - np.exp(-k2 * dt)) / k2
        var_z_step = dt

        S_t[0] = S

        cov = np.array([
            [var_x_step,  cov_xy_step, cov_xz_step],
            [cov_xy_step, var_y_step,  cov_yz_step],
            [cov_xz_step, cov_yz_step, var_z_step]
        ])

        for i in range(M):
            t = t_grid[i]
            xi0_t = np.interp(t, dtes, xi0_curve)

            EX2_t[i + 1] = np.exp(-2*k1*dt) * EX2_t[i] + var_x_step
            EY2_t[i + 1] = np.exp(-2*k2*dt) * EY2_t[i] + var_y_step
            EXY_t[i + 1] = np.exp(-(k1 + k2)*dt) * EXY_t[i] + cov_xy_step

            shocks = rng.multivariate_normal(mean=np.array([0.0, 0.0, 0.0]), cov=cov, size=I)

            shock_x = shocks[:, 0]
            shock_y = shocks[:, 1]
            dZ = shocks[:, 2]

            X_t[i + 1] = np.exp(-k1*dt)*X_t[i] + shock_x
            Y_t[i + 1] = np.exp(-k2*dt)*Y_t[i] + shock_y

            random_part = omega * (X_t[i + 1] + theta * Y_t[i + 1])
            variance_correction = 0.5 * omega**2 * (EX2_t[i + 1] + theta**2 * EY2_t[i + 1] + 2 * theta * EXY_t[i + 1])

            inst_var[i + 1] = xi0_t * np.exp(random_part - variance_correction)
            inst_vol[i + 1] = np.sqrt(inst_var[i + 1])

            S_t[i + 1] = S_t[i] * np.exp((r - 0.5 * inst_var[i + 1]) * dt + inst_vol[i + 1] * dZ)

        return S_t

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


# Simulation Code/Methods we may use later

def bsm_paths_euler(self, S: float, T: float, r: float, M: int, I: int, sigma: float) -> np.ndarray:
    # This method is a Euler discretization of the SDE generally used in the Black Scholes model
    dt = T / M
    S_paths = np.zeros((M + 1, I))
    S_paths[0] = S
    rn = np.random.standard_normal((M, I))  # use M rows and index t-1
    for t in range(1, M + 1):
        dW = np.sqrt(dt) * rn[t - 1]
        S_prev = S_paths[t - 1]
        S_paths[t] = S_prev + r * S_prev * dt + sigma * S_prev * dW
    return S_paths


def bsm_path_gbm_solution(S: float, T: float, r: float, M: int, I: int, sigma: float) -> list:
    dt = T / M
    S_paths = np.zeros((M + 1, I))
    S_paths[0] = S
    rn = np.random.standard_normal(S_paths.shape)
    # euler discretization 
    for t in range(1, M + 1):   # 1
        S_paths[t] = S_paths[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t])  # 2

    return S_paths
    

def polynomial_basis(S, degree: int):
    S = np.asarray(S)                     # ensure 1D array of shape (I,)
    return np.column_stack([np.ones_like(S), *[S**d for d in range(1, degree + 1)]])


class LeastSquaresAmerican:
    """
    We should change this so it is not a class but two functions.
    """

    def _calculate(self, S, K, T, r, sigma, M, I, otype = "call", method = "tvr", deg = 4):
        dt = T / M
        S_paths = bsm_paths_euler(S, T, r, M, I, sigma)

        if otype.lower() == "call":

            payoff = lambda S, K: np.maximum(S - K, 0.0)
            
        else:
            payoff = lambda S, K: np.maximum(K - S, 0.0)

        V = payoff(S_paths[-1], K)

        for i in range(M - 1, 0, -1):
            X = polynomial_basis(S_paths[i], deg)

            beta, *_ = np.linalg.lstsq(X, V, rcond=None)
            Vhat = X @ beta

            cont_disc = np.exp(-r * dt) * Vhat
            payoff_prev = payoff(S_paths[i], K)

            if method.lower() == "tvr":
                
                V = np.maximum(payoff_prev, cont_disc)
            
            elif method.lower() == "ls":
                exercise = payoff_prev > cont_disc
                V = np.where(exercise, payoff_prev, np.exp(-r * dt) * V)


            #V = np.maximum(payoff_prev, cont_disc)

        if method.lower() == "ls":
            P0 = payoff(S_paths[0], K)  # if you switch to the put, call payoff_put here
            return float(np.maximum(V, P0).mean())
        else:
            return float(V.mean())

    def call(self, S, K, T, r, sigma, M, I, method = "tvr", deg = 4):
        return self._calculate(S, K, T, r, sigma, M, I, otype = "call", method = method, deg = deg)

    def put(self, S, K, T, r, sigma, M, I, method = "tvr", deg = 4):
        return self._calculate(S, K, T, r, sigma, M, I, otype = "put", method = method, deg = deg)
    
def yves_american_call(S, K, T, r, sigma, M, I):
    dt = T / M
    df = np.exp(-r*dt)

    S_paths = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt +
                          sigma * np.sqrt(dt) *
                          np.random.standard_normal((M + 1, I)), axis=0))
    
    S_paths[0] = S

    h = np.maximum(S_paths - K, 0)

    V = h[-1]

    for t in range(M - 1, 0, -1):
        rg = np.polyfit(S_paths[t], V * df, 5)
        C = np.polyval(rg, S_paths[t])  # continuation values
        V = np.where(h[t] > C, h[t], V * df)
        # exercise decision
    V0 = df * np.sum(V) / I  # LSM estimator
    return V0


def yves_american_put(S, K, T, r, sigma, M, I):
    dt = T / M
    df = np.exp(-r*dt)

    S_paths = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt +
                          sigma * np.sqrt(dt) *
                          np.random.standard_normal((M + 1, I)), axis=0))
    
    S_paths[0] = S

    h = np.maximum(K - S_paths, 0)

    V = h[-1]

    for t in range(M - 1, 0, -1):
        rg = np.polyfit(S_paths[t], V * df, 5)
        C = np.polyval(rg, S_paths[t])  # continuation values
        V = np.where(h[t] > C, h[t], V * df)
        # exercise decision
    V0 = df * np.sum(V) / I  # LSM estimator
    return V0


def one_step_lsm_option(S, K, T, r, sigma, I, otype="call"):
    """
    Part (a) one-step LSM (intercept-only). Returns (price_tvr, price_ls).
    """
    paths = bsm_path_gbm_solution(S, T, r, M=1, I=I, sigma=sigma)
    ST = paths[-1, :]

    # terminal value
    if otype.lower() == "call":
        VT = np.maximum(ST - K, 0.0)
        P0 = np.maximum(S - K, 0.0)
    else:
        VT = np.maximum(K - ST, 0.0)
        P0 = np.maximum(K - S, 0.0)

    Vhat_t = VT.mean()                  # intercept-only
    cont_disc = np.exp(-r * T) * Vhat_t

    # TvR and LS coincide in one-step with intercept-only basis
    price_tvr = max(P0, cont_disc)
    price_ls  = max(P0, cont_disc) if P0 > cont_disc else (np.exp(-r * T) * VT).mean()

    return float(price_tvr), float(price_ls)


def longstaff_schwartz_american_itm_modification(S, K, T, r, sigma, M, I, otype = 'call'):
    """
    American PUT via LSMC (Longstaff–Schwartz) with ITM-only regression samples.
    - Regress UNDIsCOUNTED V_t on polynomial basis of S_{t-Δt}, using ONLY paths with payoff_put>0.
    - No TvR path; no fallback if there are zero ITM samples at a step.
    """
    dt = T / M
    S_paths = bsm_paths_euler(S, T, r, M, I, sigma)

    # terminal payoff at T
    if otype == 'call':
        payoff = lambda S, K: np.maximum(S - K, 0.0)
    else:
        payoff = lambda S, K: np.maximum(K - S, 0.0)

    V = payoff(S_paths[-1], K)

    # backward induction (LS)
    for i in range(M - 1, 0, -1):
        s_prev = S_paths[i]
        X_all = polynomial_basis(s_prev, 8)

        # ITM-only mask (PUT)
        payoff_prev = payoff(s_prev, K)
        mask_itm = payoff_prev > 0.0

        # FIT ONLY ON ITM SAMPLES (no fallback)
        beta, *_ = np.linalg.lstsq(X_all[mask_itm], V[mask_itm], rcond=None)
        #beta, *_ = np.linalg.lstsq(X_all, V, rcond=None)

        # predict continuation for ALL paths, then discount
        Vhat = X_all @ beta
        vhat2 = X_all[mask_itm] @ beta
        cont_disc = np.exp(-r * dt) * Vhat

        # LS update: exercise vs continue (continue uses discounted pathwise V_t)
        exercise = (payoff_prev > cont_disc) * mask_itm
        V = np.where(exercise, payoff_prev, np.exp(-r * dt) * V)

    # LS finalization at t=0: mean(max{V0, P(S0)})
    P0 = payoff(S_paths[0], K)
    return float(np.maximum(V, P0).mean())