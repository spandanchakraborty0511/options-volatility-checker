"""
SVI (Stochastic Volatility Inspired) Calibration Module (Phase 3).
Fits 5-parameter Raw SVI model to single-expiry options slices via non-linear least squares,
enforces no-arbitrage parameter bounds, and flags Rich/Cheap option strikes.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, Optional

import config

def raw_svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float
) -> np.ndarray:
    """
    Raw SVI Total Variance formula w(k):
    w(k; a, b, rho, m, sigma) = a + b * [ rho * (k - m) + sqrt((k - m)^2 + sigma^2) ]
    """
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))

class SVIModel:
    """
    Calibration and evaluation engine for single-expiry Raw SVI model.
    """
    def __init__(self):
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.rho: Optional[float] = None
        self.m: Optional[float] = None
        self.sigma: Optional[float] = None
        self.fitted: bool = False
        self.rmse: Optional[float] = None
        self.mae: Optional[float] = None
        self.tau: Optional[float] = None

    def fit(
        self,
        log_moneyness: np.ndarray,
        total_variance: np.ndarray,
        tau: float,
        weights: Optional[np.ndarray] = None
    ) -> "SVIModel":
        """
        Fit Raw SVI parameters (a, b, rho, m, sigma) to market log-moneyness and total variance points.
        Uses multi-start optimization for robust global convergence.
        """
        k_data = np.asarray(log_moneyness, dtype=float)
        w_data = np.asarray(total_variance, dtype=float)
        self.tau = float(tau)
        
        if len(k_data) < 5:
            raise ValueError(f"SVI fit requires at least 5 points, got {len(k_data)}")
            
        iv_market = np.sqrt(np.maximum(1e-6, w_data / self.tau))
        
        if weights is None:
            wts = np.ones_like(k_data)
        else:
            wts = np.asarray(weights, dtype=float)
            wts = wts / np.sum(wts)
            
        w_min = np.min(w_data)
        k_at_w_min = k_data[np.argmin(w_data)]
        
        # Parameter bounds
        bounds = [
            (-0.5, 1.5),            # a
            (0.001, 3.0),          # b
            (-0.99, 0.99),         # rho
            (-1.5, 1.5),           # m
            (0.001, 1.5)           # sigma
        ]
        
        # Multi-start initial points generator
        initial_guesses = []
        for m_g in [k_at_w_min, 0.0, -0.05, 0.05]:
            for rho_g in [-0.7, -0.4, -0.1]:
                for b_g in [0.05, 0.15, 0.3]:
                    for sigma_g in [0.05, 0.15]:
                        a_g = float(w_min - b_g * sigma_g * np.sqrt(1 - rho_g ** 2))
                        initial_guesses.append([a_g, b_g, rho_g, m_g, sigma_g])
                        
        def loss_function(params):
            a_p, b_p, rho_p, m_p, sigma_p = params
            
            # Guarantees w(k) >= 0 everywhere (no-arbitrage minimum variance bound)
            min_variance = a_p + b_p * sigma_p * np.sqrt(1 - rho_p ** 2)
            if min_variance < 0:
                return 1e6 + (abs(min_variance) * 1e5)
                
            w_pred = raw_svi_total_variance(k_data, a_p, b_p, rho_p, m_p, sigma_p)
            iv_pred = np.sqrt(np.maximum(1e-6, w_pred / self.tau))
            
            # Loss in IV space directly for superior fit to option market prices
            residuals = (iv_market - iv_pred) * wts
            return np.sum(residuals ** 2)
            
        best_res = None
        best_fun = float('inf')
        
        # Run top candidate initial guesses
        for x0 in initial_guesses[:8]:
            try:
                res = minimize(loss_function, x0, method='L-BFGS-B', bounds=bounds)
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                continue
                
        if best_res is None or not np.isfinite(best_fun):
            # Fallback to single SLSQP
            x0_def = [0.01, 0.1, -0.4, 0.0, 0.1]
            best_res = minimize(loss_function, x0_def, method='SLSQP', bounds=bounds)
            
        self.a, self.b, self.rho, self.m, self.sigma = best_res.x
        self.fitted = True
        
        # Calculate fit statistics
        w_fit = self.predict_total_variance(k_data)
        iv_fit = self.predict_iv(k_data, tau=self.tau)
        
        self.rmse = float(np.sqrt(np.mean((iv_market - iv_fit) ** 2)))
        self.mae = float(np.mean(np.abs(iv_market - iv_fit)))
        return self

    def predict_total_variance(self, k: np.ndarray) -> np.ndarray:
        """Predict total variance w(k) for given log-moneyness array."""
        if not self.fitted:
            raise RuntimeError("SVI Model must be fitted before prediction!")
        return raw_svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)

    def predict_iv(self, k: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
        """Predict implied volatility sigma_SVI = sqrt(w(k) / tau)."""
        t = tau if tau is not None else self.tau
        if t is None or t <= 0:
            raise ValueError("Time to expiry tau must be positive!")
        w = self.predict_total_variance(k)
        w_safe = np.maximum(0.0, w)
        return np.sqrt(w_safe / t)

    def get_params(self) -> Dict[str, float]:
        """Return fitted SVI 5 parameters dictionary."""
        return {
            'a': self.a,
            'b': self.b,
            'rho': self.rho,
            'm': self.m,
            'sigma': self.sigma,
            'rmse': self.rmse,
            'mae': self.mae
        }

def flag_rich_cheap_strikes(
    df_slice: pd.DataFrame,
    svi_model: SVIModel,
    vol_threshold: float = config.RICH_CHEAP_THRESHOLD_VOL
) -> pd.DataFrame:
    """
    Compare Market IV against SVI Model IV for each strike in clean options slice
    and assign mispricing signal flags (RICH / CHEAP / FAIR).
    """
    df = df_slice.copy()
    if not svi_model.fitted:
        raise RuntimeError("SVI Model must be fitted before flagging strikes!")
        
    k_array = df['log_moneyness'].values
    tau = df['tau'].iloc[0]
    
    df['svi_total_variance'] = svi_model.predict_total_variance(k_array)
    df['svi_iv'] = svi_model.predict_iv(k_array, tau=tau)
    
    df['iv_diff'] = df['iv'] - df['svi_iv']
    df['iv_diff_pct'] = (df['iv_diff'] / df['svi_iv']) * 100.0
    
    conditions = [
        df['iv_diff'] > vol_threshold,
        df['iv_diff'] < -vol_threshold
    ]
    choices = ['RICH', 'CHEAP']
    df['signal'] = np.select(conditions, choices, default='FAIR')
    
    return df
