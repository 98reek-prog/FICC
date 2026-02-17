import numpy as np
import matplotlib.pyplot as plt

def merton_monte_carlo_paths(asset_value, debt_value, maturity, risk_free_rate, asset_volatility, n_sims=50, n_steps=252):
    """
    Monte Carlo simulation with asset path visualization for the Merton model.
    
    Parameters:
    asset_value (float): Current value of firm's assets (A0)
    debt_value (float): Face value of debt (D)
    maturity (float): Time to maturity in years (T)
    risk_free_rate (float): Risk-free interest rate (r)
    asset_volatility (float): Volatility of firm's assets (sigma)
    n_sims (int): Number of simulation paths
    n_steps (int): Number of time steps per path
    
    Returns:
    None (plots asset paths)
    """
    
    dt = maturity / n_steps
    time_grid = np.linspace(0, maturity, n_steps+1)
    
    plt.figure(figsize=(10,6))
    
    for _ in range(n_sims):
        A = np.zeros(n_steps+1)
        A[0] = asset_value
        for t in range(1, n_steps+1):
            # Geometric Brownian Motion
            A[t] = A[t-1] * np.exp((risk_free_rate - 0.5 * asset_volatility**2) * dt +
                                   asset_volatility * np.sqrt(dt) * np.random.normal())
        plt.plot(time_grid, A, lw=0.8, alpha=0.7)
    
    # Plot debt threshold
    plt.axhline(y=debt_value, color='red', linestyle='--', label='Debt Level')
    
    plt.title("Monte Carlo Simulation of Asset Paths (Merton Model)")
    plt.xlabel("Time (years)")
    plt.ylabel("Asset Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
merton_monte_carlo_paths(
    asset_value=100,     # Firm's assets
    debt_value=80,       # Debt face value
    maturity=1,          # 1 year
    risk_free_rate=0.05, # 5% risk-free rate
    asset_volatility=0.25, # 25% volatility
    n_sims=50            # Plot 50 paths
)
