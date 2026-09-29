from flask import Flask, render_template, request
import numpy as np
import json

app = Flask(__name__)

def generate_asset_paths(S0, r, sigma, T, n_paths, n_steps):
    """Generates a 2D matrix of simulated stock paths using Geometric Brownian Motion."""
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol_scale = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((n_paths, n_steps))
    daily_log_returns = drift + vol_scale * Z
    
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(S0)
    log_paths[:, 1:] = np.log(S0) + np.cumsum(daily_log_returns, axis=1)
    
    return np.exp(log_paths)

def longstaff_schwartz_american_put(paths, K, r, T):
    """Robust evaluation of the optimal early exercise boundary for an American Put Option."""
    n_paths, n_cols = paths.shape
    dt = T / (n_cols - 1)
    discount = np.exp(-r * dt)
    
    cash_flows = np.zeros((n_paths, n_cols))
    cash_flows[:, -1] = np.maximum(K - paths[:, -1], 0)
    
    for t in range(n_cols - 2, 0, -1):
        S = paths[:, t]
        itm = (K - S) > 0  
        
        if np.any(itm):
            future_disc_to_t = np.zeros(n_paths)
            
            for i in range(n_paths):
                future_cols = np.where(cash_flows[i, t+1:] > 0)[0]
                if len(future_cols) > 0:
                    col = t + 1 + future_cols[0]
                    future_cf = cash_flows[i, col]
                    future_disc_to_t[i] = future_cf * (discount ** (col - t))
            
            X = S[itm]
            Y = future_disc_to_t[itm]
            
            valid_regression = Y > 0
            if np.sum(valid_regression) > 3:
                coeffs = np.polyfit(X[valid_regression], Y[valid_regression], 2)
                continuation_val = np.polyval(coeffs, X)
                exercise_val = K - X
                
                exercise = (exercise_val > continuation_val) & (Y > 0)
                
                itm_indices = np.where(itm)[0]
                exercising_indices = itm_indices[exercise]
                
                if len(exercising_indices) > 0:
                    cash_flows[exercising_indices, t] = exercise_val[exercise]
                    for idx in exercising_indices:
                        cash_flows[idx, t+1:] = 0  
                        
    total_pv = 0.0
    for t in range(n_cols):
        disc = np.exp(-r * t * dt)
        total_pv += np.sum(cash_flows[:, t]) * disc
        
    return total_pv / n_paths

@app.route("/", methods=["GET", "POST"])
def index():
    european_price = None
    american_price = None
    average_path = None
    sample_paths = None
    time_steps = None
    
    S0, K, r, sigma, T, n_paths = 100.0, 100.0, 0.05, 0.20, 1.0, 15000

    if request.method == "POST":
        S0 = float(request.form["S0"])
        K = float(request.form["K"])
        r = float(request.form["r"])
        sigma = float(request.form["sigma"])
        T = float(request.form["T"])
        n_paths = int(request.form["n_paths"])
        n_steps = 252

        paths = generate_asset_paths(S0, r, sigma, T, n_paths, n_steps)
        
        european_payoffs = np.maximum(K - paths[:, -1], 0)
        european_price = float(np.mean(european_payoffs) * np.exp(-r * T))
        american_price = float(longstaff_schwartz_american_put(paths, K, r, T))

        # Prepare chart data
        time_steps = [round(i * (T / n_steps), 3) for i in range(n_steps + 1)]
        average_path = np.mean(paths, axis=0).tolist()
        
        # Grab a random sample of 15 paths to display cleanly in the background
        sample_indices = np.random.choice(n_paths, size=min(15, n_paths), replace=False)
        sample_paths = paths[sample_indices, :].tolist()

    return render_template("index.html", 
                           S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=n_paths,
                           european_price=european_price, 
                           american_price=american_price,
                           time_steps=json.dumps(time_steps) if time_steps else '[]',
                           average_path=json.dumps(average_path) if average_path else '[]',
                           sample_paths=json.dumps(sample_paths) if sample_paths else '[]')

if __name__ == "__main__":
    app.run(debug=True)