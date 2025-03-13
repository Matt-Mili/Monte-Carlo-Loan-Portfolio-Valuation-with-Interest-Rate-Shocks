import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Loan Simulation Function
# -------------------------------
def simulate_loan(loan_params, rate_shock):
    """
    Simulates the annual cash flows for a loan over its term, incorporating default, prepayment, and rate shocks.
    
    Parameters:
        loan_params (dict): Loan characteristics
        rate_shock (float): Adjusted interest rate (affected by macro changes)
    
    Returns:
        cash_flows (list): Annual cash flows for the loan
    """
    principal = loan_params["principal"]
    base_rate = loan_params["annual_rate"]
    T = loan_params["term"]
    
    # Adjusted rate based on macroeconomic shock
    r = max(0, base_rate + rate_shock)  # Ensure rate is non-negative
    
    # Default and prepayment probabilities shift with macro conditions
    p_default = loan_params["p_default"] + (rate_shock * 0.5)  # Higher rates → higher default risk
    p_prepay = max(0, loan_params["p_prepay"] - (rate_shock * 0.3))  # Higher rates → lower prepayments

    # Compute scheduled loan payment
    if r == 0:
        scheduled_payment = principal / T
    else:
        scheduled_payment = principal * (r * (1 + r) ** T) / ((1 + r) ** T - 1)
    
    cash_flows = []
    outstanding = principal
    active = True

    for year in range(1, T + 1):
        if not active or outstanding <= 0:
            cash_flows.append(0.0)
            continue

        # Compute interest & principal portions
        interest = outstanding * r
        principal_component = scheduled_payment - interest

        # Adjust final payment
        if principal_component > outstanding:
            principal_component = outstanding
            scheduled_payment = interest + principal_component

        year_cf = scheduled_payment  # Base cash flow

        # Event probability check
        rand = np.random.rand()
        if rand < p_default:
            active = False  # Default occurs, no further cash flows
        elif rand < p_default + p_prepay:
            year_cf += (outstanding - principal_component)  # Prepayment occurs
            active = False

        cash_flows.append(year_cf)
        if active:
            outstanding -= principal_component
        else:
            outstanding = 0.0
            
    return cash_flows

# -------------------------------
# Discounting Function with Rate Shocks
# -------------------------------
def discount_cash_flows(cash_flows, discount_rate):
    """
    Discounts a series of cash flows at the given discount rate.
    """
    return sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cash_flows, start=1))

# -------------------------------
# Monte Carlo Portfolio Simulation with Rate Shocks
# -------------------------------
def monte_carlo_portfolio_dcf(n_loans, n_simulations, loan_params, base_discount_rate, rate_shock_std):
    """
    Simulates loan portfolio valuation incorporating interest rate shocks.
    
    Parameters:
        n_loans (int): Number of loans in portfolio
        n_simulations (int): Number of Monte Carlo runs
        loan_params (dict): Loan parameters
        base_discount_rate (float): Initial discount rate
        rate_shock_std (float): Standard deviation for rate shocks
    
    Returns:
        portfolio_values (np.array): Portfolio values across simulations
        discount_rates (np.array): Discount rates applied in each simulation
    """
    portfolio_values = []
    discount_rates = []
    T = loan_params["term"]

    for _ in range(n_simulations):
        # Simulate an interest rate shock
        rate_shock = np.random.normal(0, rate_shock_std)
        adjusted_discount_rate = max(0, base_discount_rate + rate_shock)  # Ensure non-negative rate

        # Track discount rate used in this simulation
        discount_rates.append(adjusted_discount_rate)

        # Aggregate cash flows for portfolio
        portfolio_cf = np.zeros(T)
        for _ in range(n_loans):
            loan_cf = simulate_loan(loan_params, rate_shock)
            portfolio_cf += np.array(loan_cf)

        # Compute discounted portfolio value
        pv = discount_cash_flows(portfolio_cf, adjusted_discount_rate)
        portfolio_values.append(pv)

    return np.array(portfolio_values), np.array(discount_rates)

# -------------------------------
# Set Parameters & Run Simulation
# -------------------------------
loan_params = {
    "principal": 10000.0,     # $10,000 loan
    "annual_rate": 0.10,      # 10% interest rate
    "term": 10,               # 10-year term
    "p_default": 0.02,        # 2% annual default rate
    "p_prepay": 0.05          # 5% annual prepayment rate
}

n_loans = 50                # Portfolio of 50 loans
n_simulations = 5000        # Monte Carlo iterations
base_discount_rate = 0.08   # 8% discount rate
rate_shock_std = 0.02       # Increased interest rate shock volatility for stress testing

# Run Monte Carlo simulation with rate shocks
portfolio_values, discount_rates = monte_carlo_portfolio_dcf(n_loans, n_simulations, loan_params, base_discount_rate, rate_shock_std)

# -------------------------------
# Results & Visualization
# -------------------------------
mean_value = np.mean(portfolio_values)
median_value = np.median(portfolio_values)
std_value = np.std(portfolio_values)
percentile_5 = np.percentile(portfolio_values, 5)
percentile_95 = np.percentile(portfolio_values, 95)

print("Monte Carlo DCF Valuation for Loan Portfolio (with Interest Rate Shocks & Stress Testing)")
print(f"Number of Loans: {n_loans}")
print(f"Simulations: {n_simulations}")
print(f"Mean Portfolio Value: ${mean_value:,.2f}")
print(f"Median Portfolio Value: ${median_value:,.2f}")
print(f"Standard Deviation: ${std_value:,.2f}")
print(f"5th Percentile Portfolio Value (Downside Risk): ${percentile_5:,.2f}")
print(f"95th Percentile Portfolio Value (Upside Potential): ${percentile_95:,.2f}")

# Plot portfolio valuation distribution
plt.figure(figsize=(12,6))
plt.hist(portfolio_values, bins=50, edgecolor='black', alpha=0.7)
plt.title("Portfolio DCF Value Distribution with Interest Rate Shocks")
plt.xlabel("Portfolio Value (USD)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Plot distribution of discount rates used in simulations
plt.figure(figsize=(12,6))
plt.hist(discount_rates, bins=50, edgecolor='black', alpha=0.7, color='red')
plt.title("Distribution of Discount Rates in Simulations")
plt.xlabel("Discount Rate (%)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
