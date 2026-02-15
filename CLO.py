# =============================================================================
# CLO Cashflow Model with Monte Carlo Simulation
# =============================================================================

# First, check and install required packages if needed
import subprocess
import sys
import importlib.util

def check_and_install_package(package_name):
    """Check if a package is installed, if not install it."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    return False

# Required packages
required_packages = ['pandas', 'numpy', 'scipy', 'matplotlib', 'openpyxl']

for package in required_packages:
    check_and_install_package(package)

print("All required packages are installed!")

# Now import all required libraries
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt

print("All packages loaded successfully!")

# -----------------------------------------------------------------------------
# 1. Load and prepare data with error handling
# -----------------------------------------------------------------------------
def load_excel_file():
    """Load Excel file with user input for file path if not found."""
    # Try to find the file
    file_path = "CLO Case Study Data.xlsx"
    
    # If not in current directory, prompt user
    if not os.path.exists(file_path):
        print(f"\nFile '{file_path}' not found in: {os.getcwd()}")
        print("\nPlease make sure your Excel file is in one of these locations:")
        print(f"1. Current directory: {os.getcwd()}")
        print("2. Or enter the full path below")
        
        file_path = input("\nEnter full path to Excel file (or press Enter to exit): ").strip('"').strip("'")
        
        if not file_path:
            print("No file path provided. Exiting...")
            sys.exit(1)
        
        if not os.path.exists(file_path):
            print(f"Error: Cannot find file: {file_path}")
            sys.exit(1)
    
    print(f"\nLoading file: {file_path}")
    return file_path

# Load the Excel file
try:
    file_path = load_excel_file()
    collateral_df = pd.read_excel(file_path, sheet_name="Collateral")
    tranches_df   = pd.read_excel(file_path, sheet_name="Tranches")
    transactions_df = pd.read_excel(file_path, sheet_name="Transactions")
    
    print("Excel file loaded successfully!")
    
except Exception as e:
    print(f"Error loading Excel file: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure the file is not open in another program")
    print("2. Check that all sheet names are exactly: 'Collateral', 'Tranches', 'Transactions'")
    print("3. Verify the file is a valid .xlsx file")
    sys.exit(1)

# Display actual column names to help with debugging
print("\n" + "="*60)
print("COLUMN NAMES IN EXCEL FILES")
print("="*60)
print("\nCollateral sheet columns:")
print(collateral_df.columns.tolist())
print("\nTranches sheet columns:")
print(tranches_df.columns.tolist())
print("\nTransactions sheet columns:")
print(transactions_df.columns.tolist())
print("="*60)

# Create a mapping for column names (since they might have spaces)
def get_column_name(df, possible_names):
    """Find the first matching column name from a list of possibilities."""
    for name in possible_names:
        if name in df.columns:
            return name
    # If no match found, return None and print available columns
    print(f"Could not find any of {possible_names} in columns: {df.columns.tolist()}")
    return None

# For collateral sheet
current_balance_col = get_column_name(collateral_df, ['Current_Balance', 'Current Balance', 'CurrentBalance'])
if current_balance_col is None:
    print("ERROR: Cannot find Current Balance column. Exiting.")
    sys.exit(1)

gross_spread_col = get_column_name(collateral_df, ['Gross_Spread/Margin', 'Gross Spread/Margin', 'Gross Spread'])
fixed_float_col = get_column_name(collateral_df, ['Fixed_or_Float', 'Fixed or Float', 'Fixed/Float'])
gross_coupon_col = get_column_name(collateral_df, ['Gross_Coupon', 'Gross Coupon'])
frequency_col = get_column_name(collateral_df, ['Frequency'])
next_paydate_col = get_column_name(collateral_df, ['Next_Paydate', 'Next Paydate', 'Next Pay Date'])
maturity_date_col = get_column_name(collateral_df, ['Asset_Maturity_Date', 'Asset Maturity Date', 'Maturity Date'])
amort_type_col = get_column_name(collateral_df, ['Amortization_Type', 'Amortization Type'])
rem_term_col = get_column_name(collateral_df, ['Rem_Term', 'Rem Term', 'Remaining Term'])
defaulted_flag_col = get_column_name(collateral_df, ['Defaulted_Flag', 'Defaulted Flag'])
rating_col = get_column_name(collateral_df, ['Adjusted_Moody\'s_Rating', "Adjusted Moody's Rating", 'Moody Rating'])
recovery_col = get_column_name(collateral_df, ['Predominant:_S&P_Recovery_Rate', 'Predominant: S&P Recovery Rate', 'S&P Recovery'])

# For tranches sheet
tranche_col = get_column_name(tranches_df, ['Tranche'])
type_col = get_column_name(tranches_df, ['Type'])
coupon_col = get_column_name(tranches_df, ['Coupon'])
floater_formula_col = get_column_name(tranches_df, ['Floater_Formula', 'Floater Formula'])
tranche_balance_col = get_column_name(tranches_df, ['Current_Balance', 'Current Balance', 'CurrentBalance'])

# Rename columns to standardized names for easier coding
if current_balance_col and current_balance_col != 'Current_Balance':
    collateral_df.rename(columns={current_balance_col: 'Current_Balance'}, inplace=True)

if gross_spread_col and gross_spread_col != 'Gross_Spread/Margin':
    collateral_df.rename(columns={gross_spread_col: 'Gross_Spread/Margin'}, inplace=True)

if fixed_float_col and fixed_float_col != 'Fixed_or_Float':
    collateral_df.rename(columns={fixed_float_col: 'Fixed_or_Float'}, inplace=True)

if gross_coupon_col and gross_coupon_col != 'Gross_Coupon':
    collateral_df.rename(columns={gross_coupon_col: 'Gross_Coupon'}, inplace=True)

if frequency_col and frequency_col != 'Frequency':
    collateral_df.rename(columns={frequency_col: 'Frequency'}, inplace=True)

if next_paydate_col and next_paydate_col != 'Next_Paydate':
    collateral_df.rename(columns={next_paydate_col: 'Next_Paydate'}, inplace=True)

if maturity_date_col and maturity_date_col != 'Asset_Maturity_Date':
    collateral_df.rename(columns={maturity_date_col: 'Asset_Maturity_Date'}, inplace=True)

if amort_type_col and amort_type_col != 'Amortization_Type':
    collateral_df.rename(columns={amort_type_col: 'Amortization_Type'}, inplace=True)

if rem_term_col and rem_term_col != 'Rem_Term':
    collateral_df.rename(columns={rem_term_col: 'Rem_Term'}, inplace=True)

if defaulted_flag_col and defaulted_flag_col != 'Defaulted_Flag':
    collateral_df.rename(columns={defaulted_flag_col: 'Defaulted_Flag'}, inplace=True)

if rating_col and rating_col != 'Adjusted_Moody\'s_Rating':
    collateral_df.rename(columns={rating_col: 'Adjusted_Moody\'s_Rating'}, inplace=True)

if recovery_col and recovery_col != 'Predominant:_S&P_Recovery_Rate':
    collateral_df.rename(columns={recovery_col: 'Predominant:_S&P_Recovery_Rate'}, inplace=True)

# Rename tranches columns
if tranche_col and tranche_col != 'Tranche':
    tranches_df.rename(columns={tranche_col: 'Tranche'}, inplace=True)

if type_col and type_col != 'Type':
    tranches_df.rename(columns={type_col: 'Type'}, inplace=True)

if coupon_col and coupon_col != 'Coupon':
    tranches_df.rename(columns={coupon_col: 'Coupon'}, inplace=True)

if floater_formula_col and floater_formula_col != 'Floater_Formula':
    tranches_df.rename(columns={floater_formula_col: 'Floater_Formula'}, inplace=True)

if tranche_balance_col and tranche_balance_col != 'Current_Balance':
    tranches_df.rename(columns={tranche_balance_col: 'Current_Balance'}, inplace=True)

print("\nData summary:")
print(f"Collateral shape: {collateral_df.shape}")
print(f"Tranches shape: {tranches_df.shape}")
print(f"Transactions shape: {transactions_df.shape}")
print("\nTranches:")
print(tranches_df)

# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
def get_index_rate(date, index_name='LIBOR_3MO'):
    """
    Placeholder for future interest rate scenarios.
    Returns a deterministic rate (here 3% flat).
    """
    return 0.03

def loan_cashflows(loan, start_date, index_curve):
    """
    Generate contractual monthly cashflows for a single loan.
    loan: Series with loan data.
    start_date: first payment date (typically Next_Paydate).
    index_curve: function or dict to get forward index rate.
    Returns DataFrame with columns: date, interest, principal, balance_after.
    """
    balance = loan['Current_Balance']
    rate_spread = loan['Gross_Spread/Margin']          # in percent
    fixed_float = loan['Fixed_or_Float']
    coupon = loan['Gross_Coupon']                      # all-in if fixed
    freq = loan['Frequency']                           # 'M', 'Q', 'S', 'A'
    next_pay = loan['Next_Paydate']
    maturity = loan['Asset_Maturity_Date']
    amort_type = loan['Amortization_Type']
    rem_term = loan['Rem_Term']                        # remaining months

    # Skip if no balance or already defaulted
    if balance <= 0 or loan['Defaulted_Flag'] == 'Y':
        return pd.DataFrame()

    # Frequency to months between payments
    freq_map = {'M': 1, 'Q': 3, 'S': 6, 'A': 12}
    months_per_pay = freq_map.get(freq, 1)

    # Handle NaN or missing next_pay
    if pd.isna(next_pay):
        return pd.DataFrame()
    
    # Payment dates from next_pay to maturity
    try:
        dates = pd.date_range(start=next_pay, end=maturity, freq=f'{months_per_pay}MS')
    except:
        # If date range fails, return empty
        return pd.DataFrame()

    cf = []
    for pay_date in dates:
        if balance <= 1e-6:
            break

        # Determine interest rate for this period
        if fixed_float == 'Float' and not pd.isna(rate_spread):
            idx_rate = index_curve(pay_date)
            all_in_rate = idx_rate + rate_spread / 100.0
        elif not pd.isna(coupon):
            all_in_rate = coupon / 100.0
        else:
            all_in_rate = 0.05  # default 5% if nothing else

        # Interest accrual (simplified 30/360, monthly)
        days = 30
        interest = balance * all_in_rate * days / 360.0

        # Principal payment
        if amort_type == 'Sched' and rem_term > 0 and rem_term is not None:
            principal = balance / rem_term
            rem_term -= months_per_pay
        else:
            principal = balance if pay_date == maturity else 0

        principal = min(principal, balance)
        balance -= principal

        cf.append({
            'date': pay_date,
            'interest': interest,
            'principal': principal,
            'balance_after': balance
        })

    return pd.DataFrame(cf)

# -----------------------------------------------------------------------------
# 3. Default probability and recovery mapping
# -----------------------------------------------------------------------------
# Illustrative 1‑year PD by Moody's rating (replace with actual data if available)
rating_pd = {
    'Aaa': 0.0001, 'Aa1': 0.0002, 'Aa2': 0.0003, 'Aa3': 0.0004,
    'A1': 0.0005, 'A2': 0.0007, 'A3': 0.0010,
    'Baa1': 0.0015, 'Baa2': 0.0020, 'Baa3': 0.0025,
    'Ba1': 0.0050, 'Ba2': 0.0075, 'Ba3': 0.0100,
    'B1': 0.0200, 'B2': 0.0300, 'B3': 0.0500,
    'Caa1': 0.0750, 'Caa2': 0.1000, 'Caa3': 0.1500,
    'Ca': 0.2000, 'C': 0.3000, 'D': 1.0000
}

# Map ratings to PD, handling missing values
collateral_df['PD_1y'] = collateral_df['Adjusted_Moody\'s_Rating'].map(rating_pd).fillna(0.02)

# Recovery rate (use S&P recovery field or default 50%)
collateral_df['Recovery'] = collateral_df['Predominant:_S&P_Recovery_Rate'].fillna(50) / 100.0

print(f"\nProcessed {len(collateral_df)} loans with PD and recovery rates")

# -----------------------------------------------------------------------------
# 4. Waterfall function
# -----------------------------------------------------------------------------
def apply_waterfall(available_int, available_pri, tranches, index_rate):
    """
    Distribute interest and principal to tranches according to seniority.
    tranches: DataFrame with columns Tranche, Type, Current_Balance, Coupon, Floater_Formula.
    index_rate: current index rate.
    Returns updated tranche DataFrame with allocations.
    """
    # Seniority order
    order = ['AR', 'BR', 'CR', 'DR', 'ER', 'SUBORD']
    # Work on a copy so we don't modify original
    tr = tranches.set_index('Tranche').reindex(order).copy()
    
    # Remove any tranches not in the data
    tr = tr.dropna()

    # Calculate coupon for each tranche
    for t in tr.index:
        if tr.loc[t, 'Type'] in ['SEN_FLT', 'MEZ_FLT']:
            # Extract spread from formula
            formula = str(tr.loc[t, 'Floater_Formula'])
            try:
                spread = float(formula.split('+')[-1].strip())
                tr.loc[t, 'coupon'] = index_rate + spread / 100.0
            except:
                tr.loc[t, 'coupon'] = index_rate + 0.02  # default 2% spread
        elif tr.loc[t, 'Type'] == 'JUN_SUB':
            tr.loc[t, 'coupon'] = 0.0
        else:
            tr.loc[t, 'coupon'] = tr.loc[t, 'Coupon'] / 100.0 if pd.notna(tr.loc[t, 'Coupon']) else 0.0

    # Interest due (monthly approximation)
    tr['int_due'] = tr['Current_Balance'] * tr['coupon'] * 30 / 360
    tr['int_paid'] = 0.0
    tr['int_short'] = 0.0
    tr['pri_paid'] = 0.0

    # Allocate interest sequentially
    int_remaining = available_int
    for t in order:
        if t not in tr.index:
            continue
        due = tr.loc[t, 'int_due']
        paid = min(due, int_remaining)
        tr.loc[t, 'int_paid'] = paid
        int_remaining -= paid
        tr.loc[t, 'int_short'] = due - paid

    # Allocate principal sequentially (only after interest)
    pri_remaining = available_pri
    for t in order:
        if t not in tr.index:
            continue
        if tr.loc[t, 'Current_Balance'] <= 0:
            continue
        due = tr.loc[t, 'Current_Balance']
        paid = min(due, pri_remaining)
        tr.loc[t, 'pri_paid'] = paid
        pri_remaining -= paid
        tr.loc[t, 'Current_Balance'] -= paid   # update balance

    return tr

# -----------------------------------------------------------------------------
# 5. Single deterministic scenario (no defaults)
# -----------------------------------------------------------------------------
def run_deterministic(collateral, tranches, index_curve, max_months=360):
    """
    Simulate cashflows without defaults, month by month.
    Returns history of tranche balances and cumulative payments.
    """
    print("\nRunning deterministic scenario (no defaults)...")
    
    # Copy initial balances
    tr_balances = tranches.set_index('Tranche')['Current_Balance'].copy()
    cum_payments = {t: {'interest':0.0, 'principal':0.0} for t in tr_balances.index}

    # Generate loan‑level cashflow schedules (all loans)
    loan_sched_list = []
    valid_loans = 0
    
    for idx, loan in collateral.iterrows():
        if loan['Current_Balance'] > 0 and loan['Defaulted_Flag'] != 'Y':
            sched = loan_cashflows(loan, loan['Next_Paydate'], index_curve)
            if not sched.empty:
                loan_sched_list.append(sched)
                valid_loans += 1

    print(f"Generated cashflows for {valid_loans} loans")

    if not loan_sched_list:
        print("No cashflows generated!")
        return pd.DataFrame()

    all_cf = pd.concat(loan_sched_list, ignore_index=True)
    all_cf = all_cf.groupby('date').agg({'interest':'sum', 'principal':'sum'}).reset_index()
    all_cf = all_cf.sort_values('date')
    
    print(f"Generated {len(all_cf)} payment periods")

    history = []
    for _, row in all_cf.iterrows():
        date = row['date']
        idx_rate = index_curve(date)

        # Apply waterfall
        updated = apply_waterfall(row['interest'], row['principal'], tranches, idx_rate)

        # Update tranche balances and cumulative payments
        for t in updated.index:
            if t in tr_balances.index:
                tr_balances[t] = updated.loc[t, 'Current_Balance']
                cum_payments[t]['interest'] += updated.loc[t, 'int_paid']
                cum_payments[t]['principal'] += updated.loc[t, 'pri_paid']

        # Record snapshot
        snap = {'date': date}
        snap.update(tr_balances.to_dict())
        for t in cum_payments:
            snap[f'{t}_int'] = cum_payments[t]['interest']
            snap[f'{t}_pri'] = cum_payments[t]['principal']
        history.append(snap)

    return pd.DataFrame(history)

# -----------------------------------------------------------------------------
# 6. Monte Carlo simulation with correlated defaults (simplified)
# -----------------------------------------------------------------------------
def simulate_default_times(collateral, n_sims=100, rho=0.3):
    """
    Generate correlated default times for each loan.
    Returns array of shape (n_loans, n_sims) with default flag (1=default, 0=no default).
    """
    n_loans = len(collateral)
    np.random.seed(42)
    Y = np.random.normal(size=n_sims)
    default_flags = np.zeros((n_loans, n_sims))

    for i, loan in collateral.iterrows():
        term = loan['Rem_Term']
        if term <= 0 or pd.isna(term):
            continue
            
        # Cumulative PD over term
        pd_1y = loan['PD_1y']
        pd_term = 1 - (1 - pd_1y) ** (term / 12.0)

        # Inverse normal of PD_term
        inv_norm = norm.ppf(pd_term)

        # Generate idiosyncratic shocks
        eps = np.random.normal(size=n_sims)
        latent = np.sqrt(rho) * Y + np.sqrt(1 - rho) * eps
        default_flags[i, :] = (latent < inv_norm).astype(int)

    return default_flags

def run_monte_carlo_simple(collateral, tranches, n_sims=100, rho=0.3):
    """
    Simplified Monte Carlo simulation counting defaults only.
    """
    print(f"\nRunning simplified Monte Carlo with {n_sims} simulations...")
    
    default_matrix = simulate_default_times(collateral, n_sims, rho)
    
    # Calculate statistics
    default_counts = default_matrix.sum(axis=0)
    default_rate_by_loan = default_matrix.mean(axis=1)
    
    results = pd.DataFrame({
        'simulation': range(n_sims),
        'total_defaults': default_counts,
        'default_rate': default_counts / len(collateral)
    })
    
    return results, default_rate_by_loan

# -----------------------------------------------------------------------------
# 7. Run the deterministic scenario
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("STARTING CLO CASHFLOW ANALYSIS")
print("="*60)

det_hist = run_deterministic(collateral_df, tranches_df, get_index_rate)

if not det_hist.empty:
    print("\n" + "-"*40)
    print("DETERMINISTIC SCENARIO RESULTS")
    print("-"*40)
    print("\nLast 5 periods:")
    print(det_hist.tail())
    
    # Plot AR and SUBORD balances over time
    plt.figure(figsize=(12,6))
    if 'AR' in det_hist.columns:
        plt.plot(det_hist['date'], det_hist['AR'], label='AR (Senior)', linewidth=2)
    if 'SUBORD' in det_hist.columns:
        plt.plot(det_hist['date'], det_hist['SUBORD'], label='SUBORD (Equity)', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Balance ($)')
    plt.title('Deterministic Scenario – Tranche Balances Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Final balances
    print("\nFinal Tranche Balances:")
    final = det_hist.iloc[-1]
    for t in ['AR', 'BR', 'CR', 'DR', 'ER', 'SUBORD']:
        if t in final:
            print(f"{t}: ${final[t]:,.2f}")
else:
    print("Deterministic simulation produced no output – check data.")

# -----------------------------------------------------------------------------
# 8. Run simplified Monte Carlo
# -----------------------------------------------------------------------------
print("\n" + "-"*40)
print("MONTE CARLO SIMULATION")
print("-"*40)

mc_results, loan_default_rates = run_monte_carlo_simple(collateral_df, tranches_df, n_sims=500, rho=0.3)

print("\nMonte Carlo Summary Statistics:")
print(mc_results.describe())

# Plot default distribution
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(mc_results['total_defaults'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Defaults')
plt.ylabel('Frequency')
plt.title('Distribution of Defaults Across Simulations')
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
loan_default_pct = loan_default_rates * 100
plt.hist(loan_default_pct, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Default Probability (%)')
plt.ylabel('Number of Loans')
plt.title('Default Probability by Loan')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Top 10 riskiest loans
risky_loans = collateral_df.copy()
risky_loans['default_prob'] = loan_default_rates
risky_loans = risky_loans.nlargest(10, 'default_prob')[['Asset_Name', 'Current_Balance', 'Adjusted_Moody\'s_Rating', 'default_prob']]
print("\nTop 10 Riskiest Loans:")
print(risky_loans.to_string(index=False))

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
