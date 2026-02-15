# ============================================
# TASK 1: Build PD Model
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# Generate sample data (since we don't have actual data)
# ============================================

def generate_sample_data(n_loans=10000, n_quarters=20):
    """
    Generate synthetic loan and macro data for demonstration
    """
    # Check pandas version to handle frequency changes
    import pandas as pd
    pd_version = pd.__version__
    
    # Use appropriate quarterly frequency based on pandas version
    if pd_version >= '2.2.0':
        # For newer pandas versions (2.2.0+), use 'QE' instead of 'Q'
        freq = 'QE'
    else:
        freq = 'Q'
    
    # Generate macro data
    quarters = pd.date_range(start='2018-01-01', periods=n_quarters, freq=freq)
    macro_data = pd.DataFrame({
        'quarter': quarters,
        'gdp_growth': np.random.normal(0.02, 0.01, n_quarters),  # 2% avg growth
        'unemployment': np.random.normal(0.05, 0.01, n_quarters),  # 5% avg unemployment
        'interest_rate': np.random.normal(0.03, 0.005, n_quarters),  # 3% avg rate
        'inflation': np.random.normal(0.02, 0.005, n_quarters)  # 2% avg inflation
    })
    
    # Generate loan data
    loan_data = []
    for i in range(n_loans):
        # Random quarter assignment
        quarter_idx = np.random.randint(0, n_quarters)
        quarter = quarters[quarter_idx]
        
        # Get macro conditions for that quarter
        macro_conditions = macro_data[macro_data['quarter'] == quarter].iloc[0]
        
        # Loan characteristics
        credit_score = np.random.normal(700, 50)  # Normal credit score
        ltv = np.random.uniform(0.5, 0.95)  # Loan-to-value ratio
        dti = np.random.uniform(0.1, 0.5)  # Debt-to-income ratio
        loan_amount = np.random.uniform(10000, 500000)
        employment_years = np.random.exponential(5)
        
        # Calculate default probability based on features
        log_odds = (
            -8  # intercept
            - 0.01 * (credit_score - 700)  # higher credit score -> lower PD
            + 2 * ltv  # higher LTV -> higher PD
            + 3 * dti  # higher DTI -> higher PD
            - 0.5 * employment_years  # more experience -> lower PD
            - 20 * macro_conditions['gdp_growth']  # higher GDP growth -> lower PD
            + 20 * macro_conditions['unemployment']  # higher unemployment -> higher PD
            + 10 * macro_conditions['interest_rate']  # higher rates -> higher PD
            + np.random.normal(0, 0.5)  # random noise
        )
        
        pd_true = 1 / (1 + np.exp(-log_odds))
        
        # Generate default (1) or not (0)
        default = np.random.binomial(1, min(pd_true, 0.3))  # cap at 30% for realism
        
        loan_data.append({
            'loan_id': i,
            'quarter': quarter,
            'credit_score': credit_score,
            'ltv': ltv,
            'dti': dti,
            'loan_amount': loan_amount,
            'employment_years': employment_years,
            'gdp_growth': macro_conditions['gdp_growth'],
            'unemployment': macro_conditions['unemployment'],
            'interest_rate': macro_conditions['interest_rate'],
            'inflation': macro_conditions['inflation'],
            'default': default
        })
    
    loan_data = pd.DataFrame(loan_data)
    return loan_data, macro_data

# Generate data
print("Generating sample data...")
loan_data, macro_data = generate_sample_data(n_loans=10000, n_quarters=20)
print("Data shape:", loan_data.shape)
print("\nFirst few rows:")
print(loan_data.head())
print("\nDefault rate: {:.2%}".format(loan_data['default'].mean()))

# ============================================
# Prepare data for modeling
# ============================================

# Define features for the model
feature_cols = ['credit_score', 'ltv', 'dti', 'employment_years', 
                'gdp_growth', 'unemployment', 'interest_rate', 'inflation']
target_col = 'default'

# Split data into train and test sets
X = loan_data[feature_cols]
y = loan_data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# Train Logistic Regression Model
# ============================================

print("\n" + "="*50)
print("TASK 1: BUILDING PD MODEL")
print("="*50)

logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_proba_logistic = logistic_model.predict_proba(X_test_scaled)[:, 1]
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Calculate AUC
auc_logistic = roc_auc_score(y_test, y_pred_proba_logistic)
print(f"\nLogistic Regression AUC: {auc_logistic:.4f}")

# ============================================
# Train Random Forest for comparison
# ============================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

# Calculate AUC
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"Random Forest AUC: {auc_rf:.4f}")

# ============================================
# Model Interpretation
# ============================================

# Logistic Regression Coefficients
coefficients = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': logistic_model.coef_[0]
})
coefficients['abs_coefficient'] = np.abs(coefficients['coefficient'])
coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
print("\nLogistic Regression Coefficients:")
print(coefficients)

# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nRandom Forest Feature Importance:")
print(feature_importance)

# ============================================
# Plot ROC Curves
# ============================================

plt.figure(figsize=(10, 8))

# Logistic Regression ROC
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_proba_logistic)
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (AUC = {auc_logistic:.3f})')

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})')

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - PD Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# TASK 2: Stress Scenario
# ============================================

print("\n" + "="*50)
print("TASK 2: STRESS SCENARIO")
print("="*50)

def apply_stress_scenario(loan_data, model, scaler, feature_cols, stress_scenario):
    """
    Apply macro stress scenario and calculate stressed PD and EL
    
    Parameters:
    - loan_data: original loan data
    - model: trained PD model
    - scaler: fitted scaler
    - feature_cols: list of feature names
    - stress_scenario: dict with macro shocks
    """
    
    # Create stressed data (copy to avoid modifying original)
    stressed_data = loan_data.copy()
    
    # Apply macro shocks
    for macro_var, shock_value in stress_scenario.items():
        if macro_var in stressed_data.columns:
            stressed_data[macro_var] = stressed_data[macro_var] + shock_value
    
    # Calculate stressed PD
    X_stressed = stressed_data[feature_cols]
    X_stressed_scaled = scaler.transform(X_stressed)
    
    # Get PD predictions
    stressed_pd = model.predict_proba(X_stressed_scaled)[:, 1]
    
    return stressed_pd

# Define stress scenario
stress_scenario = {
    'gdp_growth': -0.02,  # GDP growth drops by 2%
    'unemployment': 0.02,  # Unemployment increases by 2%
    'interest_rate': 0.01  # Interest rates increase by 1%
}

print("\nStress Scenario:")
for var, shock in stress_scenario.items():
    print(f"  {var}: {shock:+.2%}")

# Calculate baseline PD
X_all_scaled = scaler.transform(loan_data[feature_cols])
baseline_pd = logistic_model.predict_proba(X_all_scaled)[:, 1]

# Calculate stressed PD
stressed_pd = apply_stress_scenario(loan_data, logistic_model, scaler, feature_cols, stress_scenario)

# Add to dataframe
loan_data['baseline_pd'] = baseline_pd
loan_data['stressed_pd'] = stressed_pd

print("\nPD Statistics:")
print(f"  Baseline PD - Mean: {baseline_pd.mean():.4f}, Std: {baseline_pd.std():.4f}")
print(f"  Stressed PD - Mean: {stressed_pd.mean():.4f}, Std: {stressed_pd.std():.4f}")
print(f"  PD Increase: {(stressed_pd.mean() - baseline_pd.mean())*100:.2f}% points")

# ============================================
# Calculate Expected Loss (EL)
# ============================================

# Assume LGD (Loss Given Default) and EAD (Exposure at Default)
LGD = 0.45  # 45% loss given default (typical for unsecured loans)
EAD = loan_data['loan_amount']  # Exposure at default

# Calculate EL
loan_data['baseline_el'] = loan_data['baseline_pd'] * LGD * loan_data['loan_amount']
loan_data['stressed_el'] = loan_data['stressed_pd'] * LGD * loan_data['loan_amount']

portfolio_baseline_el = loan_data['baseline_el'].sum()
portfolio_stressed_el = loan_data['stressed_el'].sum()

print(f"\nPortfolio Expected Loss (EL):")
print(f"  Baseline EL: ${portfolio_baseline_el:,.0f}")
print(f"  Stressed EL: ${portfolio_stressed_el:,.0f}")
print(f"  EL Increase: ${portfolio_stressed_el - portfolio_baseline_el:,.0f}")
print(f"  EL Increase: {(portfolio_stressed_el/portfolio_baseline_el - 1)*100:.1f}%")

# ============================================
# TASK 3: Portfolio Risk (EL, UL, VaR)
# ============================================

print("\n" + "="*50)
print("TASK 3: PORTFOLIO RISK METRICS")
print("="*50)

def calculate_portfolio_risk(loan_data, lgd=0.45, confidence_level=0.999, n_simulations=10000):
    """
    Calculate EL, UL, and VaR using Monte Carlo simulation
    
    Parameters:
    - loan_data: dataframe with PD and loan amount
    - lgd: Loss Given Default
    - confidence_level: confidence level for VaR
    - n_simulations: number of Monte Carlo simulations
    """
    
    n_loans = len(loan_data)
    pd_values = loan_data['baseline_pd'].values
    ead_values = loan_data['loan_amount'].values
    
    # Expected Loss
    el = np.sum(pd_values * lgd * ead_values)
    
    # Individual loan risk contributions
    individual_ul = np.sqrt(pd_values * (1 - pd_values)) * lgd * ead_values
    
    # Unexpected Loss with correlation (simplified single-factor model)
    # Assume asset correlation of 0.15 (typical for corporate loans)
    rho = 0.15
    
    # Vasicek model for correlated defaults
    # Generate systematic factor
    np.random.seed(42)
    systematic_factor = np.random.normal(0, 1, n_simulations)
    
    # Simulate losses
    simulated_losses = []
    
    for sim in range(n_simulations):
        # Idiosyncratic factors
        idiosyncratic = np.random.normal(0, 1, n_loans)
        
        # Asset values (single-factor model)
        asset_values = np.sqrt(rho) * systematic_factor[sim] + np.sqrt(1 - rho) * idiosyncratic
        
        # Default thresholds (based on PD)
        # Use inverse normal CDF to get thresholds
        from scipy.stats import norm
        default_thresholds = norm.ppf(pd_values)
        
        # Determine defaults
        defaults = asset_values < default_thresholds
        
        # Calculate loss
        loss = np.sum(defaults * lgd * ead_values)
        simulated_losses.append(loss)
    
    simulated_losses = np.array(simulated_losses)
    
    # Calculate UL (standard deviation of losses)
    ul = np.std(simulated_losses)
    
    # Calculate VaR
    var = np.percentile(simulated_losses, confidence_level * 100)
    
    # Calculate Expected Shortfall (CVaR) - average loss beyond VaR
    cvar = simulated_losses[simulated_losses >= var].mean()
    
    return {
        'EL': el,
        'UL': ul,
        'VaR': var,
        'CVaR': cvar,
        'simulated_losses': simulated_losses
    }

# Calculate risk metrics
risk_metrics = calculate_portfolio_risk(loan_data)

print(f"\nPortfolio Risk Metrics (Baseline):")
print(f"  Expected Loss (EL): ${risk_metrics['EL']:,.0f}")
print(f"  Unexpected Loss (UL): ${risk_metrics['UL']:,.0f}")
print(f"  99.9% VaR: ${risk_metrics['VaR']:,.0f}")
print(f"  99.9% CVaR: ${risk_metrics['CVaR']:,.0f}")
print(f"  UL/EL Ratio: {risk_metrics['UL']/risk_metrics['EL']:.2f}")
print(f"  VaR/EL Ratio: {risk_metrics['VaR']/risk_metrics['EL']:.2f}")

# Plot loss distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(risk_metrics['simulated_losses'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(risk_metrics['EL'], color='green', linestyle='--', linewidth=2, label=f"EL = ${risk_metrics['EL']:,.0f}")
plt.axvline(risk_metrics['VaR'], color='red', linestyle='--', linewidth=2, label=f"VaR = ${risk_metrics['VaR']:,.0f}")
plt.xlabel('Portfolio Loss ($)')
plt.ylabel('Frequency')
plt.title('Loss Distribution (Monte Carlo Simulation)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(risk_metrics['simulated_losses'])
plt.ylabel('Portfolio Loss ($)')
plt.title('Loss Distribution - Box Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# TASK 4: Model Explanation
# ============================================

print("\n" + "="*50)
print("TASK 4: MODEL EXPLANATION")
print("="*50)

# ============================================
# Feature Importance Visualization
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Logistic Regression Coefficients
ax = axes[0, 0]
colors = ['red' if x < 0 else 'green' for x in coefficients['coefficient']]
ax.barh(coefficients['feature'], coefficients['coefficient'], color=colors)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Coefficient Value')
ax.set_title('Logistic Regression Coefficients\n(Positive = Higher PD, Negative = Lower PD)')
ax.grid(True, alpha=0.3)

# 2. Random Forest Feature Importance
ax = axes[0, 1]
ax.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance')
ax.grid(True, alpha=0.3)

# 3. PD Distribution - Baseline vs Stressed
ax = axes[1, 0]
ax.hist(loan_data['baseline_pd'], bins=30, alpha=0.5, label='Baseline', density=True)
ax.hist(loan_data['stressed_pd'], bins=30, alpha=0.5, label='Stressed', density=True)
ax.set_xlabel('Probability of Default (PD)')
ax.set_ylabel('Density')
ax.set_title('PD Distribution: Baseline vs Stressed Scenario')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Macro Sensitivity Analysis
ax = axes[1, 1]

# Create sensitivity analysis for key macro variables
macro_vars = ['gdp_growth', 'unemployment', 'interest_rate']
sensitivity_results = {}

# Take a representative loan (median values)
representative_loan = loan_data[feature_cols].median().to_frame().T

for macro_var in macro_vars:
    shocks = np.linspace(-0.03, 0.03, 20)
    pd_values = []
    
    for shock in shocks:
        # Create shocked data
        shocked_loan = representative_loan.copy()
        shocked_loan[macro_var] = shocked_loan[macro_var].values[0] + shock
        
        # Scale and predict
        shocked_scaled = scaler.transform(shocked_loan)
        pd_shocked = logistic_model.predict_proba(shocked_scaled)[0, 1]
        pd_values.append(pd_shocked)
    
    sensitivity_results[macro_var] = pd_values
    ax.plot(shocks*100, pd_values, label=macro_var, linewidth=2, marker='o', markersize=4)

ax.set_xlabel('Macro Variable Shock (percentage points)')
ax.set_ylabel('Predicted PD')
ax.set_title('Macro Sensitivity Analysis\n(Representative Loan)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_explanation.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Detailed Model Explanation Text
# ============================================

print("\n" + "="*50)
print("DETAILED MODEL EXPLANATION")
print("="*50)

print("\n1. MODEL DRIVERS:")
print("-" * 30)

# Sort coefficients by absolute value for interpretation
coef_sorted = coefficients.sort_values('abs_coefficient', ascending=False)
print("\nTop drivers of default (by coefficient magnitude):")
for idx, row in coef_sorted.head(3).iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"  • {row['feature']}: {direction} PD (coef: {row['coefficient']:.4f})")

print("\n2. MACRO SENSITIVITY:")
print("-" * 30)

# Calculate average sensitivity
for macro_var in macro_vars:
    # Get coefficient for macro variable
    coef = coefficients[coefficients['feature'] == macro_var]['coefficient'].values[0]
    
    # Calculate impact of 1% change
    impact_pct = np.exp(coef * 0.01) - 1  # Approximate % change in odds
    
    print(f"\n{macro_var}:")
    print(f"  • Coefficient: {coef:.4f}")
    print(f"  • Interpretation: A 1% increase in {macro_var} changes the odds of default by {impact_pct*100:.1f}%")
    
    if macro_var == 'unemployment':
        print(f"  • Business meaning: Higher unemployment reduces income, making loan repayment harder")

print("\n3. MODEL PERFORMANCE:")
print("-" * 30)
print(f"  • AUC Score: {auc_logistic:.4f}")
print(f"  • Interpretation: The model correctly ranks a random defaulting loan higher than")
print(f"    a random non-defaulting loan {auc_logistic*100:.1f}% of the time")
print(f"  • This indicates {'excellent' if auc_logistic > 0.8 else 'good' if auc_logistic > 0.7 else 'moderate'} discriminatory power")

print("\n4. STRESS TESTING INSIGHTS:")
print("-" * 30)
print(f"  • Under the stress scenario (GDP -2%, Unemployment +2%, Rates +1%):")
print(f"    - Average PD increases from {baseline_pd.mean():.2%} to {stressed_pd.mean():.2%}")
print(f"    - Portfolio EL increases by ${portfolio_stressed_el - portfolio_baseline_el:,.0f}")
print(f"    - This represents a {(portfolio_stressed_el/portfolio_baseline_el - 1)*100:.1f}% increase in expected losses")

print("\n5. PORTFOLIO RISK SUMMARY:")
print("-" * 30)
print(f"  • Expected Loss (EL): ${risk_metrics['EL']:,.0f} (predictable losses)")
print(f"  • Unexpected Loss (UL): ${risk_metrics['UL']:,.0f} (volatility around EL)")
print(f"  • 99.9% VaR: ${risk_metrics['VaR']:,.0f} (worst-case loss at 99.9% confidence)")
print(f"  • Capital buffer needed: ${risk_metrics['VaR'] - risk_metrics['EL']:,.0f} (above expected losses)")

# ============================================
# Save all results to CSV
# ============================================

# Save loan data with PDs
loan_data.to_csv('loan_data_with_pd.csv', index=False)
print("\n" + "="*50)
print("Results saved to CSV files:")
print("  • loan_data_with_pd.csv")
print("  • roc_curves.png")
print("  • loss_distribution.png")
print("  • model_explanation.png")
print("="*50)
