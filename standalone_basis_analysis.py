"""
Credit Basis Analysis - Standalone Version
Uses embedded data from the analysis output
No network connection required
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Ensure output directory exists
os.makedirs('/mnt/user-data/outputs', exist_ok=True)

print("="*70)
print("CREDIT BASIS REGRESSION ANALYSIS")
print("="*70)

# Data from your successful analysis (July-December 2025)
data_dict = {
    'date': ['2025-07-31', '2025-08-31', '2025-09-30', '2025-10-31', '2025-11-30', '2025-12-31'],
    'basis': [-21, -25, -22, -28, -31, -29],
    'OAS_pct': [0.79, 0.81, 0.76, 0.80, 0.82, 0.79],
    'TED': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'VIX': [16.719999, 15.360000, 16.280001, 17.440001, 16.350000, 14.950000],
    'SP_ret': [2.162044, 1.945744, 3.494335, 2.326084, 0.214968, -0.019938],
    'oas_bps': [79, 81, 76, 80, 82, 79],
    'cdx_bps': [58, 56, 54, 52, 51, 50]
}

# Create DataFrame
df = pd.DataFrame(data_dict)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print("\n📊 DATA SUMMARY")
print("-" * 70)
print(df)
print(f"\nData period: {df.index[0].strftime('%B %Y')} to {df.index[-1].strftime('%B %Y')}")
print(f"Number of observations: {len(df)}")

# Regression coefficients from your statsmodels output
print("\n" + "="*70)
print("REGRESSION MODEL: Basis ~ TED + VIX + SP_ret")
print("="*70)

coefficients = {
    'Constant': -0.0007,  # Calculated from residuals
    'TED': -19.5534,
    'VIX': -1.2879,
    'SP_ret': 2.7382
}

print("\nCoefficients:")
for name, value in coefficients.items():
    if name == 'Constant':
        print(f"  {name:12s}: {value:8.4f}")
    else:
        print(f"  {name:12s}: {value:8.4f}  (t-stat from original: see regression output)")

# Calculate fitted values
df['fitted'] = (coefficients['Constant'] + 
                coefficients['TED'] * df['TED'] + 
                coefficients['VIX'] * df['VIX'] + 
                coefficients['SP_ret'] * df['SP_ret'])

df['residuals'] = df['basis'] - df['fitted']

# Model fit statistics
ss_res = np.sum(df['residuals']**2)
ss_tot = np.sum((df['basis'] - df['basis'].mean())**2)
r_squared = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean(df['residuals']**2))
mae = np.mean(np.abs(df['residuals']))

print(f"\nModel Fit Statistics:")
print(f"  R-squared:            {r_squared:.3f}")
print(f"  Adjusted R-squared:   {1 - (1-r_squared)*(len(df)-1)/(len(df)-4):.3f}")
print(f"  RMSE:                 {rmse:.2f} bps")
print(f"  Mean Absolute Error:  {mae:.2f} bps")

# Current market prediction
print("\n" + "="*70)
print("FAIR VALUE PREDICTION (Current Market Conditions)")
print("="*70)

current_market = {
    'TED': 17.0,
    'VIX': 16.0,
    'SP_ret': 1.0
}

print("\nCurrent Market Inputs:")
for name, value in current_market.items():
    print(f"  {name:10s}: {value:6.1f}")

fair_value = coefficients['Constant']
print(f"\nCalculation:")
print(f"  Starting value (constant): {fair_value:8.4f}")

for name in ['TED', 'VIX', 'SP_ret']:
    contribution = coefficients[name] * current_market[name]
    fair_value += contribution
    print(f"  + {name:8s} ({current_market[name]:6.1f} × {coefficients[name]:8.4f}) = {contribution:8.2f}")

print(f"\n  ➜ Fair Value Basis:    {fair_value:7.1f} bps")
print(f"  ➜ Current Market:        -26.5 bps")
deviation = -26.5 - fair_value
print(f"  ➜ Deviation:           {deviation:+7.1f} bps")

if abs(deviation) < 5:
    interpretation = "✓ Fairly valued (within ±5 bps tolerance)"
elif deviation < 0:
    interpretation = f"⚠ Basis is {abs(deviation):.1f} bps WIDER than fair value → Potential BUY"
else:
    interpretation = f"⚠ Basis is {deviation:.1f} bps TIGHTER than fair value → Potential SELL"

print(f"\n  Interpretation: {interpretation}")

# Summary statistics
print("\n" + "="*70)
print("VARIABLE SUMMARY STATISTICS")
print("="*70)

summary = pd.DataFrame({
    'Mean': df[['basis', 'TED', 'VIX', 'SP_ret']].mean(),
    'Std Dev': df[['basis', 'TED', 'VIX', 'SP_ret']].std(),
    'Min': df[['basis', 'TED', 'VIX', 'SP_ret']].min(),
    'Max': df[['basis', 'TED', 'VIX', 'SP_ret']].max(),
})
print(summary.round(2))

# Create comprehensive visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Main analysis plot
fig1, axes = plt.subplots(2, 2, figsize=(16, 11))
fig1.suptitle('Credit Basis Regression Analysis', fontsize=18, fontweight='bold', y=0.997)

# Color scheme
color_actual = '#1f77b4'
color_fitted = '#d62728'
color_fair = '#2ca02c'
color_current = '#ff7f0e'

# Plot 1: Actual vs Fitted Basis
ax1 = axes[0, 0]
ax1.plot(df.index, df['basis'], 'o-', label='Actual Basis', 
        color=color_actual, linewidth=2.5, markersize=11, 
        markeredgewidth=2, markeredgecolor='white', zorder=3)
ax1.plot(df.index, df['fitted'], 's--', label='Fitted Basis', 
        color=color_fitted, linewidth=2.5, markersize=9, 
        markeredgewidth=2, markeredgecolor='white', zorder=3)
ax1.axhline(y=fair_value, color=color_fair, linestyle=':', linewidth=2.5, 
           label=f'Fair Value: {fair_value:.1f} bps', zorder=2)
ax1.axhline(y=-26.5, color=color_current, linestyle='-.', linewidth=2, 
           label='Current Market: -26.5 bps', alpha=0.8, zorder=2)
ax1.fill_between(df.index, df['basis'], df['fitted'], alpha=0.15, color='gray', zorder=1)

# Add R² annotation
ax1.text(0.02, 0.98, f'R² = {r_squared:.3f}', transform=ax1.transAxes,
        fontsize=12, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Basis (bps)', fontsize=12, fontweight='bold')
ax1.set_title('Actual vs Fitted Basis (CDX - OAS)', fontsize=13, fontweight='bold', pad=12)
ax1.legend(loc='lower left', fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# Plot 2: Residuals
ax2 = axes[0, 1]
colors = ['#2ca02c' if x >= 0 else '#d62728' for x in df['residuals']]
bars = ax2.bar(df.index, df['residuals'], color=colors, alpha=0.75, 
              edgecolor='black', linewidth=1.5, width=12)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=2.5, zorder=2)

# Add values on bars
for idx, val in zip(df.index, df['residuals']):
    ax2.text(idx, val + (0.4 if val > 0 else -0.4), f'{val:.1f}', 
            ha='center', va='bottom' if val > 0 else 'top', 
            fontsize=10, fontweight='bold')

# Add RMSE annotation
ax2.text(0.98, 0.98, f'RMSE = {rmse:.2f} bps', transform=ax2.transAxes,
        fontsize=12, fontweight='bold', verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals (bps)', fontsize=12, fontweight='bold')
ax2.set_title(f'Regression Residuals', fontsize=13, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.4, linestyle='--', axis='y', linewidth=0.8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# Plot 3: Market Drivers
ax3 = axes[1, 0]
ax3_vix = ax3.twinx()
ax3_sp = ax3.twinx()
ax3_sp.spines['right'].set_position(('outward', 70))

# TED (note: constant at 0.5, so just show as reference)
p1, = ax3.plot(df.index, df['TED'], 'o-', color='#1f77b4', 
              label='TED Spread', linewidth=2.5, markersize=10,
              markeredgewidth=2, markeredgecolor='white')

# VIX
p2, = ax3_vix.plot(df.index, df['VIX'], 's-', color='#d62728', 
                   label='VIX Index', linewidth=2.5, markersize=10,
                   markeredgewidth=2, markeredgecolor='white')

# S&P Return
p3, = ax3_sp.plot(df.index, df['SP_ret'], '^-', color='#2ca02c', 
                  label='S&P 500 Return', linewidth=2.5, markersize=10,
                  markeredgewidth=2, markeredgecolor='white')

ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_ylabel('TED Spread (bps)', fontsize=11, fontweight='bold', color='#1f77b4')
ax3_vix.set_ylabel('VIX Index', fontsize=11, fontweight='bold', color='#d62728')
ax3_sp.set_ylabel('S&P 500 Return (%)', fontsize=11, fontweight='bold', color='#2ca02c')
ax3.set_title('Market Risk Indicators (Independent Variables)', fontsize=13, fontweight='bold', pad=12)

ax3.tick_params(axis='y', labelcolor='#1f77b4', labelsize=10)
ax3_vix.tick_params(axis='y', labelcolor='#d62728', labelsize=10)
ax3_sp.tick_params(axis='y', labelcolor='#2ca02c', labelsize=10)

lines = [p1, p2, p3]
ax3.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=10, framealpha=0.95)
ax3.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# Plot 4: OAS vs CDX Components
ax4 = axes[1, 1]
width = 10
x_pos = np.arange(len(df))

bars1 = ax4.bar(x_pos - width/2/20, df['oas_bps'], width=width, 
               label='OAS Index', color='#ff7f0e', alpha=0.8, 
               edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x_pos + width/2/20, df['cdx_bps'], width=width, 
               label='CDX Index', color='#9467bd', alpha=0.8, 
               edgecolor='black', linewidth=1.5)

# Add values on bars
for i, (oas, cdx) in enumerate(zip(df['oas_bps'], df['cdx_bps'])):
    ax4.text(i - width/2/20, oas + 1.5, str(oas), 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.text(i + width/2/20, cdx + 1.5, str(cdx), 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.set_ylabel('Spread (bps)', fontsize=12, fontweight='bold')
ax4.set_title('Credit Spread Components\n(Basis = CDX - OAS)', fontsize=13, fontweight='bold', pad=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([d.strftime('%b\n%Y') for d in df.index])
ax4.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax4.grid(True, alpha=0.4, linestyle='--', axis='y', linewidth=0.8)

plt.tight_layout()

plot_path = '/mnt/user-data/outputs/basis_analysis_plots.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Main analysis plot saved: {plot_path}")
plt.close()

# Diagnostic plots
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Regression Diagnostics', fontsize=16, fontweight='bold', y=0.995)

# 1. Residuals vs Fitted
ax1 = axes2[0, 0]
ax1.scatter(df['fitted'], df['residuals'], s=120, alpha=0.7, 
           edgecolors='black', linewidth=2, c=colors)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
for i, (x, y) in enumerate(zip(df['fitted'], df['residuals'])):
    ax1.annotate(df.index[i].strftime('%b'), (x, y), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax1.set_xlabel('Fitted Values (bps)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Residuals (bps)', fontsize=11, fontweight='bold')
ax1.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3)

# 2. Q-Q Plot
from scipy import stats
ax2 = axes2[0, 1]
stats.probplot(df['residuals'], dist="norm", plot=ax2)
ax2.get_lines()[0].set_markerfacecolor('#1f77b4')
ax2.get_lines()[0].set_markersize(10)
ax2.get_lines()[0].set_markeredgecolor('black')
ax2.get_lines()[0].set_markeredgewidth(1.5)
ax2.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3)

# 3. Scale-Location
ax3 = axes2[1, 0]
standardized_residuals = df['residuals'] / df['residuals'].std()
ax3.scatter(df['fitted'], np.sqrt(np.abs(standardized_residuals)), 
           s=120, alpha=0.7, edgecolors='black', linewidth=2, c='#ff7f0e')
ax3.set_xlabel('Fitted Values (bps)', fontsize=11, fontweight='bold')
ax3.set_ylabel('√|Standardized Residuals|', fontsize=11, fontweight='bold')
ax3.set_title('Scale-Location Plot', fontsize=12, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3)

# 4. Residuals vs Order
ax4 = axes2[1, 1]
ax4.plot(range(len(df)), df['residuals'], 'o-', markersize=12, linewidth=2.5, 
        color='#2ca02c', markeredgewidth=2, markeredgecolor='white')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Observation Order', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residuals (bps)', fontsize=11, fontweight='bold')
ax4.set_title('Residuals vs Observation Order', fontsize=12, fontweight='bold', pad=10)
ax4.set_xticks(range(len(df)))
ax4.set_xticklabels([d.strftime('%b') for d in df.index])
ax4.grid(True, alpha=0.3)

plt.tight_layout()

diagnostics_path = '/mnt/user-data/outputs/regression_diagnostics.png'
plt.savefig(diagnostics_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Diagnostics plot saved: {diagnostics_path}")
plt.close()

# Summary report
print("\n" + "="*70)
print("ANALYSIS COMPLETE - KEY FINDINGS")
print("="*70)
print(f"""
Model Performance:
  • R-squared:           {r_squared:.3f} ({r_squared*100:.1f}% of variance explained)
  • RMSE:                {rmse:.2f} bps
  • Mean Abs Error:      {mae:.2f} bps

Fair Value Analysis:
  • Predicted fair value: {fair_value:.1f} bps
  • Current market:       -26.5 bps
  • Deviation:            {deviation:+.1f} bps
  
{interpretation}

Note: The large predicted fair value (-350 bps) is driven primarily by the
TED coefficient (-19.55) multiplied by current TED (17.0). This suggests:
  1. Limited sample size (6 observations)
  2. TED data used dummy values (0.5) during model fitting
  3. Model may need recalibration with actual current TED spreads

Recommendation: Verify current TED spread data and consider refitting the model
with real-time data for more accurate fair value estimates.
""")

print("="*70)
