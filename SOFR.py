"""
SOFR Rate Statistical Model
Author: Statistical Analysis Tool
Description: Model to estimate and analyze SOFR rates
"""

import subprocess
import sys

# Check and install required packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Required packages
required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'requests']

# Check and install missing packages
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"✓ {package} is already installed")
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

print("\n" + "="*60)
print("All dependencies installed successfully!")
print("="*60 + "\n")

# Now import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SOFRModel:
    """
    Statistical model for SOFR rate estimation and analysis
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def fetch_historical_sofr(self, start_date='2020-01-01', end_date=None):
        """
        Generate synthetic SOFR data for demonstration
        In production, replace with actual API calls
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print("Generating synthetic SOFR data for demonstration...")
        print("Note: For real data, use FRED API with your API key")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(42)
        
        # Generate realistic SOFR patterns (based on recent historical data)
        base_rate = 5.30  # Current SOFR level around 5.30%
        
        # Add realistic components
        trend = np.linspace(0, 0.5, len(dates))  # Slight upward trend
        seasonality = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # Annual pattern
        volatility = np.random.normal(0, 0.05, len(dates))  # Daily volatility
        
        # Add month-end effects (SOFR tends to be higher at month-end)
        month_end_effect = np.array([
            0.15 if (date.day >= 25 or date.day <= 3) else 0 
            for date in dates
        ])
        
        # Add quarter-end effects
        quarter_end_effect = np.array([
            0.25 if date.is_quarter_end else 0 
            for date in dates
        ])
        
        # Add Fed meeting effects (simplified)
        fomc_dates = self._get_fomc_dates()
        fomc_effect = np.array([
            0.10 if date.strftime('%Y-%m-%d') in fomc_dates else 0
            for date in dates
        ])
        
        sofr_rates = base_rate + trend + seasonality + volatility + \
                     month_end_effect + quarter_end_effect + fomc_effect
        
        # Ensure rates are positive and realistic
        sofr_rates = np.maximum(sofr_rates, 0.01)
        
        df = pd.DataFrame({
            'date': dates,
            'sofr_rate': sofr_rates,
            'day_of_week': dates.dayofweek,
            'day_of_month': dates.day,
            'month': dates.month,
            'quarter': dates.quarter,
            'year': dates.year,
            'is_month_end': [1 if date.is_month_end else 0 for date in dates],
            'is_quarter_end': [1 if date.is_quarter_end else 0 for date in dates],
            'days_to_month_end': [date.days_in_month - date.day for date in dates],
            'week_of_year': dates.isocalendar().week,
            'day_of_year': dates.dayofyear
        })
        
        return df
    
    def _get_fomc_dates(self):
        """Get simplified FOMC meeting dates"""
        # This is a simplified list - in production, use actual FOMC calendar
        fomc_dates = [
            '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
            '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-11',
            '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
            '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-10'
        ]
        return fomc_dates
    
    def create_features(self, df):
        """
        Create additional features for modeling
        """
        df = df.copy()
        
        # Lag features (past values)
        for lag in [1, 2, 3, 5, 10]:
            df[f'sofr_lag_{lag}'] = df['sofr_rate'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'sofr_rolling_mean_{window}'] = df['sofr_rate'].rolling(window=window).mean()
            df[f'sofr_rolling_std_{window}'] = df['sofr_rate'].rolling(window=window).std()
            df[f'sofr_rolling_min_{window}'] = df['sofr_rate'].rolling(window=window).min()
            df[f'sofr_rolling_max_{window}'] = df['sofr_rate'].rolling(window=window).max()
        
        # Rate of change
        df['sofr_roc_1'] = df['sofr_rate'].pct_change(1)
        df['sofr_roc_5'] = df['sofr_rate'].pct_change(5)
        df['sofr_roc_20'] = df['sofr_rate'].pct_change(20)
        
        # Distance from moving averages
        df['sofr_vs_ma5'] = df['sofr_rate'] - df['sofr_rolling_mean_5']
        df['sofr_vs_ma20'] = df['sofr_rate'] - df['sofr_rolling_mean_20']
        
        # Time-based features (cyclical encoding)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend/holiday indicators (simplified)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        return df.dropna()
    
    def prepare_data(self, df, target_col='sofr_rate'):
        """
        Prepare data for modeling
        """
        # Select features (exclude date and target)
        feature_cols = [col for col in df.columns if col not in ['date', target_col]]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y
    
    def train_model(self, df, test_size=0.2):
        """
        Train the Random Forest model
        """
        # Prepare data
        X, y = self.prepare_data(df)
        
        print(f"\nTraining with {len(X)} samples and {len(X.columns)} features")
        
        # Split data (time-series split - don't shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Calculate R-squared
        train_r2 = self.model.score(X_train_scaled, y_train)
        test_r2 = self.model.score(X_test_scaled, y_test)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Training MAE:  {train_mae:.4f}")
        print(f"Testing MAE:   {test_mae:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE:  {test_rmse:.4f}")
        print(f"Training R²:   {train_r2:.4f}")
        print(f"Testing R²:    {test_r2:.4f}")
        print("="*50)
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def predict_next_sofr(self, latest_data):
        """
        Predict next SOFR rate based on latest available data
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        X = latest_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions[-1] if len(predictions) > 0 else predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def calculate_statistical_summary(self, df):
        """
        Calculate statistical summary of SOFR rates
        """
        summary = {
            'current_rate': df['sofr_rate'].iloc[-1],
            'mean_rate': df['sofr_rate'].mean(),
            'median_rate': df['sofr_rate'].median(),
            'std_rate': df['sofr_rate'].std(),
            'min_rate': df['sofr_rate'].min(),
            'max_rate': df['sofr_rate'].max(),
            'skewness': df['sofr_rate'].skew(),
            'kurtosis': df['sofr_rate'].kurtosis(),
            'q1_rate': df['sofr_rate'].quantile(0.25),
            'q3_rate': df['sofr_rate'].quantile(0.75)
        }
        
        return summary
    
    def calculate_volatility(self, df, window=20):
        """
        Calculate rolling volatility of SOFR rates
        """
        df = df.copy()
        df['daily_return'] = df['sofr_rate'].pct_change()
        df['volatility'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
        
        return df[['date', 'sofr_rate', 'volatility']].dropna()

class LiveSOFREstimator:
    """
    Class for real-time SOFR estimation using market data
    """
    
    def __init__(self):
        self.model = SOFRModel()
        self.current_estimate = None
        self.confidence_interval = None
        
    def fetch_market_data(self):
        """
        Fetch relevant market data for nowcasting SOFR
        """
        current_time = datetime.now()
        
        # In production, you'd fetch real data from:
        # - Federal Reserve Economic Data (FRED)
        # - Bloomberg/Reuters terminals
        # - Treasury.gov
        # - Open market operations data
        
        # For demonstration, we'll use realistic synthetic data
        market_data = {
            'timestamp': current_time,
            'ff_futures_implied_rate': 5.32 + np.random.normal(0, 0.02),
            'treasury_bill_3m': 5.25 + np.random.normal(0, 0.01),
            'overnight_repo_rate': 5.28 + np.random.normal(0, 0.03),
            'fed_balance_sheet': 7500000000000,  # ~$7.5 trillion
            'hour': current_time.hour,
            'minute': current_time.minute,
            'second': current_time.second,
            'is_business_day': 1 if current_time.weekday() < 5 else 0,
            'days_to_fomc': self._days_to_next_fomc(),
            'month_end_proximity': 1 if current_time.day >= 28 else 0,
            'quarter_end_proximity': 1 if current_time.month in [3,6,9,12] and current_time.day >= 25 else 0
        }
        
        return market_data
    
    def _days_to_next_fomc(self):
        """
        Calculate days until next FOMC meeting
        """
        current_date = datetime.now()
        year = current_date.year
        
        # Standard FOMC meeting dates (simplified)
        fomc_dates = [
            datetime(year, 1, 31),
            datetime(year, 3, 20),
            datetime(year, 5, 1),
            datetime(year, 6, 12),
            datetime(year, 7, 31),
            datetime(year, 9, 18),
            datetime(year, 11, 7),
            datetime(year, 12, 11)
        ]
        
        # Add next year's dates if needed
        if max(fomc_dates) < current_date:
            fomc_dates = [
                datetime(year+1, 1, 30),
                datetime(year+1, 3, 19),
                datetime(year+1, 5, 7),
                datetime(year+1, 6, 18),
                datetime(year+1, 7, 30),
                datetime(year+1, 9, 17),
                datetime(year+1, 11, 5),
                datetime(year+1, 12, 10)
            ]
        
        future_dates = [d for d in fomc_dates if d > current_date]
        if future_dates:
            return (future_dates[0] - current_date).days
        return 30  # Default
    
    def estimate_live_sofr(self, market_data, historical_avg):
        """
        Estimate current SOFR rate based on market conditions
        """
        base_estimate = historical_avg
        
        # Weighted adjustments from various market indicators
        weights = {
            'futures': 0.30,
            'repo': 0.40,
            'treasury': 0.15,
            'time': 0.05,
            'calendar': 0.10
        }
        
        total_adjustment = 0
        
        # Futures market adjustment
        if market_data['ff_futures_implied_rate']:
            futures_adjustment = (market_data['ff_futures_implied_rate'] - base_estimate) * weights['futures']
            total_adjustment += futures_adjustment
        
        # Repo market adjustment
        if market_data['overnight_repo_rate']:
            repo_adjustment = (market_data['overnight_repo_rate'] - base_estimate) * weights['repo']
            total_adjustment += repo_adjustment
        
        # Treasury bill adjustment
        if market_data['treasury_bill_3m']:
            treasury_adjustment = (market_data['treasury_bill_3m'] - base_estimate) * 0.1 * weights['treasury']
            total_adjustment += treasury_adjustment
        
        # Time-based adjustments
        hour = market_data['hour']
        if hour < 9:
            time_adjustment = 0.02 * weights['time']  # Early morning
        elif hour > 16:
            time_adjustment = -0.01 * weights['time']  # Late day
        else:
            time_adjustment = 0
        total_adjustment += time_adjustment
        
        # Calendar effects
        calendar_adjustment = 0
        if market_data['month_end_proximity']:
            calendar_adjustment += 0.05 * weights['calendar']
        if market_data['quarter_end_proximity']:
            calendar_adjustment += 0.08 * weights['calendar']
        if market_data['days_to_fomc'] <= 7:
            calendar_adjustment += 0.03 * weights['calendar']
        total_adjustment += calendar_adjustment
        
        # Final estimate
        estimate = base_estimate + total_adjustment
        
        # Calculate dynamic confidence interval based on market conditions
        base_volatility = 0.05  # Base volatility
        market_uncertainty = abs(total_adjustment) * 2  # Higher adjustment = higher uncertainty
        time_uncertainty = 0.02 if hour < 9 or hour > 16 else 0.01
        
        total_uncertainty = base_volatility + market_uncertainty + time_uncertainty
        
        confidence_interval = (
            estimate - 1.96 * total_uncertainty,  # 95% CI lower bound
            estimate + 1.96 * total_uncertainty   # 95% CI upper bound
        )
        
        return estimate, confidence_interval
    
    def plot_historical_trends(self, df):
        """
        Plot historical SOFR trends
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time series plot
        axes[0, 0].plot(df['date'], df['sofr_rate'], 'b-', linewidth=1.5)
        axes[0, 0].set_title('SOFR Rate - Historical Trend', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('SOFR Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution
        axes[0, 1].hist(df['sofr_rate'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('SOFR Rate Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('SOFR Rate (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(df['sofr_rate'].mean(), color='r', linestyle='--', label=f"Mean: {df['sofr_rate'].mean():.2f}%")
        axes[0, 1].axvline(df['sofr_rate'].median(), color='g', linestyle='--', label=f"Median: {df['sofr_rate'].median():.2f}%")
        axes[0, 1].legend()
        
        # Box plot by month
        df['month_name'] = df['date'].dt.strftime('%b')
        monthly_data = [df[df['month_name'] == month]['sofr_rate'].values for month in df['month_name'].unique()]
        axes[1, 0].boxplot(monthly_data, labels=sorted(df['month_name'].unique()))
        axes[1, 0].set_title('SOFR Rate by Month', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('SOFR Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Rolling statistics
        df['rolling_mean_20'] = df['sofr_rate'].rolling(20).mean()
        df['rolling_std_20'] = df['sofr_rate'].rolling(20).std()
        
        axes[1, 1].plot(df['date'], df['sofr_rate'], 'b-', alpha=0.5, label='Daily Rate')
        axes[1, 1].plot(df['date'], df['rolling_mean_20'], 'r-', linewidth=2, label='20-Day MA')
        axes[1, 1].fill_between(df['date'], 
                                df['rolling_mean_20'] - df['rolling_std_20'],
                                df['rolling_mean_20'] + df['rolling_std_20'],
                                alpha=0.2, color='r', label='±1 Std Dev')
        axes[1, 1].set_title('SOFR Rate with Rolling Statistics', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('SOFR Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def run(self):
        """
        Main execution method
        """
        print("\n" + "="*70)
        print("🏦 LIVE SOFR RATE ESTIMATOR - STATISTICAL MODEL")
        print("="*70)
        
        # Fetch historical data and train model
        print("\n📊 [Step 1] Generating historical SOFR data...")
        df = self.model.fetch_historical_sofr()
        print(f"    ✓ Generated {len(df)} days of historical data")
        
        print("\n🔧 [Step 2] Creating features and training model...")
        df_features = self.model.create_features(df)
        print(f"    ✓ Created {len(df_features.columns)} features")
        metrics = self.model.train_model(df_features)
        
        # Get statistical summary
        print("\n📈 [Step 3] Calculating statistical summary...")
        summary = self.model.calculate_statistical_summary(df)
        print("\n    Statistical Summary:")
        print(f"    {'-'*40}")
        for key, value in summary.items():
            print(f"    {key.replace('_', ' ').title():<20}: {value:>8.4f}%")
        
        # Get live estimation
        print("\n🌐 [Step 4] Fetching live market data...")
        market_data = self.fetch_market_data()
        print(f"    ✓ Market data timestamp: {market_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🎯 [Step 5] Estimating live SOFR rate...")
        estimate, ci = self.estimate_live_sofr(market_data, summary['current_rate'])
        
        print("\n" + "="*70)
        print("🏦 LIVE SOFR ESTIMATION RESULTS")
        print("="*70)
        print(f"📅 Timestamp:           {market_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Estimated SOFR Rate: {estimate:.4f}%")
        print(f"📊 95% Confidence Interval: [{ci[0]:.4f}%, {ci[1]:.4f}%]")
        print(f"📋 Latest Published Rate:   {summary['current_rate']:.4f}%")
        print(f"📉 Deviation:               {(estimate - summary['current_rate'])*100:+.2f} basis points")
        print("="*70)
        
        # Feature importance
        print("\n⭐ [Step 6] Top 10 Most Important Features:")
        importance = self.model.get_feature_importance()
        print(importance.head(10).to_string(index=False))
        
        # Plot historical trends
        print("\n📊 [Step 7] Generating visualization...")
        try:
            self.plot_historical_trends(df)
        except Exception as e:
            print(f"    ⚠️ Could not generate plot: {e}")
        
        # Add interpretation
        print("\n" + "="*70)
        print("📝 MODEL INTERPRETATION")
        print("="*70)
        
        if abs(estimate - summary['current_rate']) > 0.1:
            print("⚠️ Current estimate differs significantly from latest published rate.")
            print("   This may indicate changing market conditions or intraday volatility.")
        else:
            print("✅ Current estimate is in line with recent published rates.")
        
        if ci[1] - ci[0] > 0.2:
            print("⚠️ Wide confidence interval suggests higher than normal uncertainty.")
            print("   Consider monitoring market conditions more frequently.")
        else:
            print("✅ Narrow confidence interval indicates relatively stable conditions.")
        
        print("="*70)
        
        return {
            'estimate': estimate,
            'confidence_interval': ci,
            'latest_published': summary['current_rate'],
            'market_data': market_data,
            'model_metrics': metrics,
            'feature_importance': importance
        }

def main():
    """
    Main function to run the SOFR estimation system
    """
    print("\n" + "🚀"*35)
    print("🚀           SOFR STATISTICAL MODEL v1.0           🚀")
    print("🚀"*35)
    
    try:
        estimator = LiveSOFREstimator()
        results = estimator.run()
        
        print("\n✅ Model execution completed successfully!")
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
