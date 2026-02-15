# simple_rates_dashboard.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import sys

class SimpleRatesDashboard:
    def __init__(self):
        self.instruments = ['US10Y', 'TYU4', 'US5Y', 'FVU4']
        self.prices = {inst: 100.0 for inst in self.instruments}
        self.spreads = {inst: 2.0 for inst in self.instruments}  # in bps
        self.history = {inst: [] for inst in self.instruments}
        self.timestamps = []
        
        # Setup figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.ion()  # Interactive mode on
    
    def update_prices(self):
        """Update prices with random walk"""
        current_time = datetime.now()
        self.timestamps.append(current_time)
        
        for inst in self.instruments:
            # Random price change
            change = random.gauss(0, 0.0003)
            self.prices[inst] *= (1 + change)
            
            # Random spread change
            spread_change = random.gauss(0, 0.1)
            self.spreads[inst] = max(0.5, min(5.0, self.spreads[inst] + spread_change))
            
            # Store history (keep last 20)
            self.history[inst].append(self.prices[inst])
            if len(self.history[inst]) > 20:
                self.history[inst].pop(0)
        
        # Keep timestamps consistent
        if len(self.timestamps) > 20:
            self.timestamps.pop(0)
    
    def display_terminal(self):
        """Display text information in terminal"""
        # Clear screen
        print("\033[H\033[J")
        
        print("=" * 70)
        print(f"RATES MARKET MAKING - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)
        
        print("\n📈 CURRENT QUOTES")
        print("-" * 70)
        print(f"{'Instrument':<10} {'Bid':<12} {'Ask':<12} {'Spread':<10} {'Change':<10}")
        print("-" * 70)
        
        for inst in self.instruments:
            price = self.prices[inst]
            spread = self.spreads[inst]
            spread_price = spread / 10000
            
            bid = price - spread_price/2
            ask = price + spread_price/2
            
            # Calculate % change
            if len(self.history[inst]) > 1:
                change = ((self.history[inst][-1] / self.history[inst][-2]) - 1) * 10000
                change_str = f"{change:+.1f}bps"
                if change > 0:
                    change_str = f"\033[92m{change_str}\033[0m"  # Green
                elif change < 0:
                    change_str = f"\033[91m{change_str}\033[0m"  # Red
            else:
                change_str = "0.0bps"
            
            print(f"{inst:<10} {bid:.6f}    {ask:.6f}    {spread:.2f}bps    {change_str}")
        
        # Arbitrage detection
        print("\n🎯 ARBITRAGE CHECK")
        print("-" * 70)
        basis_10y = (self.prices['US10Y'] - self.prices['TYU4']) * 10000
        basis_5y = (self.prices['US5Y'] - self.prices['FVU4']) * 10000
        
        if abs(basis_10y) > 1.5:
            color = "\033[93m"  # Yellow for warning
            action = "SELL BOND / BUY FUTURE" if basis_10y > 0 else "BUY BOND / SELL FUTURE"
            print(f"{color}10Y Basis: {basis_10y:+.2f}bps → {action}\033[0m")
        else:
            print(f"10Y Basis: {basis_10y:+.2f}bps (normal)")
        
        if abs(basis_5y) > 1.5:
            color = "\033[93m"
            action = "SELL BOND / BUY FUTURE" if basis_5y > 0 else "BUY BOND / SELL FUTURE"
            print(f"{color}5Y Basis: {basis_5y:+.2f}bps → {action}\033[0m")
        else:
            print(f"5Y Basis: {basis_5y:+.2f}bps (normal)")
        
        print("\n" + "=" * 70)
        print("Press Ctrl+C to exit | Auto-refresh every 2 seconds")
        print("=" * 70)
    
    def update_charts(self):
        """Update matplotlib charts"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Chart 1: Prices
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, inst in enumerate(self.instruments):
            if self.history[inst]:
                self.ax1.plot(range(len(self.history[inst])), self.history[inst],
                            label=inst, color=colors[idx], linewidth=2)
        
        self.ax1.set_title('Price Movement (Last 20 Updates)', fontweight='bold')
        self.ax1.set_xlabel('Update Count')
        self.ax1.set_ylabel('Price')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Chart 2: Spreads
        x_pos = np.arange(len(self.instruments))
        spreads = [self.spreads[inst] for inst in self.instruments]
        bars = self.ax2.bar(x_pos, spreads, color=colors, alpha=0.7)
        
        # Color bars based on spread value
        for i, bar in enumerate(bars):
            if spreads[i] > 3.0:
                bar.set_color('#e74c3c')  # Red for wide spreads
            elif spreads[i] < 1.5:
                bar.set_color('#27ae60')  # Green for tight spreads
        
        self.ax2.set_title('Current Bid-Ask Spreads (bps)', fontweight='bold')
        self.ax2.set_xlabel('Instrument')
        self.ax2.set_ylabel('Spread (bps)')
        self.ax2.set_xticks(x_pos)
        self.ax2.set_xticklabels(self.instruments)
        
        # Add value labels on bars
        for bar, spread in zip(bars, spreads):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                         f'{spread:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Update layout
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to update
    
    def run(self):
        """Main loop"""
        try:
            print("Starting Simple Rates Dashboard...")
            print("Terminal updates + Matplotlib charts")
            print("Close matplotlib window to stop\n")
            
            while plt.fignum_exists(self.fig.number):
                self.update_prices()
                self.display_terminal()
                self.update_charts()
                time.sleep(2)  # Update every 2 seconds
        
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
        finally:
            plt.close('all')

# Run the dashboard
if __name__ == "__main__":
    dashboard = SimpleRatesDashboard()
    dashboard.run()
